"""CPU regression, causal missing-text GRUs, reconstruction and matched RNG."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting_task'))
from forecasting_task import run
from forecasting_task.gate_diagnostics import write_gate_validation
from forecasting_task.gru_diagnostics import temporal_records
from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS, build_text_model
from forecasting_task.models.GroupGRUAttention import GroupGRUAttentionTextModel, temporal_structure
import test_context_novelty as fixtures
import analyze_context_price_group_gru_attention as analysis
from analyze_persistence_baseline import Run


def config(mode):
    return dict(hidden_size=64, pred_len=3, num_layers=1, used_col_prefixes=list(TEXT_GROUPS),
                group_gate='context_price_only', text_temporal_pool=mode,
                temporal_pool_structure=temporal_structure(mode))


def frame():
    original = fixtures.make_frame()
    dates = [*pd.date_range('2021-10-01', periods=40), *pd.date_range('2022-01-03', periods=12),
             *pd.date_range('2023-01-03', periods=12)]
    result = original.iloc[np.arange(len(dates)) % len(original)].reset_index(drop=True)
    result['date'] = [value.strftime('%Y-%m-%d') for value in dates]
    for channel in ('open', 'high', 'low', 'close'):
        result[channel] += 100 + np.arange(len(result)) * .5
    return result


class GroupGRUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self, batch=4, length=25):
        x = torch.randn(batch, length, 4)
        e = torch.randn(batch, length, 5, 384)
        price = torch.randn(batch, 3, 4)
        return x, e, price, dict(price_window=x, novelty=torch.rand(batch, 5), pair_available=torch.ones(batch, 5))

    def test_last_forward_and_initial_common_modules_rng_batches(self):
        for seed in (7, 17, 27):
            original = None
            for mode in ('legacy', 'last', 'group_gru_attention'):
                run.set_seed(seed)
                film = fixtures.ContextNoveltyTests.make_film()
                text = ContextNoveltyTextModel(64, 3, 1, group_gate='context_price_only') if mode == 'legacy' else build_text_model(config(mode), 'context_price_only')
                rng = torch.get_rng_state().clone()
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True, drop_last=True)
                run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([b[0] for b in loader]) for _ in range(3)]
                x, e, price, kw = self.inputs()
                output = text(price, x[:, -1], e[:, -1], **kw,
                              **({'text_window': e} if mode == 'group_gru_attention' else {}))
                actual = film.state_dict(), text.group_mlps.state_dict(), text.gate.state_dict(), rng, orders, output
                if original is None:
                    original = actual
                else:
                    for state, expected in zip(actual[:3], original[:3]):
                        for key in state:
                            torch.testing.assert_close(state[key], expected[key], rtol=0, atol=0)
                    torch.testing.assert_close(rng, original[3], rtol=0, atol=0)
                    for order, expected in zip(orders, original[4]):
                        torch.testing.assert_close(order, expected, rtol=0, atol=0)
                    if mode == 'last':
                        torch.testing.assert_close(output, original[5], rtol=0, atol=0)

    def test_empty_single_gaps_state_freeze_and_old_history_contributes(self):
        model = build_text_model(config('group_gru_attention'), 'context_price_only')
        for batch in (1, 4):
            x, e, price, kw = self.inputs(batch)
            e.zero_()
            e[:, -18, 0, 0] = 1  # gap is measured at fixed dataset-row positions
            e[:, -11, 0, 1] = 2
            e[:, -5, 1, 2] = 3
            output, detail = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
            beta, present = detail['temporal_beta'], detail['temporal_present']
            torch.testing.assert_close(detail['pooled_embeddings'], (e[:, -20:] * beta[..., None]).sum(1), rtol=0, atol=0)
            self.assertTrue((beta[~present] == 0).all())
            torch.testing.assert_close(beta.sum(1), detail['present'].float())
            self.assertTrue((beta[:, 15, 1] == 1).all())
            self.assertTrue((detail['temporal_gaps'][:, 2, 0] == 0).all())
            torch.testing.assert_close(detail['temporal_gaps'][:, 9, 0], torch.full((batch,), 7 / 20))
            torch.testing.assert_close(detail['temporal_states'][:, 3:9, 0], detail['temporal_states'][:, 2:3, 0].expand(-1, 6, -1), rtol=0, atol=0)
            self.assertTrue((detail['pooled_embeddings'][:, 2:] == 0).all())
            self.assertTrue((detail['contributions'][:, 2:] == 0).all())
            self.assertTrue(detail['present'][:, :2].all())
            self.assertTrue((e[:, -1] == 0).all())
            absent = model(price, x[:, -1], e[:, -1], text_window=torch.zeros_like(e), **kw)
            torch.testing.assert_close(absent, .9 * price + .1 * x[:, -1:])
            self.assertFalse(torch.equal(output, absent))

    def test_group_presence_survives_cancellation_and_masks_head_bias(self):
        model = build_text_model(config('group_gru_attention'), 'context_price_only')
        for attention in model.temporal_pool.attentions.values():
            torch.nn.init.zeros_(attention.vector.weight)
        x, e, price, kw = self.inputs(batch=1)
        e.zero_()
        e[:, -2, 0, 0], e[:, -1, 0, 0] = 1, -1
        for head in model.group_mlps.values():
            torch.nn.init.zeros_(head[0].weight)
            torch.nn.init.constant_(head[0].bias, 2)
        _, detail = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
        self.assertTrue((detail['pooled_embeddings'] == 0).all())
        self.assertTrue(detail['present'][0, 0])
        torch.testing.assert_close(detail['contributions'][0, 0], torch.full((3,), .04))
        self.assertTrue((detail['contributions'][:, 1:] == 0).all())

    def test_no_group_batch_or_outside_window_mixing(self):
        model = build_text_model(config('group_gru_attention'), 'context_price_only').eval()
        x, e, price, kw = self.inputs()
        _, first = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
        changed = e.clone()
        changed[:, :-20] += 10000
        _, same = model(price, x[:, -1], changed[:, -1], text_window=changed, return_details=True, **kw)
        torch.testing.assert_close(first['pooled_embeddings'], same['pooled_embeddings'], rtol=0, atol=0)
        changed[:, -20:, 0] *= -3
        _, other = model(price, x[:, -1], changed[:, -1], text_window=changed, return_details=True, **kw)
        torch.testing.assert_close(first['pooled_embeddings'][:, 1:], other['pooled_embeddings'][:, 1:], rtol=0, atol=0)
        torch.testing.assert_close(first['temporal_states'][:, :, 1:], other['temporal_states'][:, :, 1:], rtol=0, atol=0)
        changed[0] += 20
        _, other = model(price, x[:, -1], changed[:, -1], text_window=changed, return_details=True, **kw)
        torch.testing.assert_close(first['pooled_embeddings'][1:, 1:], other['pooled_embeddings'][1:, 1:], rtol=0, atol=0)
        ids = [{id(p) for p in gru.parameters()} for gru in model.temporal_pool.grus.values()]
        self.assertEqual(sum(map(len, ids)), len(set.union(*ids)))
        with torch.no_grad():
            model.temporal_pool.grus['macro'].weight_ih_l0.add_(1)
        _, altered_gru = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
        torch.testing.assert_close(first['temporal_states'][:, :, 1:], altered_gru['temporal_states'][:, :, 1:], rtol=0, atol=0)
        self.assertFalse(torch.equal(first['temporal_states'][:, :, 0], altered_gru['temporal_states'][:, :, 0]))
        data = frame()
        original = fixtures.make_dataset(data, 'val', seq_len=20)
        future = data.copy()
        future.loc[45:, 'close'] += 1000
        future.loc[45:, 'macro_emb0'] += 1000
        other = fixtures.make_dataset(future, 'val', seq_len=20)
        for index in range(5):
            np.testing.assert_array_equal(original[index][0], other[index][0])
            np.testing.assert_array_equal(original[index][2], other[index][2])

    def test_gradients_checkpoint_and_date_decomposition(self):
        model = build_text_model(config('group_gru_attention'), 'context_price_only')
        x, e, price, kw = self.inputs(batch=4, length=20)
        output, detail = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
        parameters = list(model.parameters())
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertFalse(hasattr(model, 'G'))
        self.assertFalse(model.fixed_alpha.requires_grad)
        output.square().mean().backward()
        for name, parameter in model.temporal_pool.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        torch.optim.Adam(parameters, lr=.001).step()
        model.eval()
        output, detail = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save({'config': config('group_gru_attention'), 'state': model.state_dict()}, path)
            saved = torch.load(path, weights_only=True)
            restored = build_text_model(saved['config'], 'context_price_only').eval()
            restored.load_state_dict(saved['state'], strict=True)
            prediction, restored_details = restored(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
            torch.testing.assert_close(prediction, output, rtol=0, atol=0)
            torch.testing.assert_close(restored_details['date_contributions'], detail['date_contributions'], rtol=0, atol=0)
        dataset = SimpleNamespace(seq_len=20, row_dates=np.array([f'2022-01-{i:02}' for i in range(1, 24)]),
                                  window_end_date=lambda index: f'2022-01-{20 + index:02}')
        groups, dates, components = temporal_records(dataset, list(range(4)), price, x[:, -1], output, detail, model.get_alpha())
        self.assertEqual(len(dates), 4 * 20 * 5)
        self.assertEqual(len(components), 4 * 3 * 4)
        self.assertEqual(len(groups), 4 * 5)

    def test_runner_best_checkpoint_and_matched_artifacts(self):
        data = frame()
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for mode in ('last', 'group_gru_attention'):
                root = Path(directory) / mode
                datasets = {flag: fixtures.make_dataset(data, flag, seq_len=20) for flag in ('train', 'val', 'test')}
                loaders = {flag: DataLoader(dataset, batch_size=4, shuffle=flag == 'train', drop_last=flag == 'train') for flag, dataset in datasets.items()}
                def factory(**kwargs):
                    return datasets[kwargs['flag']], loaders[kwargs['flag']]
                original_epoch = run._run_epoch
                epochs = []
                def epoch(*args, **kwargs):
                    result = original_epoch(*args, **kwargs)
                    if kwargs.get('optimizer') is not None:
                        epochs.append(1)
                        if len(epochs) == 2:
                            with torch.no_grad():
                                for head in args[3].group_mlps.values():
                                    head[0].bias.add_(10000)
                    return result
                argv = ['--model_type', 'FiLM', '--group_gate', 'context_price_only', '--text_temporal_pool', mode,
                        '--code_commit', 'a' * 40, '--seq_len', '20', '--label_len', '2', '--pred_len', '3',
                        '--batch_size', '4', '--e_layers', '1', '--num_epoch', '2', '--patience', '10',
                        '--data_path', 'SYN.parquet', '--logdir', str(root)]
                with patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader', side_effect=factory), \
                     patch.object(run, '_run_epoch', side_effect=epoch), patch('torch.cuda.is_available', return_value=False), \
                     patch('forecasting_task.models.FiLM.device', torch.device('cpu')), contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                summary = json.loads((root / 'summary.json').read_text())
                checkpoint = torch.load(root / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
                self.assertEqual(summary['best_epoch'], 1)
                self.assertEqual(checkpoint['text_temporal_pool'], mode)
                self.assertEqual(checkpoint['window_evidence'], summary['window_evidence'])
                controlled = dict(analysis.COMMON, seq_len=20, label_len=2, batch_size=4, num_epoch=2)
                with patch('pandas.read_parquet', return_value=data), patch.object(analysis, 'COMMON', controlled):
                    inspected = analysis.inspect_run(Run(mode, 'SYN', 7, root), analysis.ReplayCache(root), results[0][1] if results else None)
                self.assertEqual(inspected[0]['window_verification'], 'full', inspected[0])
                self.assertTrue(inspected[0]['comparable'], inspected[0])
                results.append(inspected)
                # Restore using the runner's saved seq_len rather than fixture defaults.
                with patch('forecasting_task.models.FiLM.device', torch.device('cpu')):
                    film = run.FiLM(SimpleNamespace(**checkpoint['args'])).cpu().eval()
                film.load_state_dict(checkpoint['model_state_dict'])
                text = build_text_model(checkpoint['args']['text_model_config'], 'context_price_only').eval()
                text.load_state_dict(checkpoint['text_model_state_dict'])
                with tempfile.TemporaryDirectory() as repeated:
                    diag_args = SimpleNamespace(**{**checkpoint['args'], 'logdir': repeated, 'diagnostic_best_val_mse': summary['best_val_mse']})
                    write_gate_validation(loaders['val'], datasets['val'], film, text, diag_args, torch.device('cpu'), run._prepare_batch)
                    for name in ('validation_group_diagnostics.csv', *(['validation_text_dates.csv', 'validation_prediction_components.csv'] if mode != 'last' else [])):
                        self.assertEqual((Path(repeated) / name).read_bytes(), (root / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
