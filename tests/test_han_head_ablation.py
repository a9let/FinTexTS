"""CPU head shapes, residual identity, causal masks, gradients and normal runner."""

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

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting_task'))
from forecasting_task import run
from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS, build_text_model
from forecasting_task.models.HANHeadAblation import ABLATION_MODES, head_ablation_structure
from forecasting_task.experiment_evidence import common_initialization, state_fingerprint
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset
from forecasting_task.gate_diagnostics import write_gate_validation
import test_context_novelty as fixtures
import test_group_gru_attention as core
import run_han_head_ablation_queue as queue
import analyze_context_price_han_head_ablation as analysis
from analyze_persistence_baseline import Run


def config(mode, pred_len=3):
    return dict(hidden_size=64, pred_len=pred_len, num_layers=1, group_gate='context_price_only',
                text_temporal_pool=mode, temporal_pool_structure=head_ablation_structure(mode, pred_len))


class HeadAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self, batch=4, pred_len=3):
        x, e, price = torch.randn(batch, 25, 4), torch.randn(batch, 25, 5, 384), torch.randn(batch, pred_len, 4)
        return x, e, price, dict(price_window=x, novelty=torch.rand(batch, 5), pair_available=torch.ones(batch, 5))

    def test_shapes_head_structures_masks_and_date_sums(self):
        for mode in ABLATION_MODES:
            for batch in (1, 4):
                for pred_len in (1, 3):
                    model = build_text_model(config(mode, pred_len), 'context_price_only')
                    x, e, price, kwargs = self.inputs(batch, pred_len)
                    e[:, :, 2] = 0
                    output, details = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kwargs)
                    self.assertEqual(output.shape, (batch, pred_len, 4))
                    self.assertTrue((details['contributions'][:, 2] == 0).all())
                    if mode == 'han_h32_mlp':
                        self.assertNotIn('date_contributions', details)
                        self.assertIsInstance(model.group_mlps['macro'][1], nn.GELU)
                        self.assertEqual(model.group_mlps['macro'][0].in_features, 64)
                        self.assertEqual(model.group_mlps['macro'][0].out_features, 16)
                        self.assertFalse(any(isinstance(m, nn.Dropout) for m in model.modules()))
                    else:
                        torch.testing.assert_close(details['date_contributions'].sum(1), details['contributions'], rtol=1e-5, atol=1e-6)
                    if mode in ('han_h32_linear', 'han_h16_linear'):
                        dimension = 64 if mode == 'han_h32_linear' else 32
                        self.assertEqual(model.group_mlps['macro'][0].in_features, dimension)
                        self.assertEqual(details['temporal_states'].shape[-1], dimension)
                    empty = model(price, x[:, -1], torch.zeros_like(e[:, -1]), text_window=torch.zeros_like(e), **kwargs)
                    torch.testing.assert_close(empty, .9 * price + .1 * x[:, -1:])

    def test_common_initialization_batches_b_d_encoder_and_e_a_identity(self):
        for seed in (7, 17, 27):
            collected = {}
            for mode in ABLATION_MODES:
                run.set_seed(seed)
                film = fixtures.ContextNoveltyTests.make_film()
                text = build_text_model(config(mode), 'context_price_only')
                rng = torch.get_rng_state().clone()
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True, drop_last=True)
                run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([b[0] for b in loader]) for _ in range(3)]
                x, e, price, kwargs = self.inputs()
                e[0] = 0
                e[1, -1] = 0
                output = text(price, x[:, -1], e[:, -1], text_window=e, **kwargs)
                collected[mode] = common_initialization(film, text), rng, orders, output.detach(), text
            reference = collected['last_matched']
            for mode, actual in collected.items():
                for key in ('film_sha256', 'price_gate_sha256'):
                    self.assertEqual(actual[0][key], reference[0][key])
                torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
                for order, expected in zip(actual[2], reference[2]):
                    torch.testing.assert_close(order, expected, rtol=0, atol=0)
            a, b, d, e = (collected[mode] for mode in ('last_matched', 'han_h32_linear', 'han_h32_mlp', 'last_han_h32_residual'))
            self.assertEqual(state_fingerprint(b[4].temporal_pool), state_fingerprint(d[4].temporal_pool))
            self.assertEqual(a[0]['heads_sha256'], e[0]['heads_sha256'])
            torch.testing.assert_close(a[3], e[3], rtol=0, atol=0)
            for mode in ABLATION_MODES[1:]:
                self.assertEqual(state_fingerprint(collected[mode][4].temporal_pool.projection), state_fingerprint(b[4].temporal_pool.projection))

    def test_residual_excludes_t_preserves_positions_and_masks_bias(self):
        model = build_text_model(config('last_han_h32_residual'), 'context_price_only')
        x, e, price, kwargs = self.inputs(batch=1)
        e.zero_()
        e[:, -20, 0, 0] = 1
        e[:, -2, 0, 0] = 2
        for attention in model.temporal_pool.attentions.values():
            nn.init.zeros_(attention.transform.weight)
            nn.init.zeros_(attention.transform.bias)
            with torch.no_grad():
                attention.position.copy_(torch.arange(20) / 5)
        for head in model.history_heads.values():
            nn.init.constant_(head.bias, 2)
        output, detail = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kwargs)
        torch.testing.assert_close(detail['temporal_beta'][0, [0, 18], 0], (torch.tensor([0., 18.]) / 10).softmax(0))
        self.assertTrue((detail['temporal_beta'][:, -1] == 0).all())
        self.assertFalse(detail['current_present'].any())
        torch.testing.assert_close(output, .9 * price + .1 * x[:, -1:] + .04)
        changed = e.clone()
        changed[:, -1] = 100
        _, other = model(price, x[:, -1], changed[:, -1], text_window=changed, return_details=True, **kwargs)
        torch.testing.assert_close(detail['temporal_states'], other['temporal_states'], rtol=0, atol=0)
        torch.testing.assert_close(detail['history_contributions'], other['history_contributions'], rtol=0, atol=0)
        only_today = torch.zeros_like(e)
        only_today[:, -1] = 1
        _, detail = model(price, x[:, -1], only_today[:, -1], text_window=only_today, return_details=True, **kwargs)
        self.assertTrue((detail['history_contributions'] == 0).all())
        self.assertTrue((detail['temporal_beta'] == 0).all())

    def test_zero_residual_head_can_learn_and_encoder_gradients_follow(self):
        model = build_text_model(config('last_han_h32_residual'), 'context_price_only')
        self.assertTrue(all((p == 0).all() for p in model.history_heads.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        x, e, price, kwargs = self.inputs()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(price, x[:, -1], e[:, -1], text_window=e, **kwargs)
            output.square().mean().backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.history_heads.parameters()), 0)
            encoder_gradient = sum(p.grad.abs().sum().item() for p in model.temporal_pool.parameters() if p.grad is not None)
            if step == 0:
                self.assertEqual(encoder_gradient, 0)
            else:
                self.assertGreater(encoder_gradient, 0)
            optimizer.step()

    def test_runner_checkpoint_diagnostics_resumability_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            core.frame().to_parquet(data / 'SYN.parquet', index=False)
            inspected = {}
            for mode in ABLATION_MODES:
                destination = root / mode
                output = destination / 'output'
                datasets = {flag: ContextNoveltyDataset(Dataset_Custom(20, 2, 3, flag, list(TEXT_GROUPS), root_path=str(data), data_path='SYN.parquet')) for flag in ('train', 'val', 'test')}
                loaders = {flag: DataLoader(ds, batch_size=4, shuffle=flag == 'train', drop_last=flag == 'train') for flag, ds in datasets.items()}
                def factory(**kwargs):
                    return datasets[kwargs['flag']], loaders[kwargs['flag']]
                argv = ['--model_type', 'FiLM', '--group_gate', 'context_price_only', '--text_temporal_pool', mode,
                        '--root_path', str(data), '--data_path', 'SYN.parquet', '--code_commit', 'a' * 40,
                        '--seq_len', '20', '--label_len', '2', '--batch_size', '4', '--e_layers', '1', '--num_epoch', '1',
                        '--patience', '10', '--logdir', str(output)]
                with patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader', side_effect=factory), \
                     patch('torch.cuda.is_available', return_value=False), patch('forecasting_task.models.FiLM.device', torch.device('cpu')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                saved = torch.load(output / 'best_checkpoint.pt', weights_only=True)
                settings = dict(queue.SETTINGS, seq_len=20, label_len=2, batch_size=4, e_layers=1)
                common = dict(analysis.han.prior.COMMON, seq_len=20, label_len=2, batch_size=4, num_epoch=1)
                with patch.object(queue, 'SETTINGS', settings), patch.object(analysis.han.prior, 'COMMON', common):
                    expected = queue.expected_config(mode, 'SYN', 7, 1, 'a' * 40, data)
                    marker = queue.completion(destination, expected, elapsed=2.)
                    self.assertEqual(queue.completion(destination, expected), marker)
                    inspected[mode] = analysis.inspect(Run(mode, 'SYN', 7, output), analysis.han.prior.ReplayCache(data))
                    self.assertTrue(inspected[mode][0]['comparable'], inspected[mode][0])
                with patch('forecasting_task.models.FiLM.device', torch.device('cpu')):
                    film = run.FiLM(SimpleNamespace(**saved['args'])).eval()
                film.load_state_dict(saved['model_state_dict'])
                text = build_text_model(saved['args']['text_model_config'], 'context_price_only').eval()
                text.load_state_dict(saved['text_model_state_dict'])
                with tempfile.TemporaryDirectory() as repeated:
                    args = SimpleNamespace(**{**saved['args'], 'logdir': repeated, 'diagnostic_best_val_mse': saved['metrics']['val_mse']})
                    write_gate_validation(loaders['val'], datasets['val'], film, text, args, torch.device('cpu'), run._prepare_batch)
                    for name in ('validation_group_diagnostics.csv', 'validation_text_dates.csv', 'validation_prediction_components.csv'):
                        self.assertEqual((Path(repeated) / name).read_bytes(), (output / name).read_bytes())
                with (output / 'validation_text_dates.csv').open(newline='') as handle:
                    columns = next(csv.reader(handle))
                self.assertEqual('date_contribution_h1' in columns, mode != 'han_h32_mlp')
                if mode == 'last_han_h32_residual':
                    self.assertIn('history_date_contribution_h1', columns)
                    self.assertIn('current_contribution_h1', columns)
                altered = dict(expected, lr=.002)
                with self.assertRaisesRegex(ValueError, 'differs'):
                    queue.completion(destination, altered)
                (output / 'history.jsonl').write_text('preserved incomplete artifact')
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    queue.completion(destination, expected)
            for mode in ABLATION_MODES[1:]:
                analysis.compare_ablations(inspected['last_matched'], inspected[mode])
            analysis.compare_encoders(inspected)


if __name__ == '__main__':
    unittest.main()
