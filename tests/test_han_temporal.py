"""HAN equations, masked BiGRU, common initialization and checkpoint diagnostics."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting_task'))
from forecasting_task import run
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model
from forecasting_task.models.GroupHANTemporal import HAN_MODES, HANDateAttention, masked_bigru
from forecasting_task.experiment_evidence import common_initialization
from forecasting_task.gate_diagnostics import write_gate_validation
from forecasting_task.gru_diagnostics import temporal_records
from analyze_persistence_baseline import Run
import analyze_context_price_han_temporal as analysis
import test_group_gru_attention as core
import test_context_novelty as fixtures


class HANTemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self, batch=4):
        x, e = torch.randn(batch, 25, 4), torch.randn(batch, 25, 5, 384)
        return x, e, torch.randn(batch, 3, 4), dict(price_window=x, novelty=torch.rand(batch, 5), pair_available=torch.ones(batch, 5))

    def test_paper_sigmoid_position_equation_and_uniform(self):
        attention = HANDateAttention()
        states = torch.randn(2, 20, 64)
        mask = torch.zeros(2, 20, dtype=torch.bool)
        mask[0, [2, 7, 19]] = True
        with torch.no_grad():
            attention.position.copy_(torch.arange(20) / 3)
        expected_scores = torch.sigmoid(attention.transform(states).squeeze(-1)) * attention.position
        actual = attention(states, mask)
        torch.testing.assert_close(actual[0, mask[0]], expected_scores[0, mask[0]].softmax(0))
        self.assertTrue((actual[~mask] == 0).all())
        # A fixed sigmoid output makes the original-position indexing explicit.
        with torch.no_grad():
            attention.transform.weight.zero_()
            attention.transform.bias.zero_()
        torch.testing.assert_close(attention(states, mask)[0, mask[0]], (.5 * attention.position[[2, 7, 19]]).softmax(0))
        x, e, price, kw = self.inputs()
        e[:, -19:, 2] = 0
        e[:, -20, 2, 0] = 1
        uniform = build_text_model(core.config(HAN_MODES[1]), 'context_price_only')
        _, detail = uniform(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
        learned = build_text_model(core.config(HAN_MODES[0]), 'context_price_only')
        learned.load_state_dict(uniform.state_dict())
        _, learned_details = learned(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
        torch.testing.assert_close(learned_details['temporal_states'], detail['temporal_states'], rtol=0, atol=0)
        torch.testing.assert_close(detail['temporal_beta'][:, :, 0], torch.full((4, 20), .05))
        self.assertTrue((detail['temporal_beta'][:, 0, 2] == 1).all())

    def test_both_directions_skip_missing_and_group_batch_isolation(self):
        torch.manual_seed(17)
        model = build_text_model(core.config(HAN_MODES[0]), 'context_price_only')
        x, e, price, kw = self.inputs()
        e[0, -20:] = 0
        e[1, -19:-7, 0] = 0
        _, detail = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
        values = model.temporal_pool.projection(e[:, -20:])
        for group_index, group in enumerate(TEXT_GROUPS):
            gru = model.temporal_pool.grus[group]
            self.assertTrue(gru.bidirectional)
            self.assertEqual(gru.num_layers, 1)
            self.assertEqual(gru.input_size, 32)
            for sample in range(4):
                mask = detail['temporal_present'][sample, :, group_index]
                self.assertTrue((detail['temporal_states'][sample, ~mask, group_index] == 0).all())
                if mask.any():
                    expected, _ = gru(values[sample, mask, group_index][None])
                    torch.testing.assert_close(detail['temporal_states'][sample, mask, group_index], expected[0], rtol=1e-5, atol=1e-7)
        changed = e.clone()
        changed[:, :-20] += 1000
        changed[0, -20:, 0] = 100
        _, other = model(price, x[:, -1], changed[:, -1], text_window=changed, return_details=True, **kw)
        # Changing packed batch lengths changes floating-point GEMM rounding.
        torch.testing.assert_close(detail['temporal_states'][1:], other['temporal_states'][1:], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(detail['temporal_states'][0, :, 1:], other['temporal_states'][0, :, 1:], rtol=0, atol=0)
        original_states = detail['temporal_states'].detach().clone()
        with torch.no_grad():
            model.temporal_pool.grus['macro'].weight_ih_l0_reverse.add_(2)
        _, other = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
        torch.testing.assert_close(original_states[:, :, 1:], other['temporal_states'][:, :, 1:], rtol=0, atol=0)
        # Price context controls group weights, while HAN date weights depend on text.
        _, different_price = model(price, x[:, -1] + 100, e[:, -1], text_window=e,
                                   price_window=x + torch.randn_like(x), novelty=kw['novelty'], pair_available=kw['pair_available'], return_details=True)
        torch.testing.assert_close(other['temporal_beta'], different_price['temporal_beta'], rtol=0, atol=0)

    def test_empty_single_history_bias_gradients_and_initial_states(self):
        reference = None
        for mode in HAN_MODES:
            torch.manual_seed(7)
            model = build_text_model(core.config(mode), 'context_price_only')
            if reference is None:
                reference = {key: value.clone() for key, value in model.state_dict().items()}
            else:
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
            x, e, price, kw = self.inputs(batch=1)
            e.zero_()
            e[:, -9, 0, 0] = 1
            for projection in model.temporal_pool.output_projections.values():
                torch.nn.init.constant_(projection.bias, 2)
            for head in model.group_mlps.values():
                torch.nn.init.constant_(head[0].bias, 3)
            output, detail = model(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
            self.assertTrue(detail['present'][0, 0])
            self.assertEqual(detail['temporal_beta'][0, 11, 0].item(), 1)
            self.assertTrue((detail['contributions'][:, 1:] == 0).all())
            self.assertTrue((detail['pooled_embeddings'][:, 1:] == 0).all())
            torch.testing.assert_close(detail['date_contributions'].sum(1), detail['contributions'], rtol=1e-5, atol=1e-6)
            absent = model(price, x[:, -1], e[:, -1], text_window=torch.zeros_like(e), **kw)
            torch.testing.assert_close(absent, .9 * price + .1 * x[:, -1:])
            self.assertFalse(torch.equal(output, absent))
        for mode in HAN_MODES:
            model = build_text_model(core.config(mode), 'context_price_only')
            x, e, price, kw = self.inputs()
            output = model(price, x[:, -1], e[:, -1], text_window=e, **kw)
            output.square().mean().backward()
            for key, parameter in model.temporal_pool.named_parameters():
                if mode == HAN_MODES[1] and key.startswith('attentions.'):
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertIsNotNone(parameter.grad, key)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, key)

    def test_common_initialization_rng_batches_old_modes_and_checkpoint(self):
        for seed in (7, 17, 27):
            reference = None
            for mode in ('last', 'group_gru_attention', *HAN_MODES):
                run.set_seed(seed)
                film = fixtures.ContextNoveltyTests.make_film()
                text = build_text_model(core.config(mode), 'context_price_only')
                rng = torch.get_rng_state().clone()
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True, drop_last=True)
                run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([b[0] for b in loader]) for _ in range(3)]
                current = common_initialization(film, text), rng, orders
                if reference is None:
                    reference = current
                else:
                    self.assertEqual(current[0], reference[0])
                    torch.testing.assert_close(current[1], reference[1], rtol=0, atol=0)
                    for order, expected in zip(orders, reference[2]):
                        torch.testing.assert_close(order, expected, rtol=0, atol=0)
                if mode in HAN_MODES:
                    x, e, price, kw = self.inputs()
                    text.eval()
                    output, detail = text(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'checkpoint.pt'
                        torch.save({'config': core.config(mode), 'state': text.state_dict()}, path)
                        saved = torch.load(path, weights_only=True)
                        restored = build_text_model(saved['config'], 'context_price_only').eval()
                        restored.load_state_dict(saved['state'])
                        actual, details = restored(price, x[:, -1], e[:, -1], text_window=e, return_details=True, **kw)
                        torch.testing.assert_close(actual, output, rtol=0, atol=0)
                        torch.testing.assert_close(details['date_contributions'], detail['date_contributions'], rtol=0, atol=0)

    def test_runner_saved_evidence_best_diagnostics_and_strict_matching(self):
        data = core.frame()
        runs = []
        with tempfile.TemporaryDirectory() as directory:
            for mode in (*HAN_MODES, 'last'):
                root = Path(directory) / mode
                datasets = {flag: fixtures.make_dataset(data, flag, seq_len=20) for flag in ('train', 'val', 'test')}
                loaders = {flag: DataLoader(ds, batch_size=4, shuffle=flag == 'train', drop_last=flag == 'train') for flag, ds in datasets.items()}
                def factory(**kwargs):
                    return datasets[kwargs['flag']], loaders[kwargs['flag']]
                argv = ['--model_type', 'FiLM', '--group_gate', 'context_price_only', '--text_temporal_pool', mode,
                        '--code_commit', 'a' * 40, '--seq_len', '20', '--label_len', '2', '--pred_len', '3',
                        '--batch_size', '4', '--e_layers', '1', '--num_epoch', '1', '--patience', '10',
                        '--data_path', 'SYN.parquet', '--logdir', str(root)]
                with patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader', side_effect=factory), \
                     patch('torch.cuda.is_available', return_value=False), patch('forecasting_task.models.FiLM.device', torch.device('cpu')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                summary = json.loads((root / 'summary.json').read_text())
                saved = torch.load(root / 'best_checkpoint.pt', weights_only=True)
                self.assertEqual(summary['train_batch_orders'], saved['train_batch_orders'])
                self.assertEqual(summary['common_initialization'], saved['common_initialization'])
                self.assertEqual(summary['temporal_initialization_sha256'], saved['temporal_initialization_sha256'])
                controlled = dict(analysis.prior.COMMON, seq_len=20, label_len=2, batch_size=4, num_epoch=1)
                with patch('pandas.read_parquet', return_value=data), patch.object(analysis.prior, 'COMMON', controlled):
                    result = analysis.inspect(Run(mode, 'SYN', 7, root), analysis.prior.ReplayCache(root))
                self.assertTrue(result[0]['comparable'], result[0])
                if runs:
                    self.assertEqual(analysis.compare_evidence(runs[0], result), 1)
                runs.append(result)
                if mode in HAN_MODES:
                    with patch('forecasting_task.models.FiLM.device', torch.device('cpu')):
                        film = run.FiLM(SimpleNamespace(**saved['args'])).eval()
                    film.load_state_dict(saved['model_state_dict'])
                    text = build_text_model(saved['args']['text_model_config'], 'context_price_only').eval()
                    text.load_state_dict(saved['text_model_state_dict'])
                    with tempfile.TemporaryDirectory() as repeated:
                        args = SimpleNamespace(**{**saved['args'], 'logdir': repeated, 'diagnostic_best_val_mse': summary['best_val_mse']})
                        write_gate_validation(loaders['val'], datasets['val'], film, text, args, torch.device('cpu'), run._prepare_batch)
                        for filename in ('validation_group_diagnostics.csv', 'validation_text_dates.csv', 'validation_prediction_components.csv'):
                            self.assertEqual((Path(repeated) / filename).read_bytes(), (root / filename).read_bytes())


if __name__ == '__main__':
    unittest.main()
