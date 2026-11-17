"""HAN suffix windows preserve positions, initialization and the old masked forward."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import test_han_head_ablation as core
from forecasting_task.models.HANHeadAblation import HAN_WINDOW_MODES, head_ablation_structure
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model
from forecasting_task.experiment_evidence import common_initialization, state_fingerprint
from forecasting_task.gate_diagnostics import write_gate_validation
import analyze_context_price_han_window as analysis
import run_han_head_ablation_queue as queue


class HANWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masked_han20_equivalence_positions_and_causality(self):
        for batch in (1, 4):
            for mode, rows in zip(HAN_WINDOW_MODES, (5, 10)):
                reference = build_text_model(core.config('han_h16_linear'), 'context_price_only').eval()
                model = build_text_model(core.config(mode), 'context_price_only').eval()
                # Nonconstant positions, gate and biases catch wrong offsets and masking.
                with torch.no_grad():
                    for attention in reference.temporal_pool.attentions.values():
                        attention.position.copy_(torch.linspace(-3, 4, 20))
                    reference.gate.weight.normal_()
                    for head in reference.group_mlps.values():
                        head[0].bias.fill_(2.)
                model.load_state_dict(reference.state_dict(), strict=True)
                x, text, price, kwargs = core.HeadAblationTests().inputs(batch)
                text[:, -rows:, 0] = 0  # Old text exists, but is outside the chosen suffix.
                text[:, -rows::2, 1] = 0
                text[0, -rows:, 2] = 0
                text[0, -1, 2] = 1  # One available date.
                masked = text.clone()
                masked[:, :-rows] = 0
                with torch.no_grad():
                    expected, old = reference(price, x[:, -1], masked[:, -1], text_window=masked, return_details=True, **kwargs)
                    actual, detail = model(price, x[:, -1], text[:, -1], text_window=text, return_details=True, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                for key in ('temporal_beta', 'temporal_present', 'temporal_states', 'date_contributions'):
                    torch.testing.assert_close(detail[key], old[key][:, -rows:], rtol=1e-6, atol=1e-7)
                self.assertEqual(actual.shape, (batch, 3, 4))
                self.assertTrue((detail['contributions'][:, 0] == 0).all())
                torch.testing.assert_close(detail['temporal_beta'][0, -1, 2], torch.tensor(1.))
                self.assertEqual(head_ablation_structure(mode)['original_position_indices'], list(range(20 - rows, 20)))
                self.assertEqual(model.temporal_pool.attentions['macro'].position.numel(), 20)
                changed = text.clone()
                changed[:, :-rows] = torch.randn_like(changed[:, :-rows]) * 100
                torch.testing.assert_close(model(price, x[:, -1], changed[:, -1], text_window=changed, **kwargs), actual, rtol=0, atol=0)
                # Explicit input slicing excludes future rows of an enclosing stream.
                stream = torch.cat((text, torch.randn(batch, 3, 5, 384)), 1)
                stream[:, text.shape[1]:] = 1000
                torch.testing.assert_close(model(price, x[:, -1], stream[:, text.shape[1] - 1],
                                                text_window=stream[:, :text.shape[1]], **kwargs), actual, rtol=0, atol=0)
                empty, empty_detail = model(price, x[:, -1], torch.zeros_like(text[:, -1]), text_window=torch.zeros_like(text), return_details=True, **kwargs)
                torch.testing.assert_close(empty, .9 * price + .1 * x[:, -1:])
                self.assertTrue((empty_detail['temporal_beta'] == 0).all())

    def test_initial_parameters_rng_and_batch_order_match_han20_and_last(self):
        for seed in (7, 17, 27):
            collected = {}
            for mode in ('last_matched', 'han_h16_linear', *HAN_WINDOW_MODES):
                core.run.set_seed(seed)
                film = core.fixtures.ContextNoveltyTests.make_film()
                text = build_text_model(core.config(mode), 'context_price_only')
                rng = torch.get_rng_state().clone()
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True, drop_last=True)
                core.run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([b[0] for b in loader]) for _ in range(3)]
                collected[mode] = common_initialization(film, text), rng, orders, text
            a, base = collected['last_matched'], collected['han_h16_linear']
            for mode, value in collected.items():
                for key in ('film_sha256', 'price_gate_sha256'):
                    self.assertEqual(value[0][key], a[0][key])
                torch.testing.assert_close(value[1], a[1], rtol=0, atol=0)
                for actual, expected in zip(value[2], a[2]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if mode != 'last_matched':
                    self.assertEqual(value[0]['heads_sha256'], base[0]['heads_sha256'])
                    self.assertEqual(state_fingerprint(value[3].temporal_pool), state_fingerprint(base[3].temporal_pool))
                    self.assertEqual(sum(p.numel() for p in value[3].parameters()), sum(p.numel() for p in base[3].parameters()))

    def test_checkpoint_diagnostics_and_normal_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            core.core.frame().to_parquet(data / 'SYN.parquet', index=False)
            inspected = []
            for mode, rows in zip(HAN_WINDOW_MODES, (5, 10)):
                output = root / mode / 'output'
                datasets = {flag: core.ContextNoveltyDataset(core.Dataset_Custom(20, 2, 3, flag, list(TEXT_GROUPS),
                            root_path=str(data), data_path='SYN.parquet')) for flag in ('train', 'val', 'test')}
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
                    core.run.main(argv)
                saved = torch.load(output / 'best_checkpoint.pt', weights_only=True)
                settings = dict(queue.SETTINGS, seq_len=20, label_len=2, batch_size=4, e_layers=1)
                common = dict(analysis.heads.han.prior.COMMON, seq_len=20, label_len=2, batch_size=4, num_epoch=1)
                with patch.object(queue, 'SETTINGS', settings), patch.object(analysis.heads.han.prior, 'COMMON', common):
                    marker = queue.completion(output.parent, queue.expected_config(mode, 'SYN', 7, 1, 'a' * 40, data), elapsed=1.)
                    self.assertEqual(marker['config']['structure']['rows'], rows)
                    item = analysis.inspect(core.Run(mode, 'SYN', 7, output), analysis.ReplayCache(data))
                    self.assertTrue(item[0]['comparable'], item[0])
                    inspected.append(item)
                with patch('forecasting_task.models.FiLM.device', torch.device('cpu')):
                    film = core.run.FiLM(SimpleNamespace(**saved['args'])).eval()
                film.load_state_dict(saved['model_state_dict'])
                text = build_text_model(saved['args']['text_model_config'], 'context_price_only').eval()
                text.load_state_dict(saved['text_model_state_dict'], strict=True)
                with tempfile.TemporaryDirectory() as repeated:
                    args = SimpleNamespace(**{**saved['args'], 'logdir': repeated, 'diagnostic_best_val_mse': saved['metrics']['val_mse']})
                    write_gate_validation(loaders['val'], datasets['val'], film, text, args, torch.device('cpu'), core.run._prepare_batch)
                    for name in ('validation_group_diagnostics.csv', 'validation_text_dates.csv', 'validation_prediction_components.csv'):
                        self.assertEqual((Path(repeated) / name).read_bytes(), (output / name).read_bytes())
                with (output / 'validation_text_dates.csv').open(newline='') as handle:
                    records = list(csv.DictReader(handle))
                self.assertEqual(sorted({int(r['original_position_index']) for r in records}), list(range(20 - rows, 20)))
                val = datasets['val']
                first = [r['date'] for r in records if r['window_index'] == '0' and r['group'] == 'macro']
                self.assertEqual(first, list(val.row_dates[val.seq_len - rows:val.seq_len]))
            with patch.object(analysis.heads.han.prior, 'COMMON', common):
                self.assertEqual(analysis.compare_pair(*inspected), 1)


if __name__ == '__main__':
    unittest.main()
