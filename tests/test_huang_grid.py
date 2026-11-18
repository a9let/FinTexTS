"""Parameterized Huang, fixed feature width, legacy checkpoints and dynamic diagnostics."""

import contextlib
import csv
import io
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import test_huang_query as core
from forecasting_task.models.GroupHuangQuery import HUANG_DEFAULTS, HUANG_MODES, huang_options, huang_structure
from forecasting_task.models.GroupText import build_text_model, TEXT_GROUPS
from forecasting_task.experiment_evidence import common_initialization, state_fingerprint
from forecasting_task.gate_diagnostics import write_gate_validation
import run_han_head_ablation_queue as queue
import run_huang_grid_queue as grid
import analyze_context_price_huang_grid as analysis


ROOT = Path(__file__).resolve().parents[1]
OLD_COMMIT = '35ce283'


def config(hidden=16, steps=5, lookback=10, mode=HUANG_MODES[0]):
    parameters = dict(huang_hidden_size=hidden, huang_recurrent_steps=steps, huang_attention_lookback=lookback)
    return {**core.config(mode), **parameters, 'temporal_pool_structure': huang_structure(mode, **parameters)}


class HuangGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_positive_integer_options_and_price_window_fit(self):
        for key in HUANG_DEFAULTS:
            for bad in (0, -1, 1.5, True, '16', float('inf')):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    huang_options({key: bad})
        self.assertEqual(huang_options({}), HUANG_DEFAULTS)
        with tempfile.TemporaryDirectory() as directory:
            base = ['--model_type', 'FiLM', '--group_gate', 'context_price_only', '--text_temporal_pool', HUANG_MODES[0],
                    '--code_commit', 'a' * 40, '--logdir', str(Path(directory) / 'output')]
            for flags in (['--huang_hidden_size', '0'], ['--huang_recurrent_steps', '-1'],
                          ['--huang_attention_lookback', '0'], ['--huang_recurrent_steps', '55'],
                          ['--huang_hidden_size', '16.5']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    core.core.run.parse_args(base + flags)
            # A short recurrent block needs its actual lookback, not a hard-coded 30 rows.
            args = core.core.run.parse_args(base + ['--seq_len', '15', '--huang_recurrent_steps', '5'])
            self.assertEqual(args.temporal_pool_structure['rows'], 15)

    def test_h16_shapes_five_step_causality_masks_and_checkpoint(self):
        for mode in HUANG_MODES:
            for batch in (1, 4):
                model = build_text_model(config(mode=mode), 'context_price_only').eval()
                values = core.inputs(batch)
                values[1][:, :, 1] = 0
                for group in model.temporal_pool.groups.values():
                    with torch.no_grad():
                        group.head.weight.normal_()
                        group.head.bias.fill_(2)
                output, detail = core.forward(model, values)
                self.assertEqual(output.shape, (batch, 3, 4))
                self.assertEqual(detail['huang_states'].shape, (batch, 5, 5, 16))
                self.assertEqual(detail['huang_memories'].shape, (batch, 5, 5, 16))
                self.assertEqual(detail['huang_queries'].shape, (batch, 5, 5, 32))
                self.assertEqual(detail['huang_beta'].shape, (batch, 5, 5, 10))
                self.assertTrue((detail['history_contributions'][:, 1] == 0).all())
                group = model.temporal_pool.groups['macro']
                self.assertEqual(group.U.out_features, 32)
                self.assertEqual(group.cell.input_size, 64)
                self.assertEqual(group.head.in_features, 16)
                if mode == HUANG_MODES[0]:
                    self.assertEqual((group.query.in_features, group.query.out_features), (16, 32))
                changed = [values[0], values[1].clone(), values[2], values[3]]
                changed[1][:, :49] = 999  # First used history is t-14, input row49.
                torch.testing.assert_close(core.forward(model, changed)[0], output, rtol=0, atol=0)
                changed[1] = values[1].clone()
                changed[1][:, 61:] = 999
                _, future = core.forward(model, changed)
                for key in ('huang_states', 'huang_memories', 'huang_queries', 'huang_beta'):
                    torch.testing.assert_close(detail[key][:, :2], future[key][:, :2], rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'checkpoint.pt'
                    torch.save({'config': config(mode=mode), 'state': model.state_dict()}, path)
                    saved = torch.load(path, weights_only=True)
                    restored = build_text_model(saved['config'], 'context_price_only').eval()
                    restored.load_state_dict(saved['state'], strict=True)
                    torch.testing.assert_close(core.forward(restored, values)[0], output, rtol=0, atol=0)

    def test_initial_last_identity_common_rng_batches_and_same_huang_at_equal_h(self):
        for seed in (7, 17, 27):
            records = {}
            for hidden, steps in ((32, 20), (32, 5), (16, 20), (16, 5)):
                core.core.run.set_seed(seed)
                film = core.core.fixtures.ContextNoveltyTests.make_film()
                text = build_text_model(config(hidden, steps), 'context_price_only').eval()
                rng = torch.get_rng_state().clone()
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True, drop_last=True)
                core.core.run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([b[0] for b in loader]) for _ in range(3)]
                values = core.inputs()
                values[1][0] = 0
                values[1][1, -1] = 0
                actual = core.forward(text, values, details=False)
                last = build_text_model(core.core.config('last_matched'), 'context_price_only').eval()
                last.group_mlps.load_state_dict(text.group_mlps.state_dict())
                last.gate.load_state_dict(text.gate.state_dict())
                torch.testing.assert_close(core.forward(last, values, details=False), actual, rtol=0, atol=0)
                records[hidden, steps] = common_initialization(film, text), rng, orders, state_fingerprint(text.temporal_pool)
            reference = records[32, 20]
            for record in records.values():
                self.assertEqual(record[0], reference[0])
                torch.testing.assert_close(record[1], reference[1], rtol=0, atol=0)
                for a, b in zip(record[2], reference[2]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            for hidden in (16, 32):
                self.assertEqual(records[hidden, 5][3], records[hidden, 20][3])

    def test_frozen_legacy_source_default_forward_state_and_old_checkpoint(self):
        # Compare to the actual parent implementation, with nonzero trained-like heads.
        try:
            source = subprocess.check_output(['git', 'show', f'{OLD_COMMIT}:forecasting_task/models/GroupHuangQuery.py'], cwd=ROOT)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest('Parent Git source unavailable; legacy reference check needs commit 35ce283')
        namespace = {'__name__': 'frozen_legacy_huang'}
        exec(compile(source, 'legacy_GroupHuangQuery.py', 'exec'), namespace)
        for mode in HUANG_MODES:
            legacy_config = core.config(mode)
            self.assertEqual(legacy_config['temporal_pool_structure'], namespace['huang_structure'](mode))
            torch.manual_seed(17)
            old = namespace['GroupHuangQueryTextModel'](**legacy_config).eval()
            for group in old.temporal_pool.groups.values():
                with torch.no_grad():
                    group.head.weight.normal_()
                    group.head.bias.normal_()
            values = core.inputs()
            values[1][0, :, 2] = 0
            values[1][1, -1] = 0
            expected, previous = core.forward(old, values)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'old.pt'
                torch.save({'args': {'text_model_config': legacy_config}, 'text_model_state_dict': old.state_dict()}, path)
                saved = torch.load(path, weights_only=True)
                new = build_text_model(saved['args']['text_model_config'], 'context_price_only').eval()
                new.load_state_dict(saved['text_model_state_dict'], strict=True)
                actual, detail = core.forward(new, values)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for key in previous:
                    torch.testing.assert_close(detail[key], previous[key], rtol=0, atol=0)

    def test_variable_diagnostics_normal_runner_and_saved_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            core.core.core.frame().to_parquet(data / 'SYN.parquet', index=False)
            inspected = {}
            for label, configuration in grid.CONFIGURATIONS.items():
                output = root / label / 'output'
                datasets = {flag: core.core.ContextNoveltyDataset(core.core.Dataset_Custom(30, 2, 3, flag, list(TEXT_GROUPS),
                            root_path=str(data), data_path='SYN.parquet')) for flag in ('train', 'val', 'test')}
                loaders = {flag: DataLoader(ds, batch_size=4, shuffle=flag == 'train', drop_last=flag == 'train') for flag, ds in datasets.items()}
                def factory(**kwargs):
                    return datasets[kwargs['flag']], loaders[kwargs['flag']]
                argv = ['--model_type', 'FiLM', '--group_gate', 'context_price_only', '--root_path', str(data),
                        '--data_path', 'SYN.parquet', '--code_commit', 'a' * 40, '--seq_len', '30', '--label_len', '2',
                        '--batch_size', '4', '--e_layers', '1', '--num_epoch', '1', '--patience', '10', '--logdir', str(output)]
                for key, value in configuration.items():
                    argv += ['--' + key, str(value)]
                with patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader', side_effect=factory), \
                     patch('torch.cuda.is_available', return_value=False), patch('forecasting_task.models.FiLM.device', torch.device('cpu')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    core.core.run.main(argv)
                saved = torch.load(output / 'best_checkpoint.pt', weights_only=True)
                parameters = huang_options(configuration)
                for source in (saved, saved['args'], saved['args']['text_model_config']):
                    self.assertEqual(huang_options(source), parameters)
                settings = dict(queue.SETTINGS, seq_len=30, label_len=2, batch_size=4, e_layers=1)
                common = dict(analysis.query.heads.han.prior.COMMON, seq_len=30, label_len=2, batch_size=4, num_epoch=1)
                with patch.object(queue, 'SETTINGS', settings), patch.object(analysis.query.heads.han.prior, 'COMMON', common):
                    expected = queue.expected_config(label, 'SYN', 7, 1, 'a' * 40, data, configuration=configuration)
                    queue.completion(output.parent, expected, elapsed=1.)
                    item = analysis.inspect(core.core.Run(label, 'SYN', 7, output), analysis.query.ReplayCache(data))
                    self.assertTrue(item[0]['comparable'], item[0])
                    inspected[label] = item
                with patch('forecasting_task.models.FiLM.device', torch.device('cpu')):
                    film = core.core.run.FiLM(SimpleNamespace(**saved['args'])).eval()
                film.load_state_dict(saved['model_state_dict'])
                text = build_text_model(saved['args']['text_model_config'], 'context_price_only').eval()
                text.load_state_dict(saved['text_model_state_dict'], strict=True)
                with tempfile.TemporaryDirectory() as repeated:
                    args = SimpleNamespace(**{**saved['args'], 'logdir': repeated, 'diagnostic_best_val_mse': saved['metrics']['val_mse']})
                    write_gate_validation(loaders['val'], datasets['val'], film, text, args, torch.device('cpu'), core.core.run._prepare_batch)
                    for name in ('validation_group_diagnostics.csv', 'validation_text_dates.csv', 'validation_prediction_components.csv'):
                        self.assertEqual((Path(repeated) / name).read_bytes(), (output / name).read_bytes())
                with (output / 'validation_text_dates.csv').open(newline='') as handle:
                    dates = list(csv.DictReader(handle))
                steps, lookback = parameters['huang_recurrent_steps'], parameters['huang_attention_lookback']
                self.assertEqual(len(dates), len(datasets['val']) * 5 * steps * lookback)
                self.assertEqual(dates[0]['date'], datasets['val'].row_dates[30 - steps - lookback])
                self.assertTrue(all(r['date'] < r['step_date'] <= r['input_end_date'] for r in dates))
                altered = dict(expected, huang_recurrent_steps=steps + 1)
                with self.assertRaisesRegex(ValueError, 'Huang parameters differ'):
                    queue.completion(output.parent, altered)
            with patch.object(analysis.query.heads.han.prior, 'COMMON', common):
                self.assertEqual(analysis.compare_pair(inspected['h16_s5_l10'], inspected['h16_s20_l10']), 1)


if __name__ == '__main__':
    unittest.main()
