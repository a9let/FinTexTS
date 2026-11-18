"""Standalone replace mode: no last heads, independent causal groups, CPU restore."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import test_huang_query as core
import test_huang_grid as grid_core
from forecasting_task.models.GroupHuangQuery import HUANG_REPLACE
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model
from forecasting_task.experiment_evidence import common_initialization, state_fingerprint
from forecasting_task.gate_diagnostics import write_gate_validation
import run_han_head_ablation_queue as shared
import run_huang_replace_grid_queue as queue
import analyze_context_price_huang_replace_grid as analysis


def config(hidden=16, steps=5):
    return grid_core.config(hidden=hidden, steps=steps, mode=HUANG_REPLACE)


class FailingLast(nn.Module):
    def forward(self, value):
        raise AssertionError('last head must never be evaluated in replace mode')


class HuangReplaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_no_last_heads_in_state_optimizer_or_forward(self):
        model = build_text_model(config(), 'context_price_only').eval()
        self.assertEqual(len(model.group_mlps), 0)
        self.assertFalse(any(n.startswith('group_mlps.') for n, _ in model.named_parameters()))
        self.assertFalse(any(n.startswith('group_mlps.') for n in model.state_dict()))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        self.assertEqual({id(p) for p in model.parameters()}, {id(p) for g in optimizer.param_groups for p in g['params']})
        model.group_mlps = nn.ModuleDict({g: FailingLast() for g in TEXT_GROUPS})
        values = core.inputs()
        output, detail = core.forward(model, values)
        torch.testing.assert_close(output, .9 * values[2] + .1 * values[0][:, -1:])
        self.assertNotIn('current_contributions', detail)
        self.assertNotIn('history_contributions', detail)
        for group in model.temporal_pool.groups.values():
            with torch.no_grad():
                group.head.weight.normal_()
                group.head.bias.fill_(2)
        output, detail = core.forward(model, values)
        delta = detail['contributions'].sum(1)[..., None]
        torch.testing.assert_close(output, .9 * values[2] + .1 * values[0][:, -1:] + delta)
        # The redundant last-embedding argument does not feed a second prediction head.
        x, text, price, kwargs = values
        other = model(price, x[:, -1], torch.randn_like(text[:, -1]) * 1000, text_window=text, **kwargs)
        torch.testing.assert_close(other, output, rtol=0, atol=0)

    def test_masks_group_independence_and_causality_all_four_pairs(self):
        for hidden in (16, 32):
            for steps in (5, 20):
                model = build_text_model(config(hidden, steps), 'context_price_only').eval()
                values = core.inputs(batch=1 if hidden == 16 else 3)
                values[1][:, :, 1] = 0
                values[1][:, -1, 0] = 0
                for group in model.temporal_pool.groups.values():
                    with torch.no_grad():
                        group.head.weight.normal_()
                        group.head.bias.fill_(2)
                output, detail = core.forward(model, values)
                self.assertEqual(output.shape, (values[0].shape[0], 3, 4))
                self.assertTrue((detail['contributions'][:, 1] == 0).all())
                self.assertTrue((detail['huang_states'][:, :, 1] == 0).all())
                self.assertFalse(detail['input_current_present'][:, 0].any())
                self.assertTrue(detail['huang_present'][:, 0].all())
                self.assertGreater(detail['contributions'][:, 0].abs().sum(), 0)
                changed = [values[0], values[1].clone(), values[2], values[3]]
                changed[1][:, -steps, 0] += 10
                _, other = core.forward(model, changed)
                self.assertGreater((detail['huang_queries'][:, 1, 0] - other['huang_queries'][:, 1, 0]).abs().sum(), 0)
                for key in ('huang_states', 'huang_memories', 'huang_queries', 'huang_beta'):
                    torch.testing.assert_close(detail[key][:, :, 1:], other[key][:, :, 1:], rtol=0, atol=0)
                changed[1] = values[1].clone()
                changed[1][:, -steps + 2:] = 999
                _, future = core.forward(model, changed)
                for key in ('huang_states', 'huang_memories', 'huang_queries', 'huang_beta'):
                    torch.testing.assert_close(detail[key][:, :2], future[key][:, :2], rtol=0, atol=0)
                changed[1] = torch.zeros_like(values[1])
                empty, empty_detail = core.forward(model, changed)
                torch.testing.assert_close(empty, .9 * values[2] + .1 * values[0][:, -1:])
                self.assertTrue((empty_detail['heads'] == 0).all())
                self.assertTrue((empty_detail['huang_beta'] == 0).all())

    def test_common_initialization_batches_and_same_h_huang(self):
        for seed in (7, 17, 27):
            records = {}
            for label, configuration in queue.CONFIGURATIONS.items():
                core.core.run.set_seed(seed)
                film = core.core.fixtures.ContextNoveltyTests.make_film()
                model = build_text_model(config(configuration['huang_hidden_size'], configuration['huang_recurrent_steps']), 'context_price_only')
                rng = torch.get_rng_state().clone()
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True, drop_last=True)
                core.core.run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([b[0] for b in loader]) for _ in range(3)]
                records[label] = common_initialization(film, model), rng, orders, state_fingerprint(model.temporal_pool)
            core.core.run.set_seed(seed)
            film = core.core.fixtures.ContextNoveltyTests.make_film()
            last = build_text_model(core.core.config('last_matched'), 'context_price_only')
            baseline = common_initialization(film, last)
            for record in records.values():
                for key in ('film_sha256', 'price_gate_sha256'):
                    self.assertEqual(record[0][key], baseline[key])
                torch.testing.assert_close(record[1], torch.get_rng_state(), rtol=0, atol=0)
                for a, b in zip(record[2], records['h16_s5_l10'][2]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            for hidden in (16, 32):
                self.assertEqual(records[f'h{hidden}_s5_l10'][3], records[f'h{hidden}_s20_l10'][3])

    def test_checkpoint_diagnostics_normal_runner_all_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            core.core.core.frame().to_parquet(data / 'SYN.parquet', index=False)
            inspected = {}
            for label, configuration in queue.CONFIGURATIONS.items():
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
                    argv.extend(['--' + key, str(value)])
                with patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader', side_effect=factory), \
                     patch('torch.cuda.is_available', return_value=False), patch('forecasting_task.models.FiLM.device', torch.device('cpu')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    core.core.run.main(argv)
                saved = torch.load(output / 'best_checkpoint.pt', weights_only=True)
                self.assertIsNone(saved['initial_prediction_evidence'])
                self.assertFalse(any(n.startswith('group_mlps.') for n in saved['text_model_state_dict']))
                settings = dict(shared.SETTINGS, seq_len=30, label_len=2, batch_size=4, e_layers=1)
                common = dict(analysis.query.heads.han.prior.COMMON, seq_len=30, label_len=2, batch_size=4, num_epoch=1)
                with patch.object(shared, 'SETTINGS', settings), patch.object(analysis.query.heads.han.prior, 'COMMON', common):
                    shared.completion(output.parent, shared.expected_config(label, 'SYN', 7, 1, 'a' * 40, data, configuration=configuration), elapsed=1.)
                    item = analysis.inspect(core.core.Run(label, 'SYN', 7, output), analysis.query.ReplayCache(data))
                    self.assertTrue(item[0]['comparable'], item[0])
                    inspected[label] = item
                with patch('forecasting_task.models.FiLM.device', torch.device('cpu')):
                    film = core.core.run.FiLM(SimpleNamespace(**saved['args'])).eval()
                film.load_state_dict(saved['model_state_dict'])
                model = build_text_model(saved['args']['text_model_config'], 'context_price_only').eval()
                model.load_state_dict(saved['text_model_state_dict'], strict=True)
                self.assertEqual(len(model.group_mlps), 0)
                with tempfile.TemporaryDirectory() as repeated:
                    args = SimpleNamespace(**{**saved['args'], 'logdir': repeated, 'diagnostic_best_val_mse': saved['metrics']['val_mse']})
                    write_gate_validation(loaders['val'], datasets['val'], film, model, args, torch.device('cpu'), core.core.run._prepare_batch)
                    for name in ('validation_group_diagnostics.csv', 'validation_text_dates.csv', 'validation_prediction_components.csv'):
                        self.assertEqual((Path(repeated) / name).read_bytes(), (output / name).read_bytes())
                with (output / 'validation_group_diagnostics.csv').open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                self.assertIn('huang_contribution_h1', rows[0])
                self.assertNotIn('current_contribution_h1', rows[0])
                self.assertNotIn('history_contribution_h1', rows[0])
            with patch.object(analysis.query.heads.han.prior, 'COMMON', common):
                for hidden in (16, 32):
                    self.assertEqual(analysis.compare_pair(inspected[f'h{hidden}_s5_l10'], inspected[f'h{hidden}_s20_l10']), 1)


if __name__ == '__main__':
    unittest.main()
