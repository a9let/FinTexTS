"""Independent news recurrence, causal queries, masks, RNG and CPU checkpoints."""

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

import test_han_head_ablation as core
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model
from forecasting_task.models.GroupHuangQuery import HUANG_MODES, HIDDEN, huang_structure
from forecasting_task.experiment_evidence import common_initialization, state_fingerprint
from forecasting_task.gate_diagnostics import write_gate_validation
import run_han_head_ablation_queue as queue
import analyze_context_price_huang_query as analysis


def config(mode):
    return dict(hidden_size=64, pred_len=3, num_layers=1, group_gate='context_price_only',
                text_temporal_pool=mode, temporal_pool_structure=huang_structure(mode))


def inputs(batch=3):
    x, text, price = torch.randn(batch, 64, 4), torch.randn(batch, 64, 5, 384), torch.randn(batch, 3, 4)
    return x, text, price, dict(price_window=x, novelty=torch.rand(batch, 5), pair_available=torch.ones(batch, 5))


def forward(model, values, details=True):
    x, text, price, kwargs = values
    return model(price, x[:, -1], text[:, -1], text_window=text, return_details=details, **kwargs)


class HuangQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_five_independent_modules_common_initialization_last_identity_and_batches(self):
        for seed in (7, 17, 27):
            records = {}
            for mode in ('last_matched', *HUANG_MODES):
                core.run.set_seed(seed)
                film = core.fixtures.ContextNoveltyTests.make_film()
                model = build_text_model(core.config(mode) if mode == 'last_matched' else config(mode), 'context_price_only').eval()
                rng = torch.get_rng_state().clone()
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True, drop_last=True)
                core.run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([b[0] for b in loader]) for _ in range(3)]
                values = inputs(batch=1 if seed == 7 else 4)
                values[1][0, :, 0] = 0
                values[1][0, -1] = 0
                output = forward(model, values, details=False)
                proof = ({g: state_fingerprint(m, exclude_prefixes=('query',)) for g, m in model.temporal_pool.groups.items()}
                         if mode != 'last_matched' else None)
                records[mode] = common_initialization(film, model), rng, orders, output.detach(), proof
                if mode != 'last_matched':
                    self.assertEqual(sum(isinstance(m, nn.LSTMCell) for m in model.modules()), 5)
                    self.assertEqual(len({m.U.weight.data_ptr() for m in model.temporal_pool.groups.values()}), 5)
                    self.assertEqual(len({m.V.weight.data_ptr() for m in model.temporal_pool.groups.values()}), 5)
                    self.assertEqual(len({m.cell.weight_hh.data_ptr() for m in model.temporal_pool.groups.values()}), 5)
                    self.assertTrue(all((m.head.weight == 0).all() and (m.head.bias == 0).all() for m in model.temporal_pool.groups.values()))
            base = records['last_matched']
            for value in records.values():
                self.assertEqual(value[0], base[0])
                torch.testing.assert_close(value[1], base[1], rtol=0, atol=0)
                for a, b in zip(value[2], base[2]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(value[3], base[3], rtol=0, atol=0)
            self.assertEqual(records[HUANG_MODES[0]][4], records[HUANG_MODES[1]][4])

    def test_query_feedback_group_isolation_and_causal_bounds(self):
        for mode in HUANG_MODES:
            model = build_text_model(config(mode), 'context_price_only').eval()
            values = inputs()
            output, detail = forward(model, values)
            changed = [values[0], values[1].clone(), values[2], values[3]]
            changed[1][:, 44, 0] += 10
            _, other = forward(model, changed)
            if mode == HUANG_MODES[0]:
                self.assertGreater((detail['huang_queries'][:, 1, 0] - other['huang_queries'][:, 1, 0]).abs().sum(), 0)
            else:
                torch.testing.assert_close(detail['huang_queries'], other['huang_queries'], rtol=0, atol=0)
                torch.testing.assert_close(detail['huang_queries'][:, 0], detail['huang_queries'][:, -1], rtol=0, atol=0)
            for key in ('huang_states', 'huang_memories', 'huang_queries', 'huang_beta'):
                torch.testing.assert_close(detail[key][:, :, 1:], other[key][:, :, 1:], rtol=0, atol=0)
            # Older than the earliest allowed history, and future relative to a step.
            changed[1] = values[1].clone()
            changed[1][:, :34] = 999
            _, outside = forward(model, changed)
            torch.testing.assert_close(detail['huang_states'], outside['huang_states'], rtol=0, atol=0)
            changed[1] = values[1].clone()
            changed[1][:, 50:] = 999
            _, future = forward(model, changed)
            for key in ('huang_states', 'huang_memories', 'huang_queries', 'huang_beta', 'huang_current', 'huang_history'):
                torch.testing.assert_close(detail[key][:, :6], future[key][:, :6], rtol=0, atol=0)
            self.assertEqual(output.shape, (3, 3, 4))

    def test_explicit_projection_attention_empty_masks_and_freeze(self):
        for mode in HUANG_MODES:
            model = build_text_model(config(mode), 'context_price_only').eval()
            values = inputs(batch=1)
            values[1].zero_()
            values[1][:, 34, 0] = 1
            values[1][:, 43, 0] = 2
            values[1][:, 44, 0] = 3
            for module in model.temporal_pool.groups.values():
                with torch.no_grad():
                    module.head.weight.fill_(1)
                    module.head.bias.fill_(2)
            _, detail = forward(model, values)
            group = model.temporal_pool.groups['macro']
            raw = values[1][:, 34:64, 0]
            u = torch.sigmoid(group.U(raw)) * torch.tanh(group.V(raw))
            q = detail['huang_queries'][:, 0, 0]
            logits = (u[:, [0, 9]] * q[:, None]).sum(-1) / HIDDEN ** .5
            beta = logits.softmax(1)
            torch.testing.assert_close(detail['huang_beta'][:, 0, 0, [0, 9]], beta)
            torch.testing.assert_close(detail['huang_history'][:, 0, 0], torch.tanh((beta[..., None] * u[:, [0, 9]]).sum(1)))
            torch.testing.assert_close(detail['huang_current'][:, 0, 0], u[:, 10])
            # A later recurrent query is nonzero; a learned constant can be nonzero too.
            if mode == HUANG_MODES[1]:
                with torch.no_grad():
                    group.query.fill_(.7)
                _, detail = forward(model, values)
            for step in (0, 1, 19):
                mask = raw[:, step:step + 10].abs().sum(-1) >= 1e-6
                q = detail['huang_queries'][:, step, 0]
                scores = (u[:, step:step + 10] * q[:, None]).sum(-1) / HIDDEN ** .5
                expected = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(1) * mask
                expected = expected / expected.sum(1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
                torch.testing.assert_close(detail['huang_beta'][:, step, 0], expected)
                torch.testing.assert_close(detail['huang_history'][:, step, 0], torch.tanh((expected[..., None] * u[:, step:step + 10]).sum(1)))
            zero = raw.new_zeros((1, 32))
            z, memory = group.cell(torch.cat((detail['huang_current'][:, 0, 0], detail['huang_history'][:, 0, 0]), -1), (zero, zero))
            torch.testing.assert_close(detail['huang_states'][:, 0, 0], z)
            torch.testing.assert_close(detail['huang_memories'][:, 0, 0], memory)
            self.assertTrue((detail['huang_beta'][~detail['huang_history_present']] == 0).all())
            self.assertTrue((detail['history_contributions'][:, 1:] == 0).all())
            self.assertTrue((detail['huang_states'][:, :, 1:] == 0).all())
            self.assertTrue((detail['huang_memories'][:, :, 1:] == 0).all())
            # Last observation at row44; updates stop after s54, then both memories freeze.
            self.assertFalse(detail['huang_updated'][:, 11:, 0].any())
            torch.testing.assert_close(detail['huang_states'][:, 11:, 0], detail['huang_states'][:, 10:11, 0].expand(-1, 9, -1))
            self.assertFalse(detail['current_present'].any())
            self.assertGreater(detail['history_contributions'][:, 0].abs().sum(), 0)
            values[1].zero_()
            empty, detail = forward(model, values)
            self.assertTrue((detail['history_contributions'] == 0).all())
            self.assertTrue((detail['huang_beta'] == 0).all())
            torch.testing.assert_close(empty, .9 * values[2] + .1 * values[0][:, -1:])

    def test_zero_head_learns_then_gradients_reach_queries_and_cells(self):
        for mode in HUANG_MODES:
            model = build_text_model(config(mode), 'context_price_only')
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            values = inputs()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = forward(model, values, details=False)
                output.square().mean().backward()
                for module in model.temporal_pool.groups.values():
                    self.assertGreater(module.head.weight.grad.abs().sum(), 0)
                    if step:
                        for parameter in (module.U.weight, module.V.weight, module.cell.weight_hh,
                                          module.query.weight if mode == HUANG_MODES[0] else module.query):
                            self.assertGreater(parameter.grad.abs().sum(), 0)
                optimizer.step()

    def test_runner_checkpoint_diagnostics_and_initial_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            core.core.frame().to_parquet(data / 'SYN.parquet', index=False)
            inspected = []
            for mode in HUANG_MODES:
                output = root / mode / 'output'
                datasets = {flag: core.ContextNoveltyDataset(core.Dataset_Custom(30, 2, 3, flag, list(TEXT_GROUPS),
                            root_path=str(data), data_path='SYN.parquet')) for flag in ('train', 'val', 'test')}
                loaders = {flag: DataLoader(ds, batch_size=4, shuffle=flag == 'train', drop_last=flag == 'train') for flag, ds in datasets.items()}
                def factory(**kwargs):
                    return datasets[kwargs['flag']], loaders[kwargs['flag']]
                argv = ['--model_type', 'FiLM', '--group_gate', 'context_price_only', '--text_temporal_pool', mode,
                        '--root_path', str(data), '--data_path', 'SYN.parquet', '--code_commit', 'a' * 40,
                        '--seq_len', '30', '--label_len', '2', '--batch_size', '4', '--e_layers', '1', '--num_epoch', '1',
                        '--patience', '10', '--logdir', str(output)]
                with patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader', side_effect=factory), \
                     patch('torch.cuda.is_available', return_value=False), patch('forecasting_task.models.FiLM.device', torch.device('cpu')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    core.run.main(argv)
                saved = torch.load(output / 'best_checkpoint.pt', weights_only=True)
                settings = dict(queue.SETTINGS, seq_len=30, label_len=2, batch_size=4, e_layers=1)
                common = dict(analysis.heads.han.prior.COMMON, seq_len=30, label_len=2, batch_size=4, num_epoch=1)
                with patch.object(queue, 'SETTINGS', settings), patch.object(analysis.heads.han.prior, 'COMMON', common):
                    queue.completion(output.parent, queue.expected_config(mode, 'SYN', 7, 1, 'a' * 40, data), elapsed=1.)
                    item = analysis.inspect(core.Run(mode, 'SYN', 7, output), analysis.ReplayCache(data))
                    self.assertTrue(item[0]['comparable'], item[0])
                    inspected.append(item)
                self.assertEqual(saved['initial_prediction_evidence']['max_abs_difference'], 0)
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
                    dates = list(csv.DictReader(handle))
                self.assertFalse(any('contribution' in key for key in dates[0]))
                self.assertEqual(len(dates), len(datasets['val']) * 5 * 20 * 10)
                self.assertTrue(all(row['date'] < row['step_date'] <= row['input_end_date'] for row in dates))
                self.assertEqual(dates[0]['date'], datasets['val'].row_dates[0])
            with patch.object(analysis.heads.han.prior, 'COMMON', common):
                self.assertEqual(analysis.compare_pair(*inspected), 1)


if __name__ == '__main__':
    unittest.main()
