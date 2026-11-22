import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import analyze_context_price_mean_lag_grid as report
from forecasting_task import run
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.experiment_evidence import state_fingerprint
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model, price_context
import group_gate_audit
import mean_lag_grid_audit as audit
import run_context_price_mean_lag_grid as queue
import run_han_head_ablation_queue as shared
from test_group_gru_attention import frame


ROOT = Path(__file__).resolve().parents[1]


def config(cell):
    return dict(audit.CONFIGS[cell], hidden_size=64, num_layers=1, pred_len=3)


class MeanLagGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_order_validation_and_legacy_price_features(self):
        price = torch.arange(24.).reshape(1, 6, 4).square()
        expected = torch.cat((price[:, 5]-price[:, 4], price[:, 5]-price[:, 0]), -1)
        torch.testing.assert_close(price_context(price), expected, rtol=0, atol=0)
        reordered = torch.cat((price[:, 5]-price[:, 0], price[:, 5]-price[:, 4], price[:, 5]-price[:, 3]), -1)
        torch.testing.assert_close(price_context(price, (5, 1, 2)), reordered, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            base = ['--model_type', 'FiLM', '--text_family', 'mean_price_grid',
                    '--logdir', directory, '--code_commit', 'a'*40]
            result = run.parse_args(base + ['--text_window', '21', '--price_gate_lags', '20,1,5'])
            self.assertEqual(result.text_model_config['price_gate_lags'], [20, 1, 5])
            self.assertEqual(result.gate_structure['in_features'], 22)
            with contextlib.redirect_stderr(io.StringIO()):
                for flags in (['--price_gate_lags', value] for value in ('', '1,1', '0', '-3', '1.5', '64')):
                    with self.assertRaises(SystemExit):
                        run.parse_args(base + flags)
                for window in ('0', '65'):
                    with self.assertRaises(SystemExit):
                        run.parse_args(base + ['--text_window', window])

    def test_masks_causal_suffix_gate_gradients_and_state(self):
        price = torch.randn(2, 64, 4)
        film = torch.randn(2, 3, 4)
        # The forecast receives only [:64]; include extra source rows to test the boundary.
        source = torch.zeros(2, 67, 5, 384)
        source[0, 62, 1, 0] = 2
        source[0, 60, 2, 0] = 5
        source[0, 61, 2, 0] = -5
        source[0, 63, 3, 0] = 1e-7
        for cell in audit.CONFIGS:
            with self.subTest(cell=cell):
                settings = config(cell)
                model = build_text_model(settings).eval()
                with torch.no_grad():
                    for head in model.group_mlps.values():
                        head[0].weight.fill_(1)
                        head[0].bias.fill_(3)
                extra = torch.ones(2, 5)

                def predict(model):
                    return model(film, price[:, -1], source[:, 63], text_window=source[:, :64],
                                 price_window=price, novelty=extra, pair_available=extra, return_details=True)
                output, detail = predict(model)
                self.assertEqual(tuple(output.shape), (2, 3, 4))
                self.assertEqual(detail['present'].tolist(), [[False, True, True, False, False], [False]*5])
                torch.testing.assert_close(detail['heads'][0, 1], torch.full((3,), 5.))
                torch.testing.assert_close(detail['heads'][0, 2], torch.full((3,), 3.))
                torch.testing.assert_close(detail['weights'], torch.full((2, 5), .2))
                self.assertEqual(torch.count_nonzero(detail['features'][:, -10:]).item(), 0)
                expected = .9*film + .1*price[:, -1:]
                expected[0] += .1*.2*(5 + 3)
                torch.testing.assert_close(output, expected)
                source[:, :64-settings['text_window']] = 1e3
                source[:, 64:] = -1e3
                torch.testing.assert_close(predict(model)[0], output, rtol=0, atol=0)
                source[:, :64-settings['text_window']] = 0
                output.sum().backward()
                self.assertGreater(model.gate.weight.grad[:, :-10].abs().sum().item(), 0)
                self.assertEqual(torch.count_nonzero(model.gate.weight.grad[:, -10:]).item(), 0)
                buffer = io.BytesIO()
                torch.save(dict(config=settings, text=model.state_dict()), buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=True)
                restored = build_text_model(saved['config']).eval()
                restored.load_state_dict(saved['text'], strict=True)
                torch.testing.assert_close(output, predict(restored)[0], rtol=0, atol=0)

    def test_nine_cells_preserve_common_rng_and_batch_order(self):
        starts = set()
        for cell in audit.CONFIGS:
            torch.manual_seed(17)
            # A preceding RNG consumer stands in for the price module.
            price = torch.nn.Linear(4, 4)
            model = build_text_model(config(cell))
            starts.add((state_fingerprint(price), state_fingerprint(model.group_mlps),
                        bytes(torch.get_rng_state().tolist())))
        self.assertEqual(len(starts), 1)
        first, state = audit.replay_orders(77, 32, 7, 3, 2)
        second, other_state = audit.replay_orders(77, 32, 7, 2, 2)
        self.assertEqual(first[:2], second)
        self.assertNotEqual(first[0]['sha256'], first[1]['sha256'])
        self.assertEqual(first[0]['windows'], 64)
        self.assertTrue(torch.equal(state, other_state))

    def test_synthetic_runner_normal_parity_checkpoint_and_full_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'data'
            data.mkdir()
            frame().to_parquet(data/'SYN.parquet', index=False)
            settings = dict(shared.SETTINGS, seq_len=20, label_len=2, batch_size=4, e_layers=1)
            dataset = {flag: ContextNoveltyDataset(Dataset_Custom(20, 2, 3, flag, list(TEXT_GROUPS),
                                        root_path=str(data), data_path='SYN.parquet'), include_calendar_days=True)
                       for flag in ('train', 'val', 'test')}

            def factory(**kwargs):
                flag = kwargs['flag']
                return dataset[flag], DataLoader(dataset[flag], batch_size=4, shuffle=flag=='train', drop_last=flag=='train')

            paired = []
            with (patch.object(shared, 'SETTINGS', settings), patch.object(audit, 'EPOCH_LIMIT', 1),
                  patch.object(group_gate_audit, 'EPOCH_LIMIT', 1), patch('torch.cuda.is_available', return_value=False),
                  patch('forecasting_task.models.FiLM.device', torch.device('cpu')),
                  patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader', side_effect=factory),
                  contextlib.redirect_stdout(io.StringIO())):
                for historical, cfg in ((True, audit.NORMAL), (False, audit.CONFIGS[audit.CONTROL])):
                    path = root/('normal' if historical else 'grid')
                    flags = shared.training_command(audit.MODE, 'SYN', 7, 1, audit.BASE_COMMIT, data, path/'output', cfg)[4:]
                    if historical:
                        flags += ['--memory_stage', 'normal', '--memory_cache', str(root/'cache')]
                    run.main(flags)
                    expected = shared.expected_config(audit.MODE, 'SYN', 7, 1, audit.BASE_COMMIT, data, cfg)
                    shared.completion(path, expected, elapsed=1.)
                    checked = audit.checked_run(path, cfg, data, audit.BASE_COMMIT, 'SYN', 7,
                                                 audit.prior.ReplayCache(data), historical=historical)
                    paired.append(checked)
                    if not historical:
                        saved = torch.load(path/'output'/'best_checkpoint.pt', weights_only=True)
                        self.assertEqual(saved['price_gate_lags'], [1, 5])
                        with (path/'output'/'validation_text_dates.csv').open(newline='') as handle:
                            rows = list(csv.DictReader(handle))
                        self.assertEqual(len(rows), len(dataset['val'])*5*20)
                    else:
                        artifact = path/'output'/'validation_text_dates.csv'
                        with artifact.open('ab') as handle:
                            handle.write(b'\n')
                        with self.assertRaisesRegex(ValueError, 'hashes differ'):
                            audit.checked_run(path, cfg, data, audit.BASE_COMMIT, 'SYN', 7,
                                              audit.prior.ReplayCache(data), historical=True)
            self.assertEqual(audit.compare_grid_runs(paired[0], paired[1]), 1)
            self.assertEqual(paired[0][0]['val_mse'], paired[1][0]['val_mse'])

    def test_queue_reuse_fallback_force_and_resume(self):
        self.assertEqual(len(set(queue.jobs())), 279)
        self.assertEqual(sum(job[0]=='checks' for job in queue.jobs()), 9)
        for scenario in ('all', 'one_missing', 'forced'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data, results, normal = (root/name for name in ('data', 'results', 'normal'))
                data.mkdir()
                for ticker in (*shared.TICKERS, 'NVDA'):
                    (data/f'{ticker}.parquet').touch()
                calls = []
                pairs = {key: object() for key in audit.PAIRS}
                records = [dict(ticker=t, seed=s, status='full') for t, s in audit.PAIRS]
                if scenario == 'one_missing':
                    del pairs[audit.PAIRS[-1]]
                    records[-1].update(status='rejected', reason='missing checkpoint')

                def training(cmd, log, _code):
                    flags = dict(zip(cmd[4::2], cmd[5::2]))
                    self.assertEqual(flags['--text_family'], 'mean_price_grid')
                    self.assertIn(tuple(map(int, flags['--price_gate_lags'].split(','))), audit.LAG_SETS)
                    calls.append((flags['--text_window'], flags['--price_gate_lags'], flags['--data_path'],
                                  flags['--seed'], flags['--num_epoch']))
                    return 0

                with (patch.dict(queue.os.environ, CODE_COMMIT='a'*40),
                      patch.object(queue.torch.cuda, 'is_available', return_value=True),
                      patch.object(audit, 'inspect_normal_cohort', return_value=(pairs, records)) as inspect,
                      patch.object(shared, 'execute_training', side_effect=training),
                      patch.object(shared, 'completion', return_value={'wall_seconds': 1.}),
                      contextlib.redirect_stdout(io.StringIO()) as log):
                    queue.run_queue(ROOT, data, results, normal, force=scenario=='forced')
                    queue.run_queue(ROOT, data, results, normal)
                reused = 30 if scenario=='all' else 0
                self.assertEqual(len(calls), 279-reused)
                self.assertEqual(len(set(calls)), len(calls))
                self.assertIn(f'trained={270-reused} reused={reused}', log.getvalue())
                if scenario == 'forced':
                    inspect.assert_not_called()
                decision = audit.decisions(results)
                self.assertEqual(decision['source'], 'historical' if reused else 'matched')
                if scenario == 'one_missing':
                    self.assertEqual(decision['records'][-1]['reason'], 'missing checkpoint')

    def test_report_cell_averages_and_blocking_unverified_pairs(self):
        rows = [dict(cell=cell, ticker=t, seed=s, verification='full',
                     val_mse=.02, test_mse=.014, test_mae=.10)
                for cell in audit.CONFIGS for t, s in audit.PAIRS]
        result = report.summarize(rows, 'partial')
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[audit.CONTROL]['baseline_improvement_pct'], 100*(1-.014/report.BASELINE_MSE))
        self.assertEqual(result[audit.CONTROL]['control_improvement_pct'], 0)
        self.assertEqual(len(report.fixed_slices(result, True)), 6)
        self.assertEqual(len(report.fixed_slices(result, False)), 6)
        rows[0]['verification'] = 'mismatch'
        cell = report.summarize(rows, 'partial')[rows[0]['cell']]
        self.assertIsNone(cell['val_mse'])
        self.assertIsNone(cell['control_improvement_pct'])

    def test_saved_report_selects_validation_winner_and_counts_paired_tickers(self):
        rows, grid = [], {}
        winner = audit.cell_name(40, audit.LAG_SETS[-1])
        for cell in audit.CONFIGS:
            for index, (ticker, seed) in enumerate(audit.PAIRS):
                val, test = (.018, .02) if cell == winner else (.02, .014)
                if cell == winner and index < 3:
                    val = .021  # One ticker loses on validation across all three seeds.
                row = dict(cell=cell, ticker=ticker, seed=seed, verification='full',
                           val_mse=val, test_mse=test, test_mae=.1)
                grid[cell, ticker, seed] = row
                rows.append(row)
        references = [dict(variant='original_baseline', ticker=ticker, seed=seed,
                           window_verification='partial', missing_fields='["saved validation windows"]',
                           mismatches='[]', test_mse=report.BASELINE_MSE)
                      for ticker, seed in audit.PAIRS]
        decision = dict(source='historical', reason='full audit')
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(report, 'gather', return_value=(rows, grid, None, {}, decision, [])),
              patch.object(report, 'historical_references', return_value=references),
              contextlib.redirect_stdout(io.StringIO())):
            output = Path(directory)/'report'
            report.analyze(None, None, None, None, None, output)
            summary = json.loads((output/'verification.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['validation_winner'], winner)
            self.assertEqual(summary['validation_ticker_wins'], 9)
            self.assertEqual(summary['original_baseline_status'], 'partial')
            with (output/'cells.csv').open(newline='', encoding='utf-8') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 9)
            with self.assertRaises(FileExistsError):
                report.analyze(None, None, None, None, None, output)


if __name__ == '__main__':
    unittest.main()
