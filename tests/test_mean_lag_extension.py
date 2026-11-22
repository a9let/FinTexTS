import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import analyze_context_price_mean_lag_extension as report
from forecasting_task import run
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset, gate_batch_inputs
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.experiment_evidence import source_data_evidence, state_fingerprint
from forecasting_task.group_gate_diagnostics import records
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model
from forecasting_task.window_evidence import window_evidence
import mean_lag_grid_audit as audit
import run_context_price_mean_lag_extension_180 as extension
import run_context_price_mean_lag_grid as queue
from test_group_gru_attention import frame


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'val', 'test')


def toy_data():
    original = frame()
    dates = [*pd.date_range('2021-07-01', periods=98), *pd.date_range('2022-01-03', periods=18),
             *pd.date_range('2023-01-03', periods=18)]
    result = original.iloc[np.arange(len(dates)) % len(original)].reset_index(drop=True)
    result['date'] = [str(date.date()) for date in dates]
    result['macro_emb0'] = np.arange(1, 1+len(dates), dtype=np.float32)
    return result


def datasets(frame, extended=None):
    with patch('pandas.read_parquet', return_value=frame):
        return {flag: ContextNoveltyDataset(Dataset_Custom(64, 16, 3, flag, list(TEXT_GROUPS)),
                                           extended_text_window=extended) for flag in SPLITS}


class LastPrice(nn.Module):
    def forward(self, price, *_args):
        return price[:, -1:].expand(-1, 3, -1)


class MeanLagExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.frame = toy_data()
        cls.original = datasets(cls.frame)
        cls.extended = datasets(cls.frame, 80)

    def test_original_windows_and_targets_and_real_prefix_in_every_split(self):
        self.assertEqual(source_data_evidence(self.original), source_data_evidence(self.extended))
        for flag in SPLITS:
            old, extended = self.original[flag], self.extended[flag]
            self.assertEqual(window_evidence(old), window_evidence(extended))
            for index in range(len(old)):
                before, after = old[index], extended[index]
                for field in (0, 1, 2, 3, 4, 5):
                    np.testing.assert_array_equal(before[field], after[field])
                history = after[6]['text_history']
                self.assertEqual(history.shape, (16, 5, 384))
                window = np.concatenate((history, after[2]))
                end = int(self.frame.index[self.frame['date'] == old.window_end_date(index)][0])
                start = max(0, end-79)
                dates = extended.text_window_dates(index)
                np.testing.assert_array_equal(dates, self.frame['date'].iloc[start:end+1])
                np.testing.assert_array_equal(window[-64:], before[2])
                np.testing.assert_array_equal(window[-len(dates):, 0, 0], self.frame['macro_emb0'].iloc[start:end+1])
                self.assertEqual(np.count_nonzero(window[:80-len(dates)]), 0)
                self.assertTrue((dates[:-1] < old.window_end_date(index)).all())
            evidence = extended.extended_text_evidence()
            self.assertEqual(evidence['min_rows'], 64 if flag == 'train' else 80)
            self.assertEqual(evidence['max_rows'], 80)
            self.assertEqual(evidence['windows'], len(old))

    def test_future_text_change_does_not_affect_current_window(self):
        changed = self.frame.copy()
        changed.loc[64:, 'macro_emb0'] *= 100
        rebuilt = datasets(changed, 80)['train']
        original = self.extended['train']
        np.testing.assert_array_equal(rebuilt[0][2], original[0][2])
        np.testing.assert_array_equal(rebuilt[0][6]['text_history'], original[0][6]['text_history'])
        self.assertNotEqual(rebuilt[1][2][-1, 0, 0], original[1][2][-1, 0, 0])

    def test_six_runner_inference_paths_masks_and_legacy_mean40(self):
        film = LastPrice()
        device = torch.device('cpu')
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            for lags in audit.LAG_SETS:
                fingerprints = []
                for window in (40, 60, 80):
                    with self.subTest(window=window, lags=lags):
                        args = run.parse_args(['--text_family', 'mean_price_grid', '--text_window', str(window),
                            '--price_gate_lags', ','.join(map(str, lags)), '--model_type', 'FiLM',
                            '--seq_len', '64', '--label_len', '16', '--logdir', directory, '--code_commit', 'a'*40])
                        torch.manual_seed(7)
                        model = build_text_model(args.text_model_config, args.group_weight_mode).eval()
                        fingerprints.append(state_fingerprint(model))
                        ds = (self.extended if window == 80 else self.original)['train']
                        loader = DataLoader([ds[0], ds[18]], batch_size=2)
                        batch = next(iter(loader))
                        price, text, *_other, target = run._prepare_batch(batch, device, 3, 16)
                        metadata = gate_batch_inputs(batch[6], text, device, use_window=True)
                        source = metadata.pop('text_window')
                        suffix = source[:, -window:]
                        present = suffix.abs().sum(-1) >= 1e-6
                        pooled = (suffix*present[..., None]).sum(1) / present.sum(1)[..., None].clamp_min(1)
                        expected, detail = model.core(film(price), price[:, -1], pooled, group_present=present.any(1),
                            price_window=price, return_details=True, **metadata)
                        output, new_detail = model(film(price), price[:, -1], text[:, -1], text_window=source,
                            price_window=price, return_details=True, **metadata)
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        self.assertTrue(torch.equal(new_detail['mean_present'], present))
                        self.assertEqual(torch.count_nonzero(detail['contributions'][:, 4]).item(), 0)
                        self.assertTrue((detail['heads'][:, 4] != 0).any())  # Nonzero bias, absent text.
                        loss, mae = run._run_epoch(loader, ds, film, model, args, device)
                        self.assertEqual(loss, (expected-target).square().mean().item())
                        self.assertEqual(mae, (expected-target).abs().mean().item())
                        groups, dates, _ = records(ds, [0, 18], film(price), price[:, -1], output, new_detail,
                                                   model.get_alpha(), model.family)
                        self.assertEqual(len(dates), (64+80 if window == 80 else 2*window)*5)
                        if window == 80:
                            self.assertEqual(groups[(0, 'macro')]['text_window_length'], 64)
                            self.assertAlmostEqual(pooled[0, 0, 0].item(), 32.5)
                self.assertEqual(len(set(fingerprints)), 1)

    def test_queue_stubs_only_six_checks_and_180_new_jobs(self):
        calls = []
        shared = audit.shared
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)/'data'
            results = Path(directory)/'results'
            data.mkdir()
            for ticker in (*shared.TICKERS, 'NVDA'):
                (data/f'{ticker}.parquet').touch()

            def training(command, *_args):
                flags = dict(zip(command[4::2], command[5::2]))
                calls.append(tuple(flags[key] for key in ('--text_window', '--price_gate_lags',
                                                           '--data_path', '--seed', '--num_epoch')))
                return 0

            with (patch.dict(queue.os.environ, CODE_COMMIT='a'*40),
                  patch.object(queue.torch.cuda, 'is_available', return_value=True),
                  patch.object(queue, 'select_control', side_effect=AssertionError('No historical training')),
                  patch.object(shared, 'execute_training', side_effect=training),
                  patch.object(shared, 'completion', return_value={'wall_seconds': 1.}),
                  contextlib.redirect_stdout(io.StringIO()) as log):
                for _ in range(2):
                    queue.run_queue(ROOT, data, results, None, configurations=extension.CONFIGS,
                                    scripts=extension.EXTENSION_SCRIPTS)
            self.assertEqual(len(calls), 186)
            self.assertEqual(len(set(calls)), 186)
            self.assertEqual(sum(call[-1]=='1' and call[2]=='NVDA.parquet' for call in calls), 6)
            self.assertEqual({call[0] for call in calls}, {'60','80'})
            self.assertIn('ALL 180 GRID RESULTS COMPLETE trained=180 reused=0', log.getvalue())

    def test_report_ranks_by_validation_and_suppresses_unverified_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root/'old', root/'new'
            for path in (old, new):
                path.mkdir()
                (path/'code_commit.txt').write_text('a'*40)

            def inspected(_destination, config, _data, _commit, ticker, seed, _cache):
                window, lags = config['text_window'], config['price_gate_lags']
                if window == 80 and lags == [1, 5] and (ticker, seed) == audit.PAIRS[0]:
                    raise ValueError('Target dates differ')
                return dict(val_mse={40:.02, 60:.018, 80:.019}[window],
                            test_mse={40:.01, 60:.013, 80:.008}[window], test_mae=.1), {}, dict(checkpoint_sha256='a'*64)

            with (patch.object(audit, 'checked_run', side_effect=inspected),
                  patch.object(audit, 'compare_grid_runs', return_value=11),
                  contextlib.redirect_stdout(io.StringIO())):
                table = report.analyze(new, old, root, root/'report')
            confirmed = [row for row in table if row['lags'] == '1,5,20']
            self.assertEqual([row['window'] for row in confirmed], [60, 80, 40])
            self.assertAlmostEqual(confirmed[0]['test_mse_improvement_pct'], -30.)
            mismatch = next(row for row in table if row['lags'] == '1,5' and row['window'] == 80)
            self.assertEqual(mismatch['verification'], 'mismatch')
            self.assertIsNone(mismatch['val_mse_improvement_pct'])
            with (root/'report'/'paired_runs.csv').open(encoding='utf-8', newline='') as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 270)
            self.assertIn('Target dates differ', {row['reason'] for row in records})


if __name__ == '__main__':
    unittest.main()
