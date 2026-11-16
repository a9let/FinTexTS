"""Matched-window reports, historical partial evidence and train-only returns."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import analyze_context_price_group_gru_attention as analysis
from forecasting_task.models.GroupText import TEXT_GROUPS
from forecasting_task.models.GroupGRUAttention import temporal_structure
from forecasting_task.window_evidence import SPLIT_POLICY, window_evidence
import test_context_novelty as fixtures
import test_group_gru_attention as core


class GroupGRUAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def datasets(self):
        data = core.frame()
        return {flag: fixtures.make_dataset(data, flag, seq_len=20) for flag in ('train', 'val', 'test')}

    def build_cohort(self, directory, datasets):
        root, baseline = Path(directory) / 'results', Path(directory) / 'baseline'
        evidence = {flag: window_evidence(datasets[flag]) for flag in ('val', 'test')}
        data = datasets['test']
        inputs = torch.tensor(np.stack([data[i][0] for i in range(len(data))]), dtype=torch.float32)
        truth = torch.tensor(np.stack([data[i][1][-3:] for i in range(len(data))]), dtype=torch.float32)
        for mode in analysis.VARIANTS:
            for ticker in analysis.TICKERS:
                for seed_index, seed in enumerate(analysis.SEEDS):
                    path = (baseline / 'full' / 'baseline' if mode == 'original_baseline' else root / 'full' / mode) / ticker / f'seed{seed}' / 'output'
                    path.mkdir(parents=True)
                    selected_mode = 'last' if mode == 'original_baseline' else mode
                    args = {**analysis.COMMON, 'seq_len': 20, 'label_len': 2, 'batch_size': 4, 'num_epoch': 2,
                            'task_name': 'short_term_forecast', 'e_layers': 1, 'output_attention': False,
                            'enc_in': 4, 'dec_in': 4, 'c_out': 4, 'model_type': 'FiLM', 'seed': seed,
                            'data_path': ticker + '.parquet', 'used_col_prefixes': list(TEXT_GROUPS),
                            'text_weight': .1, 'fixed_alpha': .1, 'text_weight_mode': 'fixed',
                            'group_gate': 'context_price_only', 'group_weight_mode': 'context_price_only',
                            'text_num_layers': 1, 'fusion_mode': 'blend', 'code_commit': 'a' * 40,
                            'text_temporal_pool': selected_mode, 'temporal_pool_structure': temporal_structure(selected_mode),
                            'sampling_policy': 'independent_train_generator', 'split_policy': SPLIT_POLICY,
                            'gate_structure': {'type': 'Linear', 'in_features': 18, 'out_features': 5, 'parameters': 95, 'initialization': 'zeros'},
                            'window_evidence': evidence,
                            'text_model_config': {**core.config(selected_mode), 'initial_text_weight': .1, 'fusion_mode': 'blend'}}
                    offset = {'last': .4, 'group_gru_attention': .3, 'original_baseline': .5}[mode]
                    val = {'last': .2, 'group_gru_attention': .3, 'original_baseline': .1}[mode] + seed_index * .05
                    summary = {'model': 'FiLM', 'seed': seed, 'data_path': ticker + '.parquet', 'best_epoch': 1,
                               'best_val_mse': val, 'best_test_mse': offset**2, 'best_test_mae': offset,
                               **{key: args[key] for key in ('group_gate', 'fixed_alpha', 'text_weight_mode', 'code_commit',
                                                            'text_temporal_pool', 'temporal_pool_structure', 'window_evidence', 'split_policy')}}
                    checkpoint = {'args': args, 'epoch': 1, **{key: summary[key] for key in ('group_gate', 'fixed_alpha', 'text_weight_mode',
                                   'code_commit', 'text_temporal_pool', 'temporal_pool_structure', 'window_evidence', 'split_policy')},
                                  **{'scaler_' + key: getattr(data.scaler, key + '_').tolist() for key in ('mean', 'scale', 'var')}}
                    if mode == 'original_baseline':
                        for source in (args, summary, checkpoint):
                            source.pop('window_evidence', None)
                            source.pop('split_policy', None)
                    (path / 'summary.json').write_text(json.dumps(summary))
                    torch.save(checkpoint, path / 'best_checkpoint.pt')
                    torch.save({'inputs': inputs, 'ground_truths': truth, 'outputs': truth + offset,
                                'outputs_inv': torch.full_like(truth, float('nan'))}, path / 'test_io.pt')
                    with (path / 'validation_group_diagnostics.csv').open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=['window_index', 'input_end_date', 'group'])
                        writer.writeheader()
                        for i in range(len(datasets['val'])):
                            for group in TEXT_GROUPS:
                                writer.writerow({'window_index': i, 'input_end_date': datasets['val'].window_end_date(i), 'group': group})
        return root, baseline

    def test_full_primary_partial_baseline_statistics_and_validation_selection(self):
        datasets = self.datasets()
        with tempfile.TemporaryDirectory() as directory:
            root, baseline = self.build_cohort(directory, datasets)
            controlled = dict(analysis.COMMON, seq_len=20, label_len=2, batch_size=4, num_epoch=2)
            with patch.object(analysis.ReplayCache, 'get', return_value=datasets), patch.object(analysis, 'COMMON', controlled), \
                 contextlib.redirect_stdout(io.StringIO()):
                tables, selection = analysis.analyze(root, baseline, root)
                self.assertEqual(selection['selected'], ['last'])
                self.assertFalse(selection['test_used_for_selection'])
                self.assertEqual(len(tables['per_run']), 90)
                self.assertTrue(all(r['window_verification'] == 'full' for r in tables['per_run'] if r['variant'] != 'original_baseline'))
                self.assertTrue(all(r['window_verification'] == 'partial' for r in tables['per_run'] if r['variant'] == 'original_baseline'))
                mean = next(r for r in tables['across_seeds'] if r['variant'] == 'last' and r['metric'] == 'val_mse')
                self.assertAlmostEqual(mean['mean'], .25)
                self.assertAlmostEqual(mean['sample_std'], .05)
                reduction = next(r for r in tables['test_mse_reductions'] if r['left'] == 'group_gru_attention' and r['right'] == 'original_baseline')
                self.assertAlmostEqual(reduction['improvement_pct'], 64)
                self.assertAlmostEqual(reduction['baseline_target_0_8'], .2)
                self.assertEqual(reduction['window_verification'], 'partial')
                self.assertTrue(tables['strong_movements'])
                with self.assertRaises(FileExistsError):
                    analysis.analyze(root, baseline, root)
                # A baseline contradiction suppresses this ticker's relative gain.
                path = baseline / 'full/baseline/AAPL/seed7/output/best_checkpoint.pt'
                cp = torch.load(path, weights_only=True)
                cp['scaler_mean'][0] += 1
                torch.save(cp, path)
                mismatch, _ = analysis.analyze(root, baseline, root, Path(directory) / 'baseline_mismatch')
                records = [r for r in mismatch['paired_by_ticker'] if r['right'] == 'original_baseline' and r['ticker'] == 'AAPL']
                self.assertTrue(all(r['improvement_pct'] is None for r in records))
                self.assertTrue(all(r['improvement_pct'] is None for r in mismatch['test_mse_reductions'] if r['right'] == 'original_baseline'))
                # Primary validation-target fingerprints are a mandatory paired check.
                path = root / 'full/group_gru_attention/AAPL/seed7/output/best_checkpoint.pt'
                cp = torch.load(path, weights_only=True)
                cp['window_evidence']['val']['targets_sha256'] = '0' * 64
                torch.save(cp, path)
                failed = Path(directory) / 'primary_mismatch'
                with self.assertRaisesRegex(ValueError, 'Matched comparison blocked'):
                    analysis.analyze(root, baseline, root, failed)
                self.assertEqual(json.loads((failed / 'selection.json').read_text())['status'], 'blocked')
                self.assertIn('fingerprints mismatch', (failed / 'window_verification.csv').read_text())

    def test_train_only_original_scale_return_thresholds_and_sse_share(self):
        datasets = self.datasets()
        train = datasets['train']
        thresholds, count = analysis.train_return_thresholds(train)
        raw = train.scaler.inverse_transform(train.base.data_price_x)
        explicit = np.array([[abs(raw[i + train.seq_len + h, 3] / raw[i + train.seq_len - 1, 3] - 1)
                              for h in range(3)] for i in range(len(train))])
        np.testing.assert_allclose(thresholds, np.percentile(explicit, 90, axis=0), rtol=1e-12)
        self.assertEqual(count, len(train))
        changed = core.frame()
        changed.loc[40:, 'close'] *= 1000
        other = fixtures.make_dataset(changed, 'train', seq_len=20)
        np.testing.assert_array_equal(analysis.train_return_thresholds(other)[0], thresholds)
        scaler = analysis.dataset_scaler(train)
        n = 6
        inputs = np.zeros((n, 20, 4))
        truth = np.zeros((n, 3, 4))
        last_close = scaler['mean'][3]
        for horizon in range(3):
            returns = thresholds[horizon] * np.array([.5, 1.5, .5, 1.5, .5, 1.5])
            truth[:, horizon, 3] = (last_close * (1 + returns) - scaler['mean'][3]) / scaler['scale'][3]
        last = {'inputs': inputs, 'ground_truths': truth, 'outputs': truth + 2}
        gru = {'inputs': inputs.copy(), 'ground_truths': truth.copy(), 'outputs': truth + 1}
        records, issues = analysis.movement_records('SYN', 7, {'datasets': datasets, 'scaler': scaler},
                                                   {'last': last, 'group_gru_attention': gru})
        self.assertFalse(issues)
        self.assertEqual(len(records), 6)
        for record in records:
            self.assertEqual(record['selected_windows'], 3)
            self.assertEqual(record['elements'], 12)
            self.assertAlmostEqual(record['share_of_last_sse'], .5)
            self.assertEqual(record['mse'], 4 if record['variant'] == 'last' else 1)
        gru['ground_truths'] = truth + .01
        with self.assertRaisesRegex(ValueError, 'targets mismatch'):
            analysis.movement_records('SYN', 7, {'datasets': datasets, 'scaler': scaler},
                                      {'last': last, 'group_gru_attention': gru})


if __name__ == '__main__':
    unittest.main()
