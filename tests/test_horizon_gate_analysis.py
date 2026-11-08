"""Element accounting, honest historical evidence and matched comparison reports."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import analyze_context_price_horizon_gate as analysis


class HorizonAnalysisTests(unittest.TestCase):
    def test_metrics_include_all_channels_horizons_and_small_final_batch(self):
        prediction = torch.zeros(5, 3, 4)
        target = torch.arange(60, dtype=torch.float32).reshape(5, 3, 4)
        metric = analysis.error_metrics(prediction, target)
        self.assertAlmostEqual(metric['mse'], target.double().square().mean().item())
        self.assertAlmostEqual(metric['mse_h3'], target[:, 2].double().square().mean().item())
        self.assertAlmostEqual(metric['mae'], 29.5)

    def test_full_partial_and_mismatch_evidence(self):
        evidence = {'val': {'input': 'hash'}, 'test': {'input': 'hash'}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'validation_group_diagnostics.csv').touch()
            with patch.object(analysis, 'check_saved_test_windows'), patch.object(analysis, 'check_validation_dates'):
                self.assertEqual(analysis.verification(root, evidence, {}, {}, {})[0], 'partial')
                self.assertEqual(analysis.verification(root, evidence, {'window_evidence': evidence}, {}, {})[0], 'full')
                with self.assertRaisesRegex(ValueError, 'fingerprints mismatch'):
                    analysis.verification(root, evidence, {'window_evidence': {}}, {}, {})

    def test_cohort_report_validation_selection_partial_and_strict(self):
        for matched in (False, True):
            with self.subTest(matched=matched), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                new, old = root / 'new', root / 'old'
                for base, condition in ((new, 'per_horizon'), (old, 'shared' if matched else 'context_price_only')):
                    for ticker in analysis.TICKERS:
                        for seed in analysis.SEEDS:
                            output = base / 'full' / condition / ticker / f'seed{seed}' / 'output'
                            output.mkdir(parents=True)
                            (output / 'best_checkpoint.pt').write_bytes(f'{condition}{ticker}{seed}'.encode())
                def evaluate(run, *_):
                    value = .1 if run.variant == 'shared' else .2
                    metrics = {m: value if m.startswith('val') else 1 - value for m in analysis.METRICS}
                    status = 'full' if matched or run.variant == 'per_horizon' else 'partial'
                    return metrics, status, 'synthetic evidence', {'same_windows': True}, {'code_commit': 'a' * 40}, {}, {}
                with patch.object(analysis, 'evaluate_run', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                    tables, selection = analysis.analyze(new, old, root, matched_shared=matched)
                    self.assertEqual(selection['status'], 'full' if matched else 'partial')
                    self.assertEqual(selection['selected'] if matched else selection['provisional_selected'], ['shared'])
                    self.assertFalse(selection['test_used_for_selection'])
                    self.assertEqual(len(tables['per_run']), 60)
                    self.assertTrue(all(r['comparable_tickers'] == 10 for r in tables['wins']))
                    with self.assertRaises(FileExistsError):
                        analysis.analyze(new, old, root, matched_shared=matched)
                def mismatch(run, *rest):
                    if run.ticker == 'AAPL' and run.seed == 7 and run.variant == 'shared':
                        raise ValueError('synthetic target mismatch')
                    return evaluate(run, *rest)
                failed = root / 'failure_report'
                with patch.object(analysis, 'evaluate_run', side_effect=mismatch), contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaisesRegex(ValueError, 'Comparison blocked'):
                    analysis.analyze(new, old, root, failed, matched)
                self.assertEqual(json.loads((failed / 'selection.json').read_text())['status'], 'blocked')
                self.assertIn('synthetic target mismatch', (failed / 'window_verification.csv').read_text())
                def config_failure(run, *rest):
                    if run.variant == 'shared' and run.ticker == 'AAPL' and run.seed == 7:
                        raise analysis.ConfigurationError('synthetic configuration mismatch')
                    return evaluate(run, *rest)
                unchecked = root / 'config_report'
                with patch.object(analysis, 'evaluate_run', side_effect=config_failure), contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaisesRegex(ValueError, 'Comparison blocked'):
                    analysis.analyze(new, old, root, unchecked, matched)
                self.assertIn('not_checked', (unchecked / 'window_verification.csv').read_text())


if __name__ == '__main__':
    unittest.main()
