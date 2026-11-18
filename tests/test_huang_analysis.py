"""Validation-only selection and pair-specific partial historical controls."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_context_price_huang_query as analysis
from analyze_persistence_baseline import Run


def fixture(run, cache):
    value = dict(zip(analysis.MODES, (.1, .2, .3)))[run.variant]
    row = {'variant': run.variant, 'ticker': run.ticker, 'seed': run.seed, 'output': str(run.output),
           'comparable': True, 'window_verification': 'full', 'match_verification': 'full',
           'missing_fields': '[]', 'mismatches': '[]', 'evidence_missing': '[]', 'evidence_reason': '',
           'val_mse': value, 'test_mse': 1 - value, 'test_mae': .5}
    initial = dict.fromkeys(('film_sha256', 'price_gate_sha256', 'heads_sha256'), 'a' * 64)
    args = {**analysis.heads.han.prior.COMMON, **{key: 'same' for key in analysis.heads.han.prior.PRICE_FIELDS},
            'd_model': 64, 'dropout': .1, 'novelty_policy': 'same', 'sampling_policy': 'same', 'gate_structure': {},
            'common_initialization': initial, 'huang_initialization': dict.fromkeys(analysis.TEXT_GROUPS, 'b' * 64),
            'initial_prediction_evidence': {'prediction_sha256': 'c' * 64, 'input_end_date': 'same', 'max_abs_difference': 0},
            'code_commit': ('a' if run.variant in analysis.HUANG_MODES else 'b') * 40}
    metadata = {'args': args, 'evidence': {'same': True}, 'split': {},
                'scaler': {key: [1., 2., 3., 4.] for key in ('mean', 'scale', 'var')}}
    proof = {'initialization': initial, 'source': {'same': True}, 'orders': ['epoch1']}
    return row, metadata, {}, proof


class HuangAnalysisTests(unittest.TestCase):
    def setup_roots(self, root):
        new, old = root / 'new', root / 'old'
        for mode in analysis.MODES:
            for ticker in analysis.TICKERS:
                for seed in analysis.SEEDS:
                    output = (old if mode == analysis.LAST else new) / 'full' / mode / ticker / f'seed{seed}/output'
                    output.mkdir(parents=True)
                    (output / 'best_checkpoint.pt').write_text(f'{mode}{ticker}{seed}')
        return new, old

    def test_report_validation_selection_and_partial_control(self):
        with tempfile.TemporaryDirectory() as directory:
            new, old = self.setup_roots(Path(directory))
            with patch.object(analysis, 'inspect', side_effect=fixture), contextlib.redirect_stdout(io.StringIO()):
                tables, selection = analysis.analyze(new, old)
                self.assertEqual(len(tables['per_run']), 90)
                self.assertEqual(len(tables['window_verification']), 90)
                self.assertEqual(selection['selected'], [analysis.HUANG_MODES[0]])
                self.assertFalse(selection['test_used_for_selection'])
                self.assertEqual(selection['last_comparison'], 'full')
                wins = [r for r in tables['wins'] if (r['left'], r['right'], r['metric']) == (*analysis.HUANG_MODES, 'val_mse')][0]
                self.assertEqual(wins['left_wins'], 10)
                with self.assertRaises(FileExistsError):
                    analysis.analyze(new, old)
            def partial(run, cache):
                result = fixture(run, cache)
                if run.variant == analysis.LAST and run.ticker == 'AAPL' and run.seed == 7:
                    result[0].update(comparable=False, match_verification='partial', evidence_missing='batch order')
                return result
            with patch.object(analysis, 'inspect', side_effect=partial), contextlib.redirect_stdout(io.StringIO()):
                tables, selection = analysis.analyze(new, old, output_dir=new / 'partial')
            self.assertEqual(selection['status'], 'full')
            self.assertEqual(selection['last_comparison'], 'partial')
            uncertain = [r for r in tables['paired_by_ticker'] if r['right'] == analysis.LAST and r['ticker'] == 'AAPL']
            self.assertTrue(all(r['difference'] is None and r['improvement_pct'] is None for r in uncertain))

    def test_changed_common_encoder_initialization_or_order_blocks(self):
        for field in ('encoder', 'orders', 'windows', 'prediction'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                new, old = self.setup_roots(Path(directory))
                def mismatch(run, cache):
                    result = fixture(run, cache)
                    if run.variant == analysis.HUANG_MODES[1] and run.ticker == 'AAPL' and run.seed == 7:
                        if field == 'encoder':
                            result[1]['args']['huang_initialization']['macro'] = '0' * 64
                        elif field == 'orders':
                            result[3]['orders'] = ['changed']
                        elif field == 'windows':
                            result[1]['evidence'] = {'changed': True}
                        else:
                            result[1]['args']['initial_prediction_evidence']['prediction_sha256'] = '0' * 64
                    return result
                with patch.object(analysis, 'inspect', side_effect=mismatch), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'comparison blocked'):
                    analysis.analyze(new, old)
                report = new / 'diagnostics/query_comparison'
                self.assertEqual(json.loads((report / 'selection.json').read_text())['status'], 'blocked')

    def test_missing_control_is_not_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = analysis.inspect(Run(analysis.LAST, 'AAPL', 7, root / 'missing'), analysis.ReplayCache(root))
            self.assertFalse(missing[0]['comparable'])
            record = analysis.verify_pair(fixture(Run(analysis.HUANG_MODES[0], 'AAPL', 7, root), None), missing)
            self.assertNotEqual(record['window_verification'], 'full')
            self.assertIn('summary.json', record['reason'])


if __name__ == '__main__':
    unittest.main()
