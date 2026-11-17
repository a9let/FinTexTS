"""Pair-specific evidence, validation-only selection and unavailable saved controls."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_context_price_han_window as analysis
from analyze_persistence_baseline import Run


def fixture(run, cache):
    value = {analysis.HAN_WINDOW_MODES[0]: .3, analysis.HAN_WINDOW_MODES[1]: .2,
             analysis.HAN20: .1, analysis.LAST: .4}[run.variant]
    row = {'variant': run.variant, 'ticker': run.ticker, 'seed': run.seed, 'output': str(run.output),
           'comparable': True, 'window_verification': 'full', 'match_verification': 'full',
           'missing_fields': '[]', 'mismatches': '[]', 'evidence_missing': '[]', 'evidence_reason': '',
           'val_mse': value, 'test_mse': 1 - value, 'test_mae': .5}
    initial = {'film_sha256': 'a' * 64, 'price_gate_sha256': 'b' * 64,
               'heads_sha256': ('c' if run.variant in analysis.WINDOWS else 'd') * 64}
    args = {**analysis.heads.han.prior.COMMON, **{key: 'same' for key in analysis.heads.han.prior.PRICE_FIELDS},
            'd_model': 64, 'dropout': .1, 'novelty_policy': 'same', 'sampling_policy': 'same', 'gate_structure': {},
            'common_initialization': initial, 'head_ablation_initialization': {'encoder_input_projection_sha256': 'e' * 64,
                                                                              'encoder_sha256': 'f' * 64},
            'code_commit': ('a' if run.variant in analysis.HAN_WINDOW_MODES else 'b') * 40}
    meta = {'args': args, 'evidence': {'same': True}, 'split': {},
            'scaler': {key: [1., 2., 3., 4.] for key in ('mean', 'scale', 'var')}}
    proof = {'initialization': initial, 'source': {'same': True}, 'orders': ['epoch1']}
    return row, meta, {}, proof


class HANWindowAnalysisTests(unittest.TestCase):
    def setup_roots(self, root):
        new, old = root / 'new', root / 'old'
        for mode in analysis.MODES:
            for ticker in analysis.TICKERS:
                for seed in analysis.SEEDS:
                    output = (new if mode in analysis.HAN_WINDOW_MODES else old) / 'full' / mode / ticker / f'seed{seed}' / 'output'
                    output.mkdir(parents=True)
                    (output / 'best_checkpoint.pt').write_text(f'{mode}{ticker}{seed}')
        return new, old

    def test_strict_report_and_validation_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            new, old = self.setup_roots(Path(directory))
            with patch.object(analysis, 'inspect', side_effect=fixture), contextlib.redirect_stdout(io.StringIO()):
                tables, selection = analysis.analyze(new, old)
                self.assertEqual(len(tables['per_run']), 120)
                self.assertEqual(len(tables['window_verification']), 180)
                self.assertEqual(selection['status'], 'full')
                self.assertEqual(selection['selected_among_confirmed_windows'], [analysis.HAN20])
                self.assertFalse(selection['test_used_for_selection'])
                primary = [r for r in tables['wins'] if (r['left'], r['right'], r['metric']) == (*analysis.HAN_WINDOW_MODES, 'val_mse')][0]
                self.assertEqual(primary['right_wins'], 10)
                for row in tables['across_seeds']:
                    self.assertAlmostEqual(row['sample_std'], 0.)
                with self.assertRaises(FileExistsError):
                    analysis.analyze(new, old)

    def test_partial_control_does_not_disable_other_pairs_or_claim_strict_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            new, old = self.setup_roots(Path(directory))
            def partial(run, cache):
                item = fixture(run, cache)
                if run.variant == analysis.HAN20 and run.ticker == 'AAPL' and run.seed == 7:
                    item[0].update(comparable=False, match_verification='partial', evidence_missing='initialization missing')
                return item
            with patch.object(analysis, 'inspect', side_effect=partial), contextlib.redirect_stdout(io.StringIO()):
                tables, selection = analysis.analyze(new, old)
            self.assertEqual(selection['status'], 'partial')
            self.assertNotIn(analysis.HAN20, selection['eligible_windows'])
            self.assertEqual(selection['selected_among_confirmed_windows'], [analysis.HAN_WINDOW_MODES[1]])
            primary = [r for r in tables['window_verification'] if (r['left'], r['right']) == analysis.HAN_WINDOW_MODES]
            self.assertTrue(all(r['window_verification'] == 'full' for r in primary))
            pair = [r for r in tables['paired_by_ticker'] if r['right'] == analysis.HAN20 and r['ticker'] == 'AAPL'][0]
            self.assertIsNone(pair['difference'])
            self.assertIsNone(pair['improvement_pct'])

    def test_initialization_batch_order_and_window_mismatches_block(self):
        for kind in ('encoder', 'orders', 'windows', 'heads'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                new, old = self.setup_roots(Path(directory))
                def changed(run, cache):
                    item = fixture(run, cache)
                    if run.variant == analysis.HAN_WINDOW_MODES[1] and run.ticker == 'AAPL' and run.seed == 7:
                        if kind == 'encoder':
                            item[1]['args']['head_ablation_initialization']['encoder_sha256'] = '0' * 64
                        elif kind == 'orders':
                            item[3]['orders'] = ['different']
                        elif kind == 'heads':
                            item[1]['args']['common_initialization']['heads_sha256'] = '0' * 64
                        else:
                            item[1]['evidence'] = {'different': True}
                    return item
                with patch.object(analysis, 'inspect', side_effect=changed), contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaisesRegex(ValueError, 'comparison blocked'):
                    analysis.analyze(new, old)
                report = new / 'diagnostics/window_comparison'
                self.assertEqual(json.loads((report / 'selection.json').read_text())['status'], 'blocked')
                self.assertIn('mismatch', (report / 'window_verification.csv').read_text())

    def test_missing_old_artifacts_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = analysis.inspect(Run(analysis.HAN20, 'AAPL', 7, root / 'missing'), analysis.ReplayCache(root))
            self.assertFalse(missing[0]['comparable'])
            self.assertEqual(missing[0]['window_verification'], 'not_checked')
            record = analysis.pair_record(fixture(Run(analysis.HAN_WINDOW_MODES[0], 'AAPL', 7, root), None), missing)
            self.assertNotEqual(record['window_verification'], 'full')
            self.assertIn('summary.json', record['reason'])

    def test_unexpected_new_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            new, old = self.setup_roots(Path(directory))
            with patch.object(analysis, 'inspect', side_effect=fixture):
                # An extra ticker cannot silently enter the fixed experimental cohort.
                first = new / 'full' / analysis.HAN_WINDOW_MODES[0] / 'AAPL/seed7/output/best_checkpoint.pt'
                extra = new / 'full' / analysis.HAN_WINDOW_MODES[0] / 'EXTRA/seed7/output'
                extra.mkdir(parents=True)
                (extra / 'best_checkpoint.pt').write_bytes(first.read_bytes())
                with self.assertRaisesRegex(ValueError, 'unexpected, missing or duplicate'):
                    analysis.analyze(new, old)


if __name__ == '__main__':
    unittest.main()
