"""Strict reuse, partial historical controls and validation-only HAN selection."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import analyze_context_price_han_temporal as analysis


class HANAnalysisTests(unittest.TestCase):
    def test_historical_missing_initial_text_and_batch_evidence_stays_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {'epochs_completed': 1}
            (root / 'summary.json').write_text(json.dumps(summary))
            (root / 'history.jsonl').write_text(json.dumps({'epoch': 1}) + '\n')
            torch.save({'epoch': 1}, root / 'best_checkpoint.pt')
            evidence, missing = analysis.provenance(root, {'args': {}})
            self.assertIsNone(evidence['initialization'])
            self.assertIn('common_initialization', missing)
            self.assertIn('source_data_evidence', missing)
            self.assertIn('actual train batch order by epoch', missing)

    def test_report_partial_controls_optional_matched_last_and_mismatch_block(self):
        for strict in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                results = root / 'results'
                for mode in analysis.HAN_MODES:
                    for ticker in analysis.TICKERS:
                        for seed in analysis.SEEDS:
                            path = results / 'full' / mode / ticker / f'seed{seed}' / 'output'
                            path.mkdir(parents=True)
                            (path / 'best_checkpoint.pt').write_text(f'{mode}{ticker}{seed}')
                def inspect(run, cache):
                    eligible = run.variant in analysis.HAN_MODES or strict
                    metric = {'group_han_temporal': .2, 'group_han_uniform': .1, 'last': .05, 'group_gru_attention': .01}[run.variant]
                    row = {'variant': run.variant, 'ticker': run.ticker, 'seed': run.seed, 'output': str(run.output),
                           'window_verification': 'full', 'match_verification': 'full' if eligible else 'partial',
                           'comparable': eligible, 'missing_fields': '[]', 'mismatches': '[]',
                           'evidence_missing': '[]' if eligible else '["common_initialization"]', 'evidence_reason': '',
                           'common_epochs': None, 'val_mse': metric, 'test_mse': 1 - metric, 'test_mae': .5}
                    args = {**analysis.prior.COMMON, **{key: 'same' for key in analysis.prior.PRICE_FIELDS}}
                    meta = {'args': args, 'evidence': {'same': True}, 'split': {},
                            'scaler': {k: [1, 2, 3, 4] for k in ('mean', 'scale', 'var')}}
                    evidence = {'initialization': {'same': True}, 'source': {'same': True}, 'orders': ['epoch1', 'epoch2']}
                    return row, meta, {}, evidence
                with patch.object(analysis, 'inspect', side_effect=inspect), \
                     patch.object(analysis.prior, 'movement_records', return_value=([], [])), contextlib.redirect_stdout(io.StringIO()):
                    tables, selection = analysis.analyze(results, root / 'old')
                    self.assertEqual(len(tables['per_run']), 120)
                    self.assertEqual(selection['selected'], ['last'] if strict else ['group_han_uniform'])
                    self.assertEqual(selection['last_confirmed'], strict)
                    self.assertNotIn('group_gru_attention', selection['scores'])
                    if not strict:
                        self.assertTrue(all(not r['comparable'] for r in tables['paired_by_ticker'] if r['right'] == 'last'))
                    with self.assertRaises(FileExistsError):
                        analysis.analyze(results, root / 'old')
                def altered(run, cache):
                    row, meta, predictions, evidence = inspect(run, cache)
                    if run.variant == 'group_han_uniform' and run.ticker == 'AAPL' and run.seed == 7:
                        evidence['orders'] = ['different_order']
                    return row, meta, predictions, evidence
                failed = root / 'failed'
                with patch.object(analysis, 'inspect', side_effect=altered), \
                     patch.object(analysis.prior, 'movement_records', return_value=([], [])), contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaisesRegex(ValueError, 'comparison blocked'):
                    analysis.analyze(results, root / 'old', output_dir=failed)
                self.assertIn('batch order differs', (failed / 'window_verification.csv').read_text())
                self.assertEqual(json.loads((failed / 'selection.json').read_text())['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()
