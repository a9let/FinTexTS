"""Grid cells, separate width/length effects and partial saved controls."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import analyze_context_price_huang_grid as analysis
from analyze_persistence_baseline import Run
from forecasting_task.models.GroupHuangQuery import huang_options
import test_huang_analysis as core


def fixture(run, cache):
    actual = analysis.LAST if run.variant == analysis.LAST else 'huang_recurrent_query'
    item = core.fixture(Run(actual, run.ticker, run.seed, run.output), cache)
    row, metadata, _, _ = item
    row['variant'] = run.variant
    value = {'h16_s5_l10': .1, 'h16_s20_l10': .2, 'h32_s5_l10': .3, analysis.OLD: .4, analysis.LAST: .5}[run.variant]
    row.update(val_mse=value, test_mse=1 - value, test_mae=.5)
    args = metadata['args']
    args['text_temporal_pool'] = actual
    args['code_commit'] = ('a' if run.variant in analysis.NEW else 'b') * 40
    if run.variant != analysis.LAST:
        parameters = huang_options(analysis.PARAMETERS[run.variant])
        if run.variant != analysis.OLD:
            args.update(parameters)
        hidden = parameters['huang_hidden_size']
        args['huang_initialization'] = dict.fromkeys(analysis.query.TEXT_GROUPS, str(hidden) * 32)
        args['temporal_initialization_sha256'] = str(hidden) * 32
    return item


class HuangGridAnalysisTests(unittest.TestCase):
    def test_full_huang_initial_fingerprint_records_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = Run('huang_recurrent_query', 'AAPL', 7, root)
            args = {'seed': 7, 'source_data_evidence': {'parquet_sha256': 'a' * 64},
                    'huang_initialization': dict.fromkeys(analysis.query.TEXT_GROUPS, 'b' * 64),
                    'initial_prediction_evidence': {'matches_last_matched_formula': True, 'max_abs_difference': 0,
                                                    'input_end_date': '2022-01-03', 'prediction_sha256': 'c' * 64},
                    'temporal_initialization_sha256': 'd' * 64}
            (root / 'summary.json').write_text(json.dumps(args))
            metadata = {'args': args, 'datasets': {'val': SimpleNamespace(window_end_date=lambda i: '2022-01-03')}}
            cache = SimpleNamespace(parquet_hash=lambda ticker: 'a' * 64)
            for kind in ('matching', 'different', 'missing'):
                checkpoint = dict(args)
                if kind == 'different':
                    checkpoint['temporal_initialization_sha256'] = 'e' * 64
                elif kind == 'missing':
                    checkpoint.pop('temporal_initialization_sha256')
                result = ({'comparable': True, 'match_verification': 'full'}, metadata, {}, {})
                with patch.object(analysis.query.heads, 'inspect', return_value=result), patch.object(analysis.query.torch, 'load', return_value=checkpoint):
                    inspected = analysis.query.inspect(run, cache)
                self.assertEqual(inspected[0]['comparable'], kind == 'matching')
                self.assertEqual(inspected[0]['match_verification'], {'matching': 'full', 'different': 'mismatch', 'missing': 'partial'}[kind])

    def setup_roots(self, root):
        new, old, last = root / 'new', root / 'old', root / 'last'
        for mode in analysis.MODES:
            base, label = (last, mode) if mode == analysis.LAST else (old, 'huang_recurrent_query') if mode == analysis.OLD else (new, mode)
            for ticker in analysis.query.TICKERS:
                for seed in analysis.query.SEEDS:
                    output = base / 'full' / label / ticker / f'seed{seed}/output'
                    output.mkdir(parents=True)
                    (output / 'best_checkpoint.pt').write_text(f'{mode}{ticker}{seed}')
        return new, old, last

    def test_all_four_cells_effects_and_validation_only_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            new, old, last = self.setup_roots(Path(directory))
            with patch.object(analysis, 'inspect', side_effect=fixture), contextlib.redirect_stdout(io.StringIO()) as log:
                tables, selection = analysis.analyze(new, old, last)
                self.assertEqual(len(tables['per_run']), 150)
                self.assertEqual(len(tables['window_verification']), 300)
                self.assertEqual(len(tables['grid_2x2']), 6)
                self.assertEqual(len(tables['effects_by_seed']), 36)
                self.assertEqual(len(tables['effects_across_seeds']), 12)
                self.assertEqual(selection['status'], 'full')
                self.assertEqual(selection['selected'], ['h16_s5_l10'])
                self.assertFalse(selection['test_used_for_selection'])
                self.assertIn('GRID val_mse: H / steps | 5 | 20', log.getvalue())
                effects = {r['effect']: r for r in tables['effects_across_seeds'] if r['metric'] == 'val_mse'}
                self.assertAlmostEqual(effects['hidden_at_steps5']['mean_difference'], -.2)
                self.assertAlmostEqual(effects['hidden_at_steps20']['mean_difference'], -.2)
                self.assertAlmostEqual(effects['steps_at_hidden16']['mean_difference'], -.1)
                self.assertAlmostEqual(effects['steps_at_hidden32']['mean_difference'], -.1)
                with self.assertRaises(FileExistsError):
                    analysis.analyze(new, old, last)

    def test_partial_historical_cell_keeps_new_comparisons_but_limits_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            new, old, last = self.setup_roots(Path(directory))
            def partial(run, cache):
                item = fixture(run, cache)
                if run.variant == analysis.OLD and run.ticker == 'AAPL' and run.seed == 7:
                    item[0].update(comparable=False, match_verification='partial', evidence_missing='batch order')
                return item
            with patch.object(analysis, 'inspect', side_effect=partial), contextlib.redirect_stdout(io.StringIO()):
                tables, selection = analysis.analyze(new, old, last)
            self.assertEqual(selection['status'], 'partial')
            self.assertEqual(selection['eligible_cells'], list(analysis.NEW))
            pairs = [r for r in tables['paired_by_ticker'] if r['right'] == analysis.OLD and r['ticker'] == 'AAPL']
            self.assertTrue(all(r['difference'] is None for r in pairs))
            self.assertTrue(all(r['window_verification'] == 'full' for r in tables['window_verification'] if r['left'] in analysis.NEW and r['right'] in analysis.NEW))

    def test_equal_h_query_initialization_and_batch_mismatch_block_new_grid(self):
        for field in ('query_init', 'orders'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                new, old, last = self.setup_roots(Path(directory))
                def changed(run, cache):
                    item = fixture(run, cache)
                    if run.variant == 'h16_s5_l10' and run.ticker == 'AAPL' and run.seed == 7:
                        if field == 'query_init':
                            item[1]['args']['temporal_initialization_sha256'] = '0' * 64
                        else:
                            item[3]['orders'] = ['changed']
                    return item
                with patch.object(analysis, 'inspect', side_effect=changed), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'comparison blocked'):
                    analysis.analyze(new, old, last)
                self.assertEqual(json.loads((new / 'diagnostics/grid_comparison/selection.json').read_text())['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()
