"""Replace comparisons require shared price initialization, not last head identity."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_context_price_huang_replace_grid as analysis
from analyze_persistence_baseline import Run
from forecasting_task.models.GroupHuangQuery import HUANG_REPLACE, huang_options
import test_huang_analysis as core


def fixture(run, cache):
    mode = analysis.LAST if run.variant == analysis.LAST else 'huang_recurrent_query'
    item = core.fixture(Run(mode, run.ticker, run.seed, run.output), cache)
    row, metadata, _, _ = item
    row['variant'] = run.variant
    values = {'h16_s5_l10': .1, 'h16_s20_l10': .2, 'h32_s5_l10': .3, 'h32_s20_l10': .4, analysis.LAST: .5}
    row.update(val_mse=values[run.variant], test_mse=1 - values[run.variant], test_mae=.5)
    args = metadata['args']
    args['text_temporal_pool'] = analysis.LAST if run.variant == analysis.LAST else HUANG_REPLACE
    args['code_commit'] = ('b' if run.variant == analysis.LAST else 'a') * 40
    if run.variant != analysis.LAST:
        args.update(huang_options(analysis.CONFIGURATIONS[run.variant]))
        hidden = args['huang_hidden_size']
        args['huang_initialization'] = dict.fromkeys(analysis.query.TEXT_GROUPS, str(hidden) * 32)
        args['temporal_initialization_sha256'] = str(hidden) * 32
        args['common_initialization']['heads_sha256'] = 'f' * 64
        args['initial_prediction_evidence'] = None
    return item


class ReplaceAnalysisTests(unittest.TestCase):
    def setup_roots(self, root):
        new, last = root / 'new', root / 'last'
        for mode in analysis.MODES:
            for ticker in analysis.query.TICKERS:
                for seed in analysis.query.SEEDS:
                    output = (last if mode == analysis.LAST else new) / 'full' / mode / ticker / f'seed{seed}/output'
                    output.mkdir(parents=True)
                    (output / 'best_checkpoint.pt').write_text(f'{mode}{ticker}{seed}')
        return new, last

    def test_four_fresh_cells_last_different_heads_and_validation_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            new, last = self.setup_roots(Path(directory))
            with patch.object(analysis, 'inspect', side_effect=fixture), contextlib.redirect_stdout(io.StringIO()):
                tables, selection = analysis.analyze(new, last)
                self.assertEqual(len(tables['per_run']), 150)
                self.assertEqual(len(tables['window_verification']), 300)
                self.assertEqual(len(tables['grid_2x2']), 6)
                self.assertEqual(len(tables['effects_by_seed']), 36)
                self.assertEqual(selection['selected'], ['h16_s5_l10'])
                self.assertEqual(selection['status'], 'full')
                self.assertFalse(selection['test_used_for_selection'])
                self.assertFalse(selection['historical_huang_used'])
                self.assertEqual(selection['last_comparison'], 'full')
                self.assertEqual(selection['including_last']['selected'], ['h16_s5_l10'])
                self.assertTrue(all(r['window_verification'] == 'full' for r in tables['window_verification']))
                with self.assertRaises(FileExistsError):
                    analysis.analyze(new, last)

    def test_partial_last_keeps_grid_selection_but_suppresses_strict_pair_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            new, last = self.setup_roots(Path(directory))
            def partial(run, cache):
                item = fixture(run, cache)
                if run.variant == analysis.LAST and run.ticker == 'AAPL' and run.seed == 7:
                    item[0].update(comparable=False, match_verification='partial', evidence_missing='batch order')
                return item
            with patch.object(analysis, 'inspect', side_effect=partial), contextlib.redirect_stdout(io.StringIO()):
                tables, selection = analysis.analyze(new, last)
            self.assertEqual(selection['status'], 'full')
            self.assertEqual(selection['last_comparison'], 'partial')
            self.assertEqual(selection['including_last']['selected'], [])
            uncertain = [r for r in tables['paired_by_ticker'] if r['right'] == analysis.LAST and r['ticker'] == 'AAPL']
            self.assertTrue(all(r['difference'] is None and r['improvement_pct'] is None for r in uncertain))

    def test_saved_last_can_win_over_all_replace_cells_on_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            new, last = self.setup_roots(Path(directory))
            def last_better(run, cache):
                item = fixture(run, cache)
                if run.variant == analysis.LAST:
                    item[0].update(val_mse=.01, test_mse=10.)
                return item
            with patch.object(analysis, 'inspect', side_effect=last_better), contextlib.redirect_stdout(io.StringIO()):
                _, selection = analysis.analyze(new, last)
            self.assertEqual(selection['selected_grid'], ['h16_s5_l10'])
            self.assertEqual(selection['including_last']['selected'], [analysis.LAST])
            self.assertFalse(selection['including_last']['test_used_for_selection'])

    def test_common_price_or_same_h_huang_mismatch_blocks(self):
        for field in ('film', 'gate', 'huang', 'orders'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                new, last = self.setup_roots(Path(directory))
                def mismatch(run, cache):
                    item = fixture(run, cache)
                    if run.variant == 'h16_s5_l10' and run.ticker == 'AAPL' and run.seed == 7:
                        if field == 'huang':
                            item[1]['args']['temporal_initialization_sha256'] = '0' * 64
                        elif field == 'orders':
                            item[3]['orders'] = ['different']
                        else:
                            key = 'film_sha256' if field == 'film' else 'price_gate_sha256'
                            item[1]['args']['common_initialization'][key] = '0' * 64
                    return item
                with patch.object(analysis, 'inspect', side_effect=mismatch), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'comparison blocked'):
                    analysis.analyze(new, last)
                self.assertEqual(json.loads((new / 'diagnostics/replace_grid_comparison/selection.json').read_text())['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()
