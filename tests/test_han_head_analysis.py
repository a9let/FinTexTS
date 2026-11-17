"""Five-condition selection, paired comparisons and old bridge verification."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import analyze_context_price_han_head_ablation as analysis
from forecasting_task.models.GroupText import build_text_model
from forecasting_task.experiment_evidence import common_initialization, state_fingerprint
import test_han_head_ablation as core


class HeadAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_report_uses_validation_and_blocks_common_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / 'new'
            for mode in analysis.ABLATION_MODES:
                for ticker in analysis.TICKERS:
                    for seed in analysis.SEEDS:
                        path = results / 'full' / mode / ticker / f'seed{seed}' / 'output'
                        path.mkdir(parents=True)
                        (path / 'best_checkpoint.pt').write_text(f'{mode}{ticker}{seed}')
            def inspect(run, cache):
                eligible = run.variant != analysis.BRIDGE
                value = dict(last_matched=.3, han_h32_linear=.2, han_h16_linear=.25, han_h32_mlp=.15, last_han_h32_residual=.1, group_han_temporal=.01)[run.variant]
                row = {'variant': run.variant, 'ticker': run.ticker, 'seed': run.seed, 'output': str(run.output),
                       'window_verification': 'full', 'match_verification': 'full' if eligible else 'partial',
                       'comparable': eligible, 'common_epochs': None, 'missing_fields': '[]', 'mismatches': '[]',
                       'evidence_missing': '[]', 'evidence_reason': '',
                       'val_mse': value, 'test_mse': 1 - value, 'test_mae': .5}
                heads = 'a' * 64 if run.variant in ('last_matched', 'last_han_h32_residual') else 'b' * 64
                initial = {'film_sha256': 'f' * 64, 'price_gate_sha256': 'e' * 64, 'heads_sha256': heads}
                proof = {'encoder_input_projection_sha256': 'd' * 64,
                         'encoder_sha256': ('c' if run.variant == 'han_h16_linear' else 'b') * 64}
                args = {**analysis.han.prior.COMMON, **{key: 'same' for key in analysis.han.prior.PRICE_FIELDS},
                        'code_commit': 'a' * 40, 'common_initialization': initial, 'head_ablation_initialization': proof,
                        'initial_prediction_evidence': {'matches_last_matched_formula': True, 'prediction_sha256': 'a' * 64}}
                meta = {'args': args, 'evidence': {'same': True}, 'split': {},
                        'scaler': {key: [1, 2, 3, 4] for key in ('mean', 'scale', 'var')}}
                evidence = {'initialization': initial, 'source': {'same': True}, 'orders': ['epoch1']}
                return row, meta, {}, evidence
            with patch.object(analysis, 'inspect', side_effect=inspect), contextlib.redirect_stdout(io.StringIO()):
                tables, selection = analysis.analyze(results, root / 'old')
                self.assertEqual(len(tables['per_run']), 180)
                self.assertEqual(selection['selected'], ['last_han_h32_residual'])
                self.assertFalse(selection['test_used_for_selection'])
                self.assertEqual(selection['bridge_comparison'], 'partial')
                means = {(r['left'], r['right'], r['metric']): r for r in tables['wins']}
                self.assertEqual(means['han_h16_linear', 'han_h32_linear', 'val_mse']['right_wins'], 10)
                self.assertEqual(means['han_h32_mlp', 'han_h32_linear', 'val_mse']['left_wins'], 10)
                with self.assertRaises(FileExistsError):
                    analysis.analyze(results, root / 'old')
            def altered(run, cache):
                result = inspect(run, cache)
                if run.ticker == 'AAPL' and run.seed == 7 and run.variant == 'han_h32_mlp':
                    result[1]['args']['head_ablation_initialization']['encoder_sha256'] = '0' * 64
                return result
            failed = root / 'failed'
            with patch.object(analysis, 'inspect', side_effect=altered), contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(ValueError, 'comparison blocked'):
                analysis.analyze(results, root / 'old', output_dir=failed)
            self.assertEqual(json.loads((failed / 'selection.json').read_text())['status'], 'blocked')
            self.assertIn('BiGRU/attention initialization differs', (failed / 'window_verification.csv').read_text())

    def test_old_bridge_encoder_can_be_verified_without_loading_trained_weights(self):
        args = {**analysis.han.prior.COMMON, 'seq_len': 20, 'label_len': 2, 'seed': 7,
                'task_name': 'short_term_forecast', 'e_layers': 1, 'output_attention': False,
                'enc_in': 4, 'dec_in': 4, 'c_out': 4, 'd_model': 64, 'dropout': .1}
        pairs = []
        for mode in ('han_h32_linear', analysis.BRIDGE):
            config = core.config(mode) if mode == 'han_h32_linear' else core.core.config(mode)
            with torch.random.fork_rng(devices=[]), analysis.execution_device(torch.device('cpu')):
                torch.manual_seed(7)
                film = analysis.film_module.Model(SimpleNamespace(**args))
                text = build_text_model(config, 'context_price_only')
                saved = {**args, 'text_model_config': config, 'common_initialization': common_initialization(film, text),
                         'temporal_initialization_sha256': state_fingerprint(text.temporal_pool),
                         'head_ablation_initialization': {'encoder_sha256': state_fingerprint(text.temporal_pool)}}
            metadata = {'args': saved, 'evidence': {}, 'split': {}, 'scaler': {k: [1, 2, 3, 4] for k in ('mean', 'scale', 'var')}}
            evidence = {'initialization': saved['common_initialization'], 'source': {}, 'orders': ['epoch1']}
            pairs.append(({'variant': mode}, metadata, {}, evidence))
        analysis.verify_bridge(pairs[0], pairs[1])
        pairs[1][1]['args']['temporal_initialization_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'partially'):
            analysis.verify_bridge(pairs[0], pairs[1])


if __name__ == '__main__':
    unittest.main()
