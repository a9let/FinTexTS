"""Matched saved metadata, paired statistics and validation-only selection."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

import analyze_context_price_only as analysis


class PriceOnlyAnalysisTests(unittest.TestCase):
    def fixtures(self, directory):
        roots = [Path(directory) / name for name in ("price", "reference", "baseline")]
        for variant, root in zip(analysis.VARIANTS, roots):
            for index, ticker in enumerate(analysis.TICKERS):
                for seed_index, seed in enumerate(analysis.SEEDS):
                    branch = "baseline" if variant == "original_baseline" else variant
                    output = root / "full" / branch / ticker / f"seed{seed}" / "output"
                    output.mkdir(parents=True)
                    args = {**analysis.COMMON, "seed": seed, "model_type": "FiLM", "data_path": ticker + ".parquet",
                            "task_name": "short_term_forecast", "e_layers": 2, "output_attention": False,
                            "enc_in": 4, "dec_in": 4, "c_out": 4, "fusion_mode": "blend", "text_num_layers": 1,
                            "used_col_prefixes": analysis.GROUPS, "text_weight": 0.1,
                            "group_gate": variant, "fixed_alpha": 0.1, "text_weight_mode": "fixed",
                            "group_weight_mode": variant, "text_model_config": {"group_gate": variant},
                            "novelty_policy": {"previous": "earlier present row"}, "sampling_policy": "independent_train_generator",
                            "gate_structure": {"parameters": 95}, "text_mlp_structure": [{"type": "Linear", "in_features": 384, "out_features": 3}]}
                    val = {"context_price_only": 0.3, "context_no_novelty": 0.2, "original_baseline": 1.5}[variant]
                    test = 0.3 if variant != "context_price_only" or index == 0 else 0.1
                    summary = {"model": "FiLM", "data_path": args["data_path"], "seed": seed, "best_epoch": 1,
                               "best_val_mse": val + index * 0.01 + seed_index * 0.05,
                               "best_test_mse": test + seed_index * 0.02, "best_test_mae": test / 2 + seed_index * 0.01,
                               "group_gate": variant, "fixed_alpha": 0.1, "text_weight_mode": "fixed"}
                    (output / "summary.json").write_text(json.dumps(summary))
                    torch.save({"args": args, "epoch": 1, "scaler_mean": [1., 2., 3., 4.],
                                "scaler_scale": [1.] * 4, "scaler_var": [1.] * 4}, output / "best_checkpoint.pt")
        return roots

    @staticmethod
    def run_analysis(roots):
        with contextlib.redirect_stdout(io.StringIO()):
            return analysis.analyze(*roots)

    def test_full_read_only_report_pairing_sample_std_and_validation_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = self.fixtures(directory)
            original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for root in roots for path in root.rglob("*.*")}
            tables, selection = self.run_analysis(roots)
            self.assertEqual(len(tables["per_run"]), 90)
            self.assertEqual(len(tables["by_seed"]), 9)
            self.assertEqual(len(tables["paired_by_ticker"]), 90)
            self.assertEqual(selection["selected_variants"], ["context_no_novelty"])
            self.assertFalse(selection["test_used_for_selection"])
            stats = next(r for r in tables["across_seeds"] if r["variant"] == "context_no_novelty" and r["metric"] == "val_mse")
            self.assertAlmostEqual(stats["mean"], 0.295)
            self.assertAlmostEqual(stats["sample_std"], 0.05)
            self.assertEqual(stats["std_ddof"], 1)
            wins = next(r for r in tables["wins"] if r["right"] == "context_no_novelty" and r["metric"] == "test_mse")
            self.assertEqual((wins["left_wins"], wins["right_wins"], wins["ties"]), (9, 0, 1))
            self.assertEqual(wins["comparable_tickers"], 10)
            self.assertTrue((roots[0] / "diagnostics/comparison_10/selection.json").is_file())
            for path, digest in original.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                self.run_analysis(roots)

    def test_primary_configuration_and_scaler_mismatches_stop_without_report(self):
        for field in ("seq_len", "scaler_mean"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                roots = self.fixtures(directory)
                path = roots[1] / "full/context_no_novelty/AAPL/seed7/output/best_checkpoint.pt"
                payload = torch.load(path, weights_only=True)
                if field == "seq_len":
                    payload["args"][field] = 32
                else:
                    payload[field][0] = 10
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "Cannot compare primary run"):
                    self.run_analysis(roots)
                self.assertFalse((roots[0] / "diagnostics").exists())

    def test_incompatible_baseline_is_visible_but_not_paired_or_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = self.fixtures(directory)
            path = roots[2] / "full/baseline/AAPL/seed7/output/best_checkpoint.pt"
            payload = torch.load(path, weights_only=True)
            payload["args"]["e_layers"] = 3
            torch.save(payload, path)
            tables, selection = self.run_analysis(roots)
            self.assertIn("original_baseline", selection["excluded_variants"])
            self.assertTrue(any("e_layers" in r["reason"] for r in tables["issues"]))
            rows = [r for r in tables["paired_by_ticker"] if r["right"] == "original_baseline" and r["ticker"] == "AAPL"]
            self.assertTrue(all(r["left_minus_right"] is None and not r["comparable"] for r in rows))
            baseline = next(r for r in tables["per_run"] if r["variant"] == "original_baseline" and r["ticker"] == "AAPL" and r["seed"] == 7)
            self.assertIsNotNone(baseline["test_mse"])

    def test_missing_validation_is_never_filled_using_test(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = self.fixtures(directory)
            path = roots[2] / "full/baseline/AAPL/seed7/output/summary.json"
            payload = json.loads(path.read_text())
            del payload["best_val_mse"]
            path.write_text(json.dumps(payload))
            tables, selection = self.run_analysis(roots)
            self.assertIn("original_baseline", selection["excluded_variants"])
            stats = next(r for r in tables["across_seeds"] if r["variant"] == "original_baseline" and r["metric"] == "val_mse")
            self.assertIsNone(stats["mean"])
            self.assertIsNone(stats["sample_std"])
            self.assertEqual(selection["selected_variants"], ["context_no_novelty"])
