"""Saved target-window alignment, report statistics and primary validation selection."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import analyze_context_price_mean5 as analysis
from forecasting_task.window_evidence import window_evidence, check_saved_test_windows


class WindowDataset:
    seq_len, pred_len = 64, 3
    row_dates = pd.bdate_range("2022-01-03", periods=72).strftime("%Y-%m-%d").to_numpy()

    def __len__(self):
        return 4

    def __getitem__(self, index):
        prices = np.arange(72 * 4, dtype=np.float32).reshape(72, 4) / 100
        return prices[index:index + 64], prices[index + 48:index + 67]

    def window_end_date(self, index):
        return str(self.row_dates[index + 63])


class Mean5AnalysisTests(unittest.TestCase):
    def fixture(self, directory):
        roots = [Path(directory) / name for name in ("mean5", "last", "context", "baseline")]
        dataset = WindowDataset()
        signatures = {split: window_evidence(dataset) for split in ("val", "test")}
        scalers = {split: {"mean": np.arange(4, dtype=float), "scale": np.ones(4), "var": np.ones(4)} for split in signatures}
        for variant, root in zip(analysis.VARIANTS, roots):
            gate = "context_price_only" if variant in ("mean5", "last") else variant
            folder = "baseline" if variant == "original_baseline" else gate
            pool = "mean5" if variant == "mean5" else "last"
            for ticker in analysis.TICKERS:
                for seed_index, seed in enumerate(analysis.SEEDS):
                    output = root / "full" / folder / ticker / f"seed{seed}" / "output"
                    output.mkdir(parents=True)
                    args = {**analysis.COMMON, "model_type": "FiLM", "seed": seed, "data_path": ticker + ".parquet",
                            "task_name": "short_term_forecast", "e_layers": 2, "output_attention": False,
                            "enc_in": 4, "dec_in": 4, "c_out": 4, "group_gate": gate, "group_weight_mode": gate,
                            "fixed_alpha": 0.1, "text_weight_mode": "fixed", "text_weight": 0.1,
                            "text_num_layers": 1, "used_col_prefixes": analysis.GROUPS, "fusion_mode": "blend",
                            "text_model_config": {"group_gate": gate}, "novelty_policy": {"previous": "earlier present row"},
                            "sampling_policy": "independent_train_generator", "gate_structure": {"parameters": 95},
                            "text_mlp_structure": [{"type": "Linear", "in_features": 384, "out_features": 3}]}
                    val = {"mean5": 0.3, "last": 0.2, "context_no_novelty": 0.1, "original_baseline": 0.4}[variant] + seed_index * 0.05
                    summary = {"model": "FiLM", "seed": seed, "data_path": args["data_path"], "best_epoch": 1,
                               "group_gate": gate, "fixed_alpha": 0.1, "text_weight_mode": "fixed",
                               "best_val_mse": val, "best_test_mse": 0.1 if variant == "mean5" else 0.3, "best_test_mae": 0.05}
                    if variant == "mean5":
                        args["text_temporal_pool"] = args["text_model_config"]["text_temporal_pool"] = pool
                        args["window_evidence"] = signatures
                        summary.update(text_temporal_pool=pool, window_evidence=signatures)
                    (output / "summary.json").write_text(json.dumps(summary))
                    torch.save({"args": args, "epoch": 1, "scaler_mean": [0., 1., 2., 3.],
                                "scaler_scale": [1.] * 4, "scaler_var": [1.] * 4}, output / "best_checkpoint.pt")
                    x = torch.from_numpy(np.stack([dataset[i][0] for i in range(len(dataset))]))
                    y = torch.from_numpy(np.stack([dataset[i][1][-3:] for i in range(len(dataset))]))
                    torch.save({"inputs": x, "ground_truths": y, "inputs_inv": x + 1000}, output / "test_io.pt")
                    if variant != "original_baseline":
                        with (output / "validation_group_diagnostics.csv").open("w", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=["window_index", "input_end_date", "group"])
                            writer.writeheader()
                            for index in range(len(dataset)):
                                for group in analysis.GROUPS:
                                    writer.writerow({"window_index": index, "input_end_date": dataset.window_end_date(index), "group": group})
        return roots, signatures, scalers

    def analyze(self, roots, signatures, scalers):
        with patch.object(analysis, "replay_windows", return_value=(signatures, scalers)), contextlib.redirect_stdout(io.StringIO()):
            return analysis.analyze(*roots)

    def test_four_variants_relative_statistics_read_only_and_primary_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            roots, signatures, scalers = self.fixture(directory)
            original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for root in roots for p in root.rglob("*.*")}
            tables, selection = self.analyze(roots, signatures, scalers)
            self.assertEqual(len(tables["per_run"]), 120)
            self.assertEqual(len(tables["by_seed"]), 12)
            self.assertEqual(len(tables["paired_by_ticker"]), 180)
            self.assertEqual(selection["selected_variants"], ["last"])
            self.assertFalse(selection["test_used_for_selection"])
            # The benchmark has lower val error, but the primary choice is mean5 vs last.
            self.assertEqual(set(selection["validation_scores"]), {"mean5", "last"})
            stat = next(r for r in tables["across_seeds"] if r["variant"] == "mean5" and r["metric"] == "val_mse")
            self.assertAlmostEqual(stat["mean"], 0.35)
            self.assertAlmostEqual(stat["sample_std"], 0.05)
            relative = next(r for r in tables["relative_by_seed"] if r["left"] == "mean5" and r["right"] == "last" and r["metric"] == "val_mse" and r["seed"] == "across_seeds")
            self.assertAlmostEqual(relative["relative_change_of_overall_means_pct"], 40.)
            for path, signature in original.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), signature)
            with self.assertRaises(FileExistsError):
                self.analyze(roots, signatures, scalers)

    def test_target_shift_and_saved_fingerprint_mismatch_stop_primary_comparison(self):
        for change in ("targets", "fingerprints", "dates", "scaler", "pool"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                roots, signatures, scalers = self.fixture(directory)
                output = roots[0] / "full/context_price_only/AAPL/seed7/output"
                if change == "targets":
                    path = roots[1] / "full/context_price_only/AAPL/seed7/output/test_io.pt"
                    payload = torch.load(path, weights_only=True)
                    payload["ground_truths"] += 1
                    torch.save(payload, path)
                elif change == "dates":
                    path = output / "validation_group_diagnostics.csv"
                    path.write_text(path.read_text().replace(WindowDataset().window_end_date(0), "1900-01-01"))
                else:
                    path = output / "best_checkpoint.pt"
                    payload = torch.load(path, weights_only=True)
                    if change == "fingerprints":
                        payload["args"]["window_evidence"]["val"]["targets_sha256"] = "bad"
                    elif change == "scaler":
                        payload["scaler_mean"][0] = 50
                    else:
                        payload["args"]["text_temporal_pool"] = "last"
                    torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "Cannot compare primary run"):
                    self.analyze(roots, signatures, scalers)
                self.assertFalse((roots[0] / "diagnostics").exists())

    def test_actual_dataset_replay_matches_normalized_future_targets(self):
        dates = [*pd.date_range("2021-01-01", periods=80), *pd.date_range("2022-01-01", periods=10),
                 *pd.date_range("2023-01-01", periods=10)]
        columns = {"date": [d.strftime("%Y-%m-%d") for d in dates], "ticker": ["AAPL"] * len(dates)}
        for index, channel in enumerate(("open", "high", "low", "close")):
            columns[channel] = np.arange(len(dates), dtype=float) + index
        for group in analysis.GROUPS:
            for dim in range(384):
                columns[f"{group}_emb{dim}"] = np.zeros(len(dates), np.float32)
        frame = pd.DataFrame(columns)
        with patch("pandas.read_parquet", return_value=frame):
            signatures, scalers = analysis.replay_windows("AAPL", analysis.COMMON, Path("unused"))
        self.assertEqual(signatures["val"]["windows"], 8)
        self.assertEqual(signatures["test"]["windows"], 8)
        self.assertEqual(signatures["val"]["first_input_end_date"], frame.date.iloc[79])
        np.testing.assert_allclose(scalers["val"]["mean"], np.arange(4) + 39.5)
        self.assertNotEqual(signatures["val"]["targets_sha256"], signatures["test"]["targets_sha256"])

    def test_normalized_io_required_and_inverse_arrays_are_not_used(self):
        expected = window_evidence(WindowDataset())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_io.pt"
            torch.save({"inputs_inv": torch.ones(4, 64, 4), "ground_truths_inv": torch.ones(4, 3, 4)}, path)
            with self.assertRaisesRegex(ValueError, "missing normalized"):
                check_saved_test_windows(path, expected)
