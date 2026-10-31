"""Synthetic persistence checks; no inference models, real data or training."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

import analyze_persistence_baseline as diagnostic
from forecasting_task.data_provider.dataset import Dataset_Custom


class PersistenceDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_last_price_repetition_has_no_access_to_future_prices(self):
        for batch in (1, 4):
            x = torch.arange(batch * 5 * 4, dtype=torch.float32).reshape(batch, 5, 4)
            prediction = diagnostic.persistence_prediction(x, 3)
            self.assertEqual(prediction.shape, (batch, 3, 4))
            for horizon in range(3):
                torch.testing.assert_close(prediction[:, horizon], x[:, -1], rtol=0, atol=0)
            future = torch.full((batch, 3, 4), 9999.)
            mse, mae, count = diagnostic.persistence_errors(x, future, 3)
            error = x[:, -1:].numpy().repeat(3, axis=1) - future.numpy()
            self.assertEqual(mse, np.square(error.astype(np.float64)).sum())
            self.assertEqual(mae, np.abs(error.astype(np.float64)).sum())
            self.assertEqual(count, batch * 3 * 4)
            future.mul_(100)
            torch.testing.assert_close(diagnostic.persistence_prediction(x, 3), prediction, rtol=0, atol=0)

    def test_future_targets_and_element_weighting_include_partial_batch(self):
        x = torch.arange(5 * 4 * 4, dtype=torch.float32).reshape(5, 4, 4) / 8
        y = torch.arange(5 * 5 * 4, dtype=torch.float32).reshape(5, 5, 4) / 3
        y[:, :2] = -1000  # Labels must not be scored as future targets.
        y[-1, -3:] += 80  # The incomplete final batch has a different error scale.
        dataset = TensorDataset(x, y, torch.zeros(5, 4, 5, 384), torch.zeros(5, 4, 3),
                                torch.zeros(5, 5, 3), torch.arange(5))
        error = x[:, -1:].double().expand(-1, 3, -1) - y[:, -3:].double()
        for batch_size in (1, 2, 4):
            result = diagnostic.evaluate_persistence(dataset, batch_size, 3, torch.device("cpu"))
            self.assertAlmostEqual(result["mse"], error.square().mean().item(), places=9)
            self.assertAlmostEqual(result["mae"], error.abs().mean().item(), places=9)
            self.assertEqual(result["num_elements"], 5 * 3 * 4)
            torch.testing.assert_close(result["targets"], y[:, -3:], rtol=0, atol=0)
            torch.testing.assert_close(result["inputs"], x, rtol=0, atol=0)

    @staticmethod
    def frame():
        dates = sum([list(pd.date_range(f"{year}-01-01", periods=24).strftime("%Y-%m-%d"))
                     for year in (2021, 2022, 2023)], [])
        columns = {"date": dates}
        for channel, name in enumerate(("open", "high", "low", "close")):
            columns[name] = np.arange(72, dtype=float) + channel
        for group in diagnostic.GROUPS:
            for dim in range(384):
                columns[f"{group}_emb{dim}"] = np.full(72, 0.01, dtype=np.float32)
        return pd.DataFrame(columns)

    def dataset(self, split):
        with patch("forecasting_task.data_provider.dataset.pd.read_parquet", return_value=self.frame()):
            return Dataset_Custom(4, 2, 3, split, list(diagnostic.GROUPS), data_path="T00.parquet")

    def test_dataset_replays_temporal_boundaries_and_train_only_scaler(self):
        for split, last, first_future in (("val", 23, 24), ("test", 47, 48)):
            dataset = self.dataset(split)
            np.testing.assert_allclose(dataset.scaler.mean_, np.arange(4) + 11.5)
            result = diagnostic.evaluate_persistence(dataset, 8, 3, torch.device("cpu"))
            self.assertEqual(result["num_elements"], 22 * 3 * 4)  # 8 + 8 + 6 windows.
            prediction = diagnostic.persistence_prediction(result["inputs"], 3).numpy()
            raw_prediction = prediction * dataset.scaler.scale_ + dataset.scaler.mean_
            raw_truth = result["targets"].numpy() * dataset.scaler.scale_ + dataset.scaler.mean_
            np.testing.assert_allclose(raw_prediction[0, :, 0], [last] * 3, rtol=0, atol=2e-6)
            np.testing.assert_allclose(raw_truth[0, :, 0], np.arange(first_future, first_future + 3), rtol=0, atol=2e-6)

    def test_saved_io_uses_actual_normalized_keys_and_checks_full_windows(self):
        expected = diagnostic.evaluate_persistence(self.dataset("test"), 8, 3, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_io.pt"
            payload = {"inputs": expected["inputs"], "ground_truths": expected["targets"],
                       "inputs_inv": np.full((2, 2), 1e20), "ground_truths_inv": np.full((2, 2), np.nan)}
            torch.save(payload, path)
            checked = diagnostic.check_test_io(path, expected, 3)
            self.assertEqual(checked["test_io_status"], "verified")
            self.assertEqual(checked["test_io_inputs_key"], "inputs")
            self.assertEqual(checked["test_io_targets_key"], "ground_truths")
            self.assertAlmostEqual(checked["io_persistence_test_mse"], expected["mse"])
            # Shifting both arrays leaves persistence MSE unchanged, but windows differ.
            torch.save({"inputs": expected["inputs"] + 1, "ground_truths": expected["targets"] + 1}, path)
            checked = diagnostic.check_test_io(path, expected, 3)
            self.assertEqual(checked["test_io_status"], "mismatch")
            self.assertIn("windows", checked["test_io_reason"])
            torch.save({"inputs_inv": expected["inputs"], "ground_truths_inv": expected["targets"]}, path)
            checked = diagnostic.check_test_io(path, expected, 3)
            self.assertEqual(checked["test_io_status"], "unavailable")
            self.assertIn("actual keys", checked["test_io_reason"])
            self.assertIsNone(checked["io_persistence_test_mse"])
            missing = diagnostic.check_test_io(path.with_name("missing.pt"), expected, 3)
            self.assertEqual(missing["test_io_status"], "unavailable")

    @staticmethod
    def create_run(root, variant, seed, dataset, test_result, **changes):
        output = root / variant / f"seed{seed}" / "output"
        output.mkdir(parents=True)
        args = dict(seed=seed, model_type="FiLM", data_path="T00.parquet", seq_len=4, label_len=2,
                    pred_len=3, batch_size=8, enc_in=4, dec_in=4, c_out=4,
                    group_weight_mode="fixed_uniform", fusion_mode=variant.removeprefix("no_weights_"),
                    text_num_layers=1, used_col_prefixes=list(diagnostic.GROUPS), text_weight=0.1)
        args.update(changes)
        checkpoint = {"args": args, "epoch": 5,
                      **{f"scaler_{name}": getattr(dataset.scaler, name + "_").tolist() for name in ("mean", "scale", "var")}}
        # Metadata-only checkpoint proves no inference model/state is required.
        torch.save(checkpoint, output / "best_checkpoint.pt")
        (output / "summary.json").write_text(json.dumps({"seed": seed, "best_epoch": 5,
                                                        "best_val_mse": 0.12, "best_test_mse": 0.1, "best_test_mae": 0.2}))
        torch.save({"inputs": test_result["inputs"], "ground_truths": test_result["targets"],
                    "inputs_inv": np.zeros(1)}, output / "test_io.pt")
        return diagnostic.Run(variant, "T00", seed, output)

    def test_comparability_exclusions_and_seed_independence_without_models(self):
        dataset = self.dataset("test")
        expected = diagnostic.evaluate_persistence(dataset, 8, 3, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [self.create_run(root, variant, seed, dataset, expected)
                    for variant in diagnostic.VARIANTS[1:] for seed in diagnostic.SEEDS]
            with patch("forecasting_task.data_provider.dataset.pd.read_parquet", return_value=self.frame()), \
                 patch.object(Dataset_Custom, "inverse_transform", side_effect=AssertionError("No inverse units")), \
                 patch.object(diagnostic, "evaluate_persistence", wraps=diagnostic.evaluate_persistence) as evaluate:
                rows, configuration = diagnostic.analyze_ticker("T00", runs, root, "", torch.device("cpu"))
            self.assertEqual(evaluate.call_count, 2)  # One val and one test replay, shared across seeds/modes.
            self.assertEqual(configuration["new_configurations_checked"], 6)
            diagnostic.verify_seed_independence(rows)
            self.assertTrue(all(row["comparable_test"] for row in rows))
            self.assertTrue(all(row["val_mae"] is None for row in rows if row["variant"] != "persistence"))
            old_runs = [run for run in runs if run.variant == "film_baseline"]
            for index, run in enumerate(old_runs):
                checkpoint = torch.load(run.output / "best_checkpoint.pt", weights_only=True)
                if index == 0:
                    checkpoint["args"]["seq_len"] += 1
                elif index == 1:
                    checkpoint["scaler_scale"][0] += 1
                else:
                    checkpoint["args"]["split_config"] = {"val_end": "2024-01-01"}
                torch.save(checkpoint, run.output / "best_checkpoint.pt")
            with patch("forecasting_task.data_provider.dataset.pd.read_parquet", return_value=self.frame()):
                rows, _ = diagnostic.analyze_ticker("T00", runs, root, "", torch.device("cpu"))
            excluded = [row for row in rows if row["variant"] == "film_baseline"]
            self.assertFalse(any(row["comparable_test"] for row in excluded))
            self.assertTrue(all(row["test_mse_improvement_pct"] is None and row["test_mse"] == 0.1 for row in excluded))
            for phrase, row in zip(("seq_len", "Scaler scale", "split"), excluded):
                self.assertIn(phrase, row["comparison_reason"])
            old = old_runs[0]
            checkpoint = torch.load(old.output / "best_checkpoint.pt", weights_only=True)
            checkpoint["args"]["seq_len"] = 4
            torch.save(checkpoint, old.output / "best_checkpoint.pt")
            torch.save({"inputs": expected["inputs"] + 1, "ground_truths": expected["targets"] + 1}, old.output / "test_io.pt")
            primary = next(run for run in runs if run.variant == "no_weights_blend" and run.seed == 17)
            torch.save({"inputs_inv": expected["inputs"], "ground_truths_inv": expected["targets"]}, primary.output / "test_io.pt")
            with patch("forecasting_task.data_provider.dataset.pd.read_parquet", return_value=self.frame()):
                rows, _ = diagnostic.analyze_ticker("T00", runs, root, "", torch.device("cpu"))
            old_row = next(row for row in rows if row["variant"] == "film_baseline" and row["seed"] == 7)
            self.assertFalse(old_row["comparable_test"])
            self.assertIn("windows", old_row["comparison_reason"])
            unverified = next(row for row in rows if row["variant"] == "no_weights_blend" and row["seed"] == 17)
            self.assertFalse(unverified["comparable_test"])
            self.assertTrue(unverified["comparable_val"])
            self.assertIsNone(unverified["test_mse_improvement_pct"])
            self.assertEqual(unverified["test_io_status"], "unavailable")
            rows[0]["persistence_test_mse"] += 1
            with self.assertRaisesRegex(ValueError, "not seed/fusion invariant"):
                diagnostic.verify_seed_independence(rows)
            checkpoint = torch.load(primary.output / "best_checkpoint.pt", weights_only=True)
            checkpoint["args"]["label_len"] += 1
            torch.save(checkpoint, primary.output / "best_checkpoint.pt")
            with self.assertRaisesRegex(ValueError, "Primary configuration mismatch.*label_len"):
                diagnostic.analyze_ticker("T00", runs, root, "", torch.device("cpu"))

    @staticmethod
    def make_inventory(root):
        data, results, baseline = root / "data", root / "results", root / "baseline"
        data.mkdir()
        for index in range(100):
            ticker = f"T{index:03}"
            (data / (ticker + ".parquet")).touch()
            for mode in ("blend", "residual"):
                for seed in diagnostic.SEEDS:
                    output = results / "full" / "alpha010" / mode / ticker / f"seed{seed}" / "output"
                    output.mkdir(parents=True)
                    (output / "summary.json").write_text("{}")
                    (output / "best_checkpoint.pt").write_bytes(b"placeholder")
        return data, results, baseline

    def test_inventory_requires_100_tickers_and_all_seeds_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            data, results, baseline = self.make_inventory(Path(directory))
            tickers, runs, reason = diagnostic.discover_runs(results, data, baseline)
            self.assertEqual(len(tickers), 100)
            self.assertEqual(len(runs), 900)  # Six new runs and three historical lookup rows per ticker.
            self.assertEqual(reason, "")
            target = next(run for run in runs if run.variant == "no_weights_residual").output / "summary.json"
            backup = target.with_suffix(".backup")
            target.rename(backup)
            with self.assertRaisesRegex(ValueError, "expected 100 tickers"):
                diagnostic.discover_runs(results, data, baseline)
            backup.rename(target)
            source = next(run for run in runs if run.variant == "no_weights_blend").output / "summary.json"
            target.rename(backup)
            target.hardlink_to(source)
            with self.assertRaisesRegex(ValueError, "Duplicate physical"):
                diagnostic.discover_runs(results, data, baseline)

    @staticmethod
    def fixture_rows(tickers):
        persistence = dict(val_mse=10., val_mae=2., test_mse=10., test_mae=2.)
        rows = []
        for ticker in tickers:
            for variant in diagnostic.VARIANTS:
                for seed in diagnostic.SEEDS:
                    if variant == "persistence":
                        values = persistence.copy()
                    else:
                        value = {"no_weights_blend": {7: 4., 17: 5., 27: 6.}[seed],
                                 "no_weights_residual": 12., "film_baseline": 8.}[variant]
                        values = dict(val_mse=value, val_mae=None, test_mse=value, test_mae=value / 10)
                    comparable = not (variant == "film_baseline" and ticker == tickers[0] and seed == 7)
                    rows.append(diagnostic.make_row(diagnostic.Run(variant, ticker, seed, Path("unused")), values,
                                                     {}, persistence, comparable, "synthetic mismatch" if not comparable else ""))
        return rows

    def test_aggregation_keeps_splits_distinct_and_never_pools_incomparable_runs(self):
        tickers = [f"T{index:03}" for index in range(100)]
        per_ticker, by_seed, stats, wins = diagnostic.aggregate(self.fixture_rows(tickers), tickers)
        row = next(row for row in stats if row["variant"] == "no_weights_blend" and row["metric"] == "test_mse")
        self.assertEqual((row["mean"], row["sample_std"], row["improvement_pct_mean"]), (5, 1, 50))
        missing = next(row for row in stats if row["variant"] == "no_weights_blend" and row["metric"] == "val_mae")
        self.assertIsNone(missing["mean"])
        self.assertEqual(missing["complete_seeds"], 0)
        old = next(row for row in by_seed if row["variant"] == "film_baseline" and row["seed"] == 7)
        self.assertEqual(old["test_mse"], 8)  # Raw errors remain visible, labeled with coverage.
        self.assertIsNone(old["test_mse_improvement_pct"])
        self.assertEqual(old["test_mse_comparable_tickers"], 99)
        counts = {row["variant"]: row for row in wins}
        self.assertEqual(counts["no_weights_blend"]["tickers_better"], 100)
        self.assertEqual(counts["no_weights_residual"]["tickers_better"], 0)
        self.assertEqual(counts["film_baseline"]["tickers_better"], 99)
        self.assertFalse(counts["film_baseline"]["all_100_confirmed"])
        excluded = next(row for row in per_ticker if row["variant"] == "film_baseline" and row["ticker"] == "T000")
        self.assertIsNone(excluded["test_mse_improvement_pct"])
        self.assertIsNone(diagnostic.improvement_pct(0, 0))

    def test_different_window_geometry_across_tickers_is_not_pooled(self):
        reference = dict(ticker="T000", seq_len=64, label_len=16, pred_len=3, split_policy="same fixed policy")
        other = dict(reference, ticker="T001", pred_len=6)
        with self.assertRaisesRegex(ValueError, "different pred_len"):
            diagnostic.verify_cohort_geometry([reference, other])
        other["pred_len"] = 3
        diagnostic.verify_cohort_geometry([reference, other])

    def test_full_report_is_read_only_and_cli_defaults_to_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            data, results, baseline = self.make_inventory(Path(directory))
            tickers, runs, _ = diagnostic.discover_runs(results, data, baseline)
            source = {path: path.read_bytes() for run in runs if run.variant != "film_baseline"
                      for path in (run.output / "summary.json", run.output / "best_checkpoint.pt")}
            rows = self.fixture_rows(tickers)

            def ticker_result(ticker, *args):
                return [row for row in rows if row["ticker"] == ticker], {
                    "ticker": ticker, "persistence_evaluations": 2, "seq_len": 64, "pred_len": 3,
                    "label_len": 16, "split_policy": "same fixed policy",
                }

            with patch.object(diagnostic, "analyze_ticker", side_effect=ticker_result) as evaluate, \
                 contextlib.redirect_stdout(io.StringIO()):
                diagnostic.analyze(results, data, baseline)
            self.assertEqual(evaluate.call_count, 100)
            self.assertTrue(all(call.args[-1] == torch.device("cpu") for call in evaluate.call_args_list))
            output = results / "diagnostics" / "persistence_100"
            self.assertTrue(all((output / name).is_file() for name in diagnostic.REPORTS))
            with (output / "per_run.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1200)
            with (output / "per_ticker.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 400)
            for path, content in source.items():
                self.assertEqual(path.read_bytes(), content)
            with self.assertRaises(FileExistsError):
                diagnostic.analyze(results, data, baseline)
        with patch.object(diagnostic, "analyze") as analyze:
            diagnostic.cli([])
            self.assertEqual(analyze.call_args.args[-1], "cpu")
