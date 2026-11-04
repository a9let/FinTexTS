"""Validation strength diagnostics do not construct models or compare trained heads."""

import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import analyze_redundancy5_strength as strength
from analyze_context_price_redundancy5 import COMMON
from forecasting_task.window_evidence import window_evidence


class FakeValidation:
    seq_len, pred_len = 64, 3

    def __init__(self):
        self.row_dates = pd.bdate_range("2022-01-03", periods=70).strftime("%Y-%m-%d").to_numpy()
        self.text = np.zeros((70, 5, 384), np.float32)
        self.text[59:63, 0, 0], self.text[63, 0, 1] = 1, 1
        self.text[61, 1, 2] = 7
        self.text[61, 3, 0], self.text[63, 3, 0] = 1, -1
        self.text[60, 4, 0], self.text[61, 4, 1], self.text[62, 4, 2] = 1, 2, 3
        self.scaler = SimpleNamespace(mean_=np.zeros(4), scale_=np.ones(4), var_=np.ones(4))
        self.base = SimpleNamespace(data_text_x=self.text)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        prices = np.arange(70 * 4, dtype=np.float32).reshape(70, 4) / 100
        return prices[index:index+64], prices[index+48:index+67], self.text[index:index+64]

    def window_end_date(self, index):
        return str(self.row_dates[index+63])

    def text_window_dates(self, index, rows):
        return self.row_dates[index+64-rows:index+64].tolist()


class StrengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, directory, minimal=False):
        dataset = FakeValidation()
        root = Path(directory) / "results"
        records = strength.reconstruct_dataset("unused", dataset, 0.1)
        evidence = window_evidence(dataset)
        for ticker in strength.TICKERS:
            for seed in strength.SEEDS:
                output = root / "full/context_price_only" / ticker / f"seed{seed}" / "output"
                output.mkdir(parents=True)
                args = {**COMMON, "seed": seed, "model_type": "FiLM", "data_path": ticker + ".parquet",
                        "task_name": "short_term_forecast", "e_layers": 2, "output_attention": False,
                        "enc_in": 4, "dec_in": 4, "c_out": 4, "group_gate": "context_price_only",
                        "group_weight_mode": "context_price_only", "fixed_alpha": .1, "text_weight_mode": "fixed",
                        "text_weight": .1, "text_num_layers": 1, "used_col_prefixes": list(strength.TEXT_GROUPS),
                        "fusion_mode": "blend", "text_temporal_pool": "redundancy5", "redundancy_tau": .1,
                        "text_model_config": {"group_gate": "context_price_only", "text_temporal_pool": "redundancy5", "redundancy_tau": .1},
                        "novelty_policy": {"previous": "earlier present row"}, "sampling_policy": "independent_train_generator",
                        "gate_structure": {"parameters": 95}, "text_mlp_structure": [{"type": "Linear", "in_features": 384, "out_features": 3}],
                        "window_evidence": {"val": evidence}}
                summary = {"seed": seed, "model": "FiLM", "data_path": ticker + ".parquet", "best_epoch": 3,
                           "epochs_completed": 10, "group_gate": "context_price_only", "fixed_alpha": .1,
                           "text_weight_mode": "fixed", "text_temporal_pool": "redundancy5", "redundancy_tau": .1,
                           "window_evidence": {"val": evidence}}
                (output / "summary.json").write_text(json.dumps(summary))
                torch.save({"args": args, "epoch": 3, "scaler_mean": [0.] * 4, "scaler_scale": [1.] * 4,
                            "scaler_var": [1.] * 4}, output / "best_checkpoint.pt")
                rows = []
                for record in records:
                    row = {"window_index": record["window_index"], "input_end_date": record["input_end_date"], "group": record["group"]}
                    if not minimal:
                        row.update(text_temporal_pool="redundancy5", redundancy_tau=.1, available_dates=record["present_dates"],
                                   present=record["present"], text_window_start_date=record["text_dates"][0],
                                   text_window_end_date=record["text_dates"][-1], effective_dates=record["effective_dates"])
                        for day in range(5):
                            row[f"text_date_{day+1}"] = record["text_dates"][day]
                            row[f"daily_present_{day+1}"] = record["daily_present"][day]
                            row[f"temporal_weight_{day+1}"] = record["temporal_weights"][day]
                            row[f"redundancy_r_{day+1}"] = record["r_all"][day]
                    # Deliberately different trained-head values: analysis must ignore them.
                    row["head_h1"] = seed * 10000
                    rows.append(row)
                strength.write_table(output / "validation_group_diagnostics.csv", rows)
                (output / "test_io.pt").write_bytes(b"must never be read")
        return root, dataset

    def analyze(self, root, dataset):
        with patch.object(strength, "load_validation", return_value=dataset), \
             patch("torch.nn.Linear", side_effect=AssertionError("No inference heads should be created")), \
             contextlib.redirect_stdout(io.StringIO()):
            return strength.analyze(root, top_k=3)

    def test_exact_distributions_four_repeats_one_orthogonal_and_l1_neff_distance(self):
        record = strength.measure_window(FakeValidation()[0][2][-5:], .1)[0]
        similarity = math.exp(-10)
        density = np.array([4+similarity] * 4 + [1+4*similarity])
        a = (1/density) / (1/density).sum()
        self.assertEqual(len(record["pair_cosines"]), 10)
        self.assertEqual(record["pair_cosines"].count(1.), 6)
        self.assertEqual(record["pair_cosines"].count(0.), 4)
        np.testing.assert_allclose(sorted(record["offdiag_kernel"]), [similarity] * 4 + [1.] * 6)
        np.testing.assert_allclose(record["r_present"], density)
        self.assertAlmostEqual(record["weight_l1_deviation"], np.abs(a-.2).sum())
        self.assertAlmostEqual(record["effective_dates"], 1/np.square(a).sum())
        self.assertAlmostEqual(record["effective_dates_ratio"], 1/np.square(a).sum()/5)
        self.assertAlmostEqual(record["aggregate_l2"], np.linalg.norm(np.array([a[:4].sum(), a[4]]) - [.8, .2]), places=6)

    def test_empty_single_and_cancelled_aggregates_do_not_fake_relative_distances(self):
        records = strength.measure_window(FakeValidation()[0][2][-5:], .1)
        single, empty, cancelled = records[1:4]
        self.assertEqual(single["weight_l1_deviation"], 0.)
        self.assertEqual(single["effective_dates"], 1.)
        self.assertEqual(single["aggregate_l2"], 0.)
        self.assertEqual(single["pair_cosines"], [])
        self.assertEqual(empty["present_dates"], 0)
        self.assertIsNone(empty["effective_dates"])
        self.assertIsNone(empty["aggregate_l2"])
        self.assertEqual(cancelled["present_dates"], 2)
        self.assertIsNone(cancelled["aggregate_relative_l2"])
        self.assertIsNone(cancelled["aggregate_cosine_distance"])
        self.assertEqual(cancelled["effective_dates_ratio"], 1.)
        all_empty = strength.measure_window(np.zeros((5, 5, 384), np.float32), .1)
        self.assertTrue(all(r["present_dates"] == 0 and r["aggregate_l2"] is None for r in all_empty))

    def test_report_verifies_30_runs_is_read_only_and_distinguishes_repeated_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root, dataset = self.fixture(directory)
            originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.*")}
            tables, manifest = self.analyze(root, dataset)
            self.assertEqual(manifest["runs_verified"], 30)
            self.assertEqual(manifest["unique_group_windows"], 100)
            self.assertEqual(manifest["run_group_windows"], 300)
            self.assertEqual(manifest["empty_unique_group_windows"], 20)
            self.assertEqual(manifest["fully_empty_input_windows"], 0)
            self.assertEqual(len(tables["verification"]), 30)
            unique = next(r for r in tables["quantiles_unique"] if r["group"] == "macro" and r["present_dates"] == 5 and r["metric"] == "pair_cosines")
            repeated = next(r for r in tables["quantiles"] if r["group"] == "macro" and r["present_dates"] == 5 and r["metric"] == "pair_cosines")
            self.assertEqual(repeated["samples"], 3 * unique["samples"])
            self.assertTrue((root / "diagnostics/redundancy5_strength/top_examples.csv").is_file())
            self.assertTrue(all("temporal_weight_1" in r["checked_csv_fields"] for r in tables["verification"]))
            for path, signature in originals.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), signature)
            with self.assertRaises(FileExistsError):
                self.analyze(root, dataset)

    def test_missing_optional_csv_quantities_are_recovered_from_verified_input_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root, dataset = self.fixture(directory, minimal=True)
            tables, _ = self.analyze(root, dataset)
            self.assertTrue(all("temporal_weight_1" in r["reconstructed_fields"] for r in tables["verification"]))
            self.assertTrue(all(set(r["checked_csv_fields"]) == {"window_index", "input_end_date", "group"}
                                for r in tables["verification"]))
            self.assertTrue(tables["top_examples"])

    def test_corrupted_dates_weights_densities_configuration_and_fingerprints_stop(self):
        for change in ("date", "weight", "density", "tau", "fingerprint"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root, dataset = self.fixture(directory)
                output = root / "full/context_price_only/AAPL/seed7/output"
                if change in ("date", "weight", "density"):
                    path = output / "validation_group_diagnostics.csv"
                    with path.open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    field = {"date": "text_date_1", "weight": "temporal_weight_1", "density": "redundancy_r_1"}[change]
                    rows[0][field] = "1900-01-01" if change == "date" else "99"
                    strength.write_table(output / "modified.csv", rows)
                    # Replace only this synthetic fixture file, never real artifacts.
                    (output / "modified.csv").replace(path)
                else:
                    path = output / "best_checkpoint.pt"
                    payload = torch.load(path, weights_only=True)
                    if change == "tau":
                        payload["args"]["redundancy_tau"] = .2
                    else:
                        payload["args"]["window_evidence"]["val"]["targets_sha256"] = "bad"
                    torch.save(payload, path)
                with self.assertRaises(ValueError):
                    self.analyze(root, dataset)
                self.assertFalse((root / "diagnostics").exists())

    def test_inventory_requires_exact_30_and_cli_help_does_not_analyze(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            (root / "full/context_price_only/AAPL/seed7/output/best_checkpoint.pt").unlink()
            with self.assertRaisesRegex(ValueError, "exactly 30"):
                strength.discover_runs(root)
        with patch.object(strength, "analyze", side_effect=AssertionError("No analysis on help")), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            strength.cli(["--help"])
