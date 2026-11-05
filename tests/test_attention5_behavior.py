"""Small saved-artifact fixtures; no training, models, GPU or server data."""

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
import yaml

import analyze_attention5_behavior as analysis


def history(length=12):
    return [{"epoch": e, "train_mse": 20-e, "val_mse": 1+(e-8)**2/100} for e in range(1, length+1)]


def summary_for(records):
    best = min(records, key=lambda row: row["val_mse"])
    return {"best_epoch": best["epoch"], "epochs_completed": len(records), "best_val_mse": best["val_mse"]}


class BehaviorTests(unittest.TestCase):
    def test_beta_metrics_explicit_uniform_selective_empty_and_single(self):
        uniform = analysis.beta_metrics([.5, 0, .5, 0, 0], [1, 0, 1, 0, 0])
        self.assertEqual((uniform["l1_uniform"], uniform["effective_dates"], uniform["effective_ratio"]), (0, 2, 1))
        weighted = analysis.beta_metrics([.8, 0, .2, 0, 0], [1, 0, 1, 0, 0])
        self.assertAlmostEqual(weighted["l1_uniform"], .6)
        self.assertEqual(weighted["max_beta"], .8)
        self.assertAlmostEqual(weighted["effective_dates"], 1/(.8**2+.2**2))
        self.assertAlmostEqual(weighted["effective_ratio"], 1/(.8**2+.2**2)/2)
        empty = analysis.beta_metrics([0]*5, [0]*5)
        self.assertEqual(empty["window_kind"], "empty")
        self.assertTrue(all(empty[m] is None for m in analysis.METRICS))
        single = analysis.beta_metrics([0, 0, 0, 1, 0], [0, 0, 0, 1, 0])
        self.assertEqual(single["effective_dates"], 1)
        self.assertFalse(single["selectivity_eligible"])
        for beta, mask in (([.5, 1e-14, .5, 0, 0], [1, 0, 1, 0, 0]),
                           ([.6, 0, .5, 0, 0], [1, 0, 1, 0, 0]),
                           ([float("nan"), 0, 0, 0, 0], [1, 0, 0, 0, 0]),
                           ([-.1, 1.1, 0, 0, 0], [1, 1, 0, 0, 0]), ([1, 0, 0, 0, 0], [2, 0, 0, 0, 0])):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                analysis.beta_metrics(beta, mask)

    def test_last_two_blocks_and_first_best_epoch(self):
        records = history()
        result = analysis.history_statistics(records, summary_for(records))
        self.assertEqual(result["best_epoch"], 8)
        self.assertEqual([result[k] for k in ("previous_start", "previous_end", "last_start", "last_end")], [3, 7, 8, 12])
        self.assertEqual((result["train_mse_previous5"], result["train_mse_last5"], result["train_mse_change"]), (15, 10, -5))
        self.assertAlmostEqual(result["train_mse_change_pct"], -100/3)
        self.assertAlmostEqual(result["val_mse_change"], -.05)
        short = history(9)
        result = analysis.history_statistics(short, summary_for(short))
        self.assertEqual(result["block_status"], "insufficient_epochs")
        self.assertIsNone(result["train_mse_change"])
        tied = [{"epoch": i+1, "train_mse": 2, "val_mse": v} for i, v in enumerate([3, 1, 1])]
        self.assertEqual(analysis.history_statistics(tied, summary_for(tied))["best_epoch"], 2)
        with self.assertRaises(ValueError):
            analysis.history_statistics(tied, {**summary_for(tied), "best_epoch": 3})
        with self.assertRaises(ValueError):
            analysis.history_statistics(records[1:], summary_for(records))

    def test_remaining_runs_and_paired_cohorts_do_not_hide_composition_changes(self):
        rows = []
        for seed in analysis.SEEDS:
            for variant, ticker, length, error in (("attention5", "A", 4, 1), ("attention5", "B", 2, 100),
                                                    ("mean5", "A", 2, 1), ("mean5", "B", 4, 100)):
                rows.extend({"variant": variant, "ticker": ticker, "seed": seed, "epoch": e,
                             "train_mse": error, "val_mse": error} for e in range(1, length+1))
        table = analysis.curve_statistics(pd.DataFrame(rows))
        late = table[(table.seed_scope == "all") & (table.epoch == 3)]
        available = late[late.cohort == "available"].set_index("variant")
        self.assertEqual(available.loc["attention5", "n_runs"], 3)
        self.assertEqual(available.loc["mean5", "n_runs"], 3)
        self.assertEqual(available.loc["attention5", "val_mse"], 1)
        self.assertEqual(available.loc["mean5", "val_mse"], 100)
        self.assertIn("A:seed7", available.loc["attention5", "members"])
        self.assertNotEqual(available.loc["attention5", "members"], available.loc["mean5", "members"])
        self.assertTrue((late[late.cohort == "paired"].n_runs == 0).all())
        self.assertTrue(late[late.cohort == "paired"].val_mse.isna().all())
        early = table[(table.seed_scope == "all") & (table.epoch == 2) & (table.cohort == "paired")]
        self.assertTrue((early.n_runs == 6).all())

    def fixture(self, directory):
        root = Path(directory)
        dates = ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-10"]
        gate = {"type": "Linear", "in_features": 18, "out_features": 5, "parameters": 95, "initialization": "zeros"}
        for variant, ticker, seed in itertools_product():
            output = root/f"context_price_{variant}_10_v1/full/context_price_only"/ticker/f"seed{seed}/output"
            output.mkdir(parents=True)
            records = history(9 if ticker == "AMZN" else 12)
            evidence = {"windows": 2, "inputs_sha256": "a"*64, "targets_sha256": "b"*64,
                        "dates_sha256": "c"*64}
            args = {"seed": seed, "data_path": ticker+".parquet", "text_temporal_pool": variant,
                    "group_gate": "context_price_only", "fixed_alpha": .1, "text_weight_mode": "fixed",
                    "seq_len": 64, "label_len": 16, "pred_len": 3, "batch_size": 32,
                    "num_epoch": 50, "patience": 10, "lr": .001, "model_type": "FiLM",
                    "used_col_prefixes": list(analysis.GROUPS), "fusion_mode": "blend", "text_num_layers": 1,
                    "text_weight": .1, "gate_structure": gate, "novelty_policy": {"history": "causal"},
                    "sampling_policy": "independent_train_generator", "text_mlp_structure": ["Linear384,3"],
                    "window_evidence": {"val": evidence}}
            summary = {**args, **summary_for(records)}
            summary["validation_diagnostics"] = {"windows": 2, "rows": 10,
                "checkpoint": "best_checkpoint.pt", "mse": summary["best_val_mse"]}
            (output/"args.yaml").write_text(yaml.safe_dump(args), encoding="utf-8")
            (output/"summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (output/"history.jsonl").write_text("\n".join(json.dumps(r) for r in records)+"\n", encoding="utf-8")
            rows = []
            for window in range(2):
                for group_index, group in enumerate(analysis.GROUPS):
                    masks = [[0]*5, [0, 0, 0, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 0, 0], [1]*5]
                    mask = masks[group_index]
                    count = sum(mask)
                    beta = np.array(mask, dtype=float)/max(count, 1)
                    if group_index == 2:
                        beta[[0, 4]] = [.8, .2]
                    row = {"window_index": window, "group": group, "text_temporal_pool": variant,
                           "input_end_date": dates[window+4], "text_window_start_date": dates[window],
                           "text_window_end_date": dates[window+4], "available_dates": count, "present": int(count > 0)}
                    if variant == "attention5":
                        for d in range(5):
                            row.update({f"date_d{d+1}": dates[window+d], f"age_d{d+1}": 4-d,
                                        f"present_d{d+1}": mask[d], f"beta_d{d+1}": beta[d]})
                    rows.append(row)
            with (output/"validation_group_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return root

    def test_matched_inventory_quantiles_availability_and_read_only_report_with_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.*")}
            with contextlib.redirect_stdout(io.StringIO()):
                tables = analysis.analyze(root)
            self.assertEqual(len(tables["run_summary"]), 60)
            self.assertEqual(len(tables["weights_per_window"]), 300)
            self.assertEqual(len(tables["verification"]), 60)
            metric = tables["weight_quantiles"]
            target = metric[(metric.seed_scope == "7") & (metric.group == "targetCompany") & (metric.metric == "l1_uniform")].iloc[0]
            self.assertEqual((target.n_present, target.n_windows, target.n_tickers), (2, 20, 10))
            self.assertAlmostEqual(target["mean"], .6)
            age = tables["age_weights"]
            absent = age[(age.seed_scope == "7") & (age.group == "targetCompany") & (age.age == 3)].iloc[0]
            self.assertEqual(absent.n_available, 0)
            self.assertEqual(absent.mean_beta_all_windows, 0)
            self.assertTrue(pd.isna(absent.mean_beta_if_present))
            empty = metric[metric.n_present == 0]
            self.assertTrue((empty.n_defined == 0).all())
            self.assertTrue(empty["mean"].isna().all())
            blocks = tables["block_changes_by_seed"]
            self.assertTrue((blocks.n_with_two_blocks == 9).all())
            output = root/"context_price_attention5_10_v1/diagnostics/attention_behavior"
            self.assertEqual(len(list(output.glob("*.png"))), 43)
            self.assertEqual(len(list(output.glob("learning_AAPL_seed7.png"))), 1)
            self.assertTrue(all(path.read_bytes().startswith(b"\x89PNG") for path in output.glob("*.png")))
            explanation = json.loads((output/"interpretation.json").read_text())
            self.assertIn("not evidence of selectivity", explanation["caution"])
            for path, digest in originals.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                analysis.analyze(root)

    def test_missing_runs_and_configuration_window_or_beta_mismatches_stop_before_writes(self):
        for change in ("missing", "seed", "best", "config", "count", "masked_beta", "date"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                output = root/"context_price_attention5_10_v1/full/context_price_only/AAPL/seed7/output"
                if change == "missing":
                    output.rename(output.with_name("archived_output"))
                elif change in ("seed", "best"):
                    path = output/"summary.json"
                    payload = json.loads(path.read_text())
                    payload["seed" if change == "seed" else "best_epoch"] = 999
                    path.write_text(json.dumps(payload))
                elif change == "config":
                    path = output/"args.yaml"
                    payload = yaml.safe_load(path.read_text())
                    payload["lr"] = .5
                    path.write_text(yaml.safe_dump(payload))
                else:
                    path = output/"validation_group_diagnostics.csv"
                    with path.open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    if change == "masked_beta":
                        rows[0]["beta_d1"] = .1
                    elif change == "count":
                        rows[2]["available_dates"] = 3
                    else:
                        rows[0]["date_d3"] = "2022-01-04"
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                with self.assertRaises(ValueError):
                    analysis.analyze(root)
                self.assertFalse((root/"context_price_attention5_10_v1/diagnostics").exists())


def itertools_product():
    import itertools
    return itertools.product(analysis.VARIANTS, analysis.TICKERS, analysis.SEEDS)
