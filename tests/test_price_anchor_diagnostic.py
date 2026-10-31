"""Synthetic validation-only checks; no server data, training or GPU required."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import analyze_blend_price_anchor as diagnostic
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.models.GroupText import FixedUniformTextModel, TEXT_GROUPS


class CheckedFiLM(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x, *unused):
        assert not self.training
        assert not torch.is_grad_enabled()
        offsets = torch.arange(self.pred_len, dtype=x.dtype)[None, :, None]
        return 2 * x[:, -1:, :] + offsets


class CheckedText(FixedUniformTextModel):
    def forward(self, *args):
        assert not self.training
        assert not torch.is_grad_enabled()
        return super().forward(*args)


class PriceAnchorDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)

    @staticmethod
    def make_inventory(root):
        for seed in diagnostic.SEEDS:
            for index in range(10):
                output = root / "full" / "alpha090" / "blend" / f"T{index:02}" / f"seed{seed}" / "output"
                output.mkdir(parents=True)
                (output / "best_checkpoint.pt").write_bytes(b"synthetic placeholder")
                (output / "summary.json").write_text("{}")
        (root / "selected_tickers.txt").write_text("\n".join(f"T{index:02}" for index in range(10)) + "\n")

    def test_inventory_requires_exact_unique_matching_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inventory(root)
            runs = diagnostic.discover_runs(root)
            self.assertEqual(len(runs), 30)
            self.assertEqual([run.seed for run in runs], [7] * 10 + [17] * 10 + [27] * 10)
            # A renamed ticker keeps 30 files, but invalidates the matched cohort.
            original = root / "full/alpha090/blend/T00/seed27"
            changed = root / "full/alpha090/blend/OTHER/seed27"
            changed.parent.mkdir()
            original.rename(changed)
            with self.assertRaisesRegex(ValueError, "same 10 tickers"):
                diagnostic.discover_runs(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inventory(root)
            # Renaming the checkpoint removes it from the discovery pattern.
            checkpoint = diagnostic.discover_runs(root)[0].checkpoint
            checkpoint.rename(checkpoint.with_suffix(".backup"))
            with self.assertRaisesRegex(ValueError, "exactly 30.*29"):
                diagnostic.discover_runs(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inventory(root)
            extra = root / "full/alpha090/blend/EXTRA/seed7/output"
            extra.mkdir(parents=True)
            (extra / "best_checkpoint.pt").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "exactly 30.*31"):
                diagnostic.discover_runs(root)

    def test_duplicate_physical_checkpoint_and_selected_list_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inventory(root)
            runs = diagnostic.discover_runs(root)
            # Model a duplicated physical file without depending on Windows symlink privileges.
            stat_a, stat_b = runs[0].checkpoint.stat(), runs[1].checkpoint.stat()
            original_stat = Path.stat

            def stat(path, *args, **kwargs):
                if path == runs[1].checkpoint:
                    return SimpleNamespace(st_dev=stat_a.st_dev, st_ino=stat_a.st_ino, st_size=stat_b.st_size)
                return original_stat(path, *args, **kwargs)

            with patch.object(Path, "stat", stat), self.assertRaisesRegex(ValueError, "Duplicate"):
                diagnostic.discover_runs(root)
            (root / "selected_tickers.txt").write_text("T00\n" * 10)
            with self.assertRaisesRegex(ValueError, "selected_tickers"):
                diagnostic.discover_runs(root)

    def test_three_formulas_element_weighting_and_incomplete_last_batch(self):
        samples, horizon = 5, 3
        x = torch.arange(samples * 3 * 4, dtype=torch.float32).reshape(samples, 3, 4) / 10
        y = torch.arange(samples * 5 * 4, dtype=torch.float32).reshape(samples, 5, 4) / 3
        y[-1] += 25
        text = torch.zeros(samples, 3, 5, 384)
        text[:, :-1] = 1000  # Earlier days must never be used.
        text[[0, 2, 4], -1, 0, 0] = torch.tensor([1., 2., 3.])
        dataset = TensorDataset(x, y, text, torch.zeros(samples, 3, 3), torch.zeros(samples, 5, 3), torch.arange(samples))
        model = CheckedText(64, horizon, 1, initial_text_weight=0.35)
        with torch.no_grad():
            for index, head in enumerate(model.group_mlps.values()):
                head[0].weight.zero_()
                head[0].bias.fill_(0.1 * (index + 1))
            model.group_mlps["macro"][0].weight[:, 0] = 0.25
        alpha = model.get_alpha().item()
        price = 2 * x[:, -1:, :] + torch.arange(horizon)[None, :, None]
        last = x[:, -1:, :].expand(-1, horizon, -1)
        delta = torch.zeros(samples, horizon, 4)
        for index in (0, 2, 4):
            delta[index] = (0.25 * text[index, -1, 0, 0] + 0.1) / 5
        expected_full_text = last + delta
        expected_full_text[[1, 3]] = 0  # The trained full model's absent-text behavior.
        expected = {
            "full": (1 - alpha) * price + alpha * expected_full_text,
            "no_text_delta": (1 - alpha) * price + alpha * last,
            "persistence": last,
        }
        for batch_size in (1, 2, 4):
            result = diagnostic.evaluate_batches(CheckedFiLM(horizon), model, DataLoader(dataset, batch_size=batch_size),
                                                  horizon, 2, torch.device("cpu"))
            self.assertEqual(result["num_elements"], samples * horizon * 4)
            self.assertAlmostEqual(result["alpha"], alpha)
            for variant, prediction in expected.items():
                error = prediction.double() - y[:, -horizon:].double()
                self.assertAlmostEqual(result[variant + "_mse"], error.square().mean().item(), places=5)
                self.assertAlmostEqual(result[variant + "_mae"], error.abs().mean().item(), places=6)
        self.assertNotEqual(result["full_mse"], result["no_text_delta_mse"])

    def make_real_checkpoint(self, root):
        dates = sum([list(pd.date_range(f"{year}-01-01", periods=24).strftime("%Y-%m-%d")) for year in (2021, 2022)], [])
        columns = {"date": dates}
        for channel, name in enumerate(("open", "high", "low", "close")):
            columns[name] = np.arange(48, dtype=float) + np.sin(np.arange(48)) + channel
        for group_index, group in enumerate(TEXT_GROUPS):
            for dim in range(384):
                columns[f"{group}_emb{dim}"] = np.full(48, (group_index + 1) * 0.01, dtype=np.float32)
        frame = pd.DataFrame(columns)
        args = dict(task_name="short_term_forecast", seq_len=12, label_len=2, pred_len=3, output_attention=False,
                    e_layers=1, enc_in=4, model_type="FiLM", group_weight_mode="fixed_uniform", fusion_mode="blend",
                    seed=7, used_col_prefixes=list(TEXT_GROUPS), text_num_layers=1, text_weight=0.9,
                    data_path="T00.parquet", batch_size=8)
        config = dict(hidden_size=64, pred_len=3, num_layers=1, used_col_prefixes=list(TEXT_GROUPS),
                      initial_text_weight=0.9, fusion_mode="blend", initial_group_weights=[0.2] * 5)
        args["text_model_config"] = config
        data = root / "data"
        data.mkdir()
        (data / "T00.parquet").touch()
        with patch("forecasting_task.data_provider.dataset.pd.read_parquet", return_value=frame):
            dataset = Dataset_Custom(12, 2, 3, "val", list(TEXT_GROUPS), root_path=str(data), data_path="T00.parquet")
        with diagnostic.execution_device(torch.device("cpu")):
            film = diagnostic.film_module.Model(SimpleNamespace(**args)).eval()
            model = FixedUniformTextModel(**config).eval()
            with torch.no_grad():
                model.alpha_logit.copy_(torch.logit(torch.tensor(0.23)))  # Learned value differs from 0.9.
                predictions, truths = [], []
                for batch in DataLoader(dataset, batch_size=8):
                    x, y, text, xm, ym = [value.float() for value in batch[:5]]
                    dec = torch.cat([y[:, :2], torch.zeros_like(y[:, -3:])], dim=1)
                    predictions.append(model(film(x, xm, dec, ym), x[:, -1], text[:, -1]))
                    truths.append(y[:, -3:])
                reference_mse = (torch.cat(predictions) - torch.cat(truths)).square().mean().item()
        output = root / "full/alpha090/blend/T00/seed7/output"
        output.mkdir(parents=True)
        checkpoint = {
            "args": args, "epoch": 4, "model_state_dict": film.state_dict(), "text_model_state_dict": model.state_dict(),
            **{f"scaler_{name}": getattr(dataset.scaler, name + "_").tolist() for name in ("mean", "scale", "var")},
        }
        torch.save(checkpoint, output / "best_checkpoint.pt")
        (output / "summary.json").write_text(json.dumps({"seed": 7, "best_epoch": 4, "best_val_mse": reference_mse}))
        return diagnostic.Run("T00", 7, output / "best_checkpoint.pt"), data, frame, checkpoint, reference_mse

    def test_real_film_restore_uses_validation_scaler_and_learned_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            run, data, frame, _, reference = self.make_real_checkpoint(Path(directory))
            with patch("forecasting_task.data_provider.dataset.pd.read_parquet", return_value=frame), \
                 patch.object(diagnostic, "Dataset_Custom", wraps=Dataset_Custom) as factory, \
                 patch.object(diagnostic.film_module, "device", torch.device("cuda")), \
                 patch.object(Dataset_Custom, "inverse_transform", side_effect=AssertionError("Keep normalized units")):
                result = diagnostic.evaluate_run(run, data, torch.device("cpu"))
                self.assertEqual(diagnostic.film_module.device, torch.device("cuda"))
            self.assertEqual(factory.call_args.kwargs["flag"], "val")
            self.assertAlmostEqual(result["alpha"], 0.23, places=6)
            self.assertTrue(np.isclose(result["full_mse"], reference, rtol=1e-4, atol=1e-6))
            self.assertEqual(result["num_elements"], 22 * 3 * 4)  # Batch sizes 8, 8, 6.

    def test_mse_and_scaler_mismatches_stop_with_checkpoint_context(self):
        with tempfile.TemporaryDirectory() as directory:
            run, data, frame, checkpoint, reference = self.make_real_checkpoint(Path(directory))
            run.summary.write_text(json.dumps({"seed": 7, "best_epoch": 4, "best_val_mse": reference + 10}))
            with patch("forecasting_task.data_provider.dataset.pd.read_parquet", return_value=frame):
                with self.assertRaisesRegex(ValueError, "Validation MSE mismatch") as error:
                    diagnostic.evaluate_run(run, data, torch.device("cpu"))
            self.assertIn(str(run.checkpoint), str(error.exception))
            self.assertIn("full_mse=", str(error.exception))
            self.assertIn("summary.best_val_mse=", str(error.exception))
            checkpoint["scaler_mean"][0] += 1
            torch.save(checkpoint, run.checkpoint)
            with patch("forecasting_task.data_provider.dataset.pd.read_parquet", return_value=frame), \
                 patch.object(diagnostic.film_module, "Model", side_effect=AssertionError("Must check scaler first")):
                with self.assertRaisesRegex(ValueError, "Scaler mean mismatch"):
                    diagnostic.evaluate_run(run, data, torch.device("cpu"))

    @staticmethod
    def rows_for(runs):
        rows = []
        for run in runs:
            full = {7: 1., 17: 3., 27: 5.}[run.seed]
            row = dict(ticker=run.ticker, seed=run.seed, checkpoint=str(run.checkpoint), alpha=0.23,
                       num_elements=24, full_mse=full, full_mae=full / 2, no_text_delta_mse=2.,
                       no_text_delta_mae=1., persistence_mse=4., persistence_mae=2.)
            for variant in diagnostic.VARIANTS[1:]:
                row[f"full_better_{variant}"] = int(full < row[f"{variant}_mse"])
                row[f"full_vs_{variant}_mse_change_pct"] = diagnostic.mse_change_pct(full, row[f"{variant}_mse"])
            rows.append(row)
        return rows

    def test_seed_statistics_percent_sign_and_strict_win_counts(self):
        runs = [diagnostic.Run(f"T{index:02}", seed, Path("unused")) for seed in diagnostic.SEEDS for index in range(10)]
        rows = self.rows_for(runs)
        seeds, stats, comparisons = diagnostic.aggregate(rows)
        self.assertEqual([row["full_mse"] for row in seeds], [1, 3, 5])
        mse = next(row for row in stats if row["metric"] == "full_mse")
        self.assertEqual((mse["mean"], mse["std"], mse["std_ddof"]), (3, 2, 1))
        self.assertEqual([row["full_better_count"] for row in comparisons], [10, 20])
        self.assertEqual([row["mse_change_pct"] for row in comparisons], [50, -25])
        self.assertIsNone(diagnostic.mse_change_pct(1, 0))
        self.assertIsNone(diagnostic.mse_change_pct(0, 0))

    def test_reports_after_all_checks_preserve_inputs_and_do_not_read_prediction_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inventory(root)
            runs = diagnostic.discover_runs(root)
            source = {path: path.read_bytes() for run in runs for path in (run.checkpoint, run.summary)}
            for run in runs:
                run.checkpoint.with_name("test_io.pt").write_bytes(b"must never be read")
            rows = self.rows_for(runs)
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if path.name.startswith("test_io"):
                    raise AssertionError("Prediction files must never be opened")
                return original_open(path, *args, **kwargs)

            with patch.object(diagnostic, "evaluate_run", side_effect=rows) as evaluate, \
                 patch.object(Path, "open", guarded_open), contextlib.redirect_stdout(io.StringIO()) as output:
                diagnostic.analyze(root, root / "unused-data")
            self.assertEqual(evaluate.call_count, 30)
            self.assertTrue(all(call.args[2] == torch.device("cpu") for call in evaluate.call_args_list))
            self.assertIn("full_better=10/30", output.getvalue())
            paths = diagnostic.report_paths(root)
            self.assertTrue(all(path.is_file() for path in paths))
            with paths[0].open(newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 30)
            self.assertEqual(float(saved[0]["alpha"]), 0.23)
            for path, contents in source.items():
                self.assertEqual(path.read_bytes(), contents)
            with patch.object(diagnostic, "evaluate_run") as evaluate, self.assertRaises(FileExistsError):
                diagnostic.analyze(root, root / "unused-data")
            evaluate.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inventory(root)
            rows = self.rows_for(diagnostic.discover_runs(root))
            effects = [*rows[:4], ValueError("Validation MSE mismatch at checkpoint")]
            with patch.object(diagnostic, "evaluate_run", side_effect=effects), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "MSE mismatch"):
                diagnostic.analyze(root, root / "unused-data")
            self.assertFalse(any(path.exists() for path in diagnostic.report_paths(root)))

    def test_cli_defaults_to_cpu_and_help_does_not_start_analysis(self):
        with patch.object(diagnostic, "analyze") as analyze:
            diagnostic.cli([])
            self.assertEqual(analyze.call_args.args[2], "cpu")
        with patch.object(diagnostic, "analyze") as analyze, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                diagnostic.cli(["--help"])
            self.assertEqual(error.exception.code, 0)
            analyze.assert_not_called()
