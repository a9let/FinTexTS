"""Read-only historical-control replay and pre-training manifest checks."""

import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import gate_history_control as control


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        dates = [*pd.date_range("2021-01-01", periods=80), *pd.date_range("2022-01-01", periods=10), *pd.date_range("2023-01-01", periods=10)]
        columns = {"date": [str(d.date()) for d in dates], "ticker": ["AAPL"]*len(dates)}
        for c, channel in enumerate(("open", "high", "low", "close")):
            columns[channel] = np.arange(len(dates), dtype=float)+c
        rng = np.random.default_rng(7)
        for group in control.TEXT_GROUPS:
            for d in range(384):
                columns[f"{group}_emb{d}"] = rng.normal(size=len(dates)).astype(np.float32)
        frame = pd.DataFrame(columns)
        config = {"hidden_size": 64, "pred_len": 3, "num_layers": 1, "group_gate": "context_price_only"}
        args = {**control.SETTINGS, "seed": 7, "data_path": "AAPL.parquet", "group_gate": "context_price_only",
                "group_weight_mode": "context_price_only", "fixed_alpha": .1, "text_weight_mode": "fixed",
                "novelty_policy": control.NOVELTY_POLICY, "sampling_policy": "independent_train_generator",
                "gate_structure": {"type": "Linear", "in_features": 18, "out_features": 5, "parameters": 95, "initialization": "zeros"},
                "text_model_config": config}
        datasets = {}
        with patch("pandas.read_parquet", return_value=frame):
            for split in ("val", "test"):
                datasets[split] = control.ContextNoveltyDataset(control.Dataset_Custom(64, 16, 3, split, list(control.TEXT_GROUPS), data_path="AAPL.parquet"))
        with control.execution_device(torch.device("cpu")), torch.no_grad():
            price = control.film_module.Model(SimpleNamespace(**args)).eval()
            text = control.build_text_model(config, "context_price_only").eval()
            error, count = 0., 0
            for batch in DataLoader(datasets["val"], batch_size=32):
                x, y, e, xm, ym = [v.float() for v in batch[:5]]
                truth = y[:, -3:]
                decoder = torch.cat((y[:, :16], torch.zeros_like(truth)), 1)
                prediction = text(price(x, xm, decoder, ym), x[:, -1], e, price_window=x,
                                  **{k: v.float() for k, v in batch[6].items()})
                error += (prediction.double()-truth.double()).square().sum().item()
                count += truth.numel()
        output = root/"output"
        output.mkdir()
        checkpoint = {"args": args, "epoch": 1, "model_state_dict": price.state_dict(), "text_model_state_dict": text.state_dict(),
                      **{f"scaler_{name}": getattr(datasets["val"].scaler, name+"_").tolist() for name in ("mean", "scale", "var")}}
        torch.save(checkpoint, output/"best_checkpoint.pt")
        summary = {"group_gate": "context_price_only", "seed": 7, "data_path": "AAPL.parquet", "best_epoch": 1, "best_val_mse": error/count}
        (output/"summary.json").write_text(json.dumps(summary))
        x = torch.stack([torch.from_numpy(datasets["test"][i][0]).float() for i in range(len(datasets["test"]))])
        y = torch.stack([torch.from_numpy(datasets["test"][i][1][-3:]).float() for i in range(len(datasets["test"]))])
        torch.save({"inputs": x, "ground_truths": y}, output/"test_io.pt")
        with (output/"validation_group_diagnostics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["window_index", "group", "input_end_date"])
            writer.writeheader()
            for i in range(len(datasets["val"])):
                for group in control.TEXT_GROUPS:
                    writer.writerow({"window_index": i, "group": group, "input_end_date": datasets["val"].window_end_date(i)})
        return output, frame, args, datasets, checkpoint

    def test_historical_partial_label_mse_replay_and_new_run_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            output, frame, args, datasets, _ = self.fixture(Path(directory))
            original = {p: p.read_bytes() for p in output.iterdir()}
            with patch("pandas.read_parquet", return_value=frame), patch.object(control, "DataLoader", wraps=DataLoader) as loaders:
                record = control.validate_control(output, "AAPL", 7, Path(directory))
            self.assertEqual(loaders.call_count, 1)  # Only validation inference; test arrays are read, not predicted.
            self.assertEqual(record["status"], "verified_replay")
            self.assertEqual(record["historical_window_verification"], "partial_historical_validation")
            self.assertAlmostEqual(record["replayed_val_mse"], record["saved_val_mse"])
            manifest = Path(directory)/"controls.json"
            manifest.write_text(json.dumps({"schema": "gate_text_history_control_v1", "runs": [record]}))
            current = SimpleNamespace(**copy.deepcopy({**args, "gate_text_history_mode": "history", "window_evidence": record["window_evidence"]}))
            reference = control.load_gate_control(manifest, current, list(datasets.values()))
            self.assertEqual(reference["key"], "AAPL/seed7")
            current.window_evidence["val"]["targets_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "windows/targets differ"):
                control.load_gate_control(manifest, current, list(datasets.values()))
            for path, value in original.items():
                self.assertEqual(path.read_bytes(), value)

    def test_control_mse_scaler_and_saved_targets_mismatches_rejected(self):
        for change in ("mse", "scaler", "targets"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                output, frame, _, _, checkpoint = self.fixture(Path(directory))
                if change == "mse":
                    path = output/"summary.json"
                    payload = json.loads(path.read_text())
                    payload["best_val_mse"] += 10
                    path.write_text(json.dumps(payload))
                elif change == "scaler":
                    checkpoint["scaler_mean"][0] += 10
                    torch.save(checkpoint, output/"best_checkpoint.pt")
                else:
                    path = output/"test_io.pt"
                    payload = torch.load(path, weights_only=True)
                    payload["ground_truths"] += 10
                    torch.save(payload, path)
                with patch("pandas.read_parquet", return_value=frame), self.assertRaises(ValueError) as error:
                    control.validate_control(output, "AAPL", 7, Path(directory))
                self.assertIn(str(output/"best_checkpoint.pt"), str(error.exception))

    def test_existing_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"manifest.json"
            output.write_text("preserve")
            with self.assertRaises(FileExistsError):
                control.verify_controls(Path(directory), Path(directory), output)
            self.assertEqual(output.read_text(), "preserve")
