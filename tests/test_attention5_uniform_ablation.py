"""Frozen validation interpolation; synthetic inputs, checkpoints and reports only."""

import contextlib
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
from torch.utils.data import DataLoader, Dataset

import analyze_attention5_uniform_ablation as analysis
from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS


class Windows(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(31)
        self.x = torch.randn(5, 7, 4, generator=generator)
        self.y = torch.randn(5, 5, 4, generator=generator)
        self.text = torch.randn(5, 7, 5, 384, generator=generator)
        self.text[0] = 0
        self.text[:, -5:, 1] = 0
        self.text[1, -5:, 2] = 0
        self.text[1, -1, 2] = 2
        self.y[-1] += 15  # An incomplete batch must not be underweighted.

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.text[index], torch.zeros(7, 3), torch.zeros(5, 3), index,
                {"novelty": torch.ones(5), "pair_available": torch.ones(5)})


class Price(nn.Module):
    def forward(self, x, *args):
        assert not self.training and not torch.is_grad_enabled()
        return x[:, -1:] * 2 + torch.arange(3)[None, :, None]


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_mixed_weights_preserve_masks_empty_and_single_groups(self):
        beta = torch.tensor([[.8, 0, .2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1., 0]])
        present = torch.tensor([[True, False, True, False, False], [False]*5, [False, False, False, True, False]])
        uniform = torch.tensor([[.5, 0, .5, 0, 0], [0]*5, [0, 0, 0, 1., 0]])
        for mixing in analysis.LAMBDAS:
            actual = analysis.mix_temporal_weights(beta, present, mixing)
            torch.testing.assert_close(actual, (1-mixing)*uniform+mixing*beta)
            self.assertTrue((actual[~present] == 0).all())
            torch.testing.assert_close(actual.sum(-1), torch.tensor([1., 0, 1.]))
        bad = beta.clone()
        bad[0, 1] = .01
        with self.assertRaises(ValueError):
            analysis.mix_temporal_weights(bad, present, .5)

    def test_predictions_all_three_lambdas_and_element_weighting_partial_batch(self):
        data = Windows()
        text = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool="attention5").eval()
        with torch.no_grad():
            text.attention_pool.query.weight.normal_(std=.1)
            text.gate.weight.normal_(std=.1)
            price = Price().eval()(data.x)
            original, detail = text(price, data.x[:, -1], data.text, price_window=data.x,
                                    novelty=torch.ones(5, 5), pair_available=torch.ones(5, 5), return_details=True)
            predictions = []
            for mixing in analysis.LAMBDAS:
                prediction = analysis.interpolated_prediction(text, price, data.x[:, -1], data.text, detail, mixing)
                predictions.append(prediction)
                torch.testing.assert_close(prediction[0], .9*price[0]+.1*data.x[0, -1:])
            torch.testing.assert_close(predictions[2], original, rtol=0, atol=0)
            torch.testing.assert_close(predictions[1], (predictions[0]+predictions[2])/2)
            uniform = detail["daily_present"].float()/detail["available_count"].clamp_min(1)[..., None]
            mean = (uniform[..., None]*data.text[:, -5:].transpose(1, 2)).sum(2)
            heads = torch.stack([head(mean[:, g]) for g, head in enumerate(text.group_mlps.values())], 1)
            delta = (heads*detail["present"][..., None]*detail["weights"][..., None]).sum(1)
            torch.testing.assert_close(predictions[0], .9*price+.1*(data.x[:, -1:]+delta[..., None]))
            expected = [(prediction.double()-data.y[:, -3:].double()).square().mean().item() for prediction in predictions]
        state = {k: v.clone() for k, v in text.state_dict().items()}
        for batch_size in (1, 2, 4):
            result = analysis.evaluate_batches(Price(), text, DataLoader(data, batch_size=batch_size), 3, 2, torch.device("cpu"))
            self.assertEqual(result["num_elements"], 5*3*4)
            for key, value in zip(analysis.MSE_KEYS, expected):
                self.assertAlmostEqual(result[key], value, places=5)
        for name, value in text.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in text.parameters()))

    def real_checkpoint(self, root):
        dates = [*pd.date_range("2021-01-01", periods=18), *pd.date_range("2022-01-01", periods=12), *pd.date_range("2023-01-01", periods=12)]
        values = {"date": [str(d.date()) for d in dates], "ticker": ["AAPL"]*len(dates)}
        for index, name in enumerate(("open", "high", "low", "close")):
            values[name] = np.arange(len(dates), dtype=float)+index
        generator = np.random.default_rng(31)
        for group in TEXT_GROUPS:
            for dimension in range(384):
                values[f"{group}_emb{dimension}"] = generator.normal(size=len(dates)).astype(np.float32)
        frame = pd.DataFrame(values)
        config = {"hidden_size": 64, "pred_len": 3, "num_layers": 1, "group_gate": "context_price_only",
                  "text_temporal_pool": "attention5", "attention_config": analysis.attention5_config()}
        args = {"task_name": "short_term_forecast", "seq_len": 6, "label_len": 2, "pred_len": 3,
                "output_attention": False, "e_layers": 1, "enc_in": 4, "model_type": "FiLM",
                "group_gate": "context_price_only", "group_weight_mode": "context_price_only", "fusion_mode": "blend",
                "seed": 7, "used_col_prefixes": list(TEXT_GROUPS), "text_num_layers": 1, "text_weight": .1,
                "fixed_alpha": .1, "data_path": "AAPL.parquet", "batch_size": 4, "text_temporal_pool": "attention5",
                "attention_config": analysis.attention5_config(), "text_model_config": config}
        with patch("pandas.read_parquet", return_value=frame):
            base = analysis.Dataset_Custom(6, 2, 3, "val", list(TEXT_GROUPS), root_path=str(root), data_path="AAPL.parquet")
            dataset = analysis.ContextNoveltyDataset(base)
        evidence = analysis.window_evidence(dataset)
        args["window_evidence"] = {"val": evidence}
        with analysis.execution_device(torch.device("cpu")), torch.no_grad():
            price = analysis.film_module.Model(SimpleNamespace(**args)).eval()
            text = ContextNoveltyTextModel(**config).eval()
            text.attention_pool.query.weight.normal_(std=.1)
            result = analysis.evaluate_batches(price, text, DataLoader(dataset, batch_size=4), 3, 2, torch.device("cpu"))
        output = root/"full/context_price_only/AAPL/seed7/output"
        output.mkdir(parents=True)
        checkpoint = {"args": args, "epoch": 4, "model_state_dict": price.state_dict(), "text_model_state_dict": text.state_dict(),
                      **{f"scaler_{name}": getattr(dataset.scaler, name+"_").tolist() for name in ("mean", "scale", "var")}}
        torch.save(checkpoint, output/"best_checkpoint.pt")
        summary = {"seed": 7, "best_epoch": 4, "best_val_mse": result["mse_lambda1"], "data_path": "AAPL.parquet",
                   "text_temporal_pool": "attention5", "group_gate": "context_price_only",
                   "attention_config": analysis.attention5_config(), "window_evidence": {"val": evidence}}
        (output/"summary.json").write_text(json.dumps(summary))
        return analysis.Run("attention5", "AAPL", 7, output), frame, checkpoint, result

    def test_restore_scaler_windows_val_only_and_mse_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            run, frame, _, expected = self.real_checkpoint(Path(directory))
            original_device = analysis.film_module.device
            dataset_class = analysis.Dataset_Custom
            with patch("pandas.read_parquet", return_value=frame), \
                 patch.object(analysis, "Dataset_Custom", wraps=analysis.Dataset_Custom) as factory, \
                 patch.object(dataset_class, "inverse_transform", side_effect=AssertionError("Keep normalized prices")):
                actual = analysis.evaluate_run(run, Path(directory), torch.device("cpu"))
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.call_args.kwargs["flag"], "val")
            self.assertEqual(analysis.film_module.device, original_device)
            self.assertEqual(actual["num_elements"], 10*3*4)  # Partial final batch of two.
            for key in analysis.MSE_KEYS:
                self.assertEqual(actual[key], expected[key])

    def test_mse_scaler_and_window_mismatches_stop_with_checkpoint_context(self):
        for change in ("mse", "scaler", "window"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                run, frame, checkpoint, result = self.real_checkpoint(Path(directory))
                if change == "mse":
                    path = run.output/"summary.json"
                    payload = json.loads(path.read_text())
                    payload["best_val_mse"] += 100
                    path.write_text(json.dumps(payload))
                else:
                    if change == "scaler":
                        checkpoint["scaler_mean"][0] += 10
                    else:
                        checkpoint["args"]["window_evidence"]["val"]["inputs_sha256"] = "wrong"
                    torch.save(checkpoint, run.output/"best_checkpoint.pt")
                with patch("pandas.read_parquet", return_value=frame), self.assertRaises(ValueError) as error:
                    analysis.evaluate_run(run, Path(directory), torch.device("cpu"))
                self.assertIn(str(run.output/"best_checkpoint.pt"), str(error.exception))
                if change == "mse":
                    self.assertIn("lambda=1 MSE=", str(error.exception))
                    self.assertIn("best_val_mse=", str(error.exception))

    def test_inventory_aggregation_wins_after_seed_means_read_only_and_no_test_io(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ticker, seed in itertools_product():
                output = root/f"full/context_price_only/{ticker}/seed{seed}/output"
                output.mkdir(parents=True)
                (output/"best_checkpoint.pt").write_bytes(b"synthetic")
                (output/"summary.json").write_text("{}")
                (output/"test_io.pt").write_bytes(b"must never read")
            runs = analysis.discover_runs(root)
            rows = []
            for run in runs:
                value = {7: 1., 17: 3., 27: 5.}[run.seed]
                rows.append({"ticker": run.ticker, "seed": run.seed, "mse_lambda1": value, "mse_lambda05": value/2,
                             "mse_lambda0": 2., "window_evidence": {"val": "fake"}})
            tables = analysis.aggregate(rows)
            self.assertEqual(tables["across_seeds"][2]["sample_std"], 2)
            self.assertEqual(tables["comparisons"][0]["smoothed_wins"], 10)  # Mean 2 < 3, although some seeds lose.
            self.assertEqual(tables["comparisons"][1]["improvement_pct"], 50)
            original = {p: p.read_bytes() for p in root.rglob("*.*")}
            original_open = Path.open
            def guard(path, *args, **kwargs):
                if path.name == "test_io.pt":
                    raise AssertionError("No test arrays")
                return original_open(path, *args, **kwargs)
            with patch.object(analysis, "evaluate_run", side_effect=rows) as evaluate, patch.object(Path, "open", guard), contextlib.redirect_stdout(io.StringIO()):
                analysis.analyze(root, root/"unused-data")
            self.assertEqual(evaluate.call_count, 30)
            self.assertTrue(all(call.args[2] == torch.device("cpu") for call in evaluate.call_args_list))
            for p, value in original.items():
                self.assertEqual(p.read_bytes(), value)
            with self.assertRaises(FileExistsError):
                analysis.analyze(root)
            run = runs[0]
            run.output.joinpath("best_checkpoint.pt").rename(run.output/"backup.pt")
            with self.assertRaises(ValueError):
                analysis.discover_runs(root)


def itertools_product():
    import itertools
    return itertools.product(analysis.TICKERS, analysis.SEEDS)
