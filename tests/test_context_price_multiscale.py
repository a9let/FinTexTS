"""Variable lag ordering, legacy compatibility, RNG isolation and checkpoint recovery."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_task"))
from forecasting_task import run
from forecasting_task.models.GroupText import (
    ContextNoveltyTextModel, TEXT_GROUPS, validate_price_context_lags,
    price_context_metadata, build_price_context, build_text_model,
)

LAGS = [1, 2, 5, 10, 20]


class Windows(Dataset):
    seq_len, pred_len = 64, 3

    def __init__(self):
        self.row_dates = pd.bdate_range("2022-01-03", periods=72).strftime("%Y-%m-%d").to_numpy()
        self.prices = torch.arange(72 * 4, dtype=torch.float32).view(72, 4) / 100
        self.text = torch.randn(72, 5, 384, generator=torch.Generator().manual_seed(43))
        self.text[:, 1] = 0
        self.scaler = SimpleNamespace(mean_=np.zeros(4), scale_=np.ones(4), var_=np.ones(4))
        self.seen = []

    def __len__(self):
        return 5

    def __getitem__(self, index):
        self.seen.append(index)
        return (self.prices[index:index+64], self.prices[index+48:index+67], self.text[index:index+64],
                torch.zeros(64, 3), torch.zeros(19, 3), index,
                {"novelty": torch.ones(5), "pair_available": torch.ones(5)})

    def window_end_date(self, index):
        return str(self.row_dates[index+63])

    def inverse_transform(self, values):
        return values


class MultiscaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_indices_ohcl_blocks_and_user_supplied_lag_order(self):
        t = torch.arange(24, dtype=torch.float32)
        x = torch.stack([t.square() * (channel+1) + t * channel for channel in range(4)], dim=-1)[None]
        lags = [5, 1, 20, 2, 10]
        expected = torch.tensor([[(2*23*lag-lag*lag)*(c+1)+c*lag for lag in lags for c in range(4)]], dtype=torch.float32)
        torch.testing.assert_close(build_price_context(x, lags), expected, rtol=0, atol=0)
        metadata = price_context_metadata(lags)
        self.assertEqual(metadata["price_context_lags"], lags)
        self.assertEqual(metadata["price_feature_order"], ["open", "high", "low", "close"])
        self.assertEqual(metadata["price_context_feature_order"][:4], [f"price_delta_lag5_{c}" for c in metadata["price_feature_order"]])
        self.assertEqual(metadata["gate_input_size"], 30)
        torch.testing.assert_close(build_price_context(x, [23]), x[:, -1] - x[:, 0])

    def test_invalid_lags_and_unsupported_conditions_are_rejected(self):
        for values in ([], [0], [-1], [1, 1], [1.5], [True], ["5"], [64], 5):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_price_context_lags(values, 64)
        for flags in (["--price_context_lags"], ["--price_context_lags", "0"],
                      ["--price_context_lags", "1", "1"], ["--price_context_lags", "1.5"],
                      ["--price_context_lags", "64"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", "--group_gate", "context_price_only", *flags])
        for condition in (None, "static", "context_novelty", "context_no_novelty"):
            flags = [] if condition is None else ["--group_gate", condition]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", *flags, "--price_context_lags", "5", "1"])
        model = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", price_context_lags=[1], seq_len=2)
        self.assertEqual(model.gate.in_features, 14)
        with self.assertRaises(ValueError):
            ContextNoveltyTextModel(64, 3, 1, group_gate="context_novelty", price_context_lags=LAGS)

    def test_default_legacy_config_and_explicit_1_5_match_old_nonzero_gate_forward(self):
        torch.manual_seed(17)
        config = dict(hidden_size=64, pred_len=3, num_layers=1, group_gate="context_price_only")
        legacy = build_text_model(config, "context_price_only")
        with torch.no_grad():
            legacy.gate.weight.normal_()
            legacy.gate.bias.normal_()
        explicit = ContextNoveltyTextModel(**config, price_context_lags=[1, 5])
        explicit.load_state_dict(legacy.state_dict())
        x, e, price = torch.randn(4, 24, 4), torch.randn(4, 5, 384), torch.randn(4, 3, 4)
        e[:, 1] = 0
        context = torch.cat((x[:, -1]-x[:, -2], x[:, -1]-x[:, -6]), -1)
        features = torch.cat((context, torch.zeros(4, 10)), -1)
        weights = torch.softmax(legacy.gate(features), -1)
        heads = torch.stack([head(e[:, i]) for i, head in enumerate(legacy.group_mlps.values())], 1)
        delta = (heads * (e.abs().sum(-1) >= 1e-6)[:, :, None] * weights[:, :, None]).sum(1)
        alpha = legacy.get_alpha()
        expected = (1-alpha)*price + alpha*(x[:, -1:, :] + delta[:, :, None])
        kwargs = dict(price_window=x, novelty=torch.rand(4, 5), pair_available=torch.rand(4, 5))
        for model in (legacy, explicit):
            torch.testing.assert_close(model(price, x[:, -1], e, **kwargs), expected, rtol=0, atol=0)
            self.assertEqual(model.gate_input_size, 18)

    def test_gradients_reach_all_new_price_columns_and_input_lag_rows(self):
        torch.manual_seed(27)
        model = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", price_context_lags=LAGS)
        self.assertEqual(sum(p.numel() for p in model.gate.parameters()), 155)
        self.assertTrue(all((p == 0).all() for p in model.gate.parameters()))
        x = torch.randn(4, 24, 4)
        output, detail = model(torch.randn(4, 3, 4), x[:, -1], torch.randn(4, 5, 384), price_window=x,
                               novelty=torch.ones(4, 5), pair_available=torch.ones(4, 5), return_details=True)
        (output - 7).square().mean().backward()
        gradient = model.gate.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient[:, :20].abs().sum(0) > 0).all())
        torch.testing.assert_close(gradient[:, 20:], torch.zeros(5, 10), rtol=0, atol=0)
        self.assertTrue((detail["features"][:, 20:] == 0).all())
        inputs = torch.randn(1, 24, 4, requires_grad=True)
        build_price_context(inputs, LAGS).sum().backward()
        expected = torch.zeros_like(inputs)
        expected[:, -1] = len(LAGS)
        for lag in LAGS:
            expected[:, -1-lag] = -1
        torch.testing.assert_close(inputs.grad, expected, rtol=0, atol=0)

    def test_runner_initialization_batch_order_checkpoint_and_variable_diagnostics(self):
        starts, orders = [], []
        for lags in ([1, 5], LAGS):
            with self.subTest(lags=lags), tempfile.TemporaryDirectory() as directory:
                datasets = {flag: Windows() for flag in ("train", "val", "test")}
                loaders = {flag: DataLoader(data, batch_size=2, shuffle=flag == "train") for flag, data in datasets.items()}
                def factory(**kwargs):
                    return datasets[kwargs["flag"]], loaders[kwargs["flag"]]
                original = run._run_epoch
                epochs = []
                def epoch(loader, dataset, film, text, args, device, optimizer=None, collect_io=False):
                    if optimizer is not None:
                        if not epochs:
                            starts.append(({k: v.clone() for k, v in film.state_dict().items()},
                                           {k: v.clone() for k, v in text.group_mlps.state_dict().items()}, torch.get_rng_state().clone()))
                        epochs.append(1)
                    result = original(loader, dataset, film, text, args, device, optimizer, collect_io)
                    if optimizer is not None and len(epochs) == 2:
                        with torch.no_grad():
                            for head in text.group_mlps.values():
                                head[0].bias.add_(1000)
                    return result
                argv = ["--model_type", "FiLM", "--group_gate", "context_price_only", "--seed", "7",
                        "--code_commit", "a"*40, "--e_layers", "1", "--num_epoch", "2", "--logdir", directory,
                        "--price_context_lags", *map(str, lags)]
                with patch("forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader", side_effect=factory), \
                     patch.object(run, "_run_epoch", side_effect=epoch), \
                     patch("torch.cuda.is_available", return_value=False), \
                     patch("forecasting_task.models.FiLM.device", torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                orders.append(datasets["train"].seen.copy())
                root = Path(directory)
                args = yaml.safe_load((root/"args.yaml").read_text())
                summary = json.loads((root/"summary.json").read_text())
                checkpoint = torch.load(root/"best_checkpoint.pt", weights_only=True)
                self.assertEqual(checkpoint["epoch"], 1)
                self.assertEqual(checkpoint["args"], args)
                layout = price_context_metadata(lags)
                for record in (args, summary, checkpoint):
                    for key, value in layout.items():
                        self.assertEqual(record[key], value)
                self.assertEqual(summary["window_evidence"], args["window_evidence"])
                with (root/"validation_group_diagnostics.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), len(datasets["val"])*5)
                expected = build_price_context(datasets["val"][0][0][None], lags)[0].numpy()
                for name, value in zip(layout["price_context_feature_order"], expected):
                    self.assertAlmostEqual(float(rows[0]["gate_feature_"+name]), float(value))
                self.assertEqual(json.loads(rows[0]["price_context_lags"]), lags)
                self.assertEqual(int(rows[0]["gate_input_size"]), len(lags)*4+10)
                text = build_text_model(args["text_model_config"], args["group_weight_mode"])
                text.load_state_dict(checkpoint["text_model_state_dict"])
                with patch("forecasting_task.models.FiLM.device", torch.device("cpu")):
                    price = run.FiLM(SimpleNamespace(**args)).cpu()
                price.load_state_dict(checkpoint["model_state_dict"])
                with torch.no_grad():
                    result = original(loaders["test"], datasets["test"], price, text, SimpleNamespace(**args), torch.device("cpu"), collect_io=True)
                    batch = list(next(iter(loaders["test"])))
                    changed = list(batch)
                    changed[2] = batch[2].clone()
                    changed[2][:, :-1] += 1000
                    a = original([batch], datasets["test"], price, text, SimpleNamespace(**args), torch.device("cpu"), collect_io=True)
                    b = original([changed], datasets["test"], price, text, SimpleNamespace(**args), torch.device("cpu"), collect_io=True)
                saved = torch.load(root/"test_io.pt", weights_only=False)
                torch.testing.assert_close(result[3], saved["outputs"], rtol=0, atol=0)
                torch.testing.assert_close(a[3], b[3], rtol=0, atol=0)
        for actual, expected in zip(starts[1][:2], starts[0][:2]):
            for key, value in actual.items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        torch.testing.assert_close(starts[1][2], starts[0][2], rtol=0, atol=0)
        self.assertEqual(orders[0], orders[1])
