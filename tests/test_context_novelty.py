"""CPU checks for causal novelty, comparable initialization and best-val diagnostics."""

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
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_task"))
from forecasting_task import run
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset, causal_novelty
from forecasting_task.gate_diagnostics import write_gate_validation
from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS, build_text_model


def make_frame():
    dates = [*pd.date_range("2021-12-01", periods=18), *pd.date_range("2022-01-03", periods=12),
             *pd.date_range("2023-01-03", periods=12)]
    rng = np.random.default_rng(51)
    columns = {"date": [date.strftime("%Y-%m-%d") for date in dates], "ticker": ["SYN"] * len(dates)}
    for channel in ("open", "high", "low", "close"):
        columns[channel] = np.arange(len(dates)) + rng.normal(size=len(dates))
    embeddings = rng.normal(size=(len(dates), 5, 384)).astype(np.float32)
    embeddings[:, 0] = 0
    embeddings[0, 0, 0] = 1
    embeddings[17, 0, 1] = 1  # Previous available text lies outside the six-day window.
    embeddings[18, 0, 1] = -1
    embeddings[:17, 1] = 0
    embeddings[:, 4] = 0
    for group_index, group in enumerate(TEXT_GROUPS):
        for dimension in range(384):
            columns[f"{group}_emb{dimension}"] = embeddings[:, group_index, dimension]
    return pd.DataFrame(columns)


def make_dataset(frame, flag, seq_len=6):
    with patch("pandas.read_parquet", return_value=frame):
        base = Dataset_Custom(seq_len, 2, 3, flag, list(TEXT_GROUPS), data_path="SYN.parquet")
        return ContextNoveltyDataset(base)


class ContextNoveltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def make_film():
        args = SimpleNamespace(task_name="short_term_forecast", seq_len=6, label_len=2,
                               pred_len=3, output_attention=False, e_layers=1, enc_in=4)
        with patch("forecasting_task.models.FiLM.device", torch.device("cpu")):
            return run.FiLM(args).cpu()

    def test_shapes_formulas_masks_weights_and_initial_static_equality(self):
        for batch in (1, 4):
            for presence in ("all", "partial", "none", "threshold"):
                torch.manual_seed(7)
                static = ContextNoveltyTextModel(64, 3, 1)
                torch.manual_seed(7)
                dynamic = ContextNoveltyTextModel(64, 3, 1, group_gate="context_novelty")
                price, x = torch.randn(batch, 3, 4), torch.randn(batch, 6, 4)
                embeddings = torch.randn(batch, 5, 384)
                if presence == "partial":
                    embeddings[:, 1:] = 0
                if presence == "none":
                    embeddings.zero_()
                if presence == "threshold":
                    embeddings.zero_()
                    embeddings[:, 0, 0] = 1e-6
                    embeddings[:, 1, 0] = 0.5e-6
                kwargs = dict(price_window=x, novelty=torch.rand(batch, 5), pair_available=torch.ones(batch, 5))
                expected = None
                for model in (static, dynamic):
                    actual, detail = model(price, x[:, -1], embeddings, return_details=True, **kwargs)
                    self.assertEqual(tuple(actual.shape), (batch, 3, 4))
                    torch.testing.assert_close(detail["weights"], torch.full((batch, 5), 0.2), rtol=0, atol=0)
                    torch.testing.assert_close(detail["weights"].sum(-1), torch.ones(batch))
                    heads = torch.stack([head(embeddings[:, i]) for i, head in enumerate(model.group_mlps.values())], 1)
                    mask = embeddings.abs().sum(-1) >= 1e-6
                    delta = (heads * mask[:, :, None] * 0.2).sum(1)[:, :, None].expand_as(price)
                    formula = 0.9 * price + 0.1 * (x[:, -1:, :] + delta)
                    torch.testing.assert_close(actual, formula)
                    torch.testing.assert_close(detail["contributions"].sum(1)[:, :, None].expand_as(price), 0.1 * delta)
                    if expected is None:
                        expected = actual
                    else:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if presence == "none":
                        torch.testing.assert_close(actual, 0.9 * price + 0.1 * x[:, -1:])
                    for head in model.group_mlps.values():
                        self.assertEqual(len(head), 1)
                        self.assertIsInstance(head[0], nn.Linear)
                        self.assertEqual((head[0].in_features, head[0].out_features), (384, 3))

    def test_mask_after_bias_and_no_presence_renormalization(self):
        for mode in ("static", "context_novelty"):
            model = ContextNoveltyTextModel(64, 3, 1, group_gate=mode)
            for head in model.group_mlps.values():
                nn.init.zeros_(head[0].weight)
                nn.init.constant_(head[0].bias, 10)
            if model.gate is not None:
                with torch.no_grad():
                    model.gate.bias.copy_(torch.tensor([2., -1., 1., 0., 0.5]))
            embeddings = torch.zeros(2, 5, 384)
            embeddings[0, 0, 0] = 1
            price, x = torch.zeros(2, 3, 4), torch.ones(2, 6, 4)
            prediction, detail = model(price, x[:, -1], embeddings, price_window=x,
                                      novelty=torch.zeros(2, 5), pair_available=torch.zeros(2, 5), return_details=True)
            torch.testing.assert_close(prediction[0], torch.full((3, 4), 0.1) + detail["weights"][0, 0])
            torch.testing.assert_close(prediction[1], torch.full((3, 4), 0.1))
            self.assertTrue((detail["contributions"][:, 1:] == 0).all())

    def test_novelty_first_text_gaps_cosine_threshold_and_future_independence(self):
        embeddings = np.zeros((9, 5, 384), np.float32)
        embeddings[1, 0, 0] = 1
        embeddings[4, 0, 0] = -1
        embeddings[5, 0, 1] = 1
        embeddings[6, 0, 1] = 2
        embeddings[7, 1, 0] = 1e-6
        embeddings[8, 1, 0] = 1e-6
        embeddings[7, 2, 0] = 0.5e-6
        embeddings[8, 2, 0] = 1
        novelty, pairs = causal_novelty(embeddings)
        np.testing.assert_array_equal(novelty[:, 0], [0, 0, 0, 0, 2, 1, 0, 0, 0])
        np.testing.assert_array_equal(pairs[:, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0])
        self.assertEqual(pairs[7, 1], 0)
        self.assertEqual(pairs[8, 1], 1)
        self.assertEqual(pairs[8, 2], 0)
        self.assertTrue(np.isfinite(novelty).all())
        self.assertTrue(((novelty >= 0) & (novelty <= 2)).all())
        changed = embeddings.copy()
        changed[6:] = np.random.default_rng(9).normal(size=changed[6:].shape)
        other = causal_novelty(changed)
        for actual, expected in zip(other, (novelty, pairs)):
            np.testing.assert_array_equal(actual[:6], expected[:6])
        for actual, expected in zip(causal_novelty(embeddings[:6]), (novelty, pairs)):
            np.testing.assert_array_equal(actual, expected[:6])

    def test_wrapper_preserves_base_splits_scaler_dates_and_cross_split_history(self):
        frame = make_frame()
        datasets = {flag: make_dataset(frame, flag) for flag in ("train", "val", "test")}
        for dataset in datasets.values():
            expected_mean = frame[["open", "high", "low", "close"]].iloc[:18].mean().values
            np.testing.assert_allclose(dataset.scaler.mean_, expected_mean)
            for index in (0, len(dataset) - 1):
                for actual, expected in zip(dataset[index][:6], dataset.base[index]):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(dataset.window_end_date(index), dataset.row_dates[index + 5])
        validation = datasets["val"]
        self.assertEqual(validation.window_end_date(0), frame.date.iloc[17])
        self.assertEqual(validation[0][6]["novelty"][0], 1)
        self.assertEqual(validation[0][6]["pair_available"][0], 1)
        self.assertEqual(validation[0][6]["pair_available"][1], 0)
        self.assertEqual(validation[1][6]["novelty"][0], 2)
        # Changing future text and prices cannot affect earlier validation features.
        changed = frame.copy()
        changed.loc[25:, "macro_emb0"] = 500
        changed.loc[25:, "close"] = -1000
        other = make_dataset(changed, "val")
        np.testing.assert_array_equal(validation.novelty[:13], other.novelty[:13])
        for index in range(7):
            np.testing.assert_array_equal(validation[index][0], other[index][0])
            np.testing.assert_array_equal(validation[index][6]["novelty"], other[index][6]["novelty"])

    def test_novelty_independent_of_batch_size_shuffle_and_ticker(self):
        frame = make_frame()
        dataset = make_dataset(frame, "val")
        expected = {i: dataset[i][6] for i in range(len(dataset))}
        for batch_size in (1, 4, 7):
            for shuffle in (False, True):
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
                for batch in loader:
                    for sample, index in enumerate(batch[5].tolist()):
                        for key in ("novelty", "pair_available"):
                            np.testing.assert_array_equal(batch[6][key][sample].numpy(), expected[index][key])
        unrelated = frame.copy()
        unrelated["ticker"] = "OTHER"
        unrelated.loc[:, [f"macro_emb{i}" for i in range(384)]] = 0
        other = make_dataset(unrelated, "val")
        self.assertTrue((other.novelty[:, 0] == 0).all())
        np.testing.assert_array_equal(dataset.novelty, make_dataset(frame, "val").novelty)
        bad = frame.copy()
        bad.loc[1, "ticker"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "one identified ticker"):
            make_dataset(bad, "val")

    def test_price_context_uses_only_exact_input_lags(self):
        model = ContextNoveltyTextModel(64, 3, 1, group_gate="context_novelty")
        x = torch.arange(2 * 9 * 4, dtype=torch.float32).reshape(2, 9, 4)
        n, p = torch.rand(2, 5), torch.ones(2, 5)
        _, detail = model(torch.zeros(2, 3, 4), x[:, -1], torch.zeros(2, 5, 384), price_window=x,
                          novelty=n, pair_available=p, return_details=True)
        expected = torch.cat((x[:, -1] - x[:, -2], x[:, -1] - x[:, -6], n, p), -1)
        torch.testing.assert_close(detail["features"], expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "six"):
            model(torch.zeros(2, 3, 4), x[:, -1], torch.zeros(2, 5, 384), price_window=x[:, :5], novelty=n, pair_available=p)

    def test_joint_gradients_fixed_alpha_and_unique_optimizer(self):
        film = self.make_film()
        model = ContextNoveltyTextModel(64, 3, 1, group_gate="context_novelty")
        self.assertEqual(sum(p.numel() for p in model.gate.parameters()), 95)
        self.assertTrue(all((p == 0).all() for p in model.gate.parameters()))
        self.assertFalse(hasattr(model, "G"))
        self.assertFalse(hasattr(model, "alpha_logit"))
        self.assertFalse(model.get_alpha().requires_grad)
        parameters = list(film.parameters()) + list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn(id(model.fixed_alpha), ids)
        x = torch.randn(4, 6, 4)
        prediction = model(film(x, None, None, None), x[:, -1], torch.randn(4, 5, 384), price_window=x,
                           novelty=torch.rand(4, 5), pair_available=torch.ones(4, 5))
        prediction.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in film.parameters()))
        optimizer.step()
        self.assertIsNone(model.fixed_alpha.grad)
        torch.testing.assert_close(model.fixed_alpha, torch.tensor(0.1), rtol=0, atol=0)

    def test_matched_film_heads_rng_predictions_and_training_permutations(self):
        for seed in (7, 17, 27):
            reference = None
            for gate in ("static", "context_novelty"):
                run.set_seed(seed)
                film = self.make_film()
                text = ContextNoveltyTextModel(64, 3, 1, group_gate=gate)
                rng = torch.get_rng_state().clone()
                x, e = torch.randn(3, 6, 4), torch.randn(3, 5, 384)
                prediction = text(film(x, None, None, None), x[:, -1], e, price_window=x,
                                  novelty=torch.ones(3, 5), pair_available=torch.ones(3, 5))
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True, drop_last=True)
                run._configure_gate_sampling(loader, seed)
                orders = []
                for epoch in range(3):
                    torch.rand(9 if gate == "static" else 37)
                    orders.append(torch.cat([batch[0] for batch in loader]))
                actual = film.state_dict(), text.group_mlps.state_dict(), rng, prediction.detach(), orders
                if reference is None:
                    reference = actual
                else:
                    for state, expected in zip(actual[:2], reference[:2]):
                        for key, value in state.items():
                            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                    for value, expected in zip(actual[2:4], reference[2:4]):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    for value, expected in zip(orders, reference[4]):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_invalid_configuration_is_rejected(self):
        for extra in (["--seq_len", "5"], ["--text_weight", "0.5"], ["--fusion_mode", "residual"],
                      ["--text_num_layers", "2"], ["--model_type", "DLinear"],
                      ["--group_weight_mode", "learnable"], ["--used_col_prefixes", "macro,sector"],
                      ["--initial_group_weights", "0.6", "0.1", "0.1", "0.1", "0.1"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", "--group_gate", "context_novelty", "--code_commit", "a" * 40, *extra])

    def test_runner_checkpoint_and_reproduced_best_validation_diagnostics(self):
        frame = make_frame()
        for gate in ("static", "context_novelty"):
            with self.subTest(gate=gate), tempfile.TemporaryDirectory() as directory:
                datasets = {flag: make_dataset(frame, flag) for flag in ("train", "val", "test")}
                loaders = {flag: DataLoader(dataset, batch_size=4, shuffle=flag == "train", drop_last=flag == "train")
                           for flag, dataset in datasets.items()}
                def factory(**kwargs):
                    flag = kwargs["flag"]
                    return datasets[flag], loaders[flag]
                original_epoch = run._run_epoch
                train_epochs = []
                def epoch(loader, dataset, film, text, args, device, optimizer=None, collect_io=False):
                    result = original_epoch(loader, dataset, film, text, args, device, optimizer, collect_io)
                    if optimizer is not None:
                        train_epochs.append(1)
                        if len(train_epochs) == 2:
                            # Force the final synthetic state to be worse: diagnostics
                            # must restore epoch 1 rather than export final heads.
                            with torch.no_grad():
                                for head in text.group_mlps.values():
                                    head[0].bias.add_(1000)
                    return result
                argv = ["--model_type", "FiLM", "--group_gate", gate, "--code_commit", "a" * 40,
                        "--seq_len", "6", "--label_len", "2", "--pred_len", "3", "--batch_size", "4",
                        "--e_layers", "1", "--num_epoch", "2", "--logdir", directory]
                with patch("forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader", side_effect=factory), \
                     patch.object(run, "_run_epoch", side_effect=epoch), \
                     patch("torch.cuda.is_available", return_value=False), \
                     patch("forecasting_task.models.FiLM.device", torch.device("cpu")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                root = Path(directory)
                args = yaml.safe_load((root / "args.yaml").read_text())
                history = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()]
                summary = json.loads((root / "summary.json").read_text())
                checkpoint = torch.load(root / "best_checkpoint.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["args"], args)
                self.assertEqual(checkpoint["epoch"], 1)
                self.assertEqual(summary["epochs_completed"], 2)
                self.assertEqual(checkpoint["metrics"], history[checkpoint["epoch"] - 1])
                for record in (args, summary, checkpoint, *history):
                    self.assertEqual(record["group_gate"], gate)
                    self.assertEqual(record["fixed_alpha"], 0.1)
                    self.assertEqual(record["text_weight_mode"], "fixed")
                    self.assertEqual(record["code_commit"], "a" * 40)
                self.assertNotIn("G", checkpoint["text_model_state_dict"])
                self.assertNotIn("alpha_logit", checkpoint["text_model_state_dict"])
                price, text = self.make_film(), build_text_model(args["text_model_config"], args["group_weight_mode"])
                price.load_state_dict(checkpoint["model_state_dict"])
                text.load_state_dict(checkpoint["text_model_state_dict"])
                price.eval()
                text.eval()
                with torch.no_grad():
                    result = run._run_epoch(loaders["test"], datasets["test"], price, text, SimpleNamespace(**args),
                                            torch.device("cpu"), collect_io=True)
                saved = torch.load(root / "test_io.pt", map_location="cpu", weights_only=False)
                torch.testing.assert_close(result[3], saved["outputs"], rtol=0, atol=0)
                diagnostic = summary["validation_diagnostics"]
                self.assertAlmostEqual(diagnostic["mse"], summary["best_val_mse"], places=5)
                with (root / diagnostic["table"]).open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), len(datasets["val"]) * 5)
                self.assertEqual([row["group"] for row in rows[:5]], list(TEXT_GROUPS))
                self.assertEqual(rows[0]["input_end_date"], datasets["val"].window_end_date(0))
                for row in rows:
                    for h in range(1, 4):
                        expected = 0.1 * float(row["weight"]) * int(row["present"]) * float(row[f"head_h{h}"])
                        self.assertAlmostEqual(float(row[f"contribution_h{h}"]), expected, places=6)
                for name in diagnostic["plots"]:
                    self.assertTrue((root / name).read_bytes().startswith(b"\x89PNG"))
                # Regenerate from the restored best state and require identical CSV.
                with tempfile.TemporaryDirectory() as restored:
                    diag_args = SimpleNamespace(**{**args, "logdir": restored,
                                                  "diagnostic_best_val_mse": summary["best_val_mse"]})
                    repeated = write_gate_validation(loaders["val"], datasets["val"], price, text, diag_args,
                                                     torch.device("cpu"), run._prepare_batch)
                    self.assertEqual((Path(restored) / repeated["table"]).read_bytes(), (root / diagnostic["table"]).read_bytes())
