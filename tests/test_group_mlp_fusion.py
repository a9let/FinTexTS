"""CPU-only synthetic checks. Run: python -m unittest discover -s tests -v"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

# The existing TS models import `layers`, `models` and `utils` as top-level packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_task"))
from forecasting_task import run
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.models.GroupText import TextModel, TEXT_GROUPS


INITIALIZATIONS = {
    "uniform": [0.2, 0.2, 0.2, 0.2, 0.2],
    "macro": [0.6, 0.1, 0.1, 0.1, 0.1],
    "targetCompany": [0.1, 0.1, 0.6, 0.1, 0.1],
}


class GroupFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(23)

    def test_shapes_formulas_and_shared_ohlc_delta(self):
        for batch_size in (1, 4):
            for num_layers in (1, 2, 4):
                for mode in ("blend", "residual"):
                    with self.subTest(batch=batch_size, layers=num_layers, mode=mode):
                        model = TextModel(8, 3, num_layers, fusion_mode=mode)
                        x = torch.randn(batch_size, 5, 384)
                        x[:, 1] = 0
                        if batch_size > 1:
                            x[-1] = 0
                        price = torch.randn(batch_size, 3, 4)
                        last = torch.randn(batch_size, 4)
                        with torch.no_grad():
                            model.G.copy_(torch.tensor([-2., 0., 1., 2., -1.]))
                        delta = torch.zeros(batch_size, 3)
                        weights = torch.softmax(model.G, dim=0)
                        present = x.abs().sum(-1) >= 1e-6
                        for index, group in enumerate(TEXT_GROUPS):
                            value = x[:, index]
                            # Independent reference calculation from the linear parameters.
                            for layer in model.group_mlps[group]:
                                if isinstance(layer, nn.Linear):
                                    value = torch.nn.functional.linear(value, layer.weight, layer.bias)
                                else:
                                    value = torch.relu(value)
                            delta += weights[index] * present[:, index, None] * value
                        expanded = delta[:, :, None].expand(-1, -1, 4)
                        alpha = torch.sigmoid(model.alpha_logit)
                        if mode == "blend":
                            text = (last[:, None] + expanded) * present.any(-1)[:, None, None]
                            expected = (1 - alpha) * price + alpha * text
                        else:
                            expected = price + alpha * expanded
                        actual = model(price, last, x)
                        self.assertEqual(actual.shape, (batch_size, 3, 4))
                        torch.testing.assert_close(actual, expected)
                        if mode == "residual":
                            torch.testing.assert_close(model(price, last + 123, x), actual)
                        for mlp in model.group_mlps.values():
                            linears = [layer for layer in mlp if isinstance(layer, nn.Linear)]
                            self.assertEqual(len(linears), num_layers)
                            self.assertEqual(linears[0].in_features, 384)
                            self.assertEqual(linears[-1].out_features, 3)
                        parameter_sets = [set(id(p) for p in mlp.parameters()) for mlp in model.group_mlps.values()]
                        self.assertEqual(sum(map(len, parameter_sets)), len(set.union(*parameter_sets)))

    def test_missing_groups_bias_threshold_and_no_renormalization(self):
        for num_layers in (1, 2):
            for mode in ("blend", "residual"):
                with self.subTest(layers=num_layers, mode=mode):
                    model = TextModel(8, 3, num_layers, fusion_mode=mode)
                    for index, mlp in enumerate(model.group_mlps.values()):
                        for parameter in mlp.parameters():
                            nn.init.zeros_(parameter)
                        nn.init.constant_(mlp[-1].bias, 10. * (index + 1))
                    torch.testing.assert_close(model.get_alpha(), torch.tensor(0.1))
                    torch.testing.assert_close(model.get_group_weights(), torch.full((5,), 0.2))
                    x = torch.zeros(4, 5, 384)
                    x[0, 0, 0] = 1  # Only macro contributes: 0.2 * 10 = 2.
                    x[2, 0, 0] = 1e-6  # Boundary is present.
                    x[3, :, 0] = 0.5e-6  # All individually absent despite a larger total sum.
                    price = torch.full((4, 3, 4), 7.)
                    last = torch.full((4, 4), 3.)
                    expected = price.clone() if mode == "residual" else 0.9 * price
                    expected[[0, 2]] += 0.1 * (2. if mode == "residual" else 5.)
                    torch.testing.assert_close(model(price, last, x), expected)
                    model(price, last, x).sum().backward()
                    for group in TEXT_GROUPS[1:]:
                        for parameter in model.group_mlps[group].parameters():
                            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))

    def test_joint_gradients_and_optimizer_membership(self):
        for num_layers in (1, 2):
            for mode in ("blend", "residual"):
                with self.subTest(layers=num_layers, mode=mode):
                    film = self.make_film()
                    model = TextModel(8, 3, num_layers, fusion_mode=mode)
                    # Positive hidden activations avoid accidental dead ReLUs.
                    for index, mlp in enumerate(model.group_mlps.values()):
                        for layer in mlp:
                            if isinstance(layer, nn.Linear):
                                nn.init.constant_(layer.weight, 0.001 * (index + 1))
                                nn.init.constant_(layer.bias, 0.05)
                    optimizer = torch.optim.Adam(list(film.parameters()) + list(model.parameters()), lr=0.001)
                    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(ids.count(id(model.alpha_logit)), 1)
                    self.assertEqual(set(ids), {id(p) for p in list(film.parameters()) + list(model.parameters())})
                    price_x = torch.randn(3, 12, 4)
                    embeddings = torch.rand(3, 5, 384)
                    output = model(film(price_x, None, None, None), price_x[:, -1], embeddings)
                    (output - 20).square().mean().backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in film.parameters()))
                    before = {name: p.detach().clone() for name, p in model.named_parameters()}
                    film_before = {name: p.detach().clone() for name, p in film.named_parameters()}
                    optimizer.step()
                    for name, parameter in model.named_parameters():
                        self.assertFalse(torch.equal(parameter, before[name]), name)
                    self.assertTrue(any(not torch.equal(p, film_before[name]) for name, p in film.named_parameters()))

    @staticmethod
    def make_film():
        config = SimpleNamespace(task_name="short_term_forecast", seq_len=12, label_len=2,
                                 pred_len=3, output_attention=False, e_layers=1, enc_in=4)
        with patch("forecasting_task.models.FiLM.device", torch.device("cpu")):
            return run.FiLM(config).cpu()

    def test_dataset_preserves_groups_train_scaler_and_time_splits(self):
        dates = sum([list(pd.date_range(f"{year}-01-01", periods=24).strftime("%Y-%m-%d"))
                     for year in (2021, 2022, 2023)], [])
        columns = {"date": dates}
        for index, channel in enumerate(("open", "high", "low", "close")):
            columns[channel] = np.arange(72, dtype=float) + index
        for index, group in enumerate(TEXT_GROUPS):
            for dim in range(384):
                columns[f"{group}_emb{dim}"] = np.full(72, index + 1., dtype=np.float32)
        frame = pd.DataFrame(columns)
        frame.loc[0, "sector_emb0"] = np.nan
        with patch("forecasting_task.data_provider.dataset.pd.read_parquet", return_value=frame):
            datasets = [Dataset_Custom(4, 2, 3, flag, list(TEXT_GROUPS)) for flag in ("train", "val", "test")]
        train, val, test = datasets
        for dataset in datasets:
            np.testing.assert_allclose(dataset.scaler.mean_, np.arange(4) + 11.5)
            np.testing.assert_allclose(dataset.scaler.var_, np.var(np.arange(24)))
            self.assertEqual(dataset[0][2].shape, (4, 5, 384))
            np.testing.assert_array_equal(dataset[0][2][0, :, 1], np.arange(1, 6))
        self.assertEqual(train[0][2][0, 1, 0], 0.)
        np.testing.assert_allclose(val.inverse_transform(val[0][0])[:, 0], np.arange(20, 24))
        np.testing.assert_allclose(test.inverse_transform(test[0][0])[:, 0], np.arange(44, 48))
        np.testing.assert_allclose(val.inverse_transform(val[0][1])[-3:, 0], np.arange(24, 27))
        np.testing.assert_allclose(test.inverse_transform(test[0][1])[-3:, 0], np.arange(48, 51))

    def test_cli_validation(self):
        for value in ("0", "1", "-0.1", "1.1", "nan", "inf"):
            with self.subTest(weight=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run.parse_args(["--text_weight", value])
        for option, value in (("--fusion_mode", "wrong"), ("--text_num_layers", "0"),
                              ("--used_col_prefixes", "macro,macro")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args([option, value])

    def test_initial_weights_softmax_and_linear_heads(self):
        for name, weights in INITIALIZATIONS.items():
            with self.subTest(initialization=name), tempfile.TemporaryDirectory() as directory:
                args = run.parse_args([
                    "--text_num_layers", "1", "--pred_len", "3", "--text_hidden_size", "64",
                    "--logdir", directory, "--initial_group_weights", *map(str, weights),
                ])
                model = TextModel(**args.text_model_config)
                torch.testing.assert_close(model.get_group_weights(), torch.tensor(weights))
                torch.testing.assert_close(model.G, torch.tensor(weights).log())
                for head in model.group_mlps.values():
                    self.assertEqual(len(head), 1)
                    self.assertIsInstance(head[0], nn.Linear)
                    self.assertEqual((head[0].in_features, head[0].out_features), (384, 3))
        with tempfile.TemporaryDirectory() as directory:
            args = run.parse_args(["--logdir", directory])
            self.assertEqual(args.initial_group_weights, INITIALIZATIONS["uniform"])
            model = TextModel(**args.text_model_config)
            torch.testing.assert_close(model.get_group_weights(), torch.full((5,), 0.2))
        # Validate against the selected group count, rather than hard-coding five.
        with tempfile.TemporaryDirectory() as directory:
            args = run.parse_args(["--logdir", directory, "--used_col_prefixes", "macro,filing",
                                   "--initial_group_weights", "0.7", "0.3"])
            torch.testing.assert_close(TextModel(**args.text_model_config).get_group_weights(),
                                       torch.tensor([0.7, 0.3]))

    def test_initial_weights_reject_invalid_values(self):
        invalid = [[], [0.25] * 4, [1 / 6] * 6, [0, 0.25, 0.25, 0.25, 0.25],
                   [-0.2, 0.3, 0.3, 0.3, 0.3], [0.1] * 5, [1e308] * 5,
                   ["oops", 0.2, 0.2, 0.2, 0.2]]
        invalid += [[value, 0.2, 0.2, 0.2, 0.2] for value in ("nan", "inf", "-inf")]
        for weights in invalid:
            with self.subTest(weights=weights), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run.parse_args(["--initial_group_weights", *map(str, weights)])
                with self.assertRaises(ValueError):
                    TextModel(64, 3, 1, initial_group_weights=weights)
        # Allow rounding noise, but do not silently accept materially unnormalized weights.
        model = TextModel(64, 3, 1, initial_group_weights=[0.20000001] * 5)
        torch.testing.assert_close(model.get_group_weights(), torch.full((5,), 0.2))

    def test_initial_weights_do_not_change_film_heads_or_rng(self):
        baseline_film = baseline_text = baseline_rng = None
        for name, weights in [("default", None), *INITIALIZATIONS.items()]:
            with self.subTest(initialization=name):
                run.set_seed(7)
                film = self.make_film()
                text = TextModel(64, 3, 1, initial_group_weights=weights)
                rng = torch.get_rng_state().clone()
                film_state = film.state_dict()
                text_state = {key: value for key, value in text.state_dict().items() if key != "G"}
                if baseline_film is None:
                    baseline_film, baseline_text, baseline_rng = film_state, text_state, rng
                else:
                    for key, value in film_state.items():
                        torch.testing.assert_close(value, baseline_film[key], rtol=0, atol=0)
                    for key, value in text_state.items():
                        torch.testing.assert_close(value, baseline_text[key], rtol=0, atol=0)
                    torch.testing.assert_close(rng, baseline_rng, rtol=0, atol=0)

    def test_runner_logging_best_checkpoint_early_stopping_and_eval_restore(self):
        # Two synthetic train batches, with controlled validation ordering so the
        # best checkpoint differs from the final state. No real dataset or GPU.
        configurations = [(mode, name, weights, layers) for mode in ("blend", "residual")
                          for name, weights in INITIALIZATIONS.items() for layers in (1, 2)]
        for mode, name, weights, layers in configurations:
            with self.subTest(mode=mode, initialization=name, layers=layers), tempfile.TemporaryDirectory() as directory:
                batch = (torch.randn(2, 12, 4), torch.randn(2, 5, 4),
                         torch.rand(2, 12, 5, 384), torch.zeros(2, 12, 3),
                         torch.zeros(2, 5, 3), torch.arange(2))
                dataset = SimpleNamespace(
                    scaler=SimpleNamespace(mean_=np.zeros(4), scale_=np.ones(4), var_=np.ones(4)),
                    inverse_transform=lambda array: array,
                )
                loaders = {flag: [batch] for flag in ("train", "val", "test")}
                trained_states = []
                initial_states = []
                original_run_epoch = run._run_epoch

                def epoch_wrapper(dataloader, dataset, film, text, args, device, optimizer=None, collect_io=False):
                    if optimizer is not None:
                        if not initial_states:
                            initial_states.append(text.fusion_state())
                        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                        self.assertEqual(ids.count(id(text.alpha_logit)), 1)
                        self.assertEqual(len(ids), len(set(ids)))
                    result = original_run_epoch(dataloader, dataset, film, text, args, device, optimizer, collect_io)
                    if optimizer is not None:
                        trained_states.append(text.fusion_state())
                    if dataloader is loaders["val"]:
                        return float(len(trained_states)), result[1]
                    return result

                def loader_factory(**kwargs):
                    return dataset, loaders[kwargs["flag"]]

                argv = ["--model_type", "FiLM", "--seq_len", "12", "--label_len", "2",
                        "--pred_len", "3", "--text_num_layers", str(layers), "--text_hidden_size", "8",
                        "--num_epoch", "4", "--patience", "1", "--fusion_mode", mode,
                        "--logdir", directory, "--initial_group_weights", *map(str, weights)]
                with patch.object(run, "get_dataset_dataloader", side_effect=loader_factory), \
                     patch.object(run, "_run_epoch", side_effect=epoch_wrapper), \
                     patch("torch.cuda.is_available", return_value=False), \
                     patch("forecasting_task.models.FiLM.device", torch.device("cpu")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                path = Path(directory)
                summary = json.loads((path / "summary.json").read_text())
                history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
                checkpoint = torch.load(path / "best_checkpoint.pt", map_location="cpu", weights_only=True)
                args = yaml.safe_load((path / "args.yaml").read_text())
                self.assertEqual(summary["best_epoch"], 1)
                self.assertEqual(summary["epochs_completed"], 2)
                self.assertEqual(checkpoint["epoch"], 1)
                self.assertEqual(checkpoint["args"], args)
                self.assertEqual(args["fusion_mode"], mode)
                self.assertEqual(args["used_col_prefixes"], list(TEXT_GROUPS))
                torch.testing.assert_close(torch.tensor(args["initial_group_logits"]), torch.tensor(weights).log())
                actual_initial_weights = list(initial_states[0]["group_weights"].values())
                self.assertEqual(args["initial_group_weights"], actual_initial_weights)
                self.assertEqual(summary["initial_group_weights"], actual_initial_weights)
                torch.testing.assert_close(torch.tensor(actual_initial_weights), torch.tensor(weights))
                self.assertEqual(args["text_model_config"]["initial_group_weights"], weights)
                expected_layers = ["Linear"] if layers == 1 else ["Linear", "ReLU", "Linear"]
                self.assertEqual([layer["type"] for layer in args["text_mlp_structure"]], expected_layers)
                self.assertEqual(args["text_model_config"]["initial_text_weight"], 0.1)
                self.assertEqual(summary["best_text_alpha"], history[0]["text_alpha"])
                self.assertEqual(summary["best_group_weights"], history[0]["group_weights"])
                self.assertEqual(summary["final_text_alpha"], history[-1]["text_alpha"])
                self.assertEqual(summary["final_group_weights"], history[-1]["group_weights"])
                self.assertNotEqual(summary["best_text_alpha"], summary["final_text_alpha"])
                self.assertNotEqual(summary["best_group_weights"], summary["final_group_weights"])
                self.assertNotEqual(actual_initial_weights, list(summary["best_group_weights"].values()))
                self.assertEqual(checkpoint["metrics"], history[0])
                self.assertEqual(len(checkpoint["optimizer_state_dict"]["state"]),
                                 len(checkpoint["optimizer_state_dict"]["param_groups"][0]["params"]))
                for name in ("train_seconds", "val_seconds", "test_seconds", "total_seconds"):
                    self.assertGreaterEqual(summary[name], 0)

                restored_price = self.make_film()
                restored_text = TextModel(**checkpoint["args"]["text_model_config"])
                restored_price.load_state_dict(checkpoint["model_state_dict"])
                restored_text.load_state_dict(checkpoint["text_model_state_dict"])
                restored_price.eval()
                restored_text.eval()
                self.assertEqual(restored_text.fusion_state(), trained_states[0])
                with torch.no_grad():
                    result = original_run_epoch(loaders["test"], dataset, restored_price, restored_text,
                                                SimpleNamespace(**args), torch.device("cpu"), collect_io=True)
                    # Earlier days' embeddings must never influence predictions.
                    modified_batch = list(batch)
                    modified_batch[2] = batch[2].clone()
                    modified_batch[2][:, :-1] += 1000
                    modified_result = original_run_epoch([modified_batch], dataset, restored_price, restored_text,
                                                         SimpleNamespace(**args), torch.device("cpu"), collect_io=True)
                saved = torch.load(path / "test_io.pt", map_location="cpu", weights_only=False)
                torch.testing.assert_close(result[3], saved["outputs"], rtol=0, atol=0)
                torch.testing.assert_close(modified_result[3], result[3], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
