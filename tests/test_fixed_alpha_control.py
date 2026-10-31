"""Small CPU synthetic checks; no server data or real training runs."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_task"))
from forecasting_task import run
from forecasting_task.models.GroupText import FixedAlphaControlModel, TEXT_GROUPS, build_text_model


class FixedAlphaControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def make_film():
        args = SimpleNamespace(task_name="short_term_forecast", seq_len=12, label_len=2,
                               pred_len=3, output_attention=False, e_layers=1, enc_in=4)
        with patch("forecasting_task.models.FiLM.device", torch.device("cpu")):
            return run.FiLM(args).cpu()

    def test_formulas_shapes_masks_and_anchor_survives_absent_text(self):
        for condition in ("full", "anchor_only"):
            for batch in (1, 4):
                model = FixedAlphaControlModel(64, 3, 1, control_condition=condition)
                self.assertEqual(len(model.group_mlps), 5 if condition == "full" else 0)
                for index, head in enumerate(model.group_mlps.values()):
                    self.assertEqual(len(head), 1)
                    self.assertIsInstance(head[0], nn.Linear)
                    self.assertEqual((head[0].in_features, head[0].out_features), (384, 3))
                    nn.init.zeros_(head[0].weight)
                    nn.init.constant_(head[0].bias, 10 * (index + 1))
                price, last = torch.randn(batch, 3, 4), torch.randn(batch, 4)
                for value, groups, correction in ((1., 5, 30.), (1., 1, 2.), (0., 5, 0.),
                                                   (0.5e-6, 5, 0.), (1e-6, 1, 2.)):
                    with self.subTest(condition=condition, batch=batch, value=value, groups=groups):
                        x = torch.zeros(batch, 5, 384)
                        x[:, :groups, 0] = value
                        expected = 0.5 * price + 0.5 * last[:, None]
                        if condition == "full":
                            expected = expected + 0.5 * correction
                        actual = model(price, last, x)
                        self.assertEqual(tuple(actual.shape), (batch, 3, 4))
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(model(price, last + 2, x), actual + 1)

    def test_fixed_alpha_optimizer_membership_and_gradients(self):
        for condition in ("full", "anchor_only"):
            film = self.make_film()
            text = FixedAlphaControlModel(64, 3, 1, control_condition=condition)
            self.assertEqual(text.get_alpha().item(), 0.5)
            self.assertFalse(text.get_alpha().requires_grad)
            self.assertEqual(set(dict(text.named_buffers())), {"fixed_alpha"})
            self.assertFalse(hasattr(text, "alpha_logit"))
            self.assertFalse(hasattr(text, "G"))
            params = list(film.parameters()) + list(text.parameters())
            optimizer = torch.optim.Adam(params, lr=0.001)
            ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertNotIn(id(text.get_alpha()), ids)
            self.assertEqual(len(list(text.parameters())), 10 if condition == "full" else 0)
            x = torch.randn(3, 12, 4)
            prediction = text(film(x, None, None, None), x[:, -1], torch.randn(3, 5, 384))
            (prediction - 7).square().mean().backward()
            for name, p in text.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(p.grad.abs().sum().item(), 0, name)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in film.parameters()))
            optimizer.step()
            self.assertEqual(text.get_alpha().item(), 0.5)
            self.assertIsNone(text.get_alpha().grad)

    def test_seed_preserves_film_rng_and_batch_order_between_conditions(self):
        for seed in (7, 17, 27):
            reference = None
            for condition in ("full", "anchor_only"):
                run.set_seed(seed)
                film = self.make_film()
                rng_before = torch.get_rng_state().clone()
                FixedAlphaControlModel(64, 3, 1, control_condition=condition)
                torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True, drop_last=True)
                run._configure_control_sampling(loader, seed)
                orders = []
                for epoch in range(3):
                    # Different unrelated RNG work must not change permutations.
                    torch.rand(17 if condition == "full" else 37)
                    orders.append(torch.cat([batch[0] for batch in loader]))
                if reference is None:
                    reference = film.state_dict(), orders
                else:
                    for name, p in film.state_dict().items():
                        torch.testing.assert_close(p, reference[0][name], rtol=0, atol=0)
                    for actual, expected in zip(orders, reference[1]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_explicit_configuration_and_invalid_combinations(self):
        cases = (["--text_weight", "0.1"], ["--fusion_mode", "residual"],
                 ["--group_weight_mode", "learnable"], ["--text_num_layers", "2"],
                 ["--used_col_prefixes", "macro,sector"], ["--model_type", "DLinear"],
                 ["--initial_group_weights", "0.6", "0.1", "0.1", "0.1", "0.1"])
        for extra in cases:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", "--fixed_alpha_control", "full",
                                "--code_commit", "a" * 40, *extra])

    def test_runner_artifacts_best_checkpoint_restore_and_identical_film_start(self):
        torch.manual_seed(31)
        tensors = (torch.randn(5, 12, 4), torch.randn(5, 5, 4), torch.randn(5, 12, 5, 384),
                   torch.zeros(5, 12, 3), torch.zeros(5, 5, 3), torch.arange(5))
        fake_dataset = SimpleNamespace(
            scaler=SimpleNamespace(mean_=np.zeros(4), scale_=np.ones(4), var_=np.ones(4)),
            inverse_transform=lambda array: array,
        )
        starts = []
        for condition in ("full", "anchor_only"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                loaders = {flag: DataLoader(TensorDataset(*tensors), batch_size=2, shuffle=flag == "train")
                           for flag in ("train", "val", "test")}
                original = run._run_epoch
                epochs = []

                def epoch(loader, dataset, film, text, args, device, optimizer=None, collect_io=False):
                    if optimizer is not None:
                        ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
                        self.assertEqual(ids, {id(p) for p in list(film.parameters()) + list(text.parameters())})
                        self.assertNotIn(id(text.fixed_alpha), ids)
                        if not epochs:
                            starts.append({key: value.clone() for key, value in film.state_dict().items()})
                        epochs.append(len(epochs) + 1)
                    result = original(loader, dataset, film, text, args, device, optimizer, collect_io)
                    if loader is loaders["val"]:
                        return float(len(epochs)), result[1]  # best epoch 1, final epoch 2
                    return result

                argv = ["--model_type", "FiLM", "--fixed_alpha_control", condition, "--code_commit", "a" * 40,
                        "--seq_len", "12", "--label_len", "2", "--pred_len", "3", "--num_epoch", "4",
                        "--patience", "1", "--seed", "7", "--logdir", directory]
                with patch.object(run, "get_dataset_dataloader", side_effect=lambda **kw: (fake_dataset, loaders[kw["flag"]])), \
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
                self.assertEqual(checkpoint["metrics"], history[0])
                self.assertEqual(checkpoint["epoch"], 1)
                self.assertEqual(summary["epochs_completed"], 2)
                for record in (args, summary, checkpoint, *history):
                    self.assertEqual(record["control_condition"], condition)
                    self.assertEqual(record["fixed_alpha"], 0.5)
                    self.assertEqual(record["text_weight_mode"], "fixed")
                    self.assertEqual(record["code_commit"], "a" * 40)
                for key in ("initial_text_alpha", "best_text_alpha", "final_text_alpha"):
                    self.assertEqual(summary[key], 0.5)
                self.assertTrue(torch.is_tensor(checkpoint["train_generator_state"]))
                self.assertNotIn("G", checkpoint["text_model_state_dict"])
                self.assertNotIn("alpha_logit", checkpoint["text_model_state_dict"])
                price = self.make_film()
                text = build_text_model(args["text_model_config"], args["group_weight_mode"])
                price.load_state_dict(checkpoint["model_state_dict"])
                text.load_state_dict(checkpoint["text_model_state_dict"])
                price.eval()
                text.eval()
                with torch.no_grad():
                    result = original(loaders["test"], fake_dataset, price, text, SimpleNamespace(**args),
                                      torch.device("cpu"), collect_io=True)
                saved = torch.load(root / "test_io.pt", map_location="cpu", weights_only=False)
                torch.testing.assert_close(result[3], saved["outputs"], rtol=0, atol=0)
        for name, value in starts[0].items():
            torch.testing.assert_close(value, starts[1][name], rtol=0, atol=0)
