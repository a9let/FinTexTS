"""CPU-only fixed-uniform checks, including migration of learned group weights."""

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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_task"))
from forecasting_task import run
from forecasting_task.models.GroupText import FixedUniformTextModel, TextModel, TEXT_GROUPS, build_text_model


class NoGroupWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(43)

    @staticmethod
    def make_film():
        args = SimpleNamespace(task_name="short_term_forecast", seq_len=12, label_len=2,
                               pred_len=3, output_attention=False, e_layers=1, enc_in=4)
        with patch("forecasting_task.models.FiLM.device", torch.device("cpu")):
            return run.FiLM(args).cpu()

    def test_shapes_formulas_missing_groups_and_fixed_denominator(self):
        for batch in (1, 4):
            for mode in ("blend", "residual"):
                with self.subTest(batch=batch, mode=mode):
                    model = FixedUniformTextModel(64, 3, 1, fusion_mode=mode)
                    self.assertFalse(hasattr(model, "G"))
                    self.assertNotIn("G", model.state_dict())
                    for index, head in enumerate(model.group_mlps.values()):
                        self.assertEqual(len(head), 1)
                        self.assertIsInstance(head[0], nn.Linear)
                        self.assertEqual((head[0].in_features, head[0].out_features), (384, 3))
                        nn.init.zeros_(head[0].weight)
                        nn.init.constant_(head[0].bias, 10 * (index + 1))
                    price = torch.randn(batch, 3, 4)
                    last = torch.randn(batch, 4)
                    for presence in ("all", "partial", "none", "below_threshold", "threshold"):
                        x = torch.zeros(batch, 5, 384)
                        if presence == "all":
                            x[:, :, 0] = 1
                            correction = 30.
                        elif presence in ("partial", "threshold"):
                            x[:, 0, 0] = 1 if presence == "partial" else 1e-6
                            correction = 2.  # Divide by five even with only one present group.
                        elif presence == "below_threshold":
                            x[:, :, 0] = 0.5e-6
                            correction = 0.
                        else:
                            correction = 0.
                        alpha = model.get_alpha()
                        delta = torch.full_like(price, correction)
                        if mode == "residual":
                            expected = price + alpha * delta
                        else:
                            text = last[:, None] + delta if correction else torch.zeros_like(price)
                            expected = (1 - alpha) * price + alpha * text
                        actual = model(price, last, x)
                        self.assertEqual(tuple(actual.shape), (batch, 3, 4))
                        torch.testing.assert_close(actual, expected)
                        if mode == "residual":
                            torch.testing.assert_close(model(price, last + 100, x), actual, rtol=0, atol=0)
                            if not correction:
                                torch.testing.assert_close(actual, price, rtol=0, atol=0)
                    partial = torch.zeros(batch, 5, 384)
                    partial[:, 0, 0] = 1
                    model(price, last, partial).sum().backward()
                    for group in TEXT_GROUPS[1:]:
                        for parameter in model.group_mlps[group].parameters():
                            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))

    def test_uniform_old_forward_and_arbitrary_checkpoint_migration(self):
        for mode in ("blend", "residual"):
            for batch in (1, 4):
                with self.subTest(mode=mode, batch=batch):
                    old = TextModel(64, 3, 1, fusion_mode=mode)
                    new = FixedUniformTextModel(64, 3, 1, fusion_mode=mode)
                    new.load_state_dict({key: value for key, value in old.state_dict().items() if key != "G"})
                    x = torch.randn(batch, 5, 384)
                    x[:, 1] = 0
                    if batch > 1:
                        x[-1] = 0
                    price = torch.randn(batch, 3, 4)
                    last = torch.randn(batch, 4)
                    torch.testing.assert_close(new(price, last, x), old(price, last, x))
                    # Serialize an arbitrary old checkpoint, including nonuniform learned G.
                    with torch.no_grad():
                        old.G.copy_(torch.randn(5) * 2)
                        old.alpha_logit.fill_(-0.73)
                    payload = io.BytesIO()
                    torch.save({"text_model_state_dict": old.state_dict()}, payload)
                    payload.seek(0)
                    state = torch.load(payload, map_location="cpu", weights_only=True)["text_model_state_dict"]
                    coefficients = 5 * torch.softmax(state["G"], dim=0)
                    migrated = {key: value.clone() for key, value in state.items() if key != "G"}
                    for index, group in enumerate(TEXT_GROUPS):
                        for name in ("weight", "bias"):
                            key = f"group_mlps.{group}.0.{name}"
                            migrated[key] *= coefficients[index]
                    new.load_state_dict(migrated)
                    old.eval()
                    new.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(new(price, last, x), old(price, last, x))

    def test_initial_alpha_gradients_and_optimizer_without_G(self):
        for initial_alpha in (0.01, 0.1, 0.9):
            for mode in ("blend", "residual"):
                with self.subTest(alpha=initial_alpha, mode=mode):
                    film = self.make_film()
                    model = FixedUniformTextModel(64, 3, 1, initial_text_weight=initial_alpha, fusion_mode=mode)
                    torch.testing.assert_close(model.get_alpha(), torch.tensor(initial_alpha))
                    parameters = list(film.parameters()) + list(model.parameters())
                    optimizer = torch.optim.Adam(parameters, lr=0.001)
                    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), {id(p) for p in parameters})
                    self.assertEqual(ids.count(id(model.alpha_logit)), 1)
                    self.assertEqual(len(list(model.parameters())), 11)
                    self.assertNotIn("G", dict(model.named_parameters()))
                    price_x = torch.randn(3, 12, 4)
                    output = model(film(price_x, None, None, None), price_x[:, -1], torch.randn(3, 5, 384))
                    (output - 7).square().mean().backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in film.parameters()))

    def test_same_seed_preserves_film_heads_and_rng_across_alpha_and_modes(self):
        for seed in (7, 17, 27):
            baseline = None
            for initial_alpha in (0.01, 0.1, 0.9):
                for mode in ("blend", "residual"):
                    run.set_seed(seed)
                    film = self.make_film()
                    text = FixedUniformTextModel(64, 3, 1, initial_text_weight=initial_alpha, fusion_mode=mode)
                    states = (film.state_dict(), text.group_mlps.state_dict(), torch.get_rng_state())
                    if baseline is None:
                        baseline = states
                    else:
                        for actual, expected in zip(states[:2], baseline[:2]):
                            for key, tensor in actual.items():
                                torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
                        torch.testing.assert_close(states[2], baseline[2], rtol=0, atol=0)

    def test_configuration_rejects_nonuniform_weights_hidden_layers_and_reordered_groups(self):
        cases = (["--initial_group_weights", "0.6", "0.1", "0.1", "0.1", "0.1"],
                 ["--text_num_layers", "2"], ["--used_col_prefixes", "macro,sector"],
                 ["--used_col_prefixes", "sector,macro,targetCompany,relatedCompany,filing"])
        for flags in cases:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--group_weight_mode", "fixed_uniform", *flags])
        with self.assertRaises(ValueError):
            FixedUniformTextModel(64, 3, 1, initial_group_weights=[0.6, 0.1, 0.1, 0.1, 0.1])
        for weights in (None, [0.2] * 5):
            FixedUniformTextModel(64, 3, 1, initial_group_weights=weights)

    def test_runner_checkpoint_alpha_logging_restore_and_last_day_only(self):
        for mode in ("blend", "residual"):
            for initial_alpha in (0.01, 0.1, 0.9):
                with self.subTest(mode=mode, alpha=initial_alpha), tempfile.TemporaryDirectory() as directory:
                    batch = (torch.randn(2, 12, 4), torch.randn(2, 5, 4), torch.randn(2, 12, 5, 384),
                             torch.zeros(2, 12, 3), torch.zeros(2, 5, 3), torch.arange(2))
                    dataset = SimpleNamespace(
                        scaler=SimpleNamespace(mean_=np.zeros(4), scale_=np.ones(4), var_=np.ones(4)),
                        inverse_transform=lambda array: array,
                    )
                    loaders = {flag: [batch] for flag in ("train", "val", "test")}
                    original = run._run_epoch
                    states = []
                    initial = []

                    def epoch(loader, dataset, film, text, args, device, optimizer=None, collect_io=False):
                        if optimizer is not None:
                            self.assertFalse(hasattr(text, "G"))
                            ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                            self.assertEqual(set(ids), {id(p) for p in list(film.parameters()) + list(text.parameters())})
                            self.assertEqual(ids.count(id(text.alpha_logit)), 1)
                            if not initial:
                                initial.append(text.get_alpha().item())
                        result = original(loader, dataset, film, text, args, device, optimizer, collect_io)
                        if optimizer is not None:
                            states.append(text.fusion_state())
                        if loader is loaders["val"]:
                            return float(len(states)), result[1]
                        return result

                    argv = ["--group_weight_mode", "fixed_uniform", "--model_type", "FiLM", "--seq_len", "12",
                            "--label_len", "2", "--pred_len", "3", "--text_num_layers", "1", "--num_epoch", "4",
                            "--patience", "1", "--text_weight", str(initial_alpha), "--fusion_mode", mode,
                            "--logdir", directory]
                    with patch.object(run, "get_dataset_dataloader", side_effect=lambda **kw: (dataset, loaders[kw["flag"]])), \
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
                    self.assertEqual(checkpoint["metrics"], history[0])
                    self.assertEqual(summary["epochs_completed"], 2)
                    self.assertEqual(summary["initial_text_alpha"], initial[0])
                    self.assertEqual(args["initial_text_alpha"], initial[0])
                    self.assertEqual(summary["best_text_alpha"], history[0]["text_alpha"])
                    self.assertEqual(summary["final_text_alpha"], history[1]["text_alpha"])
                    self.assertNotEqual(initial[0], summary["best_text_alpha"])
                    self.assertNotEqual(summary["best_text_alpha"], summary["final_text_alpha"])
                    self.assertNotIn("G", checkpoint["text_model_state_dict"])
                    self.assertNotIn("initial_group_logits", args)
                    for record in [args, summary, checkpoint, *history]:
                        self.assertEqual(record["group_weight_mode"], "fixed_uniform")
                    for record in [args, summary, *history]:
                        self.assertEqual(record["fixed_group_coefficients"], dict.fromkeys(TEXT_GROUPS, 0.2))
                        for key in ("group_weights", "best_group_weights", "final_group_weights"):
                            self.assertNotIn(key, record)
                    price = self.make_film()
                    text = build_text_model(args["text_model_config"], args["group_weight_mode"])
                    price.load_state_dict(checkpoint["model_state_dict"])
                    text.load_state_dict(checkpoint["text_model_state_dict"])
                    self.assertEqual(text.fusion_state(), states[0])
                    with torch.no_grad():
                        result = original(loaders["test"], dataset, price, text, SimpleNamespace(**args),
                                          torch.device("cpu"), collect_io=True)
                        changed = list(batch)
                        changed[2] = batch[2].clone()
                        changed[2][:, :-1] += 1000
                        other = original([changed], dataset, price, text, SimpleNamespace(**args),
                                         torch.device("cpu"), collect_io=True)
                    saved = torch.load(root / "test_io.pt", map_location="cpu", weights_only=False)
                    torch.testing.assert_close(result[3], saved["outputs"], rtol=0, atol=0)
                    torch.testing.assert_close(result[3], other[3], rtol=0, atol=0)
