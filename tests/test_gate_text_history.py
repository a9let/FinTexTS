"""Gate-only text history: unchanged prediction heads and causal logits."""

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
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"forecasting_task"))
from forecasting_task import run
from forecasting_task.models.GroupText import ContextNoveltyTextModel, similar_text_history, build_text_model
from forecasting_task.gate_diagnostics import write_gate_validation
import test_context_novelty as fixtures

MODES = ("none", "history", "difference", "uniform_history")


def model(mode):
    return ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", gate_text_history_mode=mode)


class GateHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_none_legacy_nonzero_price_gate_forward(self):
        torch.manual_seed(17)
        text = model("none")
        self.assertIsNone(text.gate_text_vectors)
        self.assertNotIn("gate_text_vectors", text.state_dict())
        with torch.no_grad():
            text.gate.weight.normal_()
            text.gate.bias.normal_()
        for batch in (1, 3):
            x, e, price = torch.randn(batch, 8, 4), torch.randn(batch, 8, 5, 384), torch.randn(batch, 3, 4)
            e[:, -1, 1] = 0
            output, detail = text(price, x[:, -1], e, price_window=x, novelty=torch.ones(batch, 5),
                                  pair_available=torch.ones(batch, 5), return_details=True)
            features = torch.cat((x[:, -1]-x[:, -2], x[:, -1]-x[:, -6], torch.zeros(batch, 10)), -1)
            weights = text.gate(features).softmax(-1)
            heads = torch.stack([head(e[:, -1, g]) for g, head in enumerate(text.group_mlps.values())], 1)
            present = e[:, -1].abs().sum(-1) >= 1e-6
            delta = (weights[..., None]*heads*present[..., None]).sum(1)
            expected = (1-text.fixed_alpha)*price+text.fixed_alpha*(x[:, -1:]+delta[..., None])
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(detail["weights"], weights, rtol=0, atol=0)

    def test_initial_parameters_predictions_rng_batches_match_and_no_extra_prediction_heads(self):
        for seed in (7, 17, 27):
            reference = None
            for mode in MODES:
                run.set_seed(seed)
                film = fixtures.ContextNoveltyTests.make_film().eval()
                text = model(mode).eval()
                states = film.state_dict(), text.group_mlps.state_dict(), text.gate.state_dict(), torch.get_rng_state().clone()
                self.assertIsNone(text.history_heads)
                self.assertIsNone(text.attention_pool)
                self.assertEqual(text.text_history_mode, "last")
                self.assertEqual(text.text_temporal_pool, "last")
                x, e = torch.randn(3, 6, 4), torch.randn(3, 6, 5, 384)
                e[0, -1] = 0
                with torch.no_grad():
                    prediction = text(film(x, None, None, None), x[:, -1], e, price_window=x,
                                      novelty=torch.ones(3, 5), pair_available=torch.ones(3, 5))
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True)
                run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([b[0] for b in loader]) for _ in range(3)]
                if reference is None:
                    reference = states, prediction, orders
                else:
                    self.assertEqual(text.gate_text_vectors.numel(), 1920)
                    self.assertTrue((text.gate_text_vectors == 0).all())
                    for actual, expected in zip(states[:3], reference[0][:3]):
                        for key, value in actual.items():
                            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                    torch.testing.assert_close(states[3], reference[0][3], rtol=0, atol=0)
                    torch.testing.assert_close(prediction, reference[1], rtol=0, atol=0)
                    for actual, expected in zip(orders, reference[2]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_explicit_logits_masks_no_future_and_no_history(self):
        torch.manual_seed(27)
        for mode in MODES[1:]:
            text = model(mode)
            with torch.no_grad():
                text.gate_text_vectors.normal_(std=.05)
                text.gate.weight.normal_(std=.05)
            e, x, price = torch.randn(3, 9, 5, 384), torch.randn(3, 9, 4), torch.randn(3, 3, 4)
            e[:, -1, 0] = 0
            e[:, -1, 1] = .5e-6/384  # Below the current-text threshold.
            e[:, -5:-1, 2] = 0
            kwargs = dict(price_window=x, novelty=torch.ones(3, 5), pair_available=torch.ones(3, 5), return_details=True)
            a, detail = text(price, x[:, -1], e, **kwargs)
            mean, rho, _ = similar_text_history(e, uniform=mode == "uniform_history")
            z = e[:, -1]-rho[..., None]*mean if mode == "difference" else mean
            present = e[:, -1].abs().sum(-1) >= 1e-6
            z = torch.where(present[..., None], z, torch.zeros_like(z))
            torch.testing.assert_close(detail["gate_text_input"], z, rtol=0, atol=0)
            torch.testing.assert_close(detail["gate_text_logits"], (z*text.gate_text_vectors).sum(-1), rtol=0, atol=0)
            torch.testing.assert_close(detail["weights"], (detail["price_logits"]+detail["gate_text_logits"]).softmax(-1))
            self.assertTrue((detail["gate_text_logits"][:, :2] == 0).all())
            self.assertTrue((detail["contributions"][:, :2] == 0).all())
            torch.testing.assert_close(detail["pooled_embeddings"], e[:, -1], rtol=0, atol=0)
            if mode != "difference":
                self.assertTrue((detail["gate_text_logits"][:, 2] == 0).all())
            else:
                torch.testing.assert_close(z[:, 2], e[:, -1, 2], rtol=0, atol=0)
            changed = e.clone()
            changed[:, :-5] += 1e6
            b, other = text(price, x[:, -1], changed, **kwargs)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(detail["gate_text_logits"], other["gate_text_logits"], rtol=0, atol=0)
            # A changed group's input changes only that group's text logit,
            # although softmax legitimately couples the final group weights.
            changed[:, -5:, 4] *= -2
            _, other = text(price, x[:, -1], changed, **kwargs)
            torch.testing.assert_close(detail["gate_text_logits"][:, :4], other["gate_text_logits"][:, :4], rtol=0, atol=0)
        frame = fixtures.make_frame()
        old = fixtures.make_dataset(frame, "val")
        future = frame.copy()
        future.loc[18:, "macro_emb0"] = 1e5
        new = fixtures.make_dataset(future, "val")
        for a, b in zip(similar_text_history(torch.from_numpy(old[0][2])[None])[:2],
                        similar_text_history(torch.from_numpy(new[0][2])[None])[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_joint_gradients_and_unique_optimizer_parameters(self):
        for mode in MODES[1:]:
            torch.manual_seed(7)
            text, film = model(mode), fixtures.ContextNoveltyTests.make_film()
            optimizer = torch.optim.Adam([*film.parameters(), *text.parameters()], lr=.001)
            ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(ids.count(id(text.gate_text_vectors)), 1)
            e, x = torch.randn(4, 6, 5, 384), torch.randn(4, 6, 4)
            e[:, -1, 1] = 0
            output = text(film(x, None, None, None), x[:, -1], e, price_window=x,
                          novelty=torch.ones(4, 5), pair_available=torch.ones(4, 5))
            output.square().mean().backward()
            gradient = text.gate_text_vectors.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient[1] == 0).all())
            self.assertTrue((gradient[[0, 2, 3, 4]].abs().sum(-1) > 0).all())
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in film.parameters()))

    def test_runner_checkpoint_and_logit_diagnostics_restore(self):
        for mode in MODES[1:]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                frame = fixtures.make_frame()
                datasets = {flag: fixtures.make_dataset(frame, flag) for flag in ("train", "val", "test")}
                loaders = {flag: DataLoader(data, batch_size=4, shuffle=flag == "train", drop_last=flag == "train") for flag, data in datasets.items()}
                def factory(**kwargs):
                    return datasets[kwargs["flag"]], loaders[kwargs["flag"]]
                argv = ["--model_type", "FiLM", "--group_gate", "context_price_only", "--gate_text_history_mode", mode,
                        "--code_commit", "a"*40, "--seq_len", "6", "--label_len", "2", "--batch_size", "4",
                        "--e_layers", "1", "--num_epoch", "1", "--logdir", directory]
                with patch("forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader", side_effect=factory), \
                     patch("torch.cuda.is_available", return_value=False), patch("forecasting_task.models.FiLM.device", torch.device("cpu")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                root = Path(directory)
                args = yaml.safe_load((root/"args.yaml").read_text())
                summary = json.loads((root/"summary.json").read_text())
                checkpoint = torch.load(root/"best_checkpoint.pt", weights_only=True)
                for source in (args, summary, checkpoint):
                    self.assertEqual(source["gate_text_history_mode"], mode)
                    self.assertEqual(source["gate_text_history_config"]["additional_parameters"], 1920)
                self.assertEqual(args["text_history_mode"], "last")
                self.assertEqual(args["text_temporal_pool"], "last")
                self.assertFalse(any(k.startswith("history_heads.") for k in checkpoint["text_model_state_dict"]))
                price = fixtures.ContextNoveltyTests.make_film()
                text = build_text_model(args["text_model_config"], args["group_weight_mode"])
                price.load_state_dict(checkpoint["model_state_dict"])
                text.load_state_dict(checkpoint["text_model_state_dict"])
                with torch.no_grad():
                    predictions = run._run_epoch(loaders["test"], datasets["test"], price, text, SimpleNamespace(**args), torch.device("cpu"), collect_io=True)
                torch.testing.assert_close(predictions[3], torch.load(root/"test_io.pt", weights_only=False)["outputs"], rtol=0, atol=0)
                with (root/"validation_group_diagnostics.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                for start in range(0, len(rows), 5):
                    group = rows[start:start+5]
                    logits = torch.tensor([float(r["total_logit"]) for r in group]).softmax(-1)
                    for index, row in enumerate(group):
                        self.assertEqual(row["gate_text_history_mode"], mode)
                        self.assertAlmostEqual(float(row["price_logit"])+float(row["gate_text_logit"]), float(row["total_logit"]), places=6)
                        self.assertAlmostEqual(float(row["weight"]), logits[index].item(), places=6)
                with tempfile.TemporaryDirectory() as restored:
                    diag_args = SimpleNamespace(**{**args, "logdir": restored, "diagnostic_best_val_mse": summary["best_val_mse"]})
                    result = write_gate_validation(loaders["val"], datasets["val"], price, text, diag_args, torch.device("cpu"), run._prepare_batch)
                    self.assertEqual((Path(restored)/result["table"]).read_bytes(), (root/"validation_group_diagnostics.csv").read_bytes())

    def test_cli_rejects_nonfilm_pooling_and_prediction_history(self):
        for extra in (["--model_type", "DLinear"], ["--text_temporal_pool", "mean5"],
                      ["--text_history_mode", "history"], ["--group_gate", "context_novelty"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", "--group_gate", "context_price_only", "--gate_text_history_mode", "history", *extra])

    def test_control_mismatch_stops_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixtures.make_frame()
            datasets = {flag: fixtures.make_dataset(frame, flag) for flag in ("train", "val", "test")}
            def factory(**kwargs):
                data = datasets[kwargs["flag"]]
                return data, DataLoader(data, batch_size=4)
            with patch("forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader", side_effect=factory), \
                 patch("gate_history_control.load_gate_control", side_effect=ValueError("control mismatch")) as verify, \
                 patch.object(run, "FiLM", side_effect=AssertionError("Do not build model before verifying control")), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "control mismatch"):
                run.main(["--model_type", "FiLM", "--group_gate", "context_price_only", "--gate_text_history_mode", "history",
                          "--gate_control_manifest", "bad.json", "--code_commit", "a"*40, "--seq_len", "6", "--label_len", "2",
                          "--batch_size", "4", "--logdir", directory])
            self.assertEqual(verify.call_count, 1)
