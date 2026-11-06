"""Four-row causal history, additive zero heads and unchanged last-date path."""

import contextlib
import csv
import io
import json
import math
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
from forecasting_task.models.GroupText import ContextNoveltyTextModel, similar_text_history, build_text_model, TEXT_GROUPS
from forecasting_task.gate_diagnostics import write_gate_validation
import test_context_novelty as fixtures


def model(mode="last"):
    return ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_history_mode=mode)


class TextHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_explicit_cosines_kernel_weighted_original_vectors_and_rho(self):
        e = torch.zeros(2, 8, 5, 384)
        e[:, :-5] = 999  # Outside the four-row history.
        e[:, -1, 0, 0] = 2
        e[:, -5, 0, 0] = 3
        e[:, -4, 0, 0] = -4
        e[:, -3, 0, 1] = 5
        e[:, -2, 0, 0] = .5e-6  # Below presence threshold: excluded, not a fifth key.
        e[:, -1, 1, 1] = 7
        e[:, -3, 1, 0] = 6  # Orthogonal: rho = exp(-10), not normalized weight=1.
        mean, rho, detail = similar_text_history(e)
        k = torch.tensor([1., math.exp(-20), math.exp(-10), 0])
        torch.testing.assert_close(detail["history_kernel"][:, :, 0], k.expand(2, -1))
        expected = torch.zeros(2, 384)
        expected[:, 0] = (3-4*math.exp(-20))/float(k.sum())
        expected[:, 1] = 5*math.exp(-10)/float(k.sum())
        torch.testing.assert_close(mean[:, 0], expected)
        torch.testing.assert_close(rho[:, 0], torch.ones(2), rtol=0, atol=0)
        torch.testing.assert_close(rho[:, 1], torch.full((2,), math.exp(-10)))
        torch.testing.assert_close(mean[:, 1, 0], torch.full((2,), 6.))
        self.assertTrue((mean[:, 1, 1:] == 0).all())
        self.assertTrue((mean[:, 2:] == 0).all())
        self.assertTrue((rho[:, 2:] == 0).all())

    def test_missing_current_no_history_and_difference_without_past(self):
        text = torch.zeros(1, 7, 5, 384)
        text[:, -5:-1, 0] = 1  # Current group 0 absent, despite available history.
        text[:, -1, 1, 0] = 2  # No history for group 1.
        text[:, :2, 1] = 1000  # Must not search backwards beyond four rows.
        mean, rho, _ = similar_text_history(text)
        self.assertTrue((mean == 0).all() and (rho == 0).all())
        x, price = torch.randn(1, 7, 4), torch.randn(1, 3, 4)
        for mode in ("history", "difference"):
            current = model(mode)
            with torch.no_grad():
                for head in current.history_heads.values():
                    head.weight.fill_(1)
            _, detail = current(price, x[:, -1], text, price_window=x, novelty=torch.ones(1, 5),
                                pair_available=torch.ones(1, 5), return_details=True)
            self.assertTrue((detail["history_contributions"][:, 0] == 0).all())
            expected = 0 if mode == "history" else 2
            torch.testing.assert_close(detail["history_heads"][:, 1], torch.full((1, 3), float(expected)))

    def test_initial_predictions_film_heads_gate_rng_and_batch_order_match_last(self):
        for seed in (7, 17, 27):
            reference = None
            counts = []
            for mode in ("last", "history", "difference"):
                run.set_seed(seed)
                film = fixtures.ContextNoveltyTests.make_film().eval()
                text = model(mode).eval()
                core = film.state_dict(), text.group_mlps.state_dict(), text.gate.state_dict(), torch.get_rng_state().clone()
                x, e = torch.randn(3, 6, 4), torch.randn(3, 6, 5, 384)
                e[0, -1] = 0
                e[:, -1, 1] = 0
                with torch.no_grad():
                    price = film(x, None, None, None)
                    prediction = text(price, x[:, -1], e, price_window=x, novelty=torch.ones(3, 5), pair_available=torch.ones(3, 5))
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True)
                run._configure_gate_sampling(loader, seed)
                order = [torch.cat([batch[0] for batch in loader]) for _ in range(3)]
                if reference is None:
                    reference = core, prediction, order
                    self.assertIsNone(text.history_heads)
                    old = build_text_model(dict(hidden_size=64, pred_len=3, num_layers=1, group_gate="context_price_only"), "context_price_only")
                    old.load_state_dict(text.state_dict(), strict=True)
                else:
                    for actual, expected in zip(core[:3], reference[0][:3]):
                        for name, value in actual.items():
                            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
                    torch.testing.assert_close(core[3], reference[0][3], rtol=0, atol=0)
                    torch.testing.assert_close(prediction, reference[1], rtol=0, atol=0)
                    for actual, expected in zip(order, reference[2]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    counts.append(sum(p.numel() for p in text.parameters()))
                    self.assertEqual(sum(p.numel() for p in text.history_heads.parameters()), 5*384*3)
                    self.assertTrue(all(head.bias is None and (head.weight == 0).all() for head in text.history_heads.values()))
            self.assertEqual(counts[0], counts[1])

    def test_nonzero_additive_heads_formulas_masks_and_joint_gradients(self):
        torch.manual_seed(17)
        for mode in ("history", "difference"):
            text = model(mode)
            with torch.no_grad():
                for head in text.history_heads.values():
                    head.weight.normal_(std=.01)
            for batch in (1, 3):
                e, x, price = torch.randn(batch, 9, 5, 384), torch.randn(batch, 9, 4), torch.randn(batch, 3, 4)
                e[:, -1, 1] = 0
                output, detail = text(price, x[:, -1], e, price_window=x, novelty=torch.ones(batch, 5),
                                      pair_available=torch.ones(batch, 5), return_details=True)
                mean, rho, _ = similar_text_history(e)
                expected_input = mean if mode == "history" else e[:, -1]-rho[..., None]*mean
                torch.testing.assert_close(detail["history_input"], expected_input, rtol=0, atol=0)
                extras = torch.stack([head(expected_input[:, g]) for g, head in enumerate(text.history_heads.values())], 1)
                baseline = torch.stack([head(e[:, -1, g]) for g, head in enumerate(text.group_mlps.values())], 1)
                present = e[:, -1].abs().sum(-1) >= 1e-6
                delta = ((baseline+extras)*present[..., None]*detail["weights"][..., None]).sum(1)
                torch.testing.assert_close(output, .9*price+.1*(x[:, -1:]+delta[..., None]))
                torch.testing.assert_close(detail["contributions"], detail["base_contributions"]+detail["history_contributions"])
                self.assertTrue((detail["history_contributions"][:, 1] == 0).all())
            film = fixtures.ContextNoveltyTests.make_film()
            text = model(mode)  # Verify gradients from the required zero initialization.
            optimizer = torch.optim.Adam([*film.parameters(), *text.parameters()], lr=.001)
            ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            x, e = torch.randn(4, 6, 4), torch.randn(4, 6, 5, 384)
            output = text(film(x, None, None, None), x[:, -1], e, price_window=x,
                          novelty=torch.ones(4, 5), pair_available=torch.ones(4, 5))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in text.history_heads.parameters()))
            self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in text.group_mlps.parameters()))
            self.assertGreater(text.gate.weight.grad.abs().sum().item(), 0)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in film.parameters()))
            self.assertTrue(all(ids.count(id(p)) == 1 for p in text.history_heads.parameters()))

    def test_causal_window_and_no_group_mixing(self):
        torch.manual_seed(27)
        e = torch.randn(2, 12, 5, 384, requires_grad=True)
        mean, rho, _ = similar_text_history(e)
        (mean.sum()+rho.sum()).backward()
        self.assertTrue((e.grad[:, :-5] == 0).all())
        other = e.detach().clone()
        other[:, :-5] = 999
        other[:, -5:, 1] *= -100
        b, r, _ = similar_text_history(other)
        torch.testing.assert_close(b[:, [0, 2, 3, 4]], mean[:, [0, 2, 3, 4]], rtol=0, atol=0)
        torch.testing.assert_close(r[:, [0, 2, 3, 4]], rho[:, [0, 2, 3, 4]], rtol=0, atol=0)
        frame = fixtures.make_frame()
        original = fixtures.make_dataset(frame, "val")
        changed = frame.copy()
        changed.loc[18:, "macro_emb0"] = 1e5  # After the first validation input end.
        future = fixtures.make_dataset(changed, "val")
        for a, b in zip(similar_text_history(torch.from_numpy(original[0][2])[None])[:2],
                        similar_text_history(torch.from_numpy(future[0][2])[None])[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_runner_checkpoint_and_additional_validation_diagnostics(self):
        frame = fixtures.make_frame()
        for mode in ("history", "difference", "uniform_history"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                datasets = {flag: fixtures.make_dataset(frame, flag) for flag in ("train", "val", "test")}
                loaders = {flag: DataLoader(data, batch_size=4, shuffle=flag == "train", drop_last=flag == "train") for flag, data in datasets.items()}
                def factory(**kwargs):
                    return datasets[kwargs["flag"]], loaders[kwargs["flag"]]
                original = run._run_epoch
                epochs = []
                def epoch(loader, dataset, price, text, args, device, optimizer=None, collect_io=False):
                    result = original(loader, dataset, price, text, args, device, optimizer, collect_io)
                    if optimizer is not None:
                        epochs.append(1)
                        if len(epochs) == 2:
                            with torch.no_grad():
                                for head in text.group_mlps.values():
                                    head[0].bias.add_(1000)
                    return result
                argv = ["--model_type", "FiLM", "--group_gate", "context_price_only", "--code_commit", "a"*40,
                        "--text_history_mode", mode, "--seq_len", "6", "--label_len", "2", "--pred_len", "3",
                        "--batch_size", "4", "--e_layers", "1", "--num_epoch", "2", "--logdir", directory]
                with patch("forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader", side_effect=factory), \
                     patch.object(run, "_run_epoch", side_effect=epoch), patch("torch.cuda.is_available", return_value=False), \
                     patch("forecasting_task.models.FiLM.device", torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                root = Path(directory)
                args = yaml.safe_load((root/"args.yaml").read_text())
                checkpoint = torch.load(root/"best_checkpoint.pt", weights_only=True)
                summary = json.loads((root/"summary.json").read_text())
                self.assertEqual(checkpoint["epoch"], 1)
                for record in (args, checkpoint, summary):
                    self.assertEqual(record["text_history_mode"], mode)
                    self.assertEqual(record["text_history_config"]["tau"], None if mode == "uniform_history" else .1)
                    if mode == "uniform_history":
                        self.assertEqual(record["text_history_config"]["weighting"], "uniform_present")
                    self.assertEqual(record["text_history_config"]["additional_parameters"], 5760)
                self.assertEqual(summary["window_evidence"], args["window_evidence"])
                price, text = fixtures.ContextNoveltyTests.make_film(), build_text_model(args["text_model_config"], args["group_weight_mode"])
                price.load_state_dict(checkpoint["model_state_dict"])
                text.load_state_dict(checkpoint["text_model_state_dict"])
                self.assertTrue(any(p.abs().sum() > 0 for p in text.history_heads.parameters()))
                with torch.no_grad():
                    result = original(loaders["test"], datasets["test"], price, text, SimpleNamespace(**args), torch.device("cpu"), collect_io=True)
                saved = torch.load(root/"test_io.pt", weights_only=False)
                torch.testing.assert_close(result[3], saved["outputs"], rtol=0, atol=0)
                with (root/"validation_group_diagnostics.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    self.assertEqual(row["text_history_mode"], mode)
                    for h in range(1, 4):
                        self.assertAlmostEqual(float(row[f"base_contributions_h{h}"])+float(row[f"history_contributions_h{h}"]), float(row[f"contribution_h{h}"]), places=6)
                    index = int(row["window_index"])
                    self.assertEqual((row["history_window_start_date"], row["history_current_date"]), datasets["val"].text_window_bounds(index, 5))
                self.assertEqual(len(summary["validation_diagnostics"]["plots"]), 7)
                with tempfile.TemporaryDirectory() as restored:
                    diag_args = SimpleNamespace(**{**args, "logdir": restored, "diagnostic_best_val_mse": summary["best_val_mse"]})
                    repeated = write_gate_validation(loaders["val"], datasets["val"], price, text, diag_args, torch.device("cpu"), run._prepare_batch)
                    self.assertEqual((Path(restored)/repeated["table"]).read_bytes(), (root/"validation_group_diagnostics.csv").read_bytes())

    def test_cli_rejects_incompatible_modes_and_short_windows(self):
        for extra in (["--text_temporal_pool", "mean5"], ["--group_gate", "context_novelty"],
                      ["--data_path", "fnspid/AAPL.parquet"], ["--seq_len", "4"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", "--group_gate", "context_price_only", "--code_commit", "a"*40,
                                "--text_history_mode", "history", *extra])
        with self.assertRaises(ValueError):
            similar_text_history(torch.zeros(1, 4, 5, 384))
