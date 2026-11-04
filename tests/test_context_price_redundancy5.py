"""Inverse-density weighting uses only present rows and retains original norms."""

import contextlib
import io
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_task"))
from forecasting_task import run
from forecasting_task.models.GroupText import ContextNoveltyTextModel, pool_text_window, build_text_model


class Redundancy5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_empty_one_present_holes_threshold_and_no_history_search(self):
        for batch in (1, 3):
            text = torch.zeros(batch, 9, 5, 384)
            text[:, :4] = 1000
            text[:, -3, 1, 7] = 12
            text[:, -1, 2, 2] = 0.5e-6
            text[:, -5, 3, 3] = 1e-6
            pooled, present, count, details = pool_text_window(text, "redundancy5", return_details=True)
            torch.testing.assert_close(count, torch.tensor([[0, 1, 0, 1, 0]]).expand(batch, -1))
            torch.testing.assert_close(present, count > 0)
            expected = torch.zeros(batch, 5, 384)
            expected[:, 1, 7], expected[:, 3, 3] = 12, 1e-6
            torch.testing.assert_close(pooled, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(details["temporal_weights"]).all())
            self.assertTrue((details["temporal_weights"][:, :, [0, 2, 4]] == 0).all())
            torch.testing.assert_close(details["temporal_weights"][:, 2, 1], torch.ones(batch, dtype=torch.float64))
            torch.testing.assert_close(details["redundancy_density"][:, 2, 1], torch.ones(batch, dtype=torch.float64))

    def test_identical_vectors_give_mean_and_collinear_norms_are_preserved(self):
        torch.manual_seed(31)
        for identical in (True, False):
            base = torch.randn(2, 1, 5, 384)
            norms = torch.ones(1, 5, 1, 1) if identical else torch.tensor([1., 2., 4., 8., 16.]).view(1, 5, 1, 1)
            text = base.expand(-1, 5, -1, -1) * norms
            before = text.clone()
            actual, _, _, detail = pool_text_window(text, "redundancy5", return_details=True)
            expected = pool_text_window(text, "mean5")[0]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(detail["temporal_weights"], torch.full((2, 5, 5), 0.2, dtype=torch.float64))
            torch.testing.assert_close(text, before, rtol=0, atol=0)
            if not identical:
                torch.testing.assert_close(actual, base[:, 0] * (31 / 5))

    def test_four_duplicates_and_one_orthogonal_match_formula_without_mixing_groups(self):
        text = torch.zeros(1, 5, 5, 384)
        text[:, :4, 0, 0] = 1
        text[:, 4, 0, 1] = 1
        text[:, :, 1, 8] = 3
        pooled, _, _, detail = pool_text_window(text, "redundancy5", 0.1, True)
        similarity = math.exp(-10)
        density = torch.tensor([4 + similarity] * 4 + [1 + 4 * similarity], dtype=torch.float64)
        weights = density.reciprocal() / density.reciprocal().sum()
        torch.testing.assert_close(detail["redundancy_density"][0, :, 0], density, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(detail["temporal_weights"][0, :, 0], weights, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(pooled[0, 0, :2], torch.tensor([weights[:4].sum(), weights[4]], dtype=torch.float32))
        torch.testing.assert_close(pooled[0, 1], text[0, 0, 1])
        self.assertTrue((pooled[:, 2:] == 0).all())
        self.assertAlmostEqual(detail["effective_dates"][0, 0].item(), 1 / weights.square().sum().item())

    def test_permutation_reorders_weights_and_densities_but_preserves_aggregate(self):
        torch.manual_seed(17)
        text = torch.randn(3, 5, 5, 384)
        text[:, 1, 2] = 0
        text[:, 4, 0] = text[:, 0, 0]
        order = torch.tensor([4, 0, 2, 1, 3])
        actual, mask, count, detail = pool_text_window(text, "redundancy5", return_details=True)
        other, other_mask, other_count, other_detail = pool_text_window(text[:, order], "redundancy5", return_details=True)
        torch.testing.assert_close(actual, other)
        torch.testing.assert_close(mask, other_mask)
        torch.testing.assert_close(count, other_count)
        for key in ("temporal_weights", "redundancy_density", "daily_present"):
            torch.testing.assert_close(other_detail[key], detail[key][:, order])

    def test_present_cancelled_group_bias_and_empty_group_masks_in_model(self):
        model = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool="redundancy5")
        mean = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool="mean5")
        self.assertEqual(list(model.state_dict()), list(mean.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in mean.parameters()))
        self.assertNotIn("redundancy_tau", dict(model.named_parameters()))
        for head in model.group_mlps.values():
            nn.init.zeros_(head[0].weight)
            nn.init.constant_(head[0].bias, -10)
        text = torch.zeros(2, 8, 5, 384)
        text[:, -5, 0, 0], text[:, -1, 0, 0] = 1, -1
        x, price = torch.randn(2, 8, 4), torch.randn(2, 3, 4)
        actual, detail = model(price, x[:, -1], text, price_window=x, novelty=torch.ones(2, 5),
                               pair_available=torch.ones(2, 5), return_details=True)
        torch.testing.assert_close(detail["pooled_embeddings"], torch.zeros(2, 5, 384))
        self.assertTrue(detail["present"][:, 0].all())
        torch.testing.assert_close(actual, 0.9 * price + 0.1 * x[:, -1:] - 0.2)
        self.assertTrue((detail["contributions"][:, 0] < 0).all())
        self.assertTrue((detail["contributions"][:, 1:] == 0).all())
        self.assertTrue((detail["features"][:, 8:] == 0).all())

    def test_tau_validation_and_extreme_positive_tau_are_safe(self):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", "--group_gate", "context_price_only", "--text_temporal_pool", "redundancy5",
                                "--code_commit", "a" * 40, "--redundancy_tau=" + value])
            with self.assertRaises(ValueError):
                pool_text_window(torch.zeros(1, 5, 5, 384), "redundancy5", float(value))
        text = torch.randn(1, 5, 5, 384)
        for tau in (1e-300, 1e300):
            pooled, _, _, detail = pool_text_window(text, "redundancy5", tau, True)
            self.assertTrue(torch.isfinite(pooled).all())
            self.assertTrue(torch.isfinite(detail["temporal_weights"]).all())
            torch.testing.assert_close(detail["temporal_weights"].sum(1), torch.ones(1, 5, dtype=torch.float64))

    def test_checkpoint_restores_nondefault_tau_without_new_parameters(self):
        config = dict(hidden_size=64, pred_len=3, num_layers=1, group_gate="context_price_only",
                      text_temporal_pool="redundancy5", redundancy_tau=0.23)
        model = ContextNoveltyTextModel(**config).eval()
        payload = io.BytesIO()
        torch.save({"config": config, "state": model.state_dict()}, payload)
        payload.seek(0)
        checkpoint = torch.load(payload, weights_only=True)
        restored = build_text_model(checkpoint["config"], "context_price_only").eval()
        restored.load_state_dict(checkpoint["state"])
        self.assertEqual(restored.redundancy_tau, 0.23)
        x, price, text = torch.randn(3, 8, 4), torch.randn(3, 3, 4), torch.randn(3, 8, 5, 384)
        kwargs = dict(price_window=x, novelty=torch.zeros(3, 5), pair_available=torch.zeros(3, 5))
        with torch.no_grad():
            torch.testing.assert_close(model(price, x[:, -1], text, **kwargs),
                                       restored(price, x[:, -1], text, **kwargs), rtol=0, atol=0)
