"""Recency weighting has fixed row ages, per-group normalization and no parameters."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_task"))
from forecasting_task import run
from forecasting_task.models.GroupText import ContextNoveltyTextModel, pool_text_window, build_text_model
import test_context_novelty as fixtures


class Decay5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_age_order_and_half_life_two_match_explicit_formula(self):
        for batch in (1, 3):
            text = torch.zeros(batch, 5, 5, 384)
            text[:, :, :, 0] = torch.arange(1., 6.).view(1, 5, 1)
            original = text.clone()
            pooled, mask, count, detail = pool_text_window(text, "decay5", 2., True)
            ages = torch.tensor([4., 3., 2., 1., 0.], dtype=torch.float64)
            unnormalized = 2. ** (-ages / 2.)
            expected = unnormalized / unnormalized.sum()
            torch.testing.assert_close(detail["row_ages"], ages, rtol=0, atol=0)
            torch.testing.assert_close(detail["temporal_weights"], expected.view(1, 5, 1).expand(batch, -1, 5), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(pooled, (text * expected.float().view(1, 5, 1, 1)).sum(1), rtol=0, atol=0)
            torch.testing.assert_close(mask, torch.ones(batch, 5, dtype=torch.bool))
            torch.testing.assert_close(count, torch.full((batch, 5), 5))
            torch.testing.assert_close(text, original, rtol=0, atol=0)

    def test_missing_dates_keep_their_ages_empty_single_and_threshold(self):
        text = torch.zeros(2, 9, 5, 384)
        text[:, :4] = 999  # Outside the five rows; no search into earlier history.
        text[:, -5, 0, 0], text[:, -3, 0, 0] = 3, 9
        text[:, -5, 1, 4] = 7
        text[:, -1, 3, 0] = 0.5e-6
        text[:, -1, 4, 0] = 1e-6
        pooled, mask, count, detail = pool_text_window(text, "decay5", 2., True)
        expected = torch.zeros(2, 5, 5, dtype=torch.float64)
        expected[:, 0, 0], expected[:, 2, 0] = 1/3, 2/3
        expected[:, 0, 1], expected[:, 4, 4] = 1, 1
        torch.testing.assert_close(detail["temporal_weights"], expected)
        torch.testing.assert_close(count, torch.tensor([[2, 1, 0, 0, 1]]).expand(2, -1))
        torch.testing.assert_close(mask, count > 0)
        torch.testing.assert_close(pooled[:, 0, 0], torch.full((2,), 7.))
        torch.testing.assert_close(pooled[:, 1, 4], torch.full((2,), 7.))
        self.assertTrue((pooled[:, 2:4] == 0).all())
        torch.testing.assert_close(pooled[:, 4, 0], torch.full((2,), 1e-6))

    def test_identical_vectors_preserve_original_norms_and_groups_do_not_mix(self):
        torch.manual_seed(7)
        originals = torch.randn(2, 5, 384) * 7
        text = originals[:, None].expand(-1, 5, -1, -1).clone()
        text[:, 1, 1] = 0
        text[:, 3, 2] = 0
        pooled, _, _, _ = pool_text_window(text, "decay5", 2., True)
        torch.testing.assert_close(pooled, originals)

    def test_dataset_alignment_and_future_text_cannot_change_past_aggregation(self):
        frame = fixtures.make_frame()
        dataset = fixtures.make_dataset(frame, "val")
        for index in (0, 1, len(dataset)-1):
            text = torch.from_numpy(dataset[index][2])[None]
            pooled, _, counts = pool_text_window(text, "decay5")
            raw = dataset[index][2][-5:]
            present = np.abs(raw.astype(np.float32)).sum(-1) >= 1e-6
            unnormalized = present * 2. ** (-np.arange(4, -1, -1)[:, None] / 2.)
            denominator = unnormalized.sum(0, keepdims=True)
            weights = unnormalized / np.where(denominator > 0, denominator, 1)
            expected = (raw * weights[:, :, None]).sum(0)
            np.testing.assert_allclose(pooled[0].numpy(), expected, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(counts[0].numpy(), present.sum(0))
        changed = frame.copy()
        changed.loc[18:, "macro_emb0"] = 1e6
        other = fixtures.make_dataset(changed, "val")
        for actual, expected in zip(pool_text_window(torch.from_numpy(other[0][2])[None], "decay5"),
                                    pool_text_window(torch.from_numpy(dataset[0][2])[None], "decay5")):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mask_after_head_uses_presence_even_if_weighted_vectors_cancel(self):
        model = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool="decay5")
        mean = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool="mean5")
        self.assertEqual(list(model.state_dict()), list(mean.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in mean.parameters()))
        self.assertNotIn("text_half_life", dict(model.named_parameters()))
        for head in model.group_mlps.values():
            nn.init.zeros_(head[0].weight)
            nn.init.constant_(head[0].bias, -10.)
        text = torch.zeros(3, 8, 5, 384)
        text[:, -5, 0, 0], text[:, -1, 0, 0] = 4, -1
        x, price = torch.randn(3, 8, 4), torch.randn(3, 3, 4)
        kwargs = dict(price_window=x, novelty=torch.ones(3, 5), pair_available=torch.ones(3, 5))
        actual, detail = model(price, x[:, -1], text, return_details=True, **kwargs)
        self.assertTrue(detail["present"][:, 0].all())
        torch.testing.assert_close(detail["pooled_embeddings"], torch.zeros(3, 5, 384), rtol=0, atol=0)
        torch.testing.assert_close(actual, .9 * price + .1 * x[:, -1:] - .2)
        self.assertTrue((detail["contributions"][:, 1:] == 0).all())
        self.assertTrue((detail["features"][:, 8:] == 0).all())
        empty = model(price, x[:, -1], torch.zeros_like(text), **kwargs)
        torch.testing.assert_close(empty, .9 * price + .1 * x[:, -1:])

    def test_half_life_validation_and_tiny_values_with_only_old_text(self):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", "--group_gate", "context_price_only", "--text_temporal_pool", "decay5",
                                "--code_commit", "a" * 40, "--text_half_life=" + value])
            with self.assertRaises(ValueError):
                pool_text_window(torch.zeros(1, 5, 5, 384), "decay5", float(value))
        text = torch.zeros(1, 5, 5, 384)
        text[:, 0, 0, 0] = 12
        for half_life in (1e-300, 1e300):
            pooled, _, _, detail = pool_text_window(text, "decay5", half_life, True)
            self.assertTrue(torch.isfinite(pooled).all())
            self.assertTrue(torch.isfinite(detail["temporal_weights"]).all())
            self.assertEqual(detail["temporal_weights"][0, 0, 0].item(), 1.)
            self.assertEqual(pooled[0, 0, 0].item(), 12.)

    def test_checkpoint_restores_nondefault_half_life_and_identical_eval_prediction(self):
        config = dict(hidden_size=64, pred_len=3, num_layers=1, group_gate="context_price_only",
                      text_temporal_pool="decay5", text_half_life=.7)
        model = ContextNoveltyTextModel(**config).eval()
        payload = io.BytesIO()
        torch.save({"config": config, "state": model.state_dict()}, payload)
        payload.seek(0)
        checkpoint = torch.load(payload, weights_only=True)
        restored = build_text_model(checkpoint["config"], "context_price_only").eval()
        restored.load_state_dict(checkpoint["state"])
        self.assertEqual(restored.text_half_life, .7)
        x, price, text = torch.randn(2, 8, 4), torch.randn(2, 3, 4), torch.randn(2, 8, 5, 384)
        kwargs = dict(price_window=x, novelty=torch.zeros(2, 5), pair_available=torch.zeros(2, 5))
        with torch.no_grad():
            torch.testing.assert_close(model(price, x[:, -1], text, **kwargs), restored(price, x[:, -1], text, **kwargs), rtol=0, atol=0)
