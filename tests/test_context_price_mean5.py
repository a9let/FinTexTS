"""Causal row pooling, explicit pre-pooling presence and unchanged random streams."""

import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_task"))
from forecasting_task import run
from forecasting_task.models.GroupText import ContextNoveltyTextModel, pool_text_window
import test_context_novelty as fixtures


class Mean5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_presence_counts_cancellation_threshold_and_groups_do_not_mix(self):
        for batch in (1, 3):
            text = torch.zeros(batch, 9, 5, 384)
            text[:, :4] = 1000  # Must be outside the five-row aggregation.
            text[:, -5, 0, 0], text[:, -1, 0, 0] = 2, -2
            text[:, -4, 1, 7], text[:, -2, 1, 7] = 4, 8
            text[:, -5:, 2, 0] = 0.5e-6  # Absent daily vectors must not enter the mean.
            text[:, -1, 3, 0] = 1e-6
            text[:, -5:, 4, 2] = 9
            pooled, present, counts = pool_text_window(text, "mean5")
            torch.testing.assert_close(counts, torch.tensor([[2, 2, 0, 1, 5]]).expand(batch, -1))
            torch.testing.assert_close(present, torch.tensor([[True, True, False, True, True]]).expand(batch, -1))
            expected = torch.zeros(batch, 5, 384)
            expected[:, 1, 7], expected[:, 3, 0], expected[:, 4, 2] = 6, 1e-6, 9
            torch.testing.assert_close(pooled, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(pooled).all())
            self.assertTrue((pooled[:, 0] == 0).all())
            self.assertTrue(present[:, 0].all())

    def test_cancelled_group_bias_survives_but_missing_group_bias_is_masked(self):
        model = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool="mean5")
        for head in model.group_mlps.values():
            nn.init.zeros_(head[0].weight)
            nn.init.constant_(head[0].bias, 10)
        text = torch.zeros(2, 8, 5, 384)
        text[:, -5, 0, 0], text[:, -2, 0, 0] = 1, -1
        x, price = torch.randn(2, 8, 4), torch.randn(2, 3, 4)
        kwargs = dict(price_window=x, novelty=torch.ones(2, 5), pair_available=torch.ones(2, 5))
        actual, details = model(price, x[:, -1], text, return_details=True, **kwargs)
        torch.testing.assert_close(actual, 0.9 * price + 0.1 * x[:, -1:] + 0.2)
        torch.testing.assert_close(details["weights"], torch.full((2, 5), 0.2), rtol=0, atol=0)
        self.assertTrue((details["contributions"][:, 1:] == 0).all())
        changed = text.clone()
        changed[:, -1, 1, 4] = 2
        _, other = model(price, x[:, -1], changed, return_details=True, **kwargs)
        torch.testing.assert_close(other["features"], details["features"], rtol=0, atol=0)
        torch.testing.assert_close(other["weights"], details["weights"], rtol=0, atol=0)

    def test_last_window_forward_is_exactly_legacy_last_vector_forward(self):
        torch.manual_seed(31)
        default = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only")
        explicit = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool="last")
        explicit.load_state_dict(default.state_dict())
        text, x, price = torch.randn(3, 9, 5, 384), torch.randn(3, 9, 4), torch.randn(3, 3, 4)
        text[:, -1, 2] = 0
        kwargs = dict(price_window=x, novelty=torch.rand(3, 5), pair_available=torch.ones(3, 5))
        legacy = default(price, x[:, -1], text[:, -1], **kwargs)
        actual = explicit(price, x[:, -1], text, **kwargs)
        torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
        pooled, present, count = pool_text_window(text, "last")
        torch.testing.assert_close(pooled, text[:, -1], rtol=0, atol=0)
        torch.testing.assert_close(count, present.long())

    def test_alignment_uses_current_input_rows_only_and_real_row_dates(self):
        frame = fixtures.make_frame()
        dataset = fixtures.make_dataset(frame, "val")
        for index in (0, 1, len(dataset) - 1):
            text = torch.from_numpy(dataset[index][2]).unsqueeze(0)
            pooled, present, count = pool_text_window(text, "mean5")
            last = index + dataset.seq_len
            raw = dataset.base.data_text_x[last - 5:last]
            mask = np.abs(raw.astype(np.float32)).sum(-1) >= 1e-6
            expected = np.where(mask[:, :, None], raw, 0).sum(0) / np.maximum(mask.sum(0), 1)[:, None]
            np.testing.assert_allclose(pooled[0].numpy(), expected, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(count[0].numpy(), mask.sum(0))
            self.assertEqual(dataset.text_window_bounds(index, 5),
                             (dataset.row_dates[last - 5], dataset.row_dates[last - 1]))
        # Corrupt text after the first input window: past pooling must be unchanged.
        changed = frame.copy()
        changed.loc[18:, "macro_emb0"] = 1e5
        other = fixtures.make_dataset(changed, "val")
        for actual, expected in zip(pool_text_window(torch.from_numpy(other[0][2])[None], "mean5"),
                                    pool_text_window(torch.from_numpy(dataset[0][2])[None], "mean5")):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_seed_preserves_film_heads_rng_and_train_batch_order_between_pools(self):
        for seed in (7, 17, 27):
            reference = None
            for pool in ("last", "mean5", "decay5"):
                run.set_seed(seed)
                film = fixtures.ContextNoveltyTests.make_film()
                text = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool=pool)
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True)
                run._configure_gate_sampling(loader, seed)
                order = [torch.cat([batch[0] for batch in loader]) for _ in range(3)]
                states = film.state_dict(), text.state_dict(), torch.get_rng_state(), order
                if reference is None:
                    reference = states
                else:
                    for actual, expected in zip(states[:2], reference[:2]):
                        for name, tensor in actual.items():
                            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
                    torch.testing.assert_close(states[2], reference[2], rtol=0, atol=0)
                    for a, b in zip(order, reference[3]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mean5_rejects_short_windows_other_gates_and_forward_fill(self):
        with self.assertRaises(ValueError):
            pool_text_window(torch.zeros(1, 4, 5, 384), "mean5")
        for extra in (["--group_gate", "context_no_novelty"], ["--data_path", "fnspid/AAPL.parquet"], ["--seq_len", "4"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", "--group_gate", "context_price_only", "--text_temporal_pool", "mean5",
                                "--code_commit", "a" * 40, *extra])
