"""Price-only gate ignores pair availability while presence still masks heads."""

import unittest

import torch
from torch import nn

from forecasting_task.models.GroupText import ContextNoveltyTextModel


class PriceOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pair_and_novelty_invariance_for_nonzero_gate_and_real_diagnostics(self):
        for batch in (1, 4):
            torch.manual_seed(71)
            model = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only")
            with torch.no_grad():
                model.gate.weight.normal_()
                model.gate.bias.normal_()
            x, price, embeddings = torch.randn(batch, 8, 4), torch.randn(batch, 3, 4), torch.randn(batch, 5, 384)
            novelty, pairs = torch.rand(batch, 5), torch.ones(batch, 5)
            embeddings[:, 2] = 0
            pairs[:, 2] = 0
            saved = [value.clone() for value in (x, embeddings, novelty, pairs)]
            output, detail = model(price, x[:, -1], embeddings, price_window=x, novelty=novelty,
                                   pair_available=pairs, return_details=True)
            other = model(price, x[:, -1], embeddings, price_window=x, novelty=novelty + 100,
                          pair_available=1 - pairs)
            torch.testing.assert_close(output, other, rtol=0, atol=0)
            torch.testing.assert_close(detail["features"][:, 8:], torch.zeros(batch, 10), rtol=0, atol=0)
            torch.testing.assert_close(detail["novelty"], novelty, rtol=0, atol=0)
            torch.testing.assert_close(detail["pair_available"], pairs, rtol=0, atol=0)
            for key in ("novelty_used", "pair_available_used"):
                torch.testing.assert_close(detail[key], torch.zeros(batch, 5), rtol=0, atol=0)
            for actual, expected in zip((x, embeddings, novelty, pairs), saved):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue((detail["contributions"][:, 2] == 0).all())
            for head in model.group_mlps.values():
                nn.init.zeros_(head[0].weight)
                nn.init.constant_(head[0].bias, 10)
            present, details = model(price, x[:, -1], embeddings, price_window=x, novelty=novelty,
                                     pair_available=pairs, return_details=True)
            absent = model(price, x[:, -1], torch.zeros_like(embeddings), price_window=x, novelty=novelty, pair_available=pairs)
            torch.testing.assert_close(absent, 0.9 * price + 0.1 * x[:, -1:])
            correction = details["weights"][:, [0, 1, 3, 4]].sum(1)[:, None, None]
            torch.testing.assert_close(present - absent, correction.expand_as(price))

    def test_zero_gradients_for_ten_unused_gate_columns(self):
        torch.manual_seed(17)
        model = ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only")
        self.assertEqual(sum(p.numel() for p in model.gate.parameters()), 95)
        self.assertTrue(all((p == 0).all() for p in model.gate.parameters()))
        x = torch.randn(4, 7, 4)
        output = model(torch.randn(4, 3, 4), x[:, -1], torch.randn(4, 5, 384), price_window=x,
                       novelty=torch.rand(4, 5), pair_available=torch.ones(4, 5))
        output.square().mean().backward()
        torch.testing.assert_close(model.gate.weight.grad[:, 8:], torch.zeros(5, 10), rtol=0, atol=0)
        self.assertGreater(model.gate.weight.grad[:, :8].abs().sum().item(), 0)
        self.assertFalse(model.fixed_alpha.requires_grad)
