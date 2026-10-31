"""Ablation changes only gate columns 8:13, preserving the head inputs."""

import unittest

import torch

from forecasting_task.models.GroupText import ContextNoveltyTextModel


class NoveltyAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_novelty_features_are_zeroed_without_mutating_inputs(self):
        for batch in (1, 4):
            torch.manual_seed(17)
            full = ContextNoveltyTextModel(64, 3, 1, group_gate="context_novelty")
            ablated = ContextNoveltyTextModel(64, 3, 1, group_gate="context_no_novelty")
            # Use a nonzero gate so a silent failure to mask novelty is visible.
            with torch.no_grad():
                full.gate.weight.copy_(torch.randn_like(full.gate.weight) * 0.3)
                full.gate.bias.copy_(torch.randn_like(full.gate.bias))
            ablated.load_state_dict(full.state_dict())
            price, x = torch.randn(batch, 3, 4), torch.randn(batch, 9, 4)
            embeddings = torch.randn(batch, 5, 384)
            embeddings[:, 2] = 0
            novelty = torch.rand(batch, 5)
            pairs = torch.ones(batch, 5)
            novelty[:, 2], pairs[:, 2] = 0, 0
            inputs = (price, x, embeddings, novelty, pairs)
            snapshots = [value.clone() for value in inputs]
            kwargs = dict(price_window=x, novelty=novelty, pair_available=pairs, return_details=True)
            reference, full_detail = full(price, x[:, -1], embeddings, **kwargs)
            prediction, detail = ablated(price, x[:, -1], embeddings, **kwargs)
            expected_features = full_detail["features"].clone()
            expected_features[:, 8:13] = 0
            torch.testing.assert_close(detail["features"], expected_features, rtol=0, atol=0)
            torch.testing.assert_close(detail["novelty"], novelty, rtol=0, atol=0)
            torch.testing.assert_close(detail["novelty_used"], torch.zeros_like(novelty), rtol=0, atol=0)
            torch.testing.assert_close(full_detail["novelty_used"], novelty, rtol=0, atol=0)
            for key in ("heads", "present", "pair_available"):
                torch.testing.assert_close(detail[key], full_detail[key], rtol=0, atol=0)
            for value, snapshot in zip(inputs, snapshots):
                torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
            zero_novelty_prediction = full(price, x[:, -1], embeddings, price_window=x,
                                          novelty=torch.zeros_like(novelty), pair_available=pairs)
            torch.testing.assert_close(prediction, zero_novelty_prediction, rtol=0, atol=0)
            self.assertFalse(torch.allclose(reference, prediction))
            delta = (detail["weights"][:, :, None] * detail["present"][:, :, None] * detail["heads"]).sum(1)
            torch.testing.assert_close(prediction, 0.9 * price + 0.1 * (x[:, -1:] + delta[:, :, None]))

    def test_novelty_column_gradients_zero_only_in_ablation(self):
        for mode in ("context_novelty", "context_no_novelty"):
            torch.manual_seed(27)
            model = ContextNoveltyTextModel(64, 3, 1, group_gate=mode)
            self.assertEqual(sum(p.numel() for p in model.gate.parameters()), 95)
            self.assertTrue(all((parameter == 0).all() for parameter in model.gate.parameters()))
            self.assertFalse(model.fixed_alpha.requires_grad)
            self.assertFalse(hasattr(model, "G"))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            x, embeddings = torch.randn(4, 8, 4), torch.randn(4, 5, 384)
            output = model(torch.randn(4, 3, 4), x[:, -1], embeddings, price_window=x,
                           novelty=torch.rand(4, 5) + 0.2, pair_available=torch.ones(4, 5))
            (output - 7).square().mean().backward()
            gradient = model.gate.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient[:, :8].abs().sum(0) > 0).all())
            self.assertTrue((gradient[:, 13:].abs().sum(0) > 0).all())
            if mode == "context_no_novelty":
                torch.testing.assert_close(gradient[:, 8:13], torch.zeros(5, 5), rtol=0, atol=0)
            else:
                self.assertTrue((gradient[:, 8:13].abs().sum(0) > 0).all())
            for head in model.group_mlps.values():
                self.assertTrue(all(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                    for parameter in head.parameters()))
            optimizer.step()
            if mode == "context_no_novelty":
                torch.testing.assert_close(model.gate.weight[:, 8:13], torch.zeros(5, 5), rtol=0, atol=0)
