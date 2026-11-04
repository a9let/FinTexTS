"""Attention arithmetic, causal masks, isolated initialization and exact attribution."""

import contextlib
import io
import math
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_task"))
from forecasting_task import run
from forecasting_task.models.AttentionPool import AttentionPool5, attention5_config
from forecasting_task.models.GroupText import ContextNoveltyTextModel, pool_text_window
import test_context_novelty as fixtures


def make_text_model(pool="attention5"):
    return ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool=pool)


class Attention5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_explicit_attention_masking_normalization_and_two_aggregation_forms(self):
        module = AttentionPool5()
        with torch.no_grad():
            module.query.weight.zero_()
            module.key.weight.zero_()
            module.query.weight[0, 0], module.query.weight[0, 384] = .5, 2
            module.query.weight[1, 1] = -.3
            module.key.weight[0, 1], module.key.weight[0, 384] = .7, -1
            module.key.weight[1, 0] = .4
        for batch in (1, 3):
            text = torch.zeros(batch, 8, 5, 384)
            text[:, :3] = 999  # Outside the attention window.
            text[:, -5, 0, :2] = torch.tensor([1., 2.])
            text[:, -2, 0, :2] = torch.tensor([3., -1.])
            text[:, -1, 1, :2] = torch.tensor([-4., 3.])
            text[:, -5:, 2, 0] = .5e-6  # All dates below daily presence threshold.
            text[:, -3, 3, 0] = 1e-6
            text[:, -5:, 4, :2] = torch.tensor([2., -2.])
            aggregate, present, count, detail = module(text)
            expected_A = torch.zeros(batch, 5, 5, 5)
            values = text[:, -5:].transpose(1, 2)
            for b in range(batch):
                for g in range(5):
                    dates = [d for d in range(5) if values[b, g, d].float().abs().sum() >= 1e-6]
                    for i in dates:
                        qi = [.5*float(values[b, g, i, 0]) + 2*(4-i)/4, -.3*float(values[b, g, i, 1])]
                        scores = []
                        for j in dates:
                            kj = [.7*float(values[b, g, j, 1]) - (4-j)/4, .4*float(values[b, g, j, 0])]
                            scores.append(math.exp(sum(q*k for q, k in zip(qi, kj))/4))
                        for j, score in zip(dates, scores):
                            expected_A[b, g, i, j] = score/sum(scores)
            torch.testing.assert_close(detail["attention"], expected_A)
            torch.testing.assert_close(detail["attention"].sum(-1), detail["daily_present"].float())
            torch.testing.assert_close(detail["temporal_weights"].sum(-1), present.float())
            torch.testing.assert_close(count, torch.tensor([[2, 1, 0, 1, 5]]).expand(batch, -1))
            torch.testing.assert_close(detail["temporal_weights"][:, 1, -1], torch.ones(batch), rtol=0, atol=0)
            self.assertTrue((detail["attention"][:, 2] == 0).all())
            self.assertTrue((detail["temporal_weights"][:, 2] == 0).all())
            self.assertTrue((aggregate[:, 2] == 0).all())
            z = detail["attention"] @ values
            query_mean = z.sum(-2)/count.clamp_min(1)[..., None]
            beta_sum = (detail["temporal_weights"][..., None]*values).sum(-2)
            torch.testing.assert_close(aggregate, query_mean)
            torch.testing.assert_close(aggregate, beta_sum)
            self.assertTrue(torch.isfinite(aggregate).all())

    def test_initial_attention_matches_mean5_and_preserves_cancellation_presence(self):
        torch.manual_seed(7)
        text = torch.randn(3, 9, 5, 384)
        text[:, -5:, 1] = 0
        text[:, -5:, 2] = .5e-6/384
        text[:, -5:, 3] = 0
        text[:, -4, 3], text[:, -1, 3] = 2, -2
        module = AttentionPool5()
        self.assertTrue((module.query.weight == 0).all())
        self.assertGreater(module.key.weight.abs().sum().item(), 0)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 12320)
        for layer in (module.query, module.key):
            self.assertEqual((layer.in_features, layer.out_features, layer.bias), (385, 16, None))
        actual, present, count, detail = module(text)
        expected, mean_present, mean_count = pool_text_window(text, "mean5")
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(present, mean_present, rtol=0, atol=0)
        torch.testing.assert_close(count, mean_count, rtol=0, atol=0)
        expected_A = (detail["daily_present"][..., :, None] * detail["daily_present"][..., None, :]).float()
        expected_A /= count.clamp_min(1)[..., None, None]
        torch.testing.assert_close(detail["attention"], expected_A, rtol=0, atol=0)
        self.assertTrue(present[:, 3].all())
        self.assertTrue((actual[:, 3] == 0).all())

    def test_groups_and_rows_outside_input_are_independent_and_age_enters_only_qk(self):
        module = AttentionPool5()
        with torch.no_grad():
            module.query.weight.zero_()
            module.key.weight.zero_()
            module.query.weight[0, -1], module.key.weight[0, -1] = 2, 3
        text = torch.randn(2, 12, 5, 384)
        aggregate, _, _, detail = module(text)
        ages = torch.tensor([4., 3., 2., 1., 0.])/4
        scores = (6*ages[:, None]*ages[None, :])/4
        torch.testing.assert_close(detail["attention"], scores.softmax(-1).expand(2, 5, -1, -1))
        expected = (scores.softmax(-1).mean(0)[None, :, None]*text[:, -5:].transpose(1, 2)).sum(2)
        torch.testing.assert_close(aggregate, expected)
        changed = text.clone()
        changed[:, :-5] += 10000
        changed[:, -5:, 1] *= -1000
        other = module(changed)[0]
        torch.testing.assert_close(other[:, [0, 2, 3, 4]], aggregate[:, [0, 2, 3, 4]], rtol=0, atol=0)
        constant = torch.randn(2, 1, 5, 384).expand(-1, 9, -1, -1)
        torch.testing.assert_close(module(constant)[0], constant[:, -1])
        # Real dataset alignment: changing rows after this window cannot affect it.
        frame = fixtures.make_frame()
        old = fixtures.make_dataset(frame, "val")
        changed_frame = frame.copy()
        changed_frame.loc[18:, "macro_emb0"] = 1e5
        changed_frame.loc[18:, "close"] = 1e5
        new = fixtures.make_dataset(changed_frame, "val")
        self.assertEqual(old.text_window_bounds(0, 5), new.text_window_bounds(0, 5))
        for a, b in zip(module(torch.from_numpy(old[0][2])[None])[:3], module(torch.from_numpy(new[0][2])[None])[:3]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_signed_date_and_group_contributions_reconstruct_prediction_and_mask_bias(self):
        torch.manual_seed(17)
        model = make_text_model()
        with torch.no_grad():
            model.attention_pool.query.weight.normal_(std=.03)
            model.gate.weight.normal_(std=.1)
            for i, head in enumerate(model.group_mlps.values()):
                head[0].bias.fill_(-i-.5)
        for batch in (1, 3):
            text, x, price = torch.randn(batch, 9, 5, 384), torch.randn(batch, 9, 4), torch.randn(batch, 3, 4)
            text[:, -5:, 1] = 0
            text[0] = 0  # Fully empty window retains the price anchor.
            output, detail = model(price, x[:, -1], text, price_window=x, novelty=torch.ones(batch, 5),
                                   pair_available=torch.ones(batch, 5), return_details=True)
            base = .9*price + .1*x[:, -1:]
            torch.testing.assert_close(detail["daily_contributions"].sum(2), detail["contributions"], atol=2e-7, rtol=1e-5)
            torch.testing.assert_close(base + detail["daily_contributions"].sum((1, 2))[..., None], output)
            torch.testing.assert_close(output[0], base[0])
            self.assertTrue((detail["daily_contributions"][:, 1] == 0).all())
            self.assertEqual(output.shape, (batch, 3, 4))
            for g, head in enumerate(model.group_mlps.values()):
                daily = head(text[:, -5:, g])
                expected = model.fixed_alpha*detail["weights"][:, g, None, None]*detail["temporal_weights"][:, g, :, None]*daily
                torch.testing.assert_close(detail["daily_contributions"][:, g], expected, rtol=0, atol=0)

    def test_seed_film_heads_gate_initial_predictions_and_batch_order_match_mean5(self):
        for seed in (7, 17, 27):
            reference = None
            for pool in ("mean5", "attention5"):
                run.set_seed(seed)
                film = fixtures.ContextNoveltyTests.make_film().eval()
                text = make_text_model(pool).eval()
                state = (film.state_dict(), text.group_mlps.state_dict(), text.gate.state_dict(), torch.get_rng_state().clone())
                x, embeddings = torch.randn(3, 6, 4), torch.randn(3, 6, 5, 384)
                embeddings[:, -4:, 1] = 0
                with torch.no_grad():
                    prediction = text(film(x, None, None, None), x[:, -1], embeddings, price_window=x,
                                      novelty=torch.ones(3, 5), pair_available=torch.ones(3, 5))
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True)
                run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([batch[0] for batch in loader]) for _ in range(3)]
                if reference is None:
                    reference = state, prediction, orders
                else:
                    for actual, expected in zip(state[:3], reference[0][:3]):
                        for name, value in actual.items():
                            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
                    torch.testing.assert_close(state[3], reference[0][3], rtol=0, atol=0)
                    torch.testing.assert_close(prediction, reference[1], atol=2e-7, rtol=1e-6)
                    for actual, expected in zip(orders, reference[2]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_first_step_query_then_key_gradients_and_unique_joint_optimizer(self):
        torch.manual_seed(27)
        film, text = fixtures.ContextNoveltyTests.make_film(), make_text_model()
        optimizer = torch.optim.Adam([*film.parameters(), *text.parameters()], lr=.001)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        for p in text.attention_pool.parameters():
            self.assertEqual(ids.count(id(p)), 1)
        self.assertNotIn(id(text.fixed_alpha), ids)
        self.assertFalse(any(name in ("G", "alpha_logit") for name, _ in text.named_parameters()))
        x, embeddings = torch.randn(4, 6, 4), torch.randn(4, 6, 5, 384)
        target = torch.randn(4, 3, 4)
        for step in range(2):
            optimizer.zero_grad()
            output = text(film(x, None, None, None), x[:, -1], embeddings, price_window=x,
                          novelty=torch.ones(4, 5), pair_available=torch.ones(4, 5))
            (output-target).square().mean().backward()
            query_grad, key_grad = text.attention_pool.query.weight.grad, text.attention_pool.key.weight.grad
            self.assertTrue(torch.isfinite(query_grad).all() and torch.isfinite(key_grad).all())
            self.assertGreater(query_grad.abs().sum().item(), 0)
            if step == 0:
                torch.testing.assert_close(key_grad, torch.zeros_like(key_grad), rtol=0, atol=0)
            else:
                self.assertGreater(key_grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in text.group_mlps.parameters()))
            self.assertGreater(text.gate.weight.grad.abs().sum().item(), 0)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in film.parameters()))
            optimizer.step()
        self.assertFalse(text.fixed_alpha.requires_grad)

    def test_reject_short_inputs_unsupported_conditions_and_wrong_saved_architecture(self):
        with self.assertRaises(ValueError):
            AttentionPool5()(torch.zeros(1, 4, 5, 384))
        for flags in (["--group_gate", "context_no_novelty"], ["--data_path", "fnspid/AAPL.parquet"], ["--seq_len", "4"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--model_type", "FiLM", "--group_gate", "context_price_only", "--text_temporal_pool", "attention5",
                                "--code_commit", "a"*40, *flags])
        wrong = attention5_config()
        wrong["qk_size"] = 32
        with self.assertRaises(ValueError):
            ContextNoveltyTextModel(64, 3, 1, group_gate="context_price_only", text_temporal_pool="attention5", attention_config=wrong)
