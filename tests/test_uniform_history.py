"""Uniform control shares the four-row history window and additive zero heads."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"forecasting_task"))
from forecasting_task import run
from forecasting_task.models.GroupText import similar_text_history
from test_text_history import model
import test_context_novelty as fixtures


class UniformHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_equal_weights_with_gaps_original_norms_and_no_current_row(self):
        for batch in (1, 3):
            e = torch.zeros(batch, 9, 5, 384)
            e[:, :-5] = 999
            e[:, -1] = 10  # Current row must not enter the mean.
            e[:, -5, 0, 0], e[:, -2, 0, 0] = 2, 8
            e[:, -4, 1, 1], e[:, -3, 1, 1] = 3, -9
            e[:, -5:-1, 2, 0] = .5e-6  # Absent vectors excluded before averaging.
            e[:, -3, 3, 0] = 1e-6
            mean, _, detail = similar_text_history(e, uniform=True)
            expected_weights = torch.tensor([[.5, 0, 0, 0, 0], [0, .5, 0, 0, 0],
                                             [0, .5, 0, 1, 0], [.5, 0, 0, 0, 0]])
            torch.testing.assert_close(detail["history_weights"], expected_weights.expand(batch, -1, -1), rtol=0, atol=0)
            expected = torch.zeros(batch, 5, 384)
            expected[:, 0, 0], expected[:, 1, 1], expected[:, 3, 0] = 5, -3, 1e-6
            torch.testing.assert_close(mean, expected, rtol=0, atol=0)
            self.assertTrue((detail["history_weights"][~detail["history_present"]] == 0).all())
            changed = e.clone()
            changed[:, :-5] = -1e5
            changed[:, -1] = -17  # Different similarity, identical availability.
            changed[:, -5:-1, 4] = 123
            other = similar_text_history(changed, uniform=True)[0]
            torch.testing.assert_close(mean[:, :4], other[:, :4], rtol=0, atol=0)

    def test_empty_history_and_absent_current_remain_masked_with_trained_heads(self):
        e = torch.zeros(2, 7, 5, 384)
        e[:, -1, 0] = 3
        mean, _, detail = similar_text_history(e, uniform=True)
        self.assertTrue((mean == 0).all() and (detail["history_weights"] == 0).all())
        e[:, -5:-1, 1] = 7  # History exists, current group 1 is absent.
        text = model("uniform_history")
        with torch.no_grad():
            for head in text.history_heads.values():
                head.weight.fill_(1)
        x, price = torch.randn(2, 7, 4), torch.randn(2, 3, 4)
        _, detail = text(price, x[:, -1], e, price_window=x, novelty=torch.ones(2, 5),
                         pair_available=torch.ones(2, 5), return_details=True)
        self.assertTrue((detail["history_contributions"] == 0).all())
        self.assertTrue((detail["contributions"][:, 1:] == 0).all())

    def test_one_previous_date_matches_similarity_history_for_identical_parameters(self):
        torch.manual_seed(17)
        original = model("history")
        with torch.no_grad():
            for head in original.history_heads.values():
                head.weight.normal_(std=.1)
            original.gate.weight.normal_(std=.1)
        uniform = model("uniform_history")
        uniform.load_state_dict(original.state_dict(), strict=True)
        e = torch.zeros(3, 8, 5, 384)
        e[:, -1] = torch.randn(3, 5, 384)
        for group in range(5):
            e[:, -5+group%4, group] = torch.randn(3, 384)
        e[0, -1] = 0
        x, price = torch.randn(3, 8, 4), torch.randn(3, 3, 4)
        kwargs = dict(price_window=x, novelty=torch.ones(3, 5), pair_available=torch.ones(3, 5), return_details=True)
        a, da = original(price, x[:, -1], e, **kwargs)
        b, db = uniform(price, x[:, -1], e, **kwargs)
        torch.testing.assert_close(da["history_mean"], db["history_mean"], rtol=0, atol=0)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_zero_heads_match_last_initial_film_heads_gate_rng_batch_order_and_parameter_count(self):
        for seed in (7, 17, 27):
            reference = None
            counts = []
            for mode in ("last", "history", "uniform_history"):
                run.set_seed(seed)
                film = fixtures.ContextNoveltyTests.make_film().eval()
                text = model(mode).eval()
                state = film.state_dict(), text.group_mlps.state_dict(), text.gate.state_dict(), torch.get_rng_state().clone()
                x, e = torch.randn(3, 6, 4), torch.randn(3, 6, 5, 384)
                e[0, -1] = 0
                with torch.no_grad():
                    prediction = text(film(x, None, None, None), x[:, -1], e, price_window=x,
                                      novelty=torch.ones(3, 5), pair_available=torch.ones(3, 5))
                loader = DataLoader(TensorDataset(torch.arange(31)), batch_size=4, shuffle=True)
                run._configure_gate_sampling(loader, seed)
                order = [torch.cat([batch[0] for batch in loader]) for _ in range(2)]
                if reference is None:
                    reference = state, prediction, order
                else:
                    for a, b in zip(state[:3], reference[0][:3]):
                        for key, value in a.items():
                            torch.testing.assert_close(value, b[key], rtol=0, atol=0)
                    torch.testing.assert_close(state[3], reference[0][3], rtol=0, atol=0)
                    torch.testing.assert_close(prediction, reference[1], rtol=0, atol=0)
                    for a, b in zip(order, reference[2]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertTrue(all(head.bias is None and (head.weight == 0).all() for head in text.history_heads.values()))
                    counts.append(sum(p.numel() for p in text.parameters()))
            self.assertEqual(counts[0], counts[1])

    def test_reject_incompatible_pooling(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.parse_args(["--model_type", "FiLM", "--group_gate", "context_price_only", "--text_history_mode", "uniform_history",
                            "--text_temporal_pool", "attention5", "--code_commit", "a"*40])
