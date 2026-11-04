"""Shared single-head attention within each group's five known input rows."""

import torch
from torch import nn


def attention5_config():
    return {"type": "single_head_self_attention", "rows": 5, "embedding_size": 384,
            "qk_input_size": 385, "qk_size": 16, "bias": False,
            "shared_across_groups": True, "ages": [4, 3, 2, 1, 0],
            "age_scale": 4, "values": "original_embeddings",
            "query_reduction": "mean_present", "key_mask": "present",
            "query_mask": "present", "triangular_mask": False,
            "q_initialization": "zeros", "k_initialization": "nn.Linear_default",
            "parameters": 2 * 385 * 16}


class AttentionPool5(nn.Module):
    def __init__(self):
        super().__init__()
        # Creation consumes no RNG outside this block, including Linear's default
        # initialization of Q before it is zeroed. K retains its default weights.
        with torch.random.fork_rng(devices=[]):
            self.query = nn.Linear(385, 16, bias=False)
            self.key = nn.Linear(385, 16, bias=False)
        nn.init.zeros_(self.query.weight)

    def forward(self, embeddings):
        if embeddings.ndim != 4 or embeddings.shape[2:] != (5, 384) or embeddings.shape[1] < 5:
            raise ValueError("attention5 requires text window [batch, rows>=5, 5, 384]")
        values = embeddings[:, -5:].transpose(1, 2)  # [B,G,dates,384]
        daily_present = values.float().abs().sum(-1) >= 1e-6
        count = daily_present.sum(-1)
        values = torch.where(daily_present[..., None], values, torch.zeros_like(values))
        ages = values.new_tensor([4, 3, 2, 1, 0])
        age_input = (ages / 4).view(1, 1, 5, 1).expand(*values.shape[:-1], 1)
        qk_input = torch.cat((values, age_input), -1)
        query, key = self.query(qk_input), self.key(qk_input)
        scores = (query @ key.transpose(-1, -2)) / 4  # sqrt(16)
        scores = scores.masked_fill(~daily_present[..., None, :], float("-inf"))
        # Avoid softmax(-inf,...,-inf). Empty groups are zeroed after softmax;
        # absent queries and keys are always exactly zero in the saved matrix.
        scores = torch.where((count > 0)[..., None, None], scores, torch.zeros_like(scores))
        attention = torch.softmax(scores, -1)
        attention = attention * daily_present[..., None, :] * daily_present[..., :, None]
        beta = attention.sum(-2) / count.clamp_min(1)[..., None].to(values.dtype)
        aggregate = (beta[..., None] * values).sum(-2)
        return aggregate, count > 0, count, {
            "attention": attention, "temporal_weights": beta,
            "daily_present": daily_present, "row_ages": ages,
        }
