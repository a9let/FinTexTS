"""Causal, per-group GRU keys and additive attention over original text values."""

import torch
from torch import nn

from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS, price_context


WINDOW = 20
HIDDEN_SIZE = 32


def temporal_structure(mode):
    if mode in ("group_han_temporal", "group_han_uniform"):
        from forecasting_task.models.GroupHANTemporal import han_structure
        return han_structure(mode)
    if mode == "last":
        return {"mode": mode, "rows": 1, "values": "original_group_embeddings_384"}
    if mode != "group_gru_attention":
        raise ValueError(f"Unsupported temporal pool: {mode}")
    return {"mode": mode, "rows": WINDOW, "projection": [384, HIDDEN_SIZE], "projection_bias": True,
            "projection_shared_between_groups": True, "gru_input_size": HIDDEN_SIZE + 1,
            "gru_hidden_size": HIDDEN_SIZE, "gru_num_layers": 1, "bidirectional": False,
            "gru_and_attention_shared_between_groups": False, "attention_hidden_size": HIDDEN_SIZE,
            "price_context_size": 8, "price_context_lags": [1, 5], "gap_divisor": WINDOW,
            "age_divisor": WINDOW - 1, "values": "original_group_embeddings_384",
            "presence": "float32_L1>=1e-6_before_projection", "horizon_shared_beta": True,
            "initialization": "torch_defaults_in_isolated_rng;attention_bias_zero"}


class AdditiveDateAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.context = nn.Linear(8, HIDDEN_SIZE, bias=False)
        self.age = nn.Linear(1, HIDDEN_SIZE, bias=False)
        self.bias = nn.Parameter(torch.zeros(HIDDEN_SIZE))
        self.vector = nn.Linear(HIDDEN_SIZE, 1, bias=False)

    def forward(self, states, context, ages, present):
        features = self.hidden(states) + self.context(context)[:, None] + self.age(ages) + self.bias
        scores = self.vector(torch.tanh(features)).squeeze(-1)
        scores = scores.masked_fill(~present, torch.finfo(scores.dtype).min)
        beta = torch.softmax(scores, dim=1) * present.to(scores.dtype)
        return beta / beta.sum(dim=1, keepdim=True).clamp_min(torch.finfo(beta.dtype).tiny)


class GroupGRUPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(384, HIDDEN_SIZE)
        self.grus = nn.ModuleDict({g: nn.GRU(HIDDEN_SIZE + 1, HIDDEN_SIZE, batch_first=True) for g in TEXT_GROUPS})
        self.attentions = nn.ModuleDict({g: AdditiveDateAttention() for g in TEXT_GROUPS})

    def forward(self, text_window, context):
        if text_window.ndim != 4 or text_window.shape[1] < WINDOW or text_window.shape[2:] != (5, 384):
            raise ValueError("Expected text window [batch,time>=20,5,384]")
        original = text_window[:, -WINDOW:].float()
        if not torch.isfinite(original).all():
            raise ValueError("Text history must be finite")
        present = original.abs().sum(dim=-1) >= 1e-6
        projected = self.projection(original)
        ages = torch.arange(WINDOW - 1, -1, -1, device=original.device, dtype=original.dtype) / (WINDOW - 1)
        betas, all_states, all_gaps = [], [], []
        for group_index, group in enumerate(TEXT_GROUPS):
            state = original.new_zeros((original.shape[0], HIDDEN_SIZE))
            previous = torch.full((original.shape[0],), -1, dtype=torch.long, device=original.device)
            states, gaps = [], []
            for row in range(WINDOW):
                available = present[:, row, group_index]
                gap = torch.where(available & (previous >= 0), (row - previous).to(original.dtype) / WINDOW, 0.)
                inputs = torch.cat((projected[:, row, group_index], gap[:, None]), dim=-1)
                _, candidate = self.grus[group](inputs[:, None], state[None])
                state = torch.where(available[:, None], candidate[0], state)
                previous = torch.where(available, row, previous)
                states.append(state)
                gaps.append(gap)
            states = torch.stack(states, dim=1)
            beta = self.attentions[group](states, context, ages[None, :, None], present[:, :, group_index])
            betas.append(beta)
            all_states.append(states)
            all_gaps.append(torch.stack(gaps, dim=1))
        beta = torch.stack(betas, dim=2)  # [B,20,G], shared over forecast horizons
        pooled = (original * beta[..., None]).sum(dim=1)
        return pooled, present.any(dim=1), {
            "temporal_beta": beta, "temporal_present": present, "temporal_ages": ages,
            "temporal_gaps": torch.stack(all_gaps, dim=2), "temporal_embeddings": original,
            "temporal_states": torch.stack(all_states, dim=2), "pooled_embeddings": pooled,
        }


class GroupGRUAttentionTextModel(ContextNoveltyTextModel):
    def __init__(self, *, temporal_pool_structure=None, **config):
        if config.get("group_gate") != "context_price_only":
            raise ValueError("group_gru_attention requires context_price_only")
        expected = temporal_structure("group_gru_attention")
        if temporal_pool_structure is not None and temporal_pool_structure != expected:
            raise ValueError("Unsupported saved GRU/attention structure")
        super().__init__(**config)
        # Original heads/gate are created first. Extra parameters consume no global RNG.
        with torch.random.fork_rng(devices=[]):
            self.temporal_pool = GroupGRUPool()

    def fusion_state(self):
        return {**super().fusion_state(), "text_temporal_pool": "group_gru_attention"}

    def forward(self, price_output, last_price, embeddings, *, text_window, price_window,
                novelty, pair_available, return_details=False):
        if text_window.shape[:2] != price_window.shape[:2]:
            raise ValueError("Text and price input windows must align")
        pooled, present, temporal = self.temporal_pool(text_window, price_context(price_window))
        result = super().forward(price_output, last_price, pooled, price_window=price_window,
                                 novelty=novelty, pair_available=pair_available,
                                 return_details=return_details, group_present=present)
        if not return_details:
            return result
        output, detail = result
        # Linear heads preserve an exact additive decomposition, including their biases.
        date_heads = torch.stack([self.group_mlps[g](temporal["temporal_embeddings"][:, :, j])
                                  for j, g in enumerate(TEXT_GROUPS)], dim=2)
        date_contributions = (self.fixed_alpha * detail["weights"][:, None, :, None]
                              * temporal["temporal_beta"][..., None] * date_heads)
        return output, {**detail, **temporal, "date_heads": date_heads, "date_contributions": date_contributions}
