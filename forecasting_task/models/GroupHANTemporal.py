"""Adapted temporal HAN block, Hu et al., arXiv:1712.02136v3, section 4.2."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS


HAN_MODES = ("group_han_temporal", "group_han_uniform")
WINDOW = 20


def han_structure(mode):
    if mode not in HAN_MODES:
        raise ValueError(f"Unknown HAN temporal mode: {mode}")
    return {"mode": mode, "rows": WINDOW, "input_projection": [384, 32], "input_projection_shared": True,
            "input_projection_bias": True, "gru_input_size": 32, "hidden_per_direction": 32,
            "gru_num_layers": 1, "bidirectional": True, "state_size": 64,
            "group_parameters_shared": False, "score_projection": [64, 1], "score_activation": "sigmoid",
            "position_parameters": [WINDOW], "position_index": "original_window_row_0_to_19",
            "position_initialization": "ones", "output_projection_per_group": [64, 384],
            "output_projection_bias": True, "values": "contextual_BiGRU_states_64",
            "pool_weights": "learned_HAN" if mode == HAN_MODES[0] else "uniform_over_present",
            "presence": "float32_L1>=1e-6_before_projection", "missing_recurrence": "compact_present_rows_both_directions",
            "price_query_in_date_score": False, "horizon_shared_beta": True,
            "initialization": "torch_defaults_in_isolated_rng", "paper": "arxiv:1712.02136v3 section 4.2"}


class HANDateAttention(nn.Module):
    def __init__(self, state_size=64):
        super().__init__()
        # Scalar o_i makes the paper's theta_i * o_i a scalar logit.
        self.transform = nn.Linear(state_size, 1)
        self.position = nn.Parameter(torch.ones(WINDOW))

    def forward(self, states, present):
        scores = self.position * torch.sigmoid(self.transform(states).squeeze(-1))
        scores = scores.masked_fill(~present, torch.finfo(scores.dtype).min)
        beta = torch.softmax(scores, dim=1) * present.to(scores.dtype)
        return beta / beta.sum(1, keepdim=True).clamp_min(torch.finfo(beta.dtype).tiny)


def masked_bigru(gru, values, present):
    """Drop missing rows for recurrence, then restore their original positions."""
    states = values.new_zeros((*present.shape, 2 * gru.hidden_size))
    nonempty = present.any(1).nonzero(as_tuple=False).flatten().tolist()
    if nonempty:
        sequences = [values[index, present[index]] for index in nonempty]
        packed = pack_sequence(sequences, enforce_sorted=False)
        encoded, _ = gru(packed)
        padded, lengths = pad_packed_sequence(encoded, batch_first=True)
        for row, index in enumerate(nonempty):
            states[index, present[index]] = padded[row, :int(lengths[row])]
    return states


class GroupHANPool(nn.Module):
    def __init__(self, mode, hidden_size=32, output_projection=True):
        super().__init__()
        self.mode = mode
        self.output_projection = output_projection
        self.projection = nn.Linear(384, 32)
        self.grus = nn.ModuleDict({g: nn.GRU(32, hidden_size, batch_first=True, bidirectional=True) for g in TEXT_GROUPS})
        self.attentions = nn.ModuleDict({g: HANDateAttention(2 * hidden_size) for g in TEXT_GROUPS})
        self.output_projections = nn.ModuleDict({g: nn.Linear(2 * hidden_size, 384) for g in TEXT_GROUPS}) if output_projection else nn.ModuleDict()

    def forward(self, text_window, exclude_current=False):
        if text_window.ndim != 4 or text_window.shape[1] < WINDOW or text_window.shape[2:] != (5, 384):
            raise ValueError("Expected text window [batch,time>=20,5,384]")
        original = text_window[:, -WINDOW:].float()
        if not torch.isfinite(original).all():
            raise ValueError("Text history must be finite")
        present = original.abs().sum(-1) >= 1e-6
        source_present = present
        if exclude_current:
            present = present.clone()
            present[:, -1] = False
        projected = self.projection(original)
        betas, states_by_group, values_by_group, pooled = [], [], [], []
        for index, group in enumerate(TEXT_GROUPS):
            mask = present[:, :, index]
            states = masked_bigru(self.grus[group], projected[:, :, index], mask)
            if self.mode == "group_han_uniform":
                beta = mask.to(original.dtype) / mask.sum(1, keepdim=True).clamp_min(1)
            else:
                beta = self.attentions[group](states, mask)
            latent = (beta[..., None] * states).sum(1)
            group_value = self.output_projections[group](latent) * mask.any(1)[:, None] if self.output_projection else latent
            pooled.append(group_value)
            # Distribute the output-projection bias over dates through sum(beta)=1.
            values_by_group.append(self.output_projections[group](states) if self.output_projection else states)
            betas.append(beta)
            states_by_group.append(states)
        pooled = torch.stack(pooled, 1)
        details = {
            "temporal_beta": torch.stack(betas, 2), "temporal_present": present,
            "temporal_states": torch.stack(states_by_group, 2), "date_values": torch.stack(values_by_group, 2),
            "temporal_ages": torch.arange(19, -1, -1, device=original.device, dtype=original.dtype) / 19,
            "pooled_embeddings": pooled}
        if exclude_current:
            details["temporal_source_present"] = source_present
        return pooled, present.any(1), details


class GroupHANTemporalTextModel(ContextNoveltyTextModel):
    def __init__(self, *, text_temporal_pool, temporal_pool_structure=None, **config):
        expected = han_structure(text_temporal_pool)
        if config.get("group_gate") != "context_price_only":
            raise ValueError("HAN temporal modes require context_price_only")
        if temporal_pool_structure is not None and temporal_pool_structure != expected:
            raise ValueError("Unsupported saved HAN structure")
        super().__init__(**config)
        self.text_temporal_pool = text_temporal_pool
        with torch.random.fork_rng(devices=[]):
            self.temporal_pool = GroupHANPool(text_temporal_pool)

    def fusion_state(self):
        return {**super().fusion_state(), "text_temporal_pool": self.text_temporal_pool}

    def forward(self, price_output, last_price, embeddings, *, text_window, price_window,
                novelty, pair_available, return_details=False):
        if text_window.shape[:2] != price_window.shape[:2]:
            raise ValueError("Text and price windows must align")
        pooled, present, temporal = self.temporal_pool(text_window)
        result = super().forward(price_output, last_price, pooled, price_window=price_window,
                                 novelty=novelty, pair_available=pair_available,
                                 return_details=return_details, group_present=present)
        if not return_details:
            return result
        output, detail = result
        date_heads = torch.stack([self.group_mlps[g](temporal["date_values"][:, :, index])
                                  for index, g in enumerate(TEXT_GROUPS)], 2)
        date_contributions = (self.fixed_alpha * detail["weights"][:, None, :, None]
                              * temporal["temporal_beta"][..., None] * date_heads)
        return output, {**detail, **temporal, "date_heads": date_heads, "date_contributions": date_contributions}
