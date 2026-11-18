"""Per-group adaptation of Huang et al. (2022), sections 3.1--3.2.

Only the state -> news attention -> state recurrence is transferred.
https://doi.org/10.1016/j.neucom.2021.10.092
"""

import math

import torch
from torch import nn

from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS


HUANG_MODES = ('huang_recurrent_query', 'huang_constant_query')
STEPS, HISTORY, HIDDEN = 20, 10, 32


def huang_structure(mode, pred_len=3):
    if mode not in HUANG_MODES:
        raise ValueError(f'Unknown grouped Huang query mode: {mode}')
    return {'mode': mode, 'rows': 30, 'recurrent_steps': STEPS, 'history_rows_per_step': HISTORY,
            'step_positions': 'input_suffix_t-19_to_t', 'history_positions': 's-10_to_s-1',
            'source': 'original_MiniLM_group_embeddings_384', 'groups_independent': True,
            'projection': {'U': [384, HIDDEN], 'V': [384, HIDDEN], 'bias': False,
                           'formula': 'sigmoid(Ue)*tanh(Ve)'},
            'query': {'source': 'previous_group_state' if mode == HUANG_MODES[0] else 'learned_group_constant',
                      'size': HIDDEN, 'bias': False,
                      'activation': 'tanh' if mode == HUANG_MODES[0] else 'identity',
                      'constant_initialization': 'zeros' if mode == HUANG_MODES[1] else None},
            'attention': 'masked_scaled_dot_product', 'attention_scale': math.sqrt(HIDDEN),
            'historical_summary_activation': 'tanh', 'current_summary_activation': 'none_after_u',
            'lstm_input_size': 2 * HIDDEN, 'hidden_size': HIDDEN, 'initial_state': 'zeros_per_window',
            'empty_step': 'freeze_hidden_and_cell', 'head_structure': [{'type': 'Linear', 'in_features': 384, 'out_features': pred_len}],
            'residual_head': {'type': 'Linear', 'in_features': HIDDEN, 'out_features': pred_len,
                              'weight_init': 'zeros', 'bias_init': 'zeros'},
            'presence': 'float32_L1>=1e-6_before_projection', 'date_decomposition': False,
            'paper': '10.1016/j.neucom.2021.10.092 sections 3.1-3.2'}


class GroupNewsState(nn.Module):
    """One group's projections, query, hidden state, cell memory and correction."""

    def __init__(self, mode, pred_len):
        super().__init__()
        self.mode = mode
        self.U = nn.Linear(384, HIDDEN, bias=False)
        self.V = nn.Linear(384, HIDDEN, bias=False)
        self.cell = nn.LSTMCell(2 * HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, pred_len)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Query creation cannot advance initialization of the next group's modules.
        with torch.random.fork_rng(devices=[]):
            if mode == HUANG_MODES[0]:
                self.query = nn.Linear(HIDDEN, HIDDEN, bias=False)
            else:
                self.query = nn.Parameter(torch.zeros(HIDDEN))

    def forward(self, original, return_details=False):
        present = original.abs().sum(-1) >= 1e-6
        u = torch.sigmoid(self.U(original)) * torch.tanh(self.V(original))
        u = u * present[..., None]
        z = original.new_zeros((original.shape[0], HIDDEN))
        memory = torch.zeros_like(z)
        processed = torch.zeros(original.shape[0], dtype=torch.bool, device=original.device)
        records = {key: [] for key in ('huang_beta', 'huang_history_present', 'huang_queries', 'huang_states',
                                      'huang_memories', 'huang_current', 'huang_history', 'huang_updated')}
        for s in range(HISTORY, HISTORY + STEPS):
            q = torch.tanh(self.query(z)) if self.mode == HUANG_MODES[0] else self.query.expand_as(z)
            mask = present[:, s - HISTORY:s]
            past = u[:, s - HISTORY:s]
            logits = (past * q[:, None]).sum(-1) / math.sqrt(HIDDEN)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            beta = torch.softmax(logits, dim=1) * mask.to(logits.dtype)
            beta = beta / beta.sum(1, keepdim=True).clamp_min(torch.finfo(beta.dtype).tiny)
            history = torch.tanh((beta[..., None] * past).sum(1))
            current = u[:, s]
            update = present[:, s] | mask.any(1)
            next_z, next_memory = self.cell(torch.cat((current, history), -1), (z, memory))
            z = torch.where(update[:, None], next_z, z)
            memory = torch.where(update[:, None], next_memory, memory)
            processed = processed | update
            if return_details:
                for key, value in zip(records, (beta, mask, q, z, memory, current, history, update)):
                    records[key].append(value)
        correction = self.head(z) * processed[:, None]
        detail = {key: torch.stack(values, 1) for key, values in records.items()} if return_details else {}
        return correction, processed, detail


class GroupHuangPool(nn.Module):
    def __init__(self, mode, pred_len):
        super().__init__()
        self.groups = nn.ModuleDict({g: GroupNewsState(mode, pred_len) for g in TEXT_GROUPS})

    def forward(self, text_window, return_details=False):
        if text_window.ndim != 4 or text_window.shape[1] < HISTORY + STEPS or text_window.shape[2:] != (5, 384):
            raise ValueError('Huang requires [batch,time>=30,5,384] within the price input window')
        original = text_window[:, -(HISTORY + STEPS):].float()
        if not torch.isfinite(original).all():
            raise ValueError('Text history must be finite')
        results = [self.groups[g](original[:, :, i], return_details) for i, g in enumerate(TEXT_GROUPS)]
        corrections = torch.stack([r[0] for r in results], 1)
        processed = torch.stack([r[1] for r in results], 1)
        detail = {key: torch.stack([r[2][key] for r in results], 2) for key in results[0][2]} if return_details else {}
        return corrections, processed, detail


class GroupHuangQueryTextModel(ContextNoveltyTextModel):
    def __init__(self, *, text_temporal_pool, temporal_pool_structure=None, **config):
        expected = huang_structure(text_temporal_pool, config['pred_len'])
        if config.get('group_gate') != 'context_price_only':
            raise ValueError('Grouped Huang modes require context_price_only')
        if temporal_pool_structure is not None and temporal_pool_structure != expected:
            raise ValueError('Unsupported saved Huang structure')
        super().__init__(**config)
        self.text_temporal_pool = text_temporal_pool
        with torch.random.fork_rng(devices=[]):
            self.temporal_pool = GroupHuangPool(text_temporal_pool, self.pred_len)

    def fusion_state(self):
        return {**super().fusion_state(), 'text_temporal_pool': self.text_temporal_pool}

    def forward(self, price_output, last_price, embeddings, *, text_window, price_window, novelty,
                pair_available, return_details=False):
        if text_window.shape[:2] != price_window.shape[:2]:
            raise ValueError('Text and price windows must align')
        historical, history_present, detail = self.temporal_pool(text_window, return_details)
        current_present = embeddings.float().abs().sum(-1) >= 1e-6
        current = torch.stack([self.group_mlps[g](embeddings[:, i]) for i, g in enumerate(TEXT_GROUPS)], 1)
        current = current * current_present[..., None]
        result = super().forward(price_output, last_price, embeddings, price_window=price_window,
                                 novelty=novelty, pair_available=pair_available, return_details=return_details,
                                 group_outputs=current + historical, group_present=current_present | history_present)
        if not return_details:
            return result
        output, base = result
        coefficient = self.fixed_alpha * base['weights'][..., None]
        return output, {**base, **detail, 'current_present': current_present, 'history_present': history_present,
                        'current_contributions': coefficient * current, 'history_contributions': coefficient * historical}
