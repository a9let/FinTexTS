"""Per-group adaptation of Huang et al. (2022), sections 3.1--3.2.

Only the state -> news attention -> state recurrence is transferred.
https://doi.org/10.1016/j.neucom.2021.10.092
"""

import math

import torch
from torch import nn

from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS


HUANG_MODES = ('huang_recurrent_query', 'huang_constant_query')
HUANG_REPLACE = 'huang_recurrent_replace'
HUANG_ALL_MODES = (*HUANG_MODES, HUANG_REPLACE)
STEPS, HISTORY, HIDDEN = 20, 10, 32
HUANG_DEFAULTS = {'huang_hidden_size': 32, 'huang_recurrent_steps': 20, 'huang_attention_lookback': 10}


def huang_options(config):
    """Missing fields in historical configurations mean the original 32/20/10."""
    options = {key: config.get(key, default) for key, default in HUANG_DEFAULTS.items()}
    for key, value in options.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
    return options


def huang_structure(mode, pred_len=3, **parameters):
    if mode not in HUANG_ALL_MODES:
        raise ValueError(f'Unknown grouped Huang query mode: {mode}')
    options = huang_options(parameters)
    hidden, steps, lookback = options.values()
    structure = {'mode': mode, 'rows': steps + lookback, 'recurrent_steps': steps, 'history_rows_per_step': lookback,
            'step_positions': f'input_suffix_t-{steps - 1}_to_t', 'history_positions': f's-{lookback}_to_s-1',
            'source': 'original_MiniLM_group_embeddings_384', 'groups_independent': True,
            'projection': {'U': [384, HIDDEN], 'V': [384, HIDDEN], 'bias': False,
                           'formula': 'sigmoid(Ue)*tanh(Ve)'},
            'query': {'source': 'learned_group_constant' if mode == HUANG_MODES[1] else 'previous_group_state',
                      'size': HIDDEN, 'bias': False,
                      'activation': 'identity' if mode == HUANG_MODES[1] else 'tanh',
                      'constant_initialization': 'zeros' if mode == HUANG_MODES[1] else None},
            'attention': 'masked_scaled_dot_product', 'attention_scale': math.sqrt(HIDDEN),
            'historical_summary_activation': 'tanh', 'current_summary_activation': 'none_after_u',
            'lstm_input_size': 2 * HIDDEN, 'hidden_size': hidden, 'initial_state': 'zeros_per_window',
            'empty_step': 'freeze_hidden_and_cell', 'head_structure': [{'type': 'Linear', 'in_features': 384, 'out_features': pred_len}],
            'residual_head': {'type': 'Linear', 'in_features': hidden, 'out_features': pred_len,
                              'weight_init': 'zeros', 'bias_init': 'zeros'},
            'presence': 'float32_L1>=1e-6_before_projection', 'date_decomposition': False,
            'paper': '10.1016/j.neucom.2021.10.092 sections 3.1-3.2'}
    if mode == HUANG_REPLACE:
        structure.update(text_branch='huang_replaces_last', residual_head=None,
                         head_structure=[{'type': 'Linear', 'in_features': hidden, 'out_features': pred_len}],
                         output_head_initialization={'weight': 'zeros', 'bias': 'zeros'})
    return structure


class GroupNewsState(nn.Module):
    """One group's projections, query, hidden state, cell memory and correction."""

    def __init__(self, mode, pred_len, hidden_size=32, recurrent_steps=20, attention_lookback=10):
        super().__init__()
        self.mode = mode
        self.hidden_size, self.recurrent_steps, self.attention_lookback = hidden_size, recurrent_steps, attention_lookback
        self.U = nn.Linear(384, HIDDEN, bias=False)
        self.V = nn.Linear(384, HIDDEN, bias=False)
        self.cell = nn.LSTMCell(2 * HIDDEN, hidden_size)
        self.head = nn.Linear(hidden_size, pred_len)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Query creation cannot advance initialization of the next group's modules.
        with torch.random.fork_rng(devices=[]):
            if mode != HUANG_MODES[1]:
                self.query = nn.Linear(hidden_size, HIDDEN, bias=False)
            else:
                self.query = nn.Parameter(torch.zeros(HIDDEN))

    def forward(self, original, return_details=False):
        present = original.abs().sum(-1) >= 1e-6
        u = torch.sigmoid(self.U(original)) * torch.tanh(self.V(original))
        u = u * present[..., None]
        z = original.new_zeros((original.shape[0], self.hidden_size))
        memory = torch.zeros_like(z)
        processed = torch.zeros(original.shape[0], dtype=torch.bool, device=original.device)
        records = {key: [] for key in ('huang_beta', 'huang_history_present', 'huang_queries', 'huang_states',
                                      'huang_memories', 'huang_current', 'huang_history', 'huang_updated')}
        for s in range(self.attention_lookback, self.attention_lookback + self.recurrent_steps):
            q = self.query.expand(original.shape[0], HIDDEN) if self.mode == HUANG_MODES[1] else torch.tanh(self.query(z))
            mask = present[:, s - self.attention_lookback:s]
            past = u[:, s - self.attention_lookback:s]
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
    def __init__(self, mode, pred_len, hidden_size=32, recurrent_steps=20, attention_lookback=10):
        super().__init__()
        self.rows = recurrent_steps + attention_lookback
        self.groups = nn.ModuleDict({g: GroupNewsState(mode, pred_len, hidden_size, recurrent_steps, attention_lookback) for g in TEXT_GROUPS})

    def forward(self, text_window, return_details=False):
        if text_window.ndim != 4 or text_window.shape[1] < self.rows or text_window.shape[2:] != (5, 384):
            raise ValueError(f'Huang requires [batch,time>={self.rows},5,384] within the price input window')
        original = text_window[:, -self.rows:].float()
        if not torch.isfinite(original).all():
            raise ValueError('Text history must be finite')
        results = [self.groups[g](original[:, :, i], return_details) for i, g in enumerate(TEXT_GROUPS)]
        corrections = torch.stack([r[0] for r in results], 1)
        processed = torch.stack([r[1] for r in results], 1)
        detail = {key: torch.stack([r[2][key] for r in results], 2) for key in results[0][2]} if return_details else {}
        return corrections, processed, detail


class GroupHuangQueryTextModel(ContextNoveltyTextModel):
    def __init__(self, *, text_temporal_pool, temporal_pool_structure=None,
                 huang_hidden_size=32, huang_recurrent_steps=20, huang_attention_lookback=10, **config):
        self.huang_parameters = huang_options(dict(huang_hidden_size=huang_hidden_size,
                                                  huang_recurrent_steps=huang_recurrent_steps,
                                                  huang_attention_lookback=huang_attention_lookback))
        expected = huang_structure(text_temporal_pool, config['pred_len'], **self.huang_parameters)
        if config.get('group_gate') != 'context_price_only':
            raise ValueError('Grouped Huang modes require context_price_only')
        if temporal_pool_structure is not None and temporal_pool_structure != expected:
            raise ValueError('Unsupported saved Huang structure')
        super().__init__(**config)
        self.text_temporal_pool = text_temporal_pool
        with torch.random.fork_rng(devices=[]):
            self.temporal_pool = GroupHuangPool(text_temporal_pool, self.pred_len, *self.huang_parameters.values())
        if text_temporal_pool == HUANG_REPLACE:
            # Preserve the historical RNG sequence, then discard unused last heads.
            # They are absent from state_dict and the optimizer in replace mode.
            self.group_mlps = nn.ModuleDict()

    def fusion_state(self):
        return {**super().fusion_state(), 'text_temporal_pool': self.text_temporal_pool}

    def forward(self, price_output, last_price, embeddings, *, text_window, price_window, novelty,
                pair_available, return_details=False):
        if text_window.shape[:2] != price_window.shape[:2]:
            raise ValueError('Text and price windows must align')
        historical, history_present, detail = self.temporal_pool(text_window, return_details)
        if self.text_temporal_pool == HUANG_REPLACE:
            result = super().forward(price_output, last_price, embeddings, price_window=price_window,
                                     novelty=novelty, pair_available=pair_available, return_details=return_details,
                                     group_outputs=historical, group_present=history_present)
            if not return_details:
                return result
            output, base = result
            return output, {**base, **detail, 'huang_present': history_present,
                            'huang_contributions': base['contributions'],
                            'input_current_present': embeddings.float().abs().sum(-1) >= 1e-6}
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
