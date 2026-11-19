"""One controlled weight ablation around the preserved HAN and Huang encoders."""

import torch
from torch import nn
from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS
from forecasting_task.models.HANHeadAblation import HANHeadAblationTextModel, head_ablation_structure
from forecasting_task.models.GroupHuangQuery import GroupHuangQueryTextModel, HUANG_REPLACE, huang_structure
from forecasting_task.models.GroupTLSTM import GroupTLSTM


MODE = 'group_gate_experiment'
SOURCES = {'han': '6a62808', 'huang': 'e642a6d', 'tlstm': 'Baytas KDD2017 section 3.1.2'}


def experiment_structure(family, hidden, window, weights, pred_len=3):
    if family not in ('han', 'huang', 'tlstm', 'last') or hidden not in (16, 32) or window not in (5, 20) or weights not in ('price_only', 'uniform'):
        raise ValueError('Expected family han/huang/tlstm/last, H=16/32, window=5/20 and price_only/uniform')
    return {'mode': MODE, 'family': family, 'hidden': hidden, 'window': window, 'group_weights': weights,
            'head_structure': [{'type': 'Linear', 'in_features': 384 if family == 'last' else 2 * hidden if family == 'han' else hidden,
                                'out_features': pred_len}],
            'source': SOURCES.get(family, 'last_matched'), 'calendar_delta': 'days_between_present_rows_in_window' if family == 'tlstm' else None,
            'huang_lookback': 10 if family == 'huang' else None, 'han_positions': list(range(20-window, 20)) if family == 'han' else None,
            'head_initialization': 'zeros' if family == 'huang' else 'torch_default'}


class GroupGateExperiment(nn.Module):
    def __init__(self, *, text_family, family_hidden_size, text_window, group_weight_condition,
                 temporal_pool_structure=None, **config):
        super().__init__()
        self.family, self.hidden, self.window, self.weights_mode = text_family, family_hidden_size, text_window, group_weight_condition
        expected = experiment_structure(self.family, self.hidden, self.window, self.weights_mode, config['pred_len'])
        if temporal_pool_structure is not None and temporal_pool_structure != expected:
            raise ValueError('Saved family/weight structure differs')
        config['group_gate'] = 'context_price_only'
        if self.family == 'huang':
            self.core = GroupHuangQueryTextModel(text_temporal_pool=HUANG_REPLACE, huang_hidden_size=self.hidden,
                        huang_recurrent_steps=self.window, huang_attention_lookback=10, **config)
        elif self.family == 'han':
            mode = f'han_h{self.hidden}_linear'
            if self.hidden == 16 and self.window == 5:
                mode += '_w5'
            self.core = HANHeadAblationTextModel(text_temporal_pool=mode, **config)
            self.core.temporal_pool.window_size = self.window
        else:
            self.core = ContextNoveltyTextModel(**config)
            if self.family == 'tlstm':
                with torch.random.fork_rng(devices=[]):
                    self.core.temporal_pool = GroupTLSTM(self.hidden, self.window, config['pred_len'])
                self.core.group_mlps = nn.ModuleDict()
        # Drop the gate only after all initializations; paired common modules match.
        if self.weights_mode == 'uniform':
            self.core.gate = None

    @property
    def group_mlps(self):
        return self.core.group_mlps

    @property
    def gate(self):
        return self.core.gate

    @property
    def temporal_pool(self):
        return self.core.temporal_pool

    def get_alpha(self):
        return self.core.get_alpha()

    def fusion_state(self):
        state = {**self.core.fusion_state(), 'text_family': self.family, 'family_hidden_size': self.hidden,
                 'text_window': self.window, 'group_weight_condition': self.weights_mode}
        if self.weights_mode == 'uniform':
            state['fixed_group_coefficients'] = dict.fromkeys(TEXT_GROUPS,.2)
        return state

    def forward(self, price_output, last_price, embeddings, *, text_window, price_window,
                novelty, pair_available, calendar_days=None, return_details=False):
        kwargs = dict(price_window=price_window, novelty=novelty, pair_available=pair_available, return_details=return_details)
        if self.family in ('han', 'huang'):
            return self.core(price_output, last_price, embeddings, text_window=text_window, **kwargs)
        if self.family == 'last':
            return self.core(price_output, last_price, embeddings, **kwargs)
        if text_window.shape[:2] != price_window.shape[:2] or calendar_days is None:
            raise ValueError('T-LSTM requires aligned input windows and dates')
        heads, present, detail = self.temporal_pool(text_window, calendar_days)
        result = self.core(price_output, last_price, embeddings, group_outputs=heads, group_present=present, **kwargs)
        if not return_details:
            return result
        output, base = result
        return output, {**base, **detail}
