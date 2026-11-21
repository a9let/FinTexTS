"""One controlled weight ablation around the preserved HAN and Huang encoders."""

import torch
from torch import nn
from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS
from forecasting_task.models.HANHeadAblation import HANHeadAblationTextModel, head_ablation_structure
from forecasting_task.models.GroupHuangQuery import GroupHuangQueryTextModel, HUANG_REPLACE, huang_structure
from forecasting_task.models.GroupTLSTM import GroupTLSTM
from forecasting_task.models.GroupMemory import FAMILIES, PrototypeMemory, LinearMemory, ExpertRouter


MODE = 'group_gate_experiment'
SOURCES = {'han': '6a62808', 'huang': 'e642a6d', 'tlstm': 'Baytas KDD2017 section 3.1.2'}


def experiment_structure(family, hidden, window, weights, pred_len=3, kd_beta=0.):
    if family in FAMILIES:
        if (window not in (5,20) or weights not in ('price_only','uniform')
                or hidden not in ((8,16) if family == 'memto_group' else (16,32) if family == 'titans_linear_group' else (32,))
                or kd_beta not in ((0.,.5) if family == 'xtime_group' else (0.,))):
            raise ValueError('Invalid memory experiment configuration')
        width = {'memto_group':68, 'titans_linear_group':hidden, 'xtime_group':384}[family]
        return {'mode':MODE, 'family':family, 'hidden':hidden, 'window':window, 'group_weights':weights,
                'head_structure':[{'type':'Linear','in_features':width,'out_features':pred_len}],
                'kd_beta':kd_beta, 'tau':1 if family == 'memto_group' else None,
                'prototype_selection':'minimum_squared_euclidean' if family == 'memto_group' else None}
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
                 temporal_pool_structure=None, kd_beta=0., **config):
        super().__init__()
        self.family, self.hidden, self.window, self.weights_mode = text_family, family_hidden_size, text_window, group_weight_condition
        expected = experiment_structure(self.family, self.hidden, self.window, self.weights_mode, config['pred_len'], kd_beta)
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
            if self.family in FAMILIES:
                with torch.random.fork_rng(devices=[]):
                    if self.family == 'xtime_group':
                        self.core.temporal_pool = ExpertRouter(config['pred_len'], self.core.group_mlps)
                    else:
                        factory = PrototypeMemory if self.family == 'memto_group' else LinearMemory
                        self.core.temporal_pool = nn.ModuleDict({g:factory(self.hidden) for g in TEXT_GROUPS})
                        width = expected['head_structure'][0]['in_features']
                        self.core.group_mlps = nn.ModuleDict({g:nn.Linear(width, config['pred_len']) for g in TEXT_GROUPS})
                if self.family == 'xtime_group':
                    self.core.group_mlps = nn.ModuleDict()
                elif self.family == 'memto_group':
                    self.core.temporal_pool.requires_grad_(False)
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
        if self.family in FAMILIES:
            return self.memory_forward(price_output, last_price, embeddings, text_window, kwargs)
        if text_window.shape[:2] != price_window.shape[:2] or calendar_days is None:
            raise ValueError('T-LSTM requires aligned input windows and dates')
        heads, present, detail = self.temporal_pool(text_window, calendar_days)
        result = self.core(price_output, last_price, embeddings, group_outputs=heads, group_present=present, **kwargs)
        if not return_details:
            return result
        output, base = result
        return output, {**base, **detail}

    def memory_forward(self, film, last, embeddings, text, kwargs):
        if text.shape[:2] != kwargs['price_window'].shape[:2] or text.shape[1] < self.window:
            raise ValueError('Memory model requires a complete aligned input window')
        text = text[:, -self.window:].float()
        present = text.abs().sum(-1) >= 1e-6
        details = kwargs['return_details']
        if self.family == 'xtime_group':
            if self.temporal_pool.normal_only:
                heads = self.temporal_pool.outputs(text,present,only_normal=True)[:,0]
                weights = film.new_tensor([1.,0.,0.]).expand(len(film), -1)
            else:
                heads, forecasts = self.expert_forecasts(film,last,embeddings,text,present,kwargs)
                weights = self.temporal_pool.top_two(self.temporal_pool.router(forecasts.flatten(1)))
                heads = (heads * weights[...,None,None]).sum(1)
            extra = {'expert_weights':weights}
            if not self.temporal_pool.normal_only:
                extra['expert_forecasts'] = forecasts
        else:
            outputs, diagnostics = [], []
            for i,g in enumerate(TEXT_GROUPS):
                encoder = self.temporal_pool[g]
                if self.family == 'memto_group':
                    value, scalars, diag = encoder.features(text[:,:,i],present[:,:,i])
                    value = torch.cat((value,(scalars-encoder.center)/encoder.scale),-1)
                else:
                    value, diag = encoder(text[:,:,i],present[:,:,i])
                outputs.append(self.group_mlps[g](value))
                diagnostics.append(diag)
            heads = torch.stack(outputs,1)
            extra = {'memory_date_values':torch.stack(diagnostics,2), 'memory_present':present}
        output = self.core(film,last,embeddings,group_outputs=heads,group_present=present.any(1),**kwargs)
        if details:
            prediction, detail = output
            return prediction, {**detail, **extra}
        return output

    def expert_forecasts(self, film, last, embeddings, text, present, kwargs):
        heads = self.temporal_pool.outputs(text,present)
        predictions = [self.core(film,last,embeddings,group_outputs=h,group_present=present.any(1),
                                 **{**kwargs,'return_details':False}) for h in heads.unbind(1)]
        return heads, torch.stack(predictions,1)
