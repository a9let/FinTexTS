"""Direct HAN heads and a zero-initialized historical correction to last."""

import hashlib

import torch
from torch import nn
from torch.utils.data import default_collate

from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS
from forecasting_task.models.GroupHANTemporal import GroupHANPool


ABLATION_MODES = ('last_matched', 'han_h32_linear', 'han_h16_linear', 'han_h32_mlp', 'last_han_h32_residual')


def head_ablation_structure(mode, pred_len=3):
    if mode not in ABLATION_MODES:
        raise ValueError(f'Unknown HAN head ablation: {mode}')
    hidden = 16 if mode == 'han_h16_linear' else 32
    state = 2 * hidden
    legacy = mode in ('last_matched', 'last_han_h32_residual')
    head = [{'type': 'Linear', 'in_features': 384 if legacy else state,
             'out_features': 16 if mode == 'han_h32_mlp' else pred_len}]
    if mode == 'han_h32_mlp':
        head.extend([{'type': 'GELU'}, {'type': 'Linear', 'in_features': 16, 'out_features': pred_len}])
    return {'mode': mode, 'rows': 1 if mode == 'last_matched' else 20,
            'history_used_rows': 19 if mode == 'last_han_h32_residual' else 0 if mode == 'last_matched' else 20,
            'include_current_in_history': mode not in ('last_matched', 'last_han_h32_residual'),
            'original_position_indices': [19] if mode == 'last_matched' else list(range(19 if mode == 'last_han_h32_residual' else 20)),
            'input_projection': None if mode == 'last_matched' else [384, 32],
            'hidden_per_direction': None if mode == 'last_matched' else hidden,
            'state_size': 384 if mode == 'last_matched' else state,
            'score_projection': None if mode == 'last_matched' else [state, 1],
            'score_activation': 'sigmoid', 'output_bridge': None, 'dropout': 0.,
            'head_structure': head,
            'residual_head': {'in_features': 64, 'out_features': pred_len, 'weight_init': 'zeros', 'bias_init': 'zeros'}
                             if mode == 'last_han_h32_residual' else None,
            'date_decomposition': mode != 'han_h32_mlp', 'presence': 'float32_L1>=1e-6_before_projection'}


class HANHeadAblationTextModel(ContextNoveltyTextModel):
    def __init__(self, *, text_temporal_pool, temporal_pool_structure=None, **config):
        if config.get('group_gate') != 'context_price_only':
            raise ValueError('HAN head ablations require context_price_only')
        expected = head_ablation_structure(text_temporal_pool, config['pred_len'])
        if temporal_pool_structure is not None and temporal_pool_structure != expected:
            raise ValueError('Unsupported saved head ablation structure')
        super().__init__(**config)
        self.text_temporal_pool = text_temporal_pool
        if text_temporal_pool == 'last_matched':
            return
        hidden = expected['hidden_per_direction']
        # Encoder creation precedes the condition-specific heads in every mode.
        with torch.random.fork_rng(devices=[]):
            self.temporal_pool = GroupHANPool('group_han_temporal', hidden_size=hidden, output_projection=False)
        with torch.random.fork_rng(devices=[]):
            if text_temporal_pool == 'last_han_h32_residual':
                self.history_heads = nn.ModuleDict({g: nn.Linear(64, self.pred_len) for g in TEXT_GROUPS})
                for head in self.history_heads.values():
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
            else:
                def make_head():
                    if text_temporal_pool == 'han_h32_mlp':
                        return nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.Linear(16, self.pred_len))
                    return nn.Sequential(nn.Linear(2 * hidden, self.pred_len))
                self.group_mlps = nn.ModuleDict({g: make_head() for g in TEXT_GROUPS})

    def fusion_state(self):
        return {**super().fusion_state(), 'text_temporal_pool': self.text_temporal_pool}

    def forward(self, price_output, last_price, embeddings, *, price_window, novelty,
                pair_available, text_window=None, return_details=False):
        kwargs = dict(price_window=price_window, novelty=novelty, pair_available=pair_available, return_details=return_details)
        if self.text_temporal_pool == 'last_matched':
            result = super().forward(price_output, last_price, embeddings, **kwargs)
            if not return_details:
                return result
            output, detail = result
            mask = detail['present'][:, None]
            return output, {**detail, 'temporal_beta': mask.to(embeddings.dtype), 'temporal_present': mask,
                            'temporal_ages': embeddings.new_zeros(1), 'date_contributions': detail['contributions'][:, None]}
        if text_window is None or text_window.shape[:2] != price_window.shape[:2]:
            raise ValueError('History and price windows must align')
        residual = self.text_temporal_pool == 'last_han_h32_residual'
        states, present, temporal = self.temporal_pool(text_window, exclude_current=residual)
        if residual:
            current_present = embeddings.float().abs().sum(-1) >= 1e-6
            current = torch.stack([self.group_mlps[g](embeddings[:, i]) for i, g in enumerate(TEXT_GROUPS)], 1)
            current = current * current_present[..., None]
            history = torch.stack([self.history_heads[g](states[:, i]) for i, g in enumerate(TEXT_GROUPS)], 1)
            history = history * present[..., None]
            heads = current + history
            combined_present = present | current_present
        else:
            heads = torch.stack([self.group_mlps[g](states[:, i]) for i, g in enumerate(TEXT_GROUPS)], 1)
            combined_present = present
        result = super().forward(price_output, last_price, embeddings, group_outputs=heads,
                                 group_present=combined_present, **kwargs)
        if not return_details:
            return result
        output, detail = result
        detail.update(temporal)
        coefficients = self.fixed_alpha * detail['weights'][..., None]
        if residual:
            detail['current_present'] = current_present
            detail['history_present'] = present
            detail['current_contributions'] = coefficients * current
            detail['history_contributions'] = coefficients * history
            date_heads = torch.stack([self.history_heads[g](temporal['temporal_states'][:, :, i]) for i, g in enumerate(TEXT_GROUPS)], 2)
        elif self.text_temporal_pool == 'han_h32_mlp':
            # Attention weights are saved without assigning additive output shares.
            return output, detail
        else:
            date_heads = torch.stack([self.group_mlps[g](temporal['temporal_states'][:, :, i]) for i, g in enumerate(TEXT_GROUPS)], 2)
        dates = coefficients[:, None] * temporal['temporal_beta'][..., None] * date_heads
        if residual:
            detail['history_date_contributions'] = dates
            dates = dates.clone()
            dates[:, -1] += detail['current_contributions']
        detail['date_contributions'] = dates
        return output, detail


@torch.no_grad()
def initial_prediction_evidence(model, text_model, dataset, args, device, prepare_batch):
    """Audit the actual initial A/E forecast before the optimizer is created."""
    model_training, text_training = model.training, text_model.training
    model.eval()
    text_model.eval()
    try:
        batch = default_collate([dataset[0]])
        x, text, xm, ym, decoder, _ = prepare_batch(batch, device, args.pred_len, args.label_len)
        metadata = {key: value.float().to(device) for key, value in batch[6].items()}
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            film = model(x, xm, decoder, ym)
            output = text_model(film, x[:, -1], text[:, -1], text_window=text, price_window=x, **metadata)
            reference = ContextNoveltyTextModel.forward(text_model, film, x[:, -1], text[:, -1], price_window=x, **metadata)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        digest = hashlib.sha256(output.cpu().contiguous().numpy().tobytes()).hexdigest()
        return {'input_end_date': dataset.window_end_date(0), 'prediction_sha256': digest,
                'matches_last_matched_formula': True, 'max_abs_difference': (output - reference).abs().max().item()}
    finally:
        model.train(model_training)
        text_model.train(text_training)
