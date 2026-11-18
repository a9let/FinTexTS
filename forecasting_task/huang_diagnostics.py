"""Causal per-step attention and signed branch contributions, without date attribution."""

import json

import numpy as np
import torch

from forecasting_task.models.GroupText import TEXT_GROUPS
from forecasting_task.models.GroupHuangQuery import HISTORY, STEPS


def huang_records(dataset, indices, film_output, last_price, output, detail, alpha, mode):
    current = detail['current_contributions']
    history = detail['history_contributions']
    torch.testing.assert_close(current + history, detail['contributions'], rtol=1e-5, atol=1e-6)
    film_part = (1 - alpha) * film_output
    anchor_part = alpha * last_price[:, None].expand_as(output)
    reconstructed = film_part + anchor_part + (current + history).sum(1)[..., None]
    torch.testing.assert_close(reconstructed, output, rtol=1e-5, atol=1e-6)
    arrays = {key: value.detach().cpu().numpy() for key, value in detail.items()}
    parts = [v.detach().cpu().numpy() for v in (film_part, anchor_part, output, reconstructed)]
    groups, dates, components = {}, [], []
    for sample, index in enumerate(indices):
        stop = index + dataset.seq_len
        row_dates = list(map(str, dataset.row_dates[stop - HISTORY - STEPS:stop]))
        if len(row_dates) != HISTORY + STEPS or row_dates[-1] != dataset.window_end_date(index):
            raise ValueError('Huang diagnostic dates differ from the causal input window')
        for g, group in enumerate(TEXT_GROUPS):
            record = {'text_temporal_pool': mode, 'date_decomposition': False,
                      'decomposition_kind': 'recurrent_attention_weights_only',
                      'recurrent_step_dates': json.dumps(row_dates[HISTORY:]),
                      'current_present': int(arrays['current_present'][sample, g]),
                      'history_present': int(arrays['history_present'][sample, g])}
            for horizon in range(output.shape[1]):
                record[f'current_contribution_h{horizon + 1}'] = float(arrays['current_contributions'][sample, g, horizon])
                record[f'history_contribution_h{horizon + 1}'] = float(arrays['history_contributions'][sample, g, horizon])
            groups[index, group] = record
            for step in range(STEPS):
                mask = arrays['huang_history_present'][sample, step, g]
                beta = arrays['huang_beta'][sample, step, g]
                if np.any(beta[~mask] != 0) or not np.isclose(beta.sum(), float(mask.any()), rtol=1e-6, atol=1e-7):
                    raise ValueError('Invalid Huang masked history attention')
                for past in range(HISTORY):
                    dates.append({'window_index': index, 'input_end_date': row_dates[-1], 'group': group,
                                  'step_index': step, 'step_date': row_dates[HISTORY + step],
                                  'date': row_dates[step + past], 'history_age_rows': HISTORY - past,
                                  'present': int(mask[past]), 'beta': float(beta[past]),
                                  'state_updated': int(arrays['huang_updated'][sample, step, g]),
                                  'group_weight': float(arrays['weights'][sample, g]),
                                  'decomposition_kind': 'recurrent_attention_weights_only'})
        for horizon in range(output.shape[1]):
            cur = float(arrays['current_contributions'][sample, :, horizon].sum())
            hist = float(arrays['history_contributions'][sample, :, horizon].sum())
            for channel, name in enumerate(('open', 'high', 'low', 'close')):
                record = {'window_index': index, 'input_end_date': row_dates[-1], 'horizon': horizon + 1, 'channel': name,
                          'current_text_part': cur, 'history_text_part': hist, 'text_part': cur + hist}
                for key, values in zip(('film_part', 'anchor_part', 'prediction', 'reconstructed'), parts):
                    record[key] = float(values[sample, horizon, channel])
                components.append(record)
    return groups, dates, components
