"""Raw group contributions and calendar gaps; never attribute T-LSTM output to dates."""
import torch
from forecasting_task.models.GroupText import TEXT_GROUPS


def records(dataset, indices, film, last, output, detail, alpha, family):
    film_part, anchor = (1-alpha)*film, alpha*last[:, None].expand_as(output)
    text = detail['contributions'].sum(1)[..., None].expand_as(output)
    torch.testing.assert_close(output, film_part+anchor+text, rtol=1e-5, atol=1e-6)
    a = {k: v.detach().cpu().numpy() for k, v in detail.items()}
    parts = [x.detach().cpu().numpy() for x in (film_part, anchor, text, output, film_part+anchor+text)]
    groups, dates, components = {}, [], []
    window = a['tlstm_present'].shape[1] if family == 'tlstm' else a['memory_present'].shape[1] if 'memory_present' in a else 1
    if 'mean_present' in a:
        window = a['mean_present'].shape[1]
    for sample, index in enumerate(indices):
        end = index + dataset.seq_len
        extended = family == 'mean_price_grid' and window > dataset.seq_len
        window_dates = dataset.text_window_dates(index) if extended else dataset.row_dates[end-window:end]
        for g, group in enumerate(TEXT_GROUPS):
            groups[index, group] = {'date_decomposition': False, 'text_family': family}
            if extended:
                groups[index, group]['text_window_length'] = len(window_dates)
            if 'expert_weights' in a:
                groups[index, group].update({f'expert_weight_{name}': float(weight)
                    for name, weight in zip(('normal','moderate','extreme'),a['expert_weights'][sample])})
            for row, date in enumerate(window_dates, start=window-len(window_dates)):
                item = {'window_index': index, 'input_end_date': dataset.window_end_date(index), 'group': group,
                        'date': str(date), 'present': int(a['tlstm_present'][sample,row,g] if family == 'tlstm' else a['present'][sample,g])}
                if family == 'tlstm':
                    item.update(delta_days=float(a['tlstm_delta'][sample,row,g]), decay=float(a['tlstm_decay'][sample,row,g]))
                if 'mean_present' in a:
                    item['present'] = int(a['mean_present'][sample,row,g])
                if 'memory_present' in a:
                    names = ('lsd','isd','anomaly') if family == 'memto_group' else ('gradient_norm','alpha','eta','theta')
                    item['present'] = int(a['memory_present'][sample,row,g])
                    item.update({name:float(v) for name,v in zip(names,a['memory_date_values'][sample,row,g])})
                dates.append(item)
        for h in range(output.shape[1]):
            for c, channel in enumerate(('open','high','low','close')):
                item = {'window_index': index, 'input_end_date': dataset.window_end_date(index), 'horizon': h+1, 'channel': channel}
                item.update({k: float(v[sample,h,c]) for k,v in zip(('film_part','anchor_part','text_part','prediction','reconstructed'),parts)})
                components.append(item)
    return groups, dates, components
