"""Validation-only window/group diagnostics for the selected best checkpoint."""

import csv
from pathlib import Path

import numpy as np
import torch

from forecasting_task.data_provider.context_novelty import gate_batch_inputs
from forecasting_task.models.GroupText import TEXT_GROUPS


def write_gate_validation(loader, dataset, model, text_model, args, device, prepare_batch):
    """Keep validation order and real dataset dates; never inspect test windows."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    model.eval()
    text_model.eval()
    root = Path(args.logdir)
    table_path = root / "validation_group_diagnostics.csv"
    rows, dates, weights, contributions = [], [], [], []
    temporal_dates, temporal_components = [], []
    squared_sum, absolute_sum, elements, next_index = 0., 0., 0, 0
    horizon_squared = np.zeros(args.pred_len, dtype=np.float64)
    horizon_absolute = np.zeros(args.pred_len, dtype=np.float64)
    horizon_elements = 0
    with torch.no_grad():
        for batch in loader:
            x, text, x_mark, y_mark, decoder, target = prepare_batch(batch, device, args.pred_len, args.label_len)
            metadata = gate_batch_inputs(batch[6], text, device,
                                         use_window=getattr(args, "text_temporal_pool", "last") != "last")
            film_output = model(x, x_mark, decoder, y_mark)
            output, detail = text_model(film_output, x[:, -1], text[:, -1],
                                        price_window=x, return_details=True, **metadata)
            group_records = {}
            if getattr(args, 'text_family', None) in ('tlstm', 'last', 'memto_group', 'titans_linear_group', 'xtime_group', 'mean_price_grid'):
                from forecasting_task.group_gate_diagnostics import records
                group_records, date_records, component_records = records(dataset, batch[5].tolist(), film_output,
                    x[:, -1], output, detail, text_model.get_alpha(), args.text_family)
                temporal_dates.extend(date_records)
                temporal_components.extend(component_records)
            if "huang_beta" in detail:
                from forecasting_task.huang_diagnostics import huang_records
                group_records, date_records, component_records = huang_records(
                    dataset, batch[5].tolist(), film_output, x[:, -1], output, detail, text_model.get_alpha(),
                    mode=args.text_temporal_pool)
                temporal_dates.extend(date_records)
                temporal_components.extend(component_records)
            if "temporal_beta" in detail:
                from forecasting_task.gru_diagnostics import temporal_records
                group_records, date_records, component_records = temporal_records(
                    dataset, batch[5].tolist(), film_output, x[:, -1], output, detail, text_model.get_alpha(),
                    mode=(text_model.core.text_temporal_pool if getattr(args, 'text_family', None) == 'han'
                          else getattr(args, "text_temporal_pool", "group_gru_attention")))
                temporal_dates.extend(date_records)
                temporal_components.extend(component_records)
            error = (output - target).double()
            horizon_squared += error.square().sum((0, 2)).cpu().numpy()
            horizon_absolute += error.abs().sum((0, 2)).cpu().numpy()
            horizon_elements += error.shape[0] * error.shape[2]
            squared_sum += error.square().sum().item()
            absolute_sum += error.abs().sum().item()
            elements += error.numel()
            detail = {key: value.detach().cpu().numpy() for key, value in detail.items()}
            for sample, index in enumerate(batch[5].tolist()):
                if index != next_index:
                    raise ValueError("Validation diagnostic windows must be complete and in dataset order")
                next_index += 1
                date = dataset.window_end_date(index)
                dates.append(np.datetime64(date))
                weights.append(detail["weights"][sample])
                contributions.append(detail["contributions"][sample])
                for group_index, group in enumerate(TEXT_GROUPS):
                    record = {"window_index": index, "input_end_date": date, "group": group,
                              "present": int(detail["present"][sample, group_index]),
                              "novelty": float(detail["novelty"][sample, group_index]),
                              "novelty_used": float(detail["novelty_used"][sample, group_index]),
                              "pair_available": int(detail["pair_available"][sample, group_index]),
                              "pair_available_used": int(detail["pair_available_used"][sample, group_index]),
                              "weight": float(detail["weights"][sample, group_index])}
                    record.update(group_records.get((index, group), {}))
                    if getattr(args, 'text_family', None) is not None:
                        record.update(text_family=args.text_family, group_weight_condition=args.group_weight_condition)
                    for horizon in range(args.pred_len):
                        record[f"head_h{horizon + 1}"] = float(detail["heads"][sample, group_index, horizon])
                        record[f"contribution_h{horizon + 1}"] = float(detail["contributions"][sample, group_index, horizon])
                    rows.append(record)
    if not elements or next_index != len(dataset):
        raise ValueError("Validation diagnostics require all nonempty validation windows")
    mse, mae = squared_sum / elements, absolute_sum / elements
    if not np.isclose(mse, args.diagnostic_best_val_mse, rtol=1e-5, atol=1e-7):
        raise ValueError(f"Best checkpoint validation MSE mismatch: {mse} vs {args.diagnostic_best_val_mse}")
    with table_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    weights, contributions = np.asarray(weights), np.asarray(contributions)
    plots = []

    def plot(values, title, ylabel, filename):
        figure = Figure(figsize=(11, 4), tight_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for index, group in enumerate(TEXT_GROUPS):
            axis.plot(dates, values[:, index], label=group, linewidth=1)
        axis.set(title=title, xlabel="Last input observation date", ylabel=ylabel)
        axis.legend(loc="best", ncol=3)
        axis.grid(alpha=0.2)
        path = root / filename
        with path.open("xb") as handle:
            figure.savefig(handle, format="png", dpi=140)
        plots.append(filename)

    plot(weights, "Validation group weights (best checkpoint)", "Group weight", "validation_weights.png")
    for horizon in range(args.pred_len):
        plot(contributions[:, :, horizon], f"Validation group contributions, horizon {horizon + 1}",
             "Contribution to normalized OHLC prediction", f"validation_contributions_h{horizon + 1}.png")
    result = {"table": table_path.name, "plots": plots, "windows": next_index, "rows": len(rows),
              "mse": mse, "mae": mae, "checkpoint": "best_checkpoint.pt"}
    if getattr(args, 'text_family', None) is not None:
        result['horizon_metrics'] = {**{f'val_mse_h{i+1}': float(x/horizon_elements) for i,x in enumerate(horizon_squared)},
                                     **{f'val_mae_h{i+1}': float(x/horizon_elements) for i,x in enumerate(horizon_absolute)}}
    if temporal_dates:
        from forecasting_task.gru_diagnostics import write_temporal_tables
        result.update(write_temporal_tables(root, temporal_dates, temporal_components))
    return result
