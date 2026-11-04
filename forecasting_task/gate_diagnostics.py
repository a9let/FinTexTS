"""Validation-only window/group diagnostics for the selected best checkpoint."""

import csv
from pathlib import Path

import numpy as np
import torch

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
    squared_sum, absolute_sum, elements, next_index = 0., 0., 0, 0
    with torch.no_grad():
        for batch in loader:
            x, text, x_mark, y_mark, decoder, target = prepare_batch(batch, device, args.pred_len, args.label_len)
            metadata = {key: value.float().to(device) for key, value in batch[6].items()}
            price_output = model(x, x_mark, decoder, y_mark)
            output, detail = text_model(price_output, x[:, -1], text,
                                        price_window=x, return_details=True, **metadata)
            if text_model.text_temporal_pool == "attention5":
                if not torch.allclose(detail["daily_contributions"].sum(2), detail["contributions"], rtol=1e-5, atol=1e-6):
                    raise ValueError("Attention date contributions do not reconstruct group contributions")
                reconstructed = ((1-text_model.fixed_alpha)*price_output + text_model.fixed_alpha*x[:, -1:]
                                 + detail["contributions"].sum(1).unsqueeze(-1))
                if not torch.allclose(reconstructed, output, rtol=1e-5, atol=1e-6):
                    raise ValueError("Attention contributions do not reconstruct prediction")
            error = (output - target).double()
            squared_sum += error.square().sum().item()
            absolute_sum += error.abs().sum().item()
            elements += error.numel()
            detail = {key: value.detach().cpu().numpy() for key, value in detail.items()}
            for sample, index in enumerate(batch[5].tolist()):
                if index != next_index:
                    raise ValueError("Validation diagnostic windows must be complete and in dataset order")
                next_index += 1
                date = dataset.window_end_date(index)
                pool = getattr(text_model, "text_temporal_pool", "last")
                text_start, text_end = dataset.text_window_bounds(index, 1 if pool == "last" else 5)
                dates.append(np.datetime64(date))
                weights.append(detail["weights"][sample])
                contributions.append(detail["contributions"][sample])
                for group_index, group in enumerate(TEXT_GROUPS):
                    record = {"window_index": index, "input_end_date": date, "group": group,
                              "text_temporal_pool": pool, "text_window_start_date": text_start,
                              "text_window_end_date": text_end,
                              "available_dates": int(detail["available_count"][sample, group_index]),
                              "present": int(detail["present"][sample, group_index]),
                              "novelty": float(detail["novelty"][sample, group_index]),
                              "novelty_used": float(detail["novelty_used"][sample, group_index]),
                              "pair_available": int(detail["pair_available"][sample, group_index]),
                              "pair_available_used": int(detail["pair_available_used"][sample, group_index]),
                              "weight": float(detail["weights"][sample, group_index])}
                    if pool == "attention5":
                        end = int(index) + dataset.seq_len
                        window_dates = dataset.row_dates[end-5:end]
                        for day, day_date in enumerate(window_dates):
                            record[f"date_d{day+1}"] = str(day_date)
                            record[f"age_d{day+1}"] = int(detail["row_ages"][day])
                            record[f"present_d{day+1}"] = int(detail["daily_present"][sample, group_index, day])
                            record[f"beta_d{day+1}"] = float(detail["temporal_weights"][sample, group_index, day])
                            for key in range(5):
                                record[f"attention_A_q{day+1}_k{key+1}"] = float(detail["attention"][sample, group_index, day, key])
                            for horizon in range(args.pred_len):
                                record[f"date_contribution_d{day+1}_h{horizon+1}"] = float(detail["daily_contributions"][sample, group_index, day, horizon])
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
    return {"table": table_path.name, "plots": plots, "windows": next_index, "rows": len(rows),
            "mse": mse, "mae": mae, "checkpoint": "best_checkpoint.pt"}
