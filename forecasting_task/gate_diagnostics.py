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
            output, detail = text_model(model(x, x_mark, decoder, y_mark), x[:, -1], text[:, -1],
                                        price_window=x, return_details=True, **metadata)
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
                dates.append(np.datetime64(date))
                weights.append(detail["weights"][sample])
                contributions.append(detail["contributions"][sample])
                for group_index, group in enumerate(TEXT_GROUPS):
                    record = {"window_index": index, "input_end_date": date, "group": group,
                              "present": int(detail["present"][sample, group_index]),
                              "novelty": float(detail["novelty"][sample, group_index]),
                              "pair_available": int(detail["pair_available"][sample, group_index]),
                              "weight": float(detail["weights"][sample, group_index])}
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
