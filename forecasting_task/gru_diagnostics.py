"""Validation decomposition at computed temporal/group weights of the best model."""

import csv
import json

import numpy as np
import torch

from forecasting_task.models.GroupText import TEXT_GROUPS
from forecasting_task.models.GroupGRUAttention import WINDOW


def temporal_records(dataset, indices, film_output, last_price, output, detail, alpha, mode="group_gru_attention"):
    torch.testing.assert_close(detail["date_contributions"].sum(dim=1), detail["contributions"], rtol=1e-5, atol=1e-6)
    film_part = (1 - alpha) * film_output
    anchor_part = alpha * last_price[:, None].expand_as(output)
    text_part = detail["contributions"].sum(dim=1)[..., None].expand_as(output)
    reconstructed = film_part + anchor_part + text_part
    torch.testing.assert_close(reconstructed, output, rtol=1e-5, atol=1e-6)
    arrays = {key: value.detach().cpu().numpy() for key, value in detail.items()}
    parts = [value.detach().cpu().numpy() for value in (film_part, anchor_part, text_part, output, reconstructed)]
    groups, dates, components = {}, [], []
    for sample, index in enumerate(indices):
        stop = index + dataset.seq_len
        row_dates = [str(value) for value in dataset.row_dates[stop - WINDOW:stop]]
        if len(row_dates) != WINDOW or row_dates[-1] != dataset.window_end_date(index):
            raise ValueError("GRU diagnostic dates do not match the causal input window")
        for group_index, group in enumerate(TEXT_GROUPS):
            mask = arrays["temporal_present"][sample, :, group_index]
            beta = arrays["temporal_beta"][sample, :, group_index]
            if np.any(beta[~mask] != 0) or not np.isclose(beta.sum(), float(mask.any()), rtol=1e-6, atol=1e-7):
                raise ValueError("Invalid masked temporal attention")
            groups[index, group] = {"text_temporal_pool": mode,
                                    "text_window_dates": json.dumps(row_dates),
                                    "text_window_present": json.dumps(mask.astype(int).tolist()),
                                    "text_window_beta": json.dumps(beta.tolist()),
                                    "available_dates": int(mask.sum())}
            for row, date in enumerate(row_dates):
                record = {"window_index": index, "input_end_date": row_dates[-1], "group": group,
                          "date": date, "age": float(arrays["temporal_ages"][row])}
                if "temporal_gaps" in arrays:
                    record["gap_scaled"] = float(arrays["temporal_gaps"][sample, row, group_index])
                record.update(present=int(mask[row]), beta=float(beta[row]),
                              group_weight=float(arrays["weights"][sample, group_index]))
                if "temporal_gaps" not in arrays:
                    record["representation"] = "contextual_BiGRU_state_projected_64_to_384"
                for horizon in range(output.shape[1]):
                    record[f"date_contribution_h{horizon + 1}"] = float(arrays["date_contributions"][sample, row, group_index, horizon])
                    record[f"group_contribution_h{horizon + 1}"] = float(arrays["contributions"][sample, group_index, horizon])
                dates.append(record)
        for horizon in range(output.shape[1]):
            for channel, name in enumerate(("open", "high", "low", "close")):
                record = {"window_index": index, "input_end_date": row_dates[-1], "horizon": horizon + 1, "channel": name}
                for key, values in zip(("film_part", "anchor_part", "text_part", "prediction", "reconstructed"), parts):
                    record[key] = float(values[sample, horizon, channel])
                components.append(record)
    return groups, dates, components


def write_temporal_tables(root, dates, components):
    result = {}
    for key, filename, rows in (("date_table", "validation_text_dates.csv", dates),
                                ("components_table", "validation_prediction_components.csv", components)):
        with (root / filename).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result[key] = filename
    return result
