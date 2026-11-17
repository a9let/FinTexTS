"""Validation decomposition at computed temporal/group weights of the best model."""

import csv
import json

import numpy as np
import torch

from forecasting_task.models.GroupText import TEXT_GROUPS
from forecasting_task.models.GroupGRUAttention import WINDOW


def temporal_records(dataset, indices, film_output, last_price, output, detail, alpha, mode="group_gru_attention"):
    if "date_contributions" in detail:
        torch.testing.assert_close(detail["date_contributions"].sum(dim=1), detail["contributions"], rtol=1e-5, atol=1e-6)
    film_part = (1 - alpha) * film_output
    anchor_part = alpha * last_price[:, None].expand_as(output)
    text_part = detail["contributions"].sum(dim=1)[..., None].expand_as(output)
    reconstructed = film_part + anchor_part + text_part
    torch.testing.assert_close(reconstructed, output, rtol=1e-5, atol=1e-6)
    arrays = {key: value.detach().cpu().numpy() for key, value in detail.items()}
    parts = [value.detach().cpu().numpy() for value in (film_part, anchor_part, text_part, output, reconstructed)]
    groups, dates, components = {}, [], []
    window = arrays["temporal_beta"].shape[1]
    for sample, index in enumerate(indices):
        stop = index + dataset.seq_len
        row_dates = [str(value) for value in dataset.row_dates[stop - window:stop]]
        if len(row_dates) != window or row_dates[-1] != dataset.window_end_date(index):
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
            if mode in ("last_matched", "han_h32_linear", "han_h16_linear", "han_h32_mlp", "last_han_h32_residual", "han_h16_linear_w5", "han_h16_linear_w10"):
                groups[index, group]["date_decomposition"] = "date_contributions" in arrays
            if "current_contributions" in arrays:
                groups[index, group]["current_present"] = int(arrays["current_present"][sample, group_index])
                for horizon in range(output.shape[1]):
                    groups[index, group][f"current_contribution_h{horizon + 1}"] = float(arrays["current_contributions"][sample, group_index, horizon])
                    groups[index, group][f"history_contribution_h{horizon + 1}"] = float(arrays["history_contributions"][sample, group_index, horizon])
            for row, date in enumerate(row_dates):
                record = {"window_index": index, "input_end_date": row_dates[-1], "group": group,
                          "date": date, "age": float(arrays["temporal_ages"][row])}
                if mode in ("han_h16_linear_w5", "han_h16_linear_w10"):
                    record["original_position_index"] = 20 - window + row
                if "temporal_gaps" in arrays:
                    record["gap_scaled"] = float(arrays["temporal_gaps"][sample, row, group_index])
                record.update(present=int(mask[row]), beta=float(beta[row]),
                              group_weight=float(arrays["weights"][sample, group_index]))
                if "temporal_gaps" not in arrays:
                    record["representation"] = "contextual_BiGRU_state_projected_64_to_384"
                    if mode in ("last_matched", "han_h32_linear", "han_h16_linear", "han_h32_mlp", "last_han_h32_residual", "han_h16_linear_w5", "han_h16_linear_w10"):
                        record["representation"] = "current_summary" if mode == "last_matched" else "contextual_BiGRU_state"
                        record["decomposition_kind"] = "linear_date_contributions" if "date_contributions" in arrays else "attention_weights_only"
                if "temporal_source_present" in arrays:
                    record["source_present"] = int(arrays["temporal_source_present"][sample, row, group_index])
                for horizon in range(output.shape[1]):
                    if "date_contributions" in arrays:
                        record[f"date_contribution_h{horizon + 1}"] = float(arrays["date_contributions"][sample, row, group_index, horizon])
                    if "current_contributions" in arrays:
                        record[f"current_contribution_h{horizon + 1}"] = float(arrays["current_contributions"][sample, group_index, horizon]) if row == window - 1 else 0.
                        record[f"history_date_contribution_h{horizon + 1}"] = float(arrays["history_date_contributions"][sample, row, group_index, horizon])
                    record[f"group_contribution_h{horizon + 1}"] = float(arrays["contributions"][sample, group_index, horizon])
                dates.append(record)
        for horizon in range(output.shape[1]):
            for channel, name in enumerate(("open", "high", "low", "close")):
                record = {"window_index": index, "input_end_date": row_dates[-1], "horizon": horizon + 1, "channel": name}
                for key, values in zip(("film_part", "anchor_part", "text_part", "prediction", "reconstructed"), parts):
                    record[key] = float(values[sample, horizon, channel])
                if "current_contributions" in arrays:
                    record["current_text_part"] = float(arrays["current_contributions"][sample, :, horizon].sum())
                    record["history_text_part"] = float(arrays["history_contributions"][sample, :, horizon].sum())
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
