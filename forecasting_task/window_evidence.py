"""Normalized OHLC window fingerprints; no model inference or random operations."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from forecasting_task.models.GroupText import TEXT_GROUPS


def _bytes(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype="<f4").tobytes(order="C")


def window_evidence(dataset):
    inputs, targets, dates, target_dates = (hashlib.sha256() for _ in range(4))
    if not len(dataset):
        raise ValueError("Cannot fingerprint an empty evaluation split")
    for index in range(len(dataset)):
        batch = dataset[index]
        inputs.update(_bytes(batch[0]))
        targets.update(_bytes(batch[1][-dataset.pred_len:]))
        dates.update((dataset.window_end_date(index) + "\n").encode("utf-8"))
        start = index + dataset.seq_len
        target_dates.update(("|".join(dataset.row_dates[start:start + dataset.pred_len]) + "\n").encode("utf-8"))
    return {"windows": len(dataset), "input_shape": [dataset.seq_len, 4],
            "target_shape": [dataset.pred_len, 4], "inputs_sha256": inputs.hexdigest(),
            "targets_sha256": targets.hexdigest(), "dates_sha256": dates.hexdigest(),
            "target_dates_sha256": target_dates.hexdigest(),
            "first_input_end_date": dataset.window_end_date(0),
            "last_input_end_date": dataset.window_end_date(len(dataset) - 1), "dtype": "float32"}


def check_saved_test_windows(path, expected):
    """Read only normalized inputs/ground_truths from trusted training artifacts."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, shape_key, hash_key in (("inputs", "input_shape", "inputs_sha256"),
                                      ("ground_truths", "target_shape", "targets_sha256")):
        values = payload.get(key)
        if values is None:
            raise ValueError(f"{path}: missing normalized {key}; actual keys={sorted(payload)}")
        if list(values.shape) != [expected["windows"], *expected[shape_key]]:
            raise ValueError(f"{path}: normalized {key} window shape mismatch")
        if hashlib.sha256(_bytes(values)).hexdigest() != expected[hash_key]:
            raise ValueError(f"{path}: normalized {key} windows/targets mismatch")


def check_validation_dates(path, expected):
    dates = hashlib.sha256()
    window = 0
    group_index = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["window_index"]) != window or row["group"] != TEXT_GROUPS[group_index]:
                raise ValueError(f"{path}: invalid validation window/group order")
            if group_index == 0:
                current_date = row["input_end_date"]
                dates.update((current_date + "\n").encode("utf-8"))
            elif row["input_end_date"] != current_date:
                raise ValueError(f"{path}: group dates disagree within a window")
            group_index += 1
            if group_index == len(TEXT_GROUPS):
                window += 1
                group_index = 0
    if group_index or window != expected["windows"] or dates.hexdigest() != expected["dates_sha256"]:
        raise ValueError(f"{path}: validation window dates mismatch")
