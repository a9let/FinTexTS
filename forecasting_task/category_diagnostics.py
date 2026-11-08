"""Exact arithmetic decomposition from the selected validation checkpoint."""

import csv
from pathlib import Path

import torch

from forecasting_task.category_data import CATEGORIES, PROVENANCE, GROUPS

RTOL, ATOL = 1e-5, 1e-6  # Float32 Linear/reduction rounding; no inverse units.


def category_records(detail, price, last, output, indices, dataset, args, alpha):
    news = detail["category_contributions"]
    if not torch.allclose(news.sum(2), detail["contributions"][:, :4], rtol=RTOL, atol=ATOL):
        raise ValueError("Category contributions do not reconstruct group contribution, including bias/masks")
    film_part = (1-alpha)*price
    anchor_part = alpha*last[:, None, :].expand_as(price)
    text_part = detail["contributions"].sum(1).unsqueeze(-1).expand_as(price)
    reconstructed = film_part+anchor_part+text_part
    if not torch.allclose(output, reconstructed, rtol=RTOL, atol=ATOL):
        raise ValueError("Saved components do not reconstruct OHLC broadcast prediction")
    categories, components = [], []
    for b, index in enumerate(indices):
        date = dataset.window_end_date(index)
        common = {"window_index": index, "date": date, "ticker": dataset.ticker, "seed": args.seed,
                  "category_pool": args.category_pool, "dataset_revision": PROVENANCE["dataset_revision"]}
        for g, group in enumerate(GROUPS):
            names = CATEGORIES[group] if g < 4 else ["filing_group_mean"]
            for k, name in enumerate(names):
                record = {**common, "group": group, "source_column": name,
                          "source_identifier": f"{PROVENANCE['dataset_revision']}|{dataset.ticker}|{date}|{name}",
                          "present": int(detail["category_present"][b, g, k]) if g < 4 else int(detail["present"][b, 4]),
                          "beta": float(detail["category_beta"][b, g, k]) if g < 4 else int(detail["present"][b, 4]),
                          "group_weight": float(detail["weights"][b, g])}
                for h in range(args.pred_len):
                    record[f"head_h{h+1}"] = float(detail["category_heads"][b, g, k, h] if g < 4 else detail["heads"][b, 4, h])
                    record[f"contribution_h{h+1}"] = float(news[b, g, k, h] if g < 4 else detail["contributions"][b, 4, h])
                    record[f"group_contribution_h{h+1}"] = float(detail["contributions"][b, g, h])
                categories.append(record)
        for h in range(args.pred_len):
            for c, channel in enumerate(("open", "high", "low", "close")):
                components.append({**common, "horizon": h+1, "channel": channel,
                                   "film_component": float(film_part[b, h, c]), "last_price_component": float(anchor_part[b, h, c]),
                                   "text_component": float(text_part[b, h, c]), "prediction": float(output[b, h, c]),
                                   "reconstructed": float(reconstructed[b, h, c])})
    return categories, components


def save_category_records(root, rows, components):
    for name, records in (("validation_category_diagnostics.csv", rows), ("validation_prediction_components.csv", components)):
        with (Path(root)/name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return {"categories": "validation_category_diagnostics.csv", "components": "validation_prediction_components.csv",
            "reconstruction_tolerance": {"rtol": RTOL, "atol": ATOL}, "checkpoint": "best_checkpoint.pt"}
