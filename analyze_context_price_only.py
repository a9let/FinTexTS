"""Read-only comparison of matched saved runs; no models or training entry points."""

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

from analyze_persistence_baseline import Run, load_metadata, check_configuration, check_scalers, model_metrics


RUNS = Path("/auto/datasets/kondyurin/FinTexTS/runs")
DEFAULT_RESULTS = RUNS / "context_price_only_10_v1"
DEFAULT_REFERENCE = RUNS / "context_novelty_100_v1"
DEFAULT_BASELINE = RUNS / "film_compare_50ep_v1"
TICKERS = tuple("AAPL ABBV ABT ACN ADBE ADI AMAT AMD AMGN AMZN".split())
SEEDS = (7, 17, 27)
VARIANTS = ("context_price_only", "context_no_novelty", "original_baseline")
METRICS = ("val_mse", "test_mse", "test_mae")
COMMON = {"seq_len": 64, "label_len": 16, "pred_len": 3, "batch_size": 32,
          "num_epoch": 50, "patience": 10, "lr": 0.001}
PRICE_FIELDS = ("task_name", "e_layers", "output_attention", "enc_in", "dec_in", "c_out")
GROUPS = ["macro", "sector", "targetCompany", "relatedCompany", "filing"]


def validate_config(run, summary, args, reference=None):
    check_configuration(run, args, reference)
    for key, value in COMMON.items():
        if args.get(key) != value:
            raise ValueError(f"{key} mismatch: expected {value}, found {args.get(key)}")
    for key in PRICE_FIELDS:
        if key not in args:
            raise ValueError(f"Missing FiLM configuration field: {key}")
        if reference is not None and args[key] != reference[key]:
            raise ValueError(f"FiLM {key} mismatch")
    if run.variant != "original_baseline":
        for source in (summary, args):
            if source.get("group_gate") != run.variant or source.get("fixed_alpha") != 0.1:
                raise ValueError("Condition/fixed alpha disagrees with requested experiment")
            if source.get("text_weight_mode") != "fixed":
                raise ValueError("Expected fixed text alpha")
        if (args.get("fusion_mode") != "blend" or args.get("text_num_layers") != 1
                or args.get("used_col_prefixes") != GROUPS or args.get("text_weight") != 0.1
                or args.get("group_weight_mode") != run.variant
                or args.get("text_model_config", {}).get("group_gate") != run.variant):
            raise ValueError("Group-head/fusion configuration mismatch")
        if reference is not None:
            for key in ("novelty_policy", "sampling_policy", "gate_structure", "text_mlp_structure"):
                if key not in args or key not in reference or args[key] != reference[key]:
                    raise ValueError(f"Controlled experiment {key} mismatch or unavailable")


def mean_complete(values, count):
    return float(np.mean(values)) if len(values) == count and all(x is not None for x in values) else None


def aggregate(rows):
    per_ticker, by_seed, across, paired, wins = [], [], [], [], []
    for variant in VARIANTS:
        cohort = [row for row in rows if row["variant"] == variant]
        if len(cohort) != 30 or {(r["ticker"], r["seed"]) for r in cohort} != set(itertools.product(TICKERS, SEEDS)):
            raise ValueError(f"Incomplete/duplicate cohort for {variant}")
        for ticker in TICKERS:
            group = [r for r in cohort if r["ticker"] == ticker]
            record = {"variant": variant, "ticker": ticker, "comparable": all(r["comparable"] for r in group)}
            record.update({metric: mean_complete([r[metric] for r in group], 3) for metric in METRICS})
            per_ticker.append(record)
        for seed in SEEDS:
            group = [r for r in cohort if r["seed"] == seed]
            record = {"variant": variant, "seed": seed, "comparable_tickers": sum(r["comparable"] for r in group)}
            for metric in METRICS:
                record[metric] = mean_complete([r[metric] for r in group], 10)
                record[metric + "_available_tickers"] = sum(r[metric] is not None for r in group)
            by_seed.append(record)
        for metric in METRICS:
            group = [r for r in by_seed if r["variant"] == variant]
            values = [r[metric] for r in group]
            complete = all(value is not None for value in values)
            across.append({"variant": variant, "metric": metric, "mean": mean_complete(values, 3),
                           "sample_std": float(np.std(values, ddof=1)) if complete else None,
                           "std_ddof": 1, "complete_seeds": sum(x is not None for x in values),
                           "comparable": all(r["comparable_tickers"] == 10 for r in group)})
    lookup = {(row["variant"], row["ticker"]): row for row in per_ticker}
    for left, right in itertools.combinations(VARIANTS, 2):
        for metric in METRICS:
            eligible = []
            for ticker in TICKERS:
                a, b = lookup[left, ticker], lookup[right, ticker]
                comparable = a["comparable"] and b["comparable"] and a[metric] is not None and b[metric] is not None
                difference = a[metric] - b[metric] if comparable else None
                record = {"left": left, "right": right, "ticker": ticker, "metric": metric,
                          "left_seed_mean": a[metric], "right_seed_mean": b[metric],
                          "left_minus_right": difference, "comparable": comparable}
                paired.append(record)
                if comparable:
                    eligible.append(difference)
            wins.append({"left": left, "right": right, "metric": metric,
                         "left_wins": sum(x < 0 for x in eligible), "right_wins": sum(x > 0 for x in eligible),
                         "ties": sum(x == 0 for x in eligible), "comparable_tickers": len(eligible), "expected_tickers": 10})
    scores = {row["variant"]: row["mean"] for row in across
              if row["metric"] == "val_mse" and row["comparable"] and row["mean"] is not None}
    best = min(scores.values()) if scores else None
    selection = {"selection_metric": "val_mse", "aggregation": "mean of three seed means over the same ten tickers",
                 "validation_scores": scores, "selected_variants": [v for v, score in scores.items() if score == best],
                 "excluded_variants": [v for v in VARIANTS if v not in scores], "test_used_for_selection": False}
    return {"per_ticker": per_ticker, "by_seed": by_seed, "across_seeds": across,
            "paired_by_ticker": paired, "wins": wins}, selection


def analyze(results_root=DEFAULT_RESULTS, reference_root=DEFAULT_REFERENCE, baseline_root=DEFAULT_BASELINE, output_dir=None):
    output = Path(output_dir) if output_dir else Path(results_root) / "diagnostics" / "comparison_10"
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    roots = {VARIANTS[0]: Path(results_root) / "full" / VARIANTS[0],
             VARIANTS[1]: Path(reference_root) / "full" / VARIANTS[1],
             VARIANTS[2]: Path(baseline_root) / "full" / "baseline"}
    rows, issues, physical = [], [], set()
    reference_price_args = None
    for ticker in TICKERS:
        ticker_reference = None
        for variant in VARIANTS:
            for seed in SEEDS:
                run = Run(variant, ticker, seed, roots[variant] / ticker / f"seed{seed}" / "output")
                row = {"variant": variant, "ticker": ticker, "seed": seed, "output": str(run.output),
                       "comparable": False, "comparison_reason": ""}
                values = dict.fromkeys(METRICS)
                sources = {metric + "_source": "unavailable" for metric in METRICS}
                try:
                    summary = json.loads((run.output / "summary.json").read_text(encoding="utf-8"))
                    all_values, all_sources = model_metrics(summary)
                    values.update({metric: all_values[metric] for metric in METRICS})
                    sources.update({metric + "_source": all_sources[metric + "_source"] for metric in METRICS})
                    for name in ("summary.json", "best_checkpoint.pt"):
                        path = run.output / name
                        stat = path.stat()
                        if not stat.st_size:
                            raise ValueError(f"Empty artifact: {path}")
                        identity = (stat.st_dev, stat.st_ino)
                        if identity in physical:
                            raise ValueError(f"Duplicate physical artifact: {path}")
                        physical.add(identity)
                    _, args, scaler, split = load_metadata(run)
                    validate_config(run, summary, args, ticker_reference[0] if ticker_reference else reference_price_args)
                    if ticker_reference is None:
                        ticker_reference = args, scaler, split
                    else:
                        check_scalers(ticker_reference[1], scaler, str(run.output))
                        if split != ticker_reference[2]:
                            raise ValueError("Split policy mismatch")
                    if reference_price_args is None:
                        reference_price_args = args
                    row["comparable"] = True
                    if variant != "original_baseline" and any(values[m] is None for m in METRICS):
                        raise ValueError("Missing required model metrics")
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                    if variant != "original_baseline":
                        raise ValueError(f"Cannot compare primary run {run.output}: {error}") from error
                    row["comparable"] = False
                    row["comparison_reason"] = str(error)
                    issues.append({"variant": variant, "ticker": ticker, "seed": seed, "reason": str(error)})
                for metric, value in values.items():
                    if value is None:
                        issues.append({"variant": variant, "ticker": ticker, "seed": seed, "reason": f"Unavailable {metric}"})
                rows.append({**row, **values, **sources})
    tables, selection = aggregate(rows)
    tables = {"per_run": rows, **tables, "issues": issues}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        fields = list(dict.fromkeys(key for row in records for key in row)) or ["variant", "ticker", "seed", "reason"]
        with (output / f"{name}.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    with (output / "selection.json").open("x", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, allow_nan=False)
    for name in ("by_seed", "across_seeds", "paired_by_ticker", "wins"):
        print(name.upper())
        records = tables[name]
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(selection, allow_nan=False))
    print(f"Reports: {output}")
    return tables, selection


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline_root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    analyze(args.results_root, args.reference_root, args.baseline_root, args.output_dir)


if __name__ == "__main__":
    cli()
