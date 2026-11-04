"""Read-only comparison of matched saved runs; no models or training entry points."""

import argparse
import csv
import itertools
import json
import pickle
from pathlib import Path
import sys

import numpy as np

from analyze_persistence_baseline import Run, load_metadata, check_scalers, model_metrics
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset
from forecasting_task.window_evidence import window_evidence, check_saved_test_windows, check_validation_dates
from forecasting_task.models.GroupText import validate_text_half_life


RUNS = Path("/auto/datasets/kondyurin/FinTexTS/runs")
DEFAULT_RESULTS = RUNS / "context_price_decay5_10_v1"
DEFAULT_LAST = RUNS / "context_price_only_10_v1"
DEFAULT_MEAN5 = RUNS / "context_price_mean5_10_v1"
DEFAULT_BASELINE = RUNS / "film_compare_50ep_v1"
TICKERS = tuple("AAPL ABBV ABT ACN ADBE ADI AMAT AMD AMGN AMZN".split())
SEEDS = (7, 17, 27)
VARIANTS = ("decay5", "mean5", "last", "original_baseline")
METRICS = ("val_mse", "test_mse", "test_mae")
COMMON = {"seq_len": 64, "label_len": 16, "pred_len": 3, "batch_size": 32,
          "num_epoch": 50, "patience": 10, "lr": 0.001}
PRICE_FIELDS = ("task_name", "e_layers", "output_attention", "enc_in", "dec_in", "c_out")
GROUPS = ["macro", "sector", "targetCompany", "relatedCompany", "filing"]


def validate_config(run, summary, args, reference=None):
    from analyze_context_price_only import validate_config as validate_original
    gate = "context_price_only" if run.variant in ("decay5", "mean5", "last") else run.variant
    validate_original(Run(gate, run.ticker, run.seed, run.output), summary, args, reference)
    expected = run.variant if run.variant in ("decay5", "mean5") else "last"
    for source in (args, summary, args.get("text_model_config", {})):
        if source.get("text_temporal_pool", "last") != expected:
            raise ValueError(f"text_temporal_pool mismatch for {run.variant}")
        if run.variant == "decay5" and validate_text_half_life(source.get("text_half_life")) != 2.0:
            raise ValueError("Expected text_half_life=2.0 for this experiment")


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
                          "left_minus_right": difference, "comparable": comparable,
                          "relative_change_pct": 100 * difference / b[metric] if comparable and b[metric] != 0 else None}
                paired.append(record)
                if comparable:
                    eligible.append(difference)
            wins.append({"left": left, "right": right, "metric": metric,
                         "left_wins": sum(x < 0 for x in eligible), "right_wins": sum(x > 0 for x in eligible),
                         "ties": sum(x == 0 for x in eligible), "comparable_tickers": len(eligible), "expected_tickers": 10})
    relative_seed_means = []
    seed_lookup = {(r["variant"], r["seed"]): r for r in by_seed}
    for left, right in itertools.combinations(VARIANTS, 2):
        for metric in METRICS:
            changes = []
            for seed in SEEDS:
                a, b = seed_lookup[left, seed], seed_lookup[right, seed]
                valid = (a["comparable_tickers"] == b["comparable_tickers"] == 10
                         and a[metric] is not None and b[metric] is not None and b[metric] != 0)
                value = 100 * (a[metric] - b[metric]) / b[metric] if valid else None
                relative_seed_means.append({"left": left, "right": right, "metric": metric, "seed": seed,
                                            "relative_change_pct": value})
                changes.append(value)
            relative_seed_means.append({"left": left, "right": right, "metric": metric, "seed": "across_seeds",
                                        "relative_change_pct": mean_complete(changes, 3),
                                        "relative_change_of_overall_means_pct":
                                            100 * (np.mean([seed_lookup[left, s][metric] for s in SEEDS])
                                                   - np.mean([seed_lookup[right, s][metric] for s in SEEDS]))
                                            / np.mean([seed_lookup[right, s][metric] for s in SEEDS])
                                            if all(v is not None for v in changes) else None,
                                        "sample_std": float(np.std(changes, ddof=1)) if all(v is not None for v in changes) else None})
    scores = {row["variant"]: row["mean"] for row in across
              if row["variant"] in ("decay5", "mean5") and row["metric"] == "val_mse"
              and row["comparable"] and row["mean"] is not None}
    best = min(scores.values()) if scores else None
    selection = {"selection_metric": "val_mse", "aggregation": "mean of three seed means over the same ten tickers",
                 "validation_scores": scores, "selected_variants": [v for v, score in scores.items() if score == best],
                 "excluded_variants": [v for v in ("decay5", "mean5") if v not in scores],
                 "benchmarks": ["last", "original_baseline"], "test_used_for_selection": False}
    return {"per_ticker": per_ticker, "by_seed": by_seed, "across_seeds": across,
            "paired_by_ticker": paired, "wins": wins, "relative_by_seed": relative_seed_means}, selection


def replay_windows(ticker, args, data_root):
    evidence, scalers = {}, {}
    for split in ("val", "test"):
        base = Dataset_Custom(args["seq_len"], args["label_len"], args["pred_len"], split, GROUPS,
                              root_path=str(data_root), data_path=ticker + ".parquet")
        dataset = ContextNoveltyDataset(base)
        evidence[split] = window_evidence(dataset)
        scalers[split] = {name: np.asarray(getattr(dataset.scaler, name + "_")) for name in ("mean", "scale", "var")}
    return evidence, scalers


VERIFICATION_FLAGS = ("val_fingerprint_verified", "test_fingerprint_verified", "saved_test_arrays_verified", "saved_val_dates_verified")


def verify_windows(run, args, summary, replayed):
    record = dict.fromkeys(VERIFICATION_FLAGS, False)
    record.update(window_verification="partial", window_verification_detail="")
    notes = []
    try:
        sources = {"checkpoint_args": args.get("window_evidence"), "summary": summary.get("window_evidence")}
        sidecar = run.output / "window_evidence.json"
        if sidecar.exists():
            sources["sidecar"] = json.loads(sidecar.read_text(encoding="utf-8"))
        for source, saved in sources.items():
            if saved is None:
                continue
            if not isinstance(saved, dict):
                raise ValueError(f"Invalid {source} window fingerprints")
            for split in ("val", "test"):
                if split not in saved:
                    notes.append(f"{source}: missing {split} fingerprint")
                elif saved[split] != replayed[split]:
                    record[f"{split}_fingerprint_verified"] = False
                    raise ValueError(f"{source}: {split} window fingerprint mismatch")
                else:
                    record[f"{split}_fingerprint_verified"] = True
        io_path = run.output / "test_io.pt"
        if io_path.exists():
            try:
                check_saved_test_windows(io_path, replayed["test"])
                record["saved_test_arrays_verified"] = True
            except ValueError as error:
                if "missing normalized" not in str(error):
                    raise
                notes.append(str(error))
        else:
            notes.append("saved normalized test arrays unavailable")
        date_path = run.output / "validation_group_diagnostics.csv"
        if date_path.exists():
            check_validation_dates(date_path, replayed["val"])
            record["saved_val_dates_verified"] = True
        else:
            notes.append("saved validation input-end dates unavailable")
        if not record["val_fingerprint_verified"]:
            notes.append("historical validation target values not confirmed; replayed targets are not saved targets")
        if not record["test_fingerprint_verified"]:
            notes.append("saved test date fingerprints unavailable; normalized arrays checked only when present")
        if all(record[flag] for flag in VERIFICATION_FLAGS):
            record["window_verification"] = "full"
            notes.append("val/test saved fingerprints, normalized test arrays and saved validation dates match replay")
    except (ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        record["window_verification"] = "mismatch"
        notes.append(str(error))
    except OSError as error:
        notes.append(f"verification evidence unavailable: {error}")
    record["window_verification_detail"] = "; ".join(notes)
    return record


def verification_summary(rows):
    summary = []
    for variant in VARIANTS:
        cohort = [r for r in rows if r["variant"] == variant]
        for status in ("full", "partial", "mismatch", "not_checked"):
            group = [r for r in cohort if r["window_verification"] == status]
            summary.append({"variant": variant, "window_verification": status, "runs": len(group),
                            **{flag: sum(row[flag] for row in group) for flag in VERIFICATION_FLAGS}})
    return summary


def analyze(results_root=DEFAULT_RESULTS, mean5_root=DEFAULT_MEAN5, last_root=DEFAULT_LAST,
            baseline_root=DEFAULT_BASELINE, data_root=Path("/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts"), output_dir=None):
    output = Path(output_dir) if output_dir else Path(results_root) / "diagnostics" / "comparison_10"
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    roots = {"decay5": Path(results_root) / "full/context_price_only", "mean5": Path(mean5_root) / "full/context_price_only",
             "last": Path(last_root) / "full/context_price_only", "original_baseline": Path(baseline_root) / "full/baseline"}
    rows, issues, physical, window_rows = [], [], set(), []
    price_reference = None
    for ticker in TICKERS:
        reference = None
        replayed = actual_scalers = None
        for variant in VARIANTS:
            for seed in SEEDS:
                run = Run(variant, ticker, seed, roots[variant] / ticker / f"seed{seed}" / "output")
                row = {"variant": variant, "ticker": ticker, "seed": seed, "output": str(run.output),
                       "comparable": False, "comparison_reason": "", "window_verification": "not_checked",
                       "window_verification_detail": "configuration/scaler checks not completed",
                       **dict.fromkeys(VERIFICATION_FLAGS, False)}
                values = dict.fromkeys(METRICS)
                sources = {metric + "_source": "unavailable" for metric in METRICS}
                try:
                    summary = json.loads((run.output / "summary.json").read_text(encoding="utf-8"))
                    all_values, all_sources = model_metrics(summary)
                    values = {metric: all_values[metric] for metric in METRICS}
                    sources = {metric + "_source": all_sources[metric + "_source"] for metric in METRICS}
                    for name in ("summary.json", "best_checkpoint.pt", "test_io.pt"):
                        path = run.output / name
                        if name == "test_io.pt" and not path.exists():
                            continue  # Report unavailable evidence separately from a contradiction.
                        stat = path.stat()
                        identity = stat.st_dev, stat.st_ino
                        if not stat.st_size or identity in physical:
                            raise ValueError(f"Empty or duplicate physical artifact: {path}")
                        physical.add(identity)
                    _, args, scaler, split = load_metadata(run)
                    validate_config(run, summary, args, reference[0] if reference else price_reference)
                    if replayed is None:
                        replayed, actual_scalers = replay_windows(ticker, args, data_root)
                        for name, signature in replayed.items():
                            window_rows.append({"ticker": ticker, "split": name, **signature})
                    for name in ("val", "test"):
                        check_scalers(actual_scalers[name], scaler, str(run.output))
                    if reference is not None:
                        check_scalers(reference[1], scaler, str(run.output))
                        if split != reference[2]:
                            raise ValueError("Split policy mismatch")
                    row.update(verify_windows(run, args, summary, replayed))
                    if row["window_verification"] == "mismatch":
                        raise ValueError(row["window_verification_detail"])
                    if variant in ("decay5", "mean5"):
                        if row["window_verification"] != "full":
                            raise ValueError("Primary comparison requires full saved window evidence: " + row["window_verification_detail"])
                        if any(values[m] is None for m in METRICS):
                            raise ValueError("Missing required primary metric")
                    elif not row["saved_test_arrays_verified"]:
                        raise ValueError("Benchmark comparison requires normalized saved test arrays")
                    if reference is None:
                        reference = args, scaler, split
                    if price_reference is None:
                        price_reference = args
                    row["comparable"] = True
                except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
                    row["comparison_reason"] = str(error)
                    issues.append({"variant": variant, "ticker": ticker, "seed": seed, "reason": str(error)})
                for metric, value in values.items():
                    if value is None:
                        issues.append({"variant": variant, "ticker": ticker, "seed": seed, "reason": f"Unavailable {metric}"})
                rows.append({**row, **values, **sources})
    tables, selection = aggregate(rows)
    blocked = any(not r["comparable"] for r in rows if r["variant"] in ("decay5", "mean5"))
    selection["selection_status"] = "blocked_primary_comparability" if blocked else "validation_only"
    if blocked:
        selection["selected_variants"] = []
    tables = {"per_run": rows, **tables, "issues": issues, "target_windows": window_rows,
              "window_verification": [{key: row[key] for key in ("variant", "ticker", "seed", "window_verification",
                                        "window_verification_detail", *VERIFICATION_FLAGS)} for row in rows],
              "window_verification_summary": verification_summary(rows)}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        fields = list(dict.fromkeys(key for row in records for key in row)) or ["variant", "ticker", "seed", "reason"]
        with (output / f"{name}.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    with (output / "selection.json").open("x", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, allow_nan=False)
    print("WINDOW_VERIFICATION: full = saved val/test fingerprints + test arrays + val dates; partial = incomplete historical evidence; mismatch = contradiction; not_checked = failed earlier checks")
    for name in ("window_verification_summary", "issues", "by_seed", "across_seeds", "relative_by_seed", "paired_by_ticker", "wins"):
        print(name.upper())
        records = tables[name]
        fields = list(dict.fromkeys(key for record in records for key in record)) or ["reason"]
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(selection, allow_nan=False))
    print(f"Reports: {output}")
    return tables, selection


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--mean5_root", type=Path, default=DEFAULT_MEAN5)
    parser.add_argument("--last_root", type=Path, default=DEFAULT_LAST)
    parser.add_argument("--baseline_root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--data_root", type=Path, default=Path("/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts"))
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    _, selection = analyze(**vars(args))
    if selection["selection_status"] == "blocked_primary_comparability":
        raise SystemExit("Primary comparison failed; verification report was saved and no pooling variant was selected")


if __name__ == "__main__":
    cli()
