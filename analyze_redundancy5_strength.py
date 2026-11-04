"""Validation-only, head-independent strength of redundancy5 on the same raw windows."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from analyze_context_price_redundancy5 import DEFAULT_RESULTS, TICKERS, SEEDS, validate_config
from analyze_persistence_baseline import Run, load_metadata, check_scalers
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset
from forecasting_task.models.GroupText import TEXT_GROUPS, pool_text_window
from forecasting_task.window_evidence import window_evidence


DEFAULT_DATA = Path("/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts")
QUANTILES = (("min", 0), ("p01", .01), ("p05", .05), ("p10", .1), ("p25", .25),
             ("p50", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1))
METRICS = {"pair_cosines": "unordered present-date pairs (i<j)",
           "offdiag_kernel": "unordered present-date pairs (i<j); diagonal excluded",
           "r_present": "present dates; density includes self-similarity",
           "weight_l1_deviation": "nonempty group windows",
           "effective_dates": "nonempty group windows", "effective_dates_ratio": "nonempty group windows",
           "aggregate_l2": "nonempty group windows; raw embedding units",
           "aggregate_l1": "nonempty group windows; raw embedding units",
           "aggregate_relative_l2": "nonempty group windows with nonzero mean5 norm",
           "aggregate_cosine_distance": "nonempty group windows with both aggregate norms nonzero"}


def discover_runs(results_root):
    root = Path(results_root) / "full/context_price_only"
    paths = sorted(root.glob("*/seed*/output/best_checkpoint.pt"))
    expected = {(ticker, seed) for ticker in TICKERS for seed in SEEDS}
    runs, seen, physical = [], set(), set()
    for path in paths:
        seed_name = path.parents[1].name
        if seed_name not in {f"seed{seed}" for seed in SEEDS}:
            raise ValueError(f"Unexpected seed directory: {path}")
        key = path.parents[2].name, int(seed_name[4:])
        if key in seen:
            raise ValueError(f"Duplicate run: {path}")
        seen.add(key)
        for name in ("best_checkpoint.pt", "summary.json", "validation_group_diagnostics.csv"):
            artifact = path.with_name(name)
            if not artifact.is_file() or not artifact.stat().st_size:
                raise ValueError(f"Missing/empty artifact: {artifact}")
            stat = artifact.stat()
            identity = stat.st_dev, stat.st_ino
            if identity in physical:
                raise ValueError(f"Duplicate physical artifact: {artifact}")
            physical.add(identity)
        runs.append(Run("redundancy5", *key, path.parent))
    if len(paths) != 30 or seen != expected:
        raise ValueError(f"Expected exactly 30 completed validation runs; missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    return runs


def load_validation(ticker, args, data_root):
    base = Dataset_Custom(args["seq_len"], args["label_len"], args["pred_len"], "val", list(TEXT_GROUPS),
                          root_path=str(data_root), data_path=ticker + ".parquet")
    return ContextNoveltyDataset(base)


@torch.no_grad()
def measure_window(text, tau):
    """Both aggregates use these SAME original five vectors; no trained heads."""
    tensor = torch.as_tensor(np.asarray(text, dtype=np.float32)).unsqueeze(0)
    if tensor.shape != (1, 5, 5, 384) or not torch.isfinite(tensor).all():
        raise ValueError("Expected five finite input rows with five separate 384-dimensional groups")
    aggregate, present, counts, detail = pool_text_window(tensor, "redundancy5", tau, True)
    mean = pool_text_window(tensor, "mean5")[0]
    if not torch.isfinite(aggregate).all() or not torch.isfinite(mean).all():
        raise ValueError("Reconstructed aggregates must be finite")
    records = []
    for group_index, group in enumerate(TEXT_GROUPS):
        mask = detail["daily_present"][0, :, group_index].numpy()
        a = detail["temporal_weights"][0, :, group_index].numpy()
        density = detail["redundancy_density"][0, :, group_index].numpy()
        original = tensor[0, :, group_index].double().numpy()
        n = int(counts[0, group_index])
        record = {"group": group, "present_dates": n, "present": int(present[0, group_index]),
                  "daily_present": mask.astype(int).tolist(), "temporal_weights": a.tolist(),
                  "r_all": density.tolist(), "original_norms": np.linalg.norm(original, axis=-1).tolist(),
                  "pair_cosines": [], "offdiag_kernel": [], "r_present": density[mask].tolist(),
                  **dict.fromkeys((key for key in METRICS if key not in ("pair_cosines", "offdiag_kernel", "r_present")))}
        if n:
            raw = original[mask]
            units = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
            cosine = np.clip(units @ units.T, -1, 1)
            np.fill_diagonal(cosine, 1.)
            kernel = np.exp(-(1 - cosine) / tau)
            # Independently confirm density reconstruction against the actual pool helper.
            if not np.allclose(kernel.sum(-1), density[mask], rtol=1e-10, atol=1e-12):
                raise ValueError("Reconstructed kernel/density disagrees with redundancy5")
            upper = np.triu_indices(n, 1)
            red, ordinary = aggregate[0, group_index].double().numpy(), mean[0, group_index].double().numpy()
            distance = float(np.linalg.norm(red - ordinary))
            red_norm, mean_norm = float(np.linalg.norm(red)), float(np.linalg.norm(ordinary))
            effective = float(1 / np.square(a[mask]).sum())
            record.update(pair_cosines=cosine[upper].tolist(), offdiag_kernel=kernel[upper].tolist(),
                          weight_l1_deviation=float(np.abs(a[mask] - 1 / n).sum()), effective_dates=effective,
                          effective_dates_ratio=effective / n, aggregate_l2=distance,
                          aggregate_l1=float(np.abs(red - ordinary).sum()),
                          aggregate_relative_l2=distance / mean_norm if mean_norm > 0 else None,
                          aggregate_cosine_distance=float(1 - np.clip(np.dot(red / red_norm, ordinary / mean_norm), -1, 1))
                          if red_norm > 0 and mean_norm > 0 else None,
                          mean5_norm=mean_norm, redundancy5_norm=red_norm)
        else:
            record.update(mean5_norm=0., redundancy5_norm=0.)
        records.append(record)
    return records


def reconstruct_dataset(ticker, dataset, tau):
    records = []
    for index in range(len(dataset)):
        # Dataset_Custom x contains ONLY input rows; the target y is never pooled.
        text = dataset[index][2][-5:]
        dates = dataset.text_window_dates(index, 5)
        measured = measure_window(text, tau)
        fully_empty = all(record["present_dates"] == 0 for record in measured)
        for record in measured:
            records.append({"ticker": ticker, "window_index": index,
                            "input_end_date": dataset.window_end_date(index), "text_dates": dates,
                            "all_groups_empty": fully_empty,
                            "text_window_start_date": dates[0], "text_window_end_date": dates[-1],
                            "tau": tau, **record})
    return records


def verify_csv(path, expected_records):
    checked, reconstructed = set(), {"pair_cosines", "offdiag_kernel", "aggregate_distances"}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"window_index", "input_end_date", "group"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: diagnostic lacks required window/date/group identifiers")
        for expected in expected_records:
            row = next(reader, None)
            context = f"{path}: window={expected['window_index']}, group={expected['group']}"
            if row is None or row["window_index"] != str(expected["window_index"]) or row["group"] != expected["group"] or row["input_end_date"] != expected["input_end_date"]:
                raise ValueError(f"{context}: missing, duplicate, reordered or misdated validation row")
            checked.update(("window_index", "input_end_date", "group"))

            def compare(key, value):
                saved = row.get(key)
                if saved in (None, ""):
                    if value is not None:
                        reconstructed.add(key)
                    return
                if value is None:
                    raise ValueError(f"{context}: {key} must be blank for an empty window")
                if isinstance(value, str):
                    matches = saved == value
                else:
                    number = float(saved)
                    matches = math.isfinite(number) and np.isclose(number, value, rtol=1e-6, atol=1e-8)
                if not matches:
                    raise ValueError(f"{context}: {key} mismatch: saved={saved}, reconstructed={value}")
                checked.add(key)

            for key, value in (("text_temporal_pool", "redundancy5"), ("redundancy_tau", expected["tau"]),
                               ("available_dates", expected["present_dates"]), ("present", expected["present"]),
                               ("text_window_start_date", expected["text_dates"][0]),
                               ("text_window_end_date", expected["text_dates"][-1]),
                               ("effective_dates", expected["effective_dates"])):
                compare(key, value)
            for day in range(5):
                compare(f"text_date_{day+1}", expected["text_dates"][day])
                compare(f"daily_present_{day+1}", expected["daily_present"][day])
                compare(f"temporal_weight_{day+1}", expected["temporal_weights"][day])
                compare(f"redundancy_r_{day+1}", expected["r_all"][day])
        if next(reader, None) is not None:
            raise ValueError(f"{path}: unexpected extra/duplicate validation rows")
    return sorted(checked), sorted(reconstructed)


def quantiles(records, scope):
    result = []
    for group in TEXT_GROUPS:
        for n in range(6):
            cohort = [r for r in records if r["group"] == group and r["present_dates"] == n]
            for metric, unit in METRICS.items():
                values = []
                for row in cohort:
                    value = row[metric]
                    if isinstance(value, list):
                        values.extend(value)
                    elif value is not None:
                        values.append(value)
                result.append({"scope": scope, "group": group, "present_dates": n, "metric": metric,
                               "unit": unit, "group_windows": len(cohort), "samples": len(values),
                               "mean": float(np.mean(values)) if values else None,
                               **{name: float(np.quantile(values, probability)) if values else None
                                  for name, probability in QUANTILES}})
    return result


def top_examples(records, top_k, stratified=False):
    cohorts = [[r for r in records if r["present_dates"] > 0]]
    if stratified:
        cohorts = [[r for r in records if r["group"] == group and r["present_dates"] == n]
                   for group in TEXT_GROUPS for n in range(1, 6)]
    return [{"rank": rank, **row} for cohort in cohorts for rank, row in enumerate(
        sorted(cohort, key=lambda r: (-r["aggregate_l2"], r["ticker"], r["window_index"], r["group"]))[:top_k], 1)]


def write_table(path, records):
    fields = list(dict.fromkeys(key for row in records for key in row)) or ["ticker", "group", "window_index"]
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def analyze(results_root=DEFAULT_RESULTS, data_root=DEFAULT_DATA, output_dir=None, top_k=10):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    output = Path(output_dir) if output_dir else Path(results_root) / "diagnostics/redundancy5_strength"
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    runs = discover_runs(results_root)
    unique, repeated, verification, cache = [], [], [], {}
    reference_args = None
    for run in runs:
        summary, args, scaler, _ = load_metadata(run)
        validate_config(run, summary, args, reference_args)
        if "fnspid" in args["data_path"]:
            raise ValueError(f"Forward-filled data are not allowed: {run.output}")
        epochs, best = summary.get("epochs_completed"), summary.get("best_epoch")
        if not isinstance(epochs, int) or not isinstance(best, int) or not 1 <= best <= epochs:
            raise ValueError(f"Incomplete run or invalid best epoch: {run.output}")
        tau = float(args["redundancy_tau"])
        if reference_args is None:
            reference_args = args
        if run.ticker not in cache:
            dataset = load_validation(run.ticker, args, data_root)
            evidence = window_evidence(dataset)  # VAL ONLY; no test_io or test evaluation.
            records = reconstruct_dataset(run.ticker, dataset, tau)
            actual_scaler = {name: np.asarray(getattr(dataset.scaler, name + "_")) for name in ("mean", "scale", "var")}
            text_hash = hashlib.sha256(np.asarray(dataset.base.data_text_x, dtype="<f4").tobytes()).hexdigest()
            cache[run.ticker] = evidence, records, actual_scaler, text_hash
            unique.extend(records)
        evidence, records, actual_scaler, text_hash = cache[run.ticker]
        check_scalers(actual_scaler, scaler, str(run.output))
        fingerprint_sources = []
        candidates = {"checkpoint_args": args.get("window_evidence"), "summary": summary.get("window_evidence")}
        sidecar = run.output / "window_evidence.json"
        if sidecar.exists():
            candidates["sidecar"] = json.loads(sidecar.read_text(encoding="utf-8"))
        for name, saved in candidates.items():
            if saved is not None and "val" in saved:
                if saved["val"] != evidence:
                    raise ValueError(f"{run.output}: {name} validation fingerprint mismatch")
                fingerprint_sources.append(name)
        csv_path = run.output / "validation_group_diagnostics.csv"
        checked, recovered = verify_csv(csv_path, records)
        verification.append({"ticker": run.ticker, "seed": run.seed, "best_epoch": best, "tau": tau,
                             "code_commit": args.get("code_commit", "unavailable"),
                             "validation_windows": evidence["windows"], "diagnostic_rows": len(records),
                             "validation_fingerprints": fingerprint_sources,
                             "validation_evidence": "saved_OHLC_target_date_fingerprints" if fingerprint_sources else "partial: dates/configuration/scaler only",
                             "checked_csv_fields": checked, "reconstructed_fields": recovered,
                             "replayed_validation_text_sha256": text_hash,
                             "embedding_evidence": "current parquet values; no historical raw-embedding fingerprint saved",
                             "source_csv": str(csv_path), "source_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()})
        repeated.extend({**row, "seed": run.seed} for row in records)
        print(f"VERIFIED {len(verification)}/30 ticker={run.ticker} seed={run.seed} val_windows={evidence['windows']}", flush=True)
    counts = []
    for group in TEXT_GROUPS:
        total = sum(r["group"] == group for r in unique)
        for n in range(6):
            count = sum(r["group"] == group and r["present_dates"] == n for r in unique)
            counts.append({"group": group, "present_dates": n, "unique_group_windows": count,
                           "run_group_windows": count * 3, "fraction_within_group": count / total})
    tables = {"quantiles": quantiles(repeated, "30_saved_runs"),
              "quantiles_unique": quantiles(unique, "10_unique_ticker_window_groups"),
              "counts": counts, "per_window": unique,
              "empty_windows": [r for r in unique if r["present_dates"] == 0],
              "fully_empty_windows": [{**{k: r[k] for k in ("ticker", "window_index", "input_end_date", "text_dates")},
                                       "represented_runs": len(SEEDS)}
                                      for r in unique if r["group"] == TEXT_GROUPS[0] and r["all_groups_empty"]],
              "top_examples": top_examples(unique, top_k),
              "top_examples_by_group_N": top_examples(unique, top_k, True), "verification": verification}
    manifest = {"runs_verified": len(verification), "tickers": list(TICKERS), "seeds": list(SEEDS),
                "split": "validation", "tau_source": "saved configuration; no tuning or override",
                "tau": float(reference_args["redundancy_tau"]), "quantile_method": "linear",
                "unique_group_windows": len(unique), "run_group_windows": len(repeated),
                "empty_unique_group_windows": sum(r["present_dates"] == 0 for r in unique),
                "fully_empty_input_windows": len(tables["fully_empty_windows"]),
                "pair_sampling": "unordered present-date pairs i<j; no diagonal in cosine/K distributions",
                "density_sampling": "present rows; self-similarity included in r_i",
                "aggregate_comparison": "same original five vectors, model float32 pooling outputs, double-precision distance; no heads",
                "seed_repetition": "pooling has no trained parameters; same ticker/window/group repeated across three verified seeds",
                "embedding_verification_limit": "saved temporal diagnostics checked where available; no historical raw-embedding-value fingerprint exists",
                "results_root": str(results_root), "data_root": str(data_root)}
    manifest["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        write_table(output / f"{name}.csv", records)
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    def display(title, records, fields):
        print(title)
        writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})

    display("QUANTILES BY GROUP AND N (unique windows; 30-run quantiles also saved)",
            [r for r in tables["quantiles_unique"] if r["samples"]],
            ["group", "present_dates", "metric", "samples", *[name for name, _ in QUANTILES]])
    display("EMPTY/N COUNTS", counts, list(counts[0]))
    display("LARGEST AGGREGATE CHANGES (same raw vectors; no trained-head comparison)", tables["top_examples"],
            ["rank", "ticker", "group", "present_dates", "window_index", "input_end_date", "text_dates",
             "weight_l1_deviation", "effective_dates", "effective_dates_ratio", "aggregate_l2", "aggregate_relative_l2"])
    print(f"Reports: {output}")
    return tables, manifest


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args(argv)
    analyze(**vars(args))


if __name__ == "__main__":
    cli()
