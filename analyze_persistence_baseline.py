"""Compare saved FiLM experiments with persistence on replayed val/test windows.

No forecasting model, runner entry point or optimizer is imported or constructed.
Only trusted, locally generated training artifacts are read.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from forecasting_task.data_provider.dataset import Dataset_Custom


DEFAULT_DATA = Path("/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts")
DEFAULT_RESULTS = Path("/auto/datasets/kondyurin/FinTexTS/runs/group_no_weights_alpha_v1")
DEFAULT_BASELINE = Path("/auto/datasets/kondyurin/FinTexTS/runs/film_compare_50ep_v1")
SEEDS = (7, 17, 27)
GROUPS = ("macro", "sector", "targetCompany", "relatedCompany", "filing")
VARIANTS = ("persistence", "no_weights_blend", "no_weights_residual", "film_baseline")
ERRORS = ("val_mse", "val_mae", "test_mse", "test_mae")
REPORTS = ("per_run.csv", "per_ticker.csv", "by_seed.csv", "across_seeds.csv", "wins.csv", "issues.csv", "configurations.csv")
# Dataset_Custom has fixed date boundaries and no split CLI arguments.
SPLIT_POLICY = {
    "data_start": "2019-01-01", "data_end": "2023-12-19",
    "train_start": "2019-01-01", "train_end": "2022-01-01",
    "val_start": "2022-01-01", "val_end": "2023-01-01",
    "test_start": "2023-01-01", "test_end": "2024-01-01",
    "end_bounds": "exclusive", "scaler_fit": "train_only", "channels": ["open", "high", "low", "close"],
    "eval_shuffle": False, "eval_drop_last": False,
}


@dataclass(frozen=True)
class Run:
    variant: str
    ticker: str
    seed: int
    output: Path


def discover_runs(results_root, data_root, baseline_root):
    data_paths = sorted(path for path in Path(data_root).glob("*.parquet") if path.is_file())
    tickers = [path.stem for path in data_paths]
    if len(tickers) != 100 or len({ticker.casefold() for ticker in tickers}) != 100:
        raise ValueError(f"Expected exactly 100 distinct ticker parquet files; found {len(tickers)}")
    runs, physical_files = [], set()
    expected = {(ticker, seed) for ticker in tickers for seed in SEEDS}
    for mode in ("blend", "residual"):
        root = Path(results_root) / "full" / "alpha010" / mode
        paths = sorted(root.glob("*/seed*/output/summary.json"))
        keys = set()
        for path in paths:
            seed_dir = path.parents[1].name
            if seed_dir not in {f"seed{seed}" for seed in SEEDS}:
                raise ValueError(f"Unexpected seed directory: {path}")
            key = (path.parents[2].name, int(seed_dir[4:]))
            if key in keys:
                raise ValueError(f"Duplicate run: {path}")
            keys.add(key)
            for required in (path, path.with_name("best_checkpoint.pt")):
                if not required.is_file() or not required.stat().st_size:
                    raise ValueError(f"Missing/empty required artifact: {required}")
                stat = required.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in physical_files:
                    raise ValueError(f"Duplicate physical artifact: {required}")
                physical_files.add(identity)
            runs.append(Run("no_weights_" + mode, *key, path.parent))
        if len(paths) != 300 or keys != expected:
            raise ValueError(f"{mode}: expected 100 tickers x seeds {SEEDS}; missing={sorted(expected - keys)}, extra={sorted(keys - expected)}")
    # Missing or incompatible historical baselines are reported, not silently pooled.
    baseline_paths = list((Path(baseline_root) / "full" / "baseline").glob("*/seed*/output/summary.json"))
    seen, baseline_inventory_error = set(), ""
    for path in baseline_paths:
        seed_dir = path.parents[1].name
        key = (path.parents[2].name, seed_dir)
        if key in seen:
            baseline_inventory_error = "Duplicate historical baseline artifact/run"
        seen.add(key)
        for artifact in (path, path.with_name("best_checkpoint.pt")):
            if artifact.exists():
                stat = artifact.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in physical_files:
                    baseline_inventory_error = "Duplicate historical baseline artifact/run"
                physical_files.add(identity)
    expected_baseline = {(ticker, f"seed{seed}") for ticker, seed in expected}
    if seen - expected_baseline:
        baseline_inventory_error = "Historical baseline has unexpected ticker/seed directories"
    for ticker in tickers:
        for seed in SEEDS:
            runs.append(Run("film_baseline", ticker, seed, Path(baseline_root) / "full" / "baseline" / ticker / f"seed{seed}" / "output"))
    return tickers, runs, baseline_inventory_error


def load_metadata(run):
    summary_path, checkpoint_path = run.output / "summary.json", run.output / "best_checkpoint.pt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # Read metadata only; no model state is loaded into an inference model.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = checkpoint.get("args")
    if args is None:
        args = yaml.safe_load((run.output / "args.yaml").read_text(encoding="utf-8"))
    if not isinstance(args, dict):
        raise ValueError("Saved configuration is not a dictionary")
    scaler = {}
    for name in ("mean", "scale", "var"):
        values = np.asarray(checkpoint[f"scaler_{name}"], dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved OHLC scaler {name}")
        scaler[name] = values
    if np.any(scaler["scale"] <= 0) or np.any(scaler["var"] < 0):
        raise ValueError("Saved scaler has invalid scale/variance")
    if summary.get("seed", run.seed) != run.seed or args["seed"] != run.seed:
        raise ValueError("Saved seed disagrees with directory")
    if "data_path" in summary and Path(str(summary["data_path"]).replace("\\", "/")).name != run.ticker + ".parquet":
        raise ValueError("Summary data_path disagrees with ticker")
    for summary_key, args_key in (("model", "model_type"), ("fusion_mode", "fusion_mode"),
                                  ("group_weight_mode", "group_weight_mode"), ("initial_text_weight", "text_weight")):
        left, right = summary.get(summary_key), args.get(args_key)
        if summary_key == "model":
            left, right = str(left).lower(), str(right).lower()
        if summary_key in summary and args_key in args and left != right:
            raise ValueError(f"Summary {summary_key} disagrees with checkpoint configuration")
    if "best_epoch" in summary and checkpoint.get("epoch") != summary["best_epoch"]:
        raise ValueError("Checkpoint epoch disagrees with summary best_epoch")
    split = split_parameters(args, checkpoint)
    return summary, args, scaler, split


def split_parameters(args, checkpoint):
    """Check any saved split overrides against the actual fixed Dataset_Custom contract."""
    for source in (args, checkpoint):
        for key, value in source.items():
            if key in ("split_config", "split_params", "split_policy"):
                if not isinstance(value, dict):
                    raise ValueError(f"Unrecognized {key}; split comparability unconfirmed")
                for name, declared in value.items():
                    if name not in SPLIT_POLICY or declared != SPLIT_POLICY[name]:
                        raise ValueError(f"Saved split parameter differs or is unsupported: {name}={declared}")
            elif key in SPLIT_POLICY:
                if value != SPLIT_POLICY[key]:
                    raise ValueError(f"Saved split parameter differs: {key}={value}")
            elif ("split" in key.lower() or key.lower() in {
                "train_ratio", "val_ratio", "test_ratio", "train_size", "val_size", "test_size",
                "train_end_date", "val_end_date", "test_end_date", "start_date", "end_date",
            }):
                raise ValueError(f"Unsupported saved split parameter {key}; cannot replay it silently")
    return SPLIT_POLICY.copy()


def check_configuration(run, args, reference=None):
    if str(args.get("model_type", "")).lower() != "film":
        raise ValueError("Expected FiLM configuration")
    name = Path(str(args["data_path"]).replace("\\", "/")).name
    if name != run.ticker + ".parquet":
        raise ValueError(f"data_path disagrees with ticker {run.ticker}")
    if reference is not None and ("fnspid" in args["data_path"]) != ("fnspid" in reference["data_path"]):
        raise ValueError("Dataset_Custom forward-fill preprocessing differs")
    for key in ("seq_len", "pred_len", "label_len"):
        if not isinstance(args[key], int) or args[key] < (0 if key == "label_len" else 1):
            raise ValueError(f"Invalid {key}")
        if reference is not None and args[key] != reference[key]:
            raise ValueError(f"{key} mismatch: model={args[key]}, reference={reference[key]}")
    if args["label_len"] > args["seq_len"]:
        raise ValueError("label_len exceeds seq_len")
    for key in ("enc_in", "dec_in", "c_out"):
        if key in args and args[key] != 4:
            raise ValueError(f"{key} is not four OHLC channels")
    if run.variant.startswith("no_weights_"):
        mode = run.variant.removeprefix("no_weights_")
        if (args.get("group_weight_mode") != "fixed_uniform" or args.get("fusion_mode") != mode
                or args.get("text_num_layers") != 1 or tuple(args.get("used_col_prefixes", [])) != GROUPS
                or not np.isclose(args.get("text_weight", np.nan), 0.1, rtol=0, atol=1e-7)):
            raise ValueError("Not the requested fixed_uniform/linear/initial-alpha=0.1 configuration")


def check_scalers(actual, saved, context):
    for name in ("mean", "scale", "var"):
        if not np.allclose(actual[name], saved[name], rtol=1e-7, atol=1e-9):
            raise ValueError(f"Scaler {name} mismatch ({context}): {actual[name].tolist()} != {saved[name].tolist()}")


def model_metrics(summary):
    values, sources = {}, {}
    for metric in ERRORS:
        key = next((key for key in ("best_" + metric, metric) if key in summary), None)
        value = summary[key] if key else None
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid model metric {key}")
        values[metric], sources[metric + "_source"] = value, key or "unavailable"
    return values, sources


@torch.no_grad()
def persistence_prediction(inputs, pred_len):
    if inputs.ndim != 3 or inputs.shape[1] < 1 or inputs.shape[-1] != 4 or pred_len < 1:
        raise ValueError("Expected nonempty [windows,time,OHLC] input and positive horizon")
    return inputs[:, -1:, :].expand(-1, pred_len, -1)


@torch.no_grad()
def persistence_errors(inputs, targets, pred_len):
    if inputs.ndim != 3 or targets.ndim != 3 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("Expected matching [windows,time,OHLC] inputs and targets")
    if inputs.shape[0] == 0 or inputs.shape[1] < 1 or inputs.shape[-1] != 4 or targets.shape[1:] != (pred_len, 4):
        raise ValueError("Unexpected input/target horizon or OHLC shape")
    prediction = persistence_prediction(inputs, pred_len)
    error = prediction.double() - targets.double()
    if not torch.isfinite(error).all():
        raise ValueError("Nonfinite normalized persistence errors")
    return error.square().sum().item(), error.abs().sum().item(), error.numel()


@torch.no_grad()
def evaluate_persistence(dataset, batch_size, pred_len, device):
    mse_sum = mae_sum = 0.0
    elements = 0
    inputs, targets = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    for batch in loader:
        x = batch[0].float().to(device)
        truth = batch[1].float().to(device)[:, -pred_len:, :]
        mse, mae, count = persistence_errors(x, truth, pred_len)
        mse_sum += mse
        mae_sum += mae
        elements += count
        inputs.append(x.cpu())
        targets.append(truth.cpu())
    if elements == 0:
        raise ValueError("No evaluation windows")
    return {"mse": mse_sum / elements, "mae": mae_sum / elements, "num_elements": elements,
            "inputs": torch.cat(inputs), "targets": torch.cat(targets)}


def window_digest(result):
    digest = hashlib.sha256()
    for name in ("inputs", "targets"):
        array = result[name].numpy()
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def check_test_io(path, expected, pred_len):
    result = {"test_io_status": "unavailable", "test_io_reason": "", "test_io_keys": "",
              "test_io_inputs_key": "", "test_io_targets_key": "", "io_persistence_test_mse": None,
              "io_persistence_test_mae": None}
    if not path.is_file():
        result["test_io_reason"] = f"Missing {path}"
        return result
    # Trusted training files contain NumPy *_inv arrays, so weights_only=True
    # cannot decode the existing format. Those arrays are NEVER used below.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError, EOFError) as error:
        result["test_io_status"], result["test_io_reason"] = "mismatch", f"Cannot decode {path}: {error}"
        return result
    if not isinstance(payload, dict):
        result["test_io_reason"] = "File is not a dictionary of saved arrays"
        return result
    result["test_io_keys"] = json.dumps(sorted(map(str, payload)))
    if "inputs" not in payload or "ground_truths" not in payload:
        result["test_io_reason"] = "Normalized inputs/ground_truths unavailable; actual keys=" + result["test_io_keys"]
        return result
    result["test_io_inputs_key"], result["test_io_targets_key"] = "inputs", "ground_truths"
    try:
        x = torch.as_tensor(payload["inputs"]).float()
        truth = torch.as_tensor(payload["ground_truths"]).float()
        mse, mae, count = persistence_errors(x, truth, pred_len)
        result["io_persistence_test_mse"], result["io_persistence_test_mae"] = mse / count, mae / count
        for key, actual in (("inputs", x), ("targets", truth)):
            reference = expected[key]
            if actual.shape != reference.shape or not torch.allclose(actual, reference, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Saved {key} windows/ordering/values differ from replayed windows")
        for metric in ("mse", "mae"):
            if not np.isclose(result[f"io_persistence_test_{metric}"], expected[metric], rtol=1e-5, atol=1e-7):
                raise ValueError(f"Saved-window persistence {metric} differs from Dataset_Custom")
    except (ValueError, TypeError, RuntimeError) as error:
        result["test_io_status"], result["test_io_reason"] = "mismatch", str(error)
        return result
    result["test_io_status"] = "verified"
    return result


def improvement_pct(persistence, model):
    return 100 * (persistence - model) / persistence if persistence != 0 else None


def make_row(run, values, sources, persistence, comparable=True, reason="", io_status=None):
    row = {"variant": run.variant, "ticker": run.ticker, "seed": run.seed, "output": str(run.output),
           **values, **sources, "comparable_val": comparable, "comparable_test": comparable,
           "comparison_reason": reason, **(io_status or {})}
    for metric in ERRORS:
        row["persistence_" + metric] = persistence[metric]
        row[metric + "_improvement_pct"] = improvement_pct(persistence[metric], values[metric]) if comparable and values[metric] is not None else None
    return row


def verify_seed_independence(rows):
    for ticker in sorted({row["ticker"] for row in rows}):
        cohort = [row for row in rows if row["ticker"] == ticker and row["variant"] != "film_baseline"]
        for metric in ERRORS:
            values = [row["persistence_" + metric] for row in cohort]
            if not values or not np.allclose(values, values[0], rtol=0, atol=1e-12):
                raise ValueError(f"Persistence is not seed/fusion invariant: {ticker} {metric}")


def verify_cohort_geometry(configurations):
    reference = configurations[0]
    for configuration in configurations[1:]:
        for key in ("seq_len", "label_len", "pred_len", "split_policy"):
            if configuration[key] != reference[key]:
                raise ValueError(f"Cannot aggregate different {key}: {reference['ticker']} vs {configuration['ticker']}")


def complete_mean(values, count):
    return float(np.mean(values)) if len(values) == count and all(value is not None for value in values) else None


def aggregate(rows, tickers):
    per_ticker, by_seed, stats, wins = [], [], [], []
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        for ticker in tickers:
            cohort = [row for row in variant_rows if row["ticker"] == ticker]
            if {row["seed"] for row in cohort} != set(SEEDS) or len(cohort) != 3:
                raise ValueError(f"Incomplete/duplicate seed results for {variant}/{ticker}")
            record = {"variant": variant, "ticker": ticker}
            for metric in ERRORS:
                model = complete_mean([row[metric] for row in cohort], 3)
                reference = complete_mean([row["persistence_" + metric] for row in cohort], 3)
                eligible = sum(row["comparable_" + metric.split("_")[0]] and row[metric] is not None for row in cohort)
                record[metric], record["persistence_" + metric] = model, reference
                record[metric + "_comparable_seeds"] = eligible
                record[metric + "_improvement_pct"] = improvement_pct(reference, model) if eligible == 3 else None
            record["test_mse_better_than_persistence"] = record["test_mse"] < record["persistence_test_mse"] if record["test_mse_comparable_seeds"] == 3 else None
            per_ticker.append(record)
        for seed in SEEDS:
            cohort = [row for row in variant_rows if row["seed"] == seed]
            record = {"variant": variant, "seed": seed, "expected_tickers": 100}
            for metric in ERRORS:
                model = complete_mean([row[metric] for row in cohort], 100)
                reference = complete_mean([row["persistence_" + metric] for row in cohort], 100)
                eligible = sum(row["comparable_" + metric.split("_")[0]] and row[metric] is not None for row in cohort)
                record[metric] = model
                record[metric + "_available_tickers"] = sum(row[metric] is not None for row in cohort)
                record[metric + "_comparable_tickers"] = eligible
                record[metric + "_improvement_pct"] = improvement_pct(reference, model) if eligible == 100 else None
            by_seed.append(record)
        for metric in ERRORS:
            cohort = [row for row in by_seed if row["variant"] == variant]
            errors = [row[metric] for row in cohort]
            improvements = [row[metric + "_improvement_pct"] for row in cohort]
            stats.append({"variant": variant, "metric": metric, "mean": complete_mean(errors, 3),
                          "sample_std": float(np.std(errors, ddof=1)) if all(value is not None for value in errors) else None,
                          "improvement_pct_mean": complete_mean(improvements, 3),
                          "improvement_pct_sample_std": float(np.std(improvements, ddof=1)) if all(value is not None for value in improvements) else None,
                          "std_ddof": 1, "complete_seeds": sum(value is not None for value in errors),
                          "fully_comparable_seeds": sum(row[metric + "_comparable_tickers"] == 100 for row in cohort)})
        ticker_rows = [row for row in per_ticker if row["variant"] == variant]
        eligible = [row for row in ticker_rows if row["test_mse_better_than_persistence"] is not None]
        wins.append({"variant": variant, "tickers_better": sum(row["test_mse_better_than_persistence"] for row in eligible),
                     "comparable_tickers": len(eligible), "total_tickers": 100, "all_100_confirmed": len(eligible) == 100})
    return per_ticker, by_seed, stats, wins


def analyze_ticker(ticker, runs, data_root, baseline_inventory_error, device):
    cohort = [run for run in runs if run.ticker == ticker]
    new_runs = [run for run in cohort if run.variant.startswith("no_weights_")]
    metadata = {}
    for run in new_runs:
        try:
            metadata[run] = load_metadata(run)
        except Exception as error:
            raise ValueError(f"Cannot verify primary run {run.output}: {error}") from error
    reference_run = next(run for run in new_runs if run.variant == "no_weights_blend" and run.seed == 7)
    _, reference_args, saved_scaler, reference_split = metadata[reference_run]
    check_configuration(reference_run, reference_args)
    for run, (_, args, scaler, split) in metadata.items():
        try:
            check_configuration(run, args, reference_args)
            check_scalers(saved_scaler, scaler, str(run.output))
            if split != reference_split:
                raise ValueError("Split parameters differ")
        except ValueError as error:
            raise ValueError(f"Primary configuration mismatch at {run.output}: {error}") from error
    results = {}
    for split in ("val", "test"):
        dataset = Dataset_Custom(
            seq_len=reference_args["seq_len"], label_len=reference_args["label_len"], pred_len=reference_args["pred_len"],
            flag=split, used_col_prefixes=list(GROUPS), root_path=str(data_root), data_path=ticker + ".parquet",
        )
        actual_scaler = {name: np.asarray(getattr(dataset.scaler, name + "_")) for name in ("mean", "scale", "var")}
        check_scalers(actual_scaler, saved_scaler, f"Dataset_Custom {ticker}/{split}")
        results[split] = evaluate_persistence(dataset, reference_args["batch_size"], reference_args["pred_len"], device)
    persistence = {f"{split}_{metric}": results[split][metric] for split in ("val", "test") for metric in ("mse", "mae")}
    rows = []
    for seed in SEEDS:
        rows.append(make_row(Run("persistence", ticker, seed, Path(data_root) / (ticker + ".parquet")), persistence,
                             {metric + "_source": "Dataset_Custom" for metric in ERRORS}, persistence))
    for run in new_runs:
        summary = metadata[run][0]
        values, sources = model_metrics(summary)
        if any(values[key] is None for key in ("val_mse", "test_mse", "test_mae")):
            raise ValueError(f"Required model metrics missing in {run.output / 'summary.json'}")
        io_status = check_test_io(run.output / "test_io.pt", results["test"], reference_args["pred_len"])
        if io_status["test_io_status"] == "mismatch":
            raise ValueError(f"Primary test_io mismatch at {run.output}: {io_status['test_io_reason']}")
        row = make_row(run, values, sources, persistence, io_status=io_status)
        if io_status["test_io_status"] == "unavailable":
            # The new cohort specifically requires an independent saved-window test check.
            row["comparable_test"] = False
            row["comparison_reason"] = "New-model test windows unconfirmed: " + io_status["test_io_reason"]
            for metric in ("test_mse", "test_mae"):
                row[metric + "_improvement_pct"] = None
        rows.append(row)
    for run in (run for run in cohort if run.variant == "film_baseline"):
        values, sources = dict.fromkeys(ERRORS), {metric + "_source": "unavailable" for metric in ERRORS}
        comparable, reason, io_status = False, "", None
        try:
            # Preserve available summary errors even if configuration cannot be verified.
            summary = json.loads((run.output / "summary.json").read_text(encoding="utf-8"))
            values, sources = model_metrics(summary)
            if baseline_inventory_error:
                raise ValueError(baseline_inventory_error)
            _, args, scaler, split = load_metadata(run)
            check_configuration(run, args, reference_args)
            check_scalers(saved_scaler, scaler, str(run.output))
            if split != reference_split:
                raise ValueError("Historical baseline split parameters differ")
            io_status = check_test_io(run.output / "test_io.pt", results["test"], reference_args["pred_len"])
            if io_status["test_io_status"] == "mismatch":
                raise ValueError(io_status["test_io_reason"])
            comparable = True
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, pickle.UnpicklingError, EOFError) as error:
            reason = str(error)
        rows.append(make_row(run, values, sources, persistence, comparable, reason, io_status))
    configuration = {"ticker": ticker, **{key: reference_args[key] for key in ("seq_len", "label_len", "pred_len")},
                     "reference_checkpoint": str(reference_run.output / "best_checkpoint.pt"),
                     "split_policy": json.dumps(reference_split, sort_keys=True),
                     "split_policy_source": "Dataset_Custom fixed boundaries; saved overrides checked",
                     "seed_independence": "matched geometry/split/scalers; deterministic result reused across seeds",
                     **{f"scaler_{name}": json.dumps(saved_scaler[name].tolist()) for name in saved_scaler},
                     **{f"{split}_num_elements": results[split]["num_elements"] for split in results},
                     **{f"{split}_window_sha256": window_digest(results[split]) for split in results},
                     "persistence_evaluations": 2, "new_configurations_checked": len(new_runs)}
    return rows, configuration


def write_reports(output, tables):
    paths = [output / name for name in REPORTS]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Existing diagnostics in {output}; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths, tables):
        # The issues table can be empty; retain its documented schema.
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["variant", "ticker", "seed", "output", "reason"]
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def analyze(results_root=DEFAULT_RESULTS, data_root=DEFAULT_DATA, baseline_root=DEFAULT_BASELINE, device="cpu"):
    tickers, runs, baseline_inventory_error = discover_runs(results_root, data_root, baseline_root)
    output = Path(results_root) / "diagnostics" / "persistence_100"
    if any((output / name).exists() for name in REPORTS):
        raise FileExistsError(f"Existing diagnostic CSV in {output}")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu")
    rows, configurations = [], []
    for index, ticker in enumerate(tickers, 1):
        ticker_rows, configuration = analyze_ticker(ticker, runs, data_root, baseline_inventory_error, device)
        rows.extend(ticker_rows)
        configurations.append(configuration)
        print(f"PERSISTENCE {index}/100 ticker={ticker} val/test replayed; six configurations/scalers checked", flush=True)
    verify_seed_independence(rows)
    verify_cohort_geometry(configurations)
    per_ticker, by_seed, stats, wins = aggregate(rows, tickers)
    issues = []
    for row in rows:
        reasons = [row.get("comparison_reason", ""), row.get("test_io_reason", "")]
        missing = [metric for metric in ERRORS if row[metric] is None]
        if missing:
            reasons.append("Unavailable summary metrics: " + ",".join(missing))
        if any(reasons):
            issues.append({key: row[key] for key in ("variant", "ticker", "seed", "output")} | {"reason": "; ".join(filter(None, reasons))})
    write_reports(output, [rows, per_ticker, by_seed, stats, wins, issues, configurations])
    print("\nMeans across 100 tickers for each seed; blank/None means incomplete coverage:")
    for row in by_seed:
        print(f"{row['variant']} seed={row['seed']}: " + " ".join(
            f"{metric}={row[metric]} improvement_pct={row[metric + '_improvement_pct']} "
            f"comparable={row[metric + '_comparable_tickers']}/100" for metric in ERRORS))
    print("\nMean and sample std between three seed means (ddof=1):")
    for row in stats:
        print(f"{row['variant']} {row['metric']}: mean={row['mean']} std={row['sample_std']} "
              f"improvement_pct={row['improvement_pct_mean']} +/- {row['improvement_pct_sample_std']} "
              f"fully_comparable_seeds={row['fully_comparable_seeds']}/3")
    print("\nStocks with lower mean-over-seeds test MSE than persistence:")
    for row in wins:
        print(f"{row['variant']}: wins={row['tickers_better']}, confirmed={row['comparable_tickers']}/100, "
              f"all_100_confirmed={row['all_100_confirmed']}")
    print(f"\nIssues/unavailable fields: {len(issues)}; details in {output / 'issues.csv'}")
    for variant in VARIANTS:
        variant_issues = [row for row in issues if row["variant"] == variant]
        print(f"  {variant}: issues/unavailable fields in {len(variant_issues)} runs")
        for row in variant_issues[:2]:
            print(f"    {row['ticker']}/seed{row['seed']}: {row['reason']}")
    print(f"CSV reports: {output}; improvement_pct > 0 means the model beats persistence.")
    return rows, per_ticker, by_seed, stats, wins


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline_root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num_threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.num_threads < 1:
        parser.error("--num_threads must be positive")
    torch.set_num_threads(args.num_threads)
    try:
        analyze(args.results_root, args.data_root, args.baseline_root, args.device)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    cli()
