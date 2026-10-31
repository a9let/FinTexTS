"""Validation-only price-anchor diagnostic for alpha090/blend best checkpoints.

Run inside Docker: python /workspace/FinTexTS/analyze_blend_price_anchor.py
No runner entry points, training, test split or saved prediction files are used.
"""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.models import FiLM as film_module
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model


DEFAULT_RESULTS = Path("/auto/datasets/kondyurin/FinTexTS/runs/group_no_weights_alpha_v1")
DEFAULT_DATA = Path("/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts")
SEEDS = (7, 17, 27)
VARIANTS = ("full", "no_text_delta", "persistence")
REPORT_NAMES = ("runs.csv", "by_seed.csv", "across_seeds.csv", "comparisons.csv")
METRICS = tuple(f"{variant}_{metric}" for variant in VARIANTS for metric in ("mse", "mae"))
PERCENT_METRICS = tuple(f"full_vs_{variant}_mse_change_pct" for variant in VARIANTS[1:])


@dataclass(frozen=True)
class Run:
    ticker: str
    seed: int
    checkpoint: Path

    @property
    def summary(self):
        return self.checkpoint.with_name("summary.json")


def discover_runs(results_root):
    """Require the complete, unique 10-ticker x 3-seed cohort before loading models."""
    root = Path(results_root)
    paths = sorted((root / "full" / "alpha090" / "blend").glob("*/seed*/output/best_checkpoint.pt"))
    if len(paths) != 30:
        raise ValueError(f"Expected exactly 30 alpha090/blend checkpoints; found {len(paths)} in {root}")
    seen_keys, seen_paths, seen_inodes = set(), set(), set()
    by_seed = {seed: set() for seed in SEEDS}
    runs = []
    for path in paths:
        seed_name = path.parents[1].name
        if seed_name not in {f"seed{seed}" for seed in SEEDS}:
            raise ValueError(f"Unexpected seed directory: {path}")
        seed = int(seed_name[4:])
        ticker = path.parents[2].name
        key = (ticker.casefold(), seed)
        resolved = path.resolve(strict=True)
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if key in seen_keys or resolved in seen_paths or inode in seen_inodes:
            raise ValueError(f"Duplicate run/checkpoint: {path}")
        seen_keys.add(key)
        seen_paths.add(resolved)
        seen_inodes.add(inode)
        run = Run(ticker, seed, path)
        for required in (run.checkpoint, run.summary):
            if not required.is_file() or required.stat().st_size == 0:
                raise ValueError(f"Missing or empty input: {required}")
        by_seed[seed].add(ticker)
        runs.append(run)
    if any(len(tickers) != 10 for tickers in by_seed.values()) or any(
        tickers != by_seed[SEEDS[0]] for tickers in by_seed.values()
    ):
        raise ValueError(f"Expected the same 10 tickers for seeds {SEEDS}; found {by_seed}")
    selected_path = root / "selected_tickers.txt"
    if selected_path.exists():
        selected = selected_path.read_text(encoding="utf-8").splitlines()
        if len(selected) != 10 or len(set(selected)) != 10 or set(selected) != by_seed[SEEDS[0]]:
            raise ValueError(f"Checkpoint tickers disagree with {selected_path}")
    return sorted(runs, key=lambda run: (run.seed, run.ticker))


@contextmanager
def execution_device(device):
    # Existing FiLM uses this module-level device inside forward as well as init.
    # Scope the execution override here; do not change the model implementation.
    previous = film_module.device
    film_module.device = device
    try:
        yield
    finally:
        film_module.device = previous


def check_scaler(dataset, checkpoint, path):
    for name in ("mean", "scale", "var"):
        saved = np.asarray(checkpoint[f"scaler_{name}"], dtype=np.float64)
        actual = np.asarray(getattr(dataset.scaler, name + "_"), dtype=np.float64)
        if (saved.shape != (4,) or actual.shape != saved.shape or not np.isfinite(saved).all()
                or not np.isfinite(actual).all() or not np.allclose(actual, saved, rtol=1e-7, atol=1e-9)):
            raise ValueError(f"Scaler {name} mismatch for {path}: checkpoint={saved.tolist()}, data={actual.tolist()}")


@torch.no_grad()
def evaluate_batches(film, text_model, loader, pred_len, label_len, device):
    film.eval()
    text_model.eval()
    sums = {variant: {"mse": 0.0, "mae": 0.0} for variant in VARIANTS}
    count = 0
    alpha = text_model.get_alpha()
    if not torch.isfinite(alpha).all():
        raise ValueError("Nonfinite checkpoint alpha")
    for batch in loader:
        price_x, price_y, text_x, x_mark, y_mark = [value.float().to(device) for value in batch[:5]]
        truth = price_y[:, -pred_len:, :]
        last_price = price_x[:, -1, :]
        dec_inp = torch.cat([price_y[:, :label_len, :], torch.zeros_like(truth)], dim=1)
        film_output = film(price_x, x_mark, dec_inp, y_mark)
        persistence = last_price[:, None, :].expand(-1, pred_len, -1)
        predictions = {
            "full": text_model(film_output, last_price, text_x[:, -1, :, :]),
            # Deliberately retain last price even if a sample has no text.
            "no_text_delta": (1 - alpha) * film_output + alpha * persistence,
            "persistence": persistence,
        }
        if truth.ndim != 3 or truth.shape[-1] != 4:
            raise ValueError(f"Expected OHLC ground truth, got {tuple(truth.shape)}")
        for variant, prediction in predictions.items():
            if prediction.shape != truth.shape:
                raise ValueError(f"{variant} prediction shape {tuple(prediction.shape)} != {tuple(truth.shape)}")
            error = prediction.double() - truth.double()
            sums[variant]["mse"] += error.square().sum().item()
            sums[variant]["mae"] += error.abs().sum().item()
        count += truth.numel()
    if count == 0:
        raise ValueError("Validation contains no forecast elements")
    result = {f"{variant}_{metric}": sums[variant][metric] / count for variant in VARIANTS for metric in ("mse", "mae")}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite validation metrics")
    return {"alpha": alpha.item(), "num_elements": count, **result}


def mse_change_pct(full, reference):
    """Negative means full improves MSE; a zero denominator is undefined (CSV blank)."""
    return 100 * (full - reference) / reference if reference != 0 else None


@torch.no_grad()
def evaluate_run(run, data_root, device):
    checkpoint = torch.load(run.checkpoint, map_location="cpu", weights_only=True)
    saved_args = checkpoint["args"]
    args = SimpleNamespace(**saved_args)
    summary = json.loads(run.summary.read_text(encoding="utf-8"))
    if (args.model_type.lower() != "film" or args.group_weight_mode != "fixed_uniform"
            or args.fusion_mode != "blend" or args.seed != run.seed
            or tuple(args.used_col_prefixes) != TEXT_GROUPS or args.text_num_layers != 1
            or not np.isclose(args.text_weight, 0.9, rtol=0, atol=1e-7)):
        raise ValueError(f"Checkpoint is not the requested FiLM/fixed_uniform/blend/initial-alpha=0.9 run: {run.checkpoint}")
    config = args.text_model_config
    if (config["fusion_mode"] != "blend" or tuple(config["used_col_prefixes"]) != TEXT_GROUPS
            or config["pred_len"] != args.pred_len or config["num_layers"] != 1
            or not np.isclose(config["initial_text_weight"], 0.9, rtol=0, atol=1e-7)):
        raise ValueError(f"Conflicting text model configuration: {run.checkpoint}")
    if summary["seed"] != run.seed or checkpoint["epoch"] != summary["best_epoch"]:
        raise ValueError(f"Summary/checkpoint run or best epoch mismatch: {run.checkpoint}")
    data_name = Path(str(args.data_path).replace("\\", "/")).name
    if data_name != run.ticker + ".parquet":
        raise ValueError(f"Checkpoint data_path does not match ticker {run.ticker}: {run.checkpoint}")
    data_path = Path(data_root) / data_name
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    expected_mse = float(summary["best_val_mse"])
    if not math.isfinite(expected_mse) or expected_mse < 0:
        raise ValueError(f"Invalid best_val_mse in {run.summary}")
    dataset = Dataset_Custom(
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        flag="val", used_col_prefixes=args.used_col_prefixes,
        root_path=str(data_root), data_path=data_name,
    )
    check_scaler(dataset, checkpoint, run.checkpoint)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    with execution_device(device):
        film = film_module.Model(SimpleNamespace(**saved_args)).to(device)
        text_model = build_text_model(config, args.group_weight_mode).to(device)
        film.load_state_dict(checkpoint["model_state_dict"], strict=True)
        text_model.load_state_dict(checkpoint["text_model_state_dict"], strict=True)
        metrics = evaluate_batches(film, text_model, loader, args.pred_len, args.label_len, device)
    actual_mse = metrics["full_mse"]
    if not np.isclose(actual_mse, expected_mse, rtol=1e-4, atol=1e-6):
        raise ValueError(
            f"Validation MSE mismatch for {run.checkpoint}: full_mse={actual_mse:.17g}, "
            f"summary.best_val_mse={expected_mse:.17g} (rtol=1e-4, atol=1e-6)"
        )
    return {
        "ticker": run.ticker, "seed": run.seed, "checkpoint": str(run.checkpoint),
        "data_path": str(data_path), "best_epoch": checkpoint["epoch"], "summary_best_val_mse": expected_mse,
        **metrics,
        **{f"full_vs_{variant}_mse_change_pct": mse_change_pct(actual_mse, metrics[f"{variant}_mse"]) for variant in VARIANTS[1:]},
        **{f"full_better_{variant}": int(actual_mse < metrics[f"{variant}_mse"]) for variant in VARIANTS[1:]},
    }


def aggregate(rows):
    by_seed = []
    for seed in SEEDS:
        cohort = [row for row in rows if row["seed"] == seed]
        if len(cohort) != 10 or len({row["ticker"] for row in cohort}) != 10:
            raise ValueError(f"Expected 10 distinct ticker results for seed {seed}")
        record = {"seed": seed, "runs": len(cohort)}
        for metric in ("alpha", *METRICS):
            record[metric] = float(np.mean([row[metric] for row in cohort]))
        for variant in VARIANTS[1:]:
            record[f"full_vs_{variant}_mse_change_pct"] = mse_change_pct(record["full_mse"], record[f"{variant}_mse"])
            record[f"full_better_{variant}"] = sum(row[f"full_better_{variant}"] for row in cohort)
        by_seed.append(record)
    stats = []
    for metric in ("alpha", *METRICS, *PERCENT_METRICS):
        values = [row[metric] for row in by_seed]
        stats.append({
            "metric": metric, "mean": float(np.mean(values)) if None not in values else None,
            "std": float(np.std(values, ddof=1)) if None not in values else None,
            "n_seeds": 3, "std_ddof": 1,
        })
    mean_full = float(np.mean([row["full_mse"] for row in by_seed]))
    comparisons = []
    for variant in VARIANTS[1:]:
        mean_reference = float(np.mean([row[f"{variant}_mse"] for row in by_seed]))
        comparisons.append({
            "reference": variant, "full_better_count": sum(row[f"full_better_{variant}"] for row in rows),
            "runs": len(rows), "full_mse_mean": mean_full, "reference_mse_mean": mean_reference,
            "mse_change_pct": mse_change_pct(mean_full, mean_reference),
        })
    return by_seed, stats, comparisons


def report_paths(results_root):
    directory = Path(results_root) / "diagnostics" / "price_anchor"
    return [directory / name for name in REPORT_NAMES]


def write_reports(paths, tables):
    # No training artifacts are written; existing diagnostic CSVs are preserved too.
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Diagnostic CSV already exists in {paths[0].parent}; preserve or move the previous report first")
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths, tables):
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def analyze(results_root=DEFAULT_RESULTS, data_root=DEFAULT_DATA, device="cpu"):
    runs = discover_runs(results_root)
    paths = report_paths(results_root)
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Existing diagnostic CSVs in {paths[0].parent}; refusing to overwrite")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use the default --device cpu")
    print(f"Validated 30 unique checkpoints; split=validation device={device}", flush=True)
    rows = []
    for run in runs:
        row = evaluate_run(run, data_root, device)
        rows.append(row)
        print(f"VALIDATION {len(rows)}/30 ticker={run.ticker} seed={run.seed} "
              f"alpha={row['alpha']:.8f} full_mse={row['full_mse']:.8f}", flush=True)
    by_seed, stats, comparisons = aggregate(rows)
    write_reports(paths, [rows, by_seed, stats, comparisons])
    print("\nPer-seed means over 10 tickers:")
    for row in by_seed:
        print(f"seed={row['seed']} alpha={row['alpha']:.8g} " + " ".join(f"{name}={row[name]:.8g}" for name in METRICS))
        for name in PERCENT_METRICS:
            print(f"  {name}={row[name] if row[name] is not None else 'undefined'}")
    print("\nMean and sample std between 3 seeds (ddof=1):")
    for row in stats:
        print(f"{row['metric']}: mean={row['mean']} std={row['std']}")
    print("\nMSE change = 100*(full-reference)/reference; negative means full is better.")
    for row in comparisons:
        print(f"vs {row['reference']}: change_pct={row['mse_change_pct']} "
              f"full_better={row['full_better_count']}/{row['runs']}")
    print(f"CSV reports: {paths[0].parent}")
    return rows, by_seed, stats, comparisons


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num_threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.num_threads < 1:
        parser.error("--num_threads must be positive")
    torch.set_num_threads(args.num_threads)
    try:
        analyze(args.results_root, args.data_root, args.device)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    cli()
