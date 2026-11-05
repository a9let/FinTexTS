"""Validation-only beta/uniform interpolation using frozen attention5 checkpoints."""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_blend_price_anchor import execution_device, check_scaler
from analyze_persistence_baseline import Run, split_parameters
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset
from forecasting_task.models import FiLM as film_module
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model
from forecasting_task.models.AttentionPool import attention5_config
from forecasting_task.window_evidence import window_evidence, check_validation_dates


DEFAULT_RESULTS = Path("/auto/datasets/kondyurin/FinTexTS/runs/context_price_attention5_10_v1")
DEFAULT_DATA = Path("/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts")
TICKERS = tuple("AAPL ABBV ABT ACN ADBE ADI AMAT AMD AMGN AMZN".split())
SEEDS = (7, 17, 27)
LAMBDAS = (0., .5, 1.)
MSE_KEYS = ("mse_lambda0", "mse_lambda05", "mse_lambda1")


def discover_runs(results_root):
    root = Path(results_root)/"full/context_price_only"
    paths = sorted(root.glob("*/seed*/output/best_checkpoint.pt"))
    expected = {root/ticker/f"seed{seed}/output/best_checkpoint.pt" for ticker, seed in itertools.product(TICKERS, SEEDS)}
    if set(paths) != expected or len(paths) != 30:
        raise ValueError(f"{root}: expected exactly 30 checkpoints for the same ten tickers and seeds {SEEDS}; found {len(paths)}")
    physical, runs = set(), []
    for ticker, seed in itertools.product(TICKERS, SEEDS):
        output = root/ticker/f"seed{seed}/output"
        for name in ("best_checkpoint.pt", "summary.json"):
            path = output/name
            stat = path.stat()
            identity = stat.st_dev, stat.st_ino
            if not stat.st_size or identity in physical:
                raise ValueError(f"Empty/duplicate artifact: {path}")
            physical.add(identity)
        runs.append(Run("attention5", ticker, seed, output))
    selected = Path(results_root)/"selected_tickers.txt"
    if selected.exists() and sorted(selected.read_text(encoding="utf-8").splitlines()) != sorted(TICKERS):
        raise ValueError(f"Ticker list disagrees with {selected}")
    return runs


def mix_temporal_weights(beta, present, mixing):
    if mixing not in LAMBDAS or beta.shape != present.shape or beta.shape[-1] != 5:
        raise ValueError("Expected matching five-date beta/mask and lambda in {0,0.5,1}")
    if present.dtype != torch.bool or not torch.isfinite(beta).all() or (beta < 0).any():
        raise ValueError("Invalid beta or presence mask")
    if (beta.masked_select(~present) != 0).any():
        raise ValueError("Absent dates must have zero beta")
    count = present.sum(-1, keepdim=True)
    if not torch.allclose(beta.sum(-1), (count[..., 0] > 0).to(beta.dtype), rtol=0, atol=1e-6):
        raise ValueError("Beta must sum to one for nonempty groups and zero for empty groups")
    uniform = present.to(beta.dtype)/count.clamp_min(1).to(beta.dtype)
    return (1-mixing)*uniform + mixing*beta


def interpolated_prediction(text_model, price_output, last_price, text_window, detail, mixing):
    beta = mix_temporal_weights(detail["temporal_weights"], detail["daily_present"], mixing)
    values = text_window[:, -5:].transpose(1, 2)
    values = torch.where(detail["daily_present"][..., None], values, torch.zeros_like(values))
    aggregate = (beta[..., None]*values).sum(-2)
    heads = torch.stack([text_model.group_mlps[group](aggregate[:, index])
                         for index, group in enumerate(TEXT_GROUPS)], dim=1)
    masked = heads*detail["present"][..., None].to(heads.dtype)
    delta = (detail["weights"][..., None]*masked).sum(1).unsqueeze(-1).expand_as(price_output)
    return (1-text_model.fixed_alpha)*price_output + text_model.fixed_alpha*(last_price[:, None]+delta)


@torch.no_grad()
def evaluate_batches(film, text_model, loader, pred_len, label_len, device):
    film.eval()
    text_model.eval()
    sums = np.zeros(3, dtype=np.float64)
    elements = 0
    max_prediction_difference = 0.
    for batch in loader:
        x, y, text, x_mark, y_mark = [v.float().to(device) for v in batch[:5]]
        truth = y[:, -pred_len:]
        decoder = torch.cat((y[:, :label_len], torch.zeros_like(truth)), dim=1)
        metadata = {key: value.float().to(device) for key, value in batch[6].items()}
        price = film(x, x_mark, decoder, y_mark)
        original, detail = text_model(price, x[:, -1], text, price_window=x, return_details=True, **metadata)
        for index, mixing in enumerate(LAMBDAS):
            prediction = interpolated_prediction(text_model, price, x[:, -1], text, detail, mixing)
            if prediction.shape != truth.shape or prediction.ndim != 3 or prediction.shape[-1] != 4:
                raise ValueError("Forecast/target must match [batch,pred_len,4]")
            if mixing == 1:
                max_prediction_difference = max(max_prediction_difference, (prediction-original).abs().max().item())
                if not torch.allclose(prediction, original, rtol=1e-5, atol=1e-6):
                    raise ValueError("lambda=1 predictions disagree with the original trained forward")
            sums[index] += (prediction.double()-truth.double()).square().sum().item()
        elements += truth.numel()
    if not elements or not np.isfinite(sums).all():
        raise ValueError("Empty validation split or nonfinite MSE")
    return {**dict(zip(MSE_KEYS, (sums/elements).tolist())), "num_elements": elements,
            "lambda1_max_prediction_difference": max_prediction_difference}


@torch.no_grad()
def evaluate_run(run, data_root, device):
    path = run.output/"best_checkpoint.pt"
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved = checkpoint["args"]
        summary = json.loads((run.output/"summary.json").read_text(encoding="utf-8"))
        args = SimpleNamespace(**saved)
        if (args.model_type.lower() != "film" or args.seed != run.seed or summary["seed"] != run.seed
                or checkpoint["epoch"] != summary["best_epoch"]
                or args.group_gate != "context_price_only" or args.group_weight_mode != "context_price_only"
                or args.text_temporal_pool != "attention5" or args.fusion_mode != "blend"
                or tuple(args.used_col_prefixes) != TEXT_GROUPS or args.text_num_layers != 1
                or args.text_weight != .1 or args.fixed_alpha != .1):
            raise ValueError("Checkpoint is not the requested best FiLM/attention5/fixed-alpha=0.1 run")
        for source in (saved, summary, args.text_model_config):
            if (source.get("attention_config") != attention5_config() or source.get("text_temporal_pool") != "attention5"
                    or source.get("group_gate") != "context_price_only" or source.get("price_context_lags", [1, 5]) != [1, 5]):
                raise ValueError("Saved attention architecture/gate configuration mismatch")
        split_parameters(saved, checkpoint)
        data_name = Path(str(args.data_path).replace("\\", "/")).name
        if data_name != run.ticker+".parquet" or "fnspid" in args.data_path:
            raise ValueError("Checkpoint data/preprocessing does not match requested ticker")
        if Path(str(summary["data_path"]).replace("\\", "/")).name != data_name:
            raise ValueError("Summary ticker mismatch")
        expected_mse = float(summary["best_val_mse"])
        if not math.isfinite(expected_mse) or expected_mse < 0:
            raise ValueError("Invalid saved best validation MSE")
        base = Dataset_Custom(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
                              flag="val", used_col_prefixes=args.used_col_prefixes,
                              root_path=str(data_root), data_path=data_name)
        dataset = ContextNoveltyDataset(base)
        check_scaler(dataset, checkpoint, path)
        evidence = window_evidence(dataset)
        for source in (saved, summary):
            if source.get("window_evidence", {}).get("val") != evidence:
                raise ValueError("Validation window/date/target fingerprints mismatch")
        sidecar = run.output/"window_evidence.json"
        if sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")).get("val") != evidence:
            raise ValueError("Sidecar validation fingerprints mismatch")
        diagnostics = run.output/"validation_group_diagnostics.csv"
        if diagnostics.exists():
            check_validation_dates(diagnostics, evidence)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
        with execution_device(device):
            film = film_module.Model(SimpleNamespace(**saved)).to(device)
            text = build_text_model(args.text_model_config, args.group_weight_mode).to(device)
            film.load_state_dict(checkpoint["model_state_dict"], strict=True)
            text.load_state_dict(checkpoint["text_model_state_dict"], strict=True)
            if (text.text_temporal_pool != "attention5" or text.attention_pool is None
                    or not torch.isclose(text.fixed_alpha, torch.tensor(.1, device=device), rtol=0, atol=1e-8)):
                raise ValueError("Restored architecture/alpha mismatch")
            film.requires_grad_(False)
            text.requires_grad_(False)
            metrics = evaluate_batches(film, text, loader, args.pred_len, args.label_len, device)
        if not np.isclose(metrics["mse_lambda1"], expected_mse, rtol=1e-4, atol=1e-6):
            raise ValueError(f"lambda=1 MSE={metrics['mse_lambda1']:.17g} != best_val_mse={expected_mse:.17g} (rtol=1e-4, atol=1e-6)")
        return {"ticker": run.ticker, "seed": run.seed, "checkpoint": str(path), "best_epoch": checkpoint["epoch"],
                "summary_best_val_mse": expected_mse, "windows": len(dataset), "scaler_verified": True,
                "window_evidence": evidence, **metrics}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise ValueError(f"{path}: {error}") from error


def aggregate(rows):
    if len(rows) != 30 or {(r["ticker"], r["seed"]) for r in rows} != set(itertools.product(TICKERS, SEEDS)):
        raise ValueError("Expected exactly 30 unique ticker/seed results")
    by_seed, per_ticker = [], []
    for seed in SEEDS:
        group = [r for r in rows if r["seed"] == seed]
        by_seed.append({"seed": seed, "tickers": 10, **{key: float(np.mean([r[key] for r in group])) for key in MSE_KEYS}})
    for ticker in TICKERS:
        group = [r for r in rows if r["ticker"] == ticker]
        per_ticker.append({"ticker": ticker, "seeds": 3, **{key: float(np.mean([r[key] for r in group])) for key in MSE_KEYS}})
    across = [{"lambda": mixing, "mean": float(np.mean([r[key] for r in by_seed])),
               "sample_std": float(np.std([r[key] for r in by_seed], ddof=1)), "std_ddof": 1}
              for mixing, key in zip(LAMBDAS, MSE_KEYS)]
    comparisons = []
    for mixing, key in zip(LAMBDAS[:2], MSE_KEYS[:2]):
        for row in per_ticker:
            row[key+"_minus_lambda1"] = row[key]-row["mse_lambda1"]
            row[key+"_improvement_pct"] = 100*(row["mse_lambda1"]-row[key])/row["mse_lambda1"] if row["mse_lambda1"] else None
        reference = across[2]["mean"]
        value = across[int(mixing*2)]["mean"]
        comparisons.append({"lambda": mixing, "reference_lambda": 1, "smoothed_wins": sum(r[key] < r["mse_lambda1"] for r in per_ticker),
                            "smoothed_losses": sum(r[key] > r["mse_lambda1"] for r in per_ticker),
                            "ties": sum(r[key] == r["mse_lambda1"] for r in per_ticker), "tickers": 10,
                            "mean_mse_difference": value-reference,
                            "improvement_pct": 100*(reference-value)/reference if reference else None})
    return {"by_seed": by_seed, "across_seeds": across, "per_ticker": per_ticker, "comparisons": comparisons}


def analyze(results_root=DEFAULT_RESULTS, data_root=DEFAULT_DATA, output_dir=None, device="cpu"):
    output = Path(output_dir) if output_dir else Path(results_root)/"diagnostics/uniform_ablation"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Report already exists: {output}")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable; CPU is the default")
    runs = discover_runs(results_root)
    rows = []
    for index, run in enumerate(runs, 1):
        row = evaluate_run(run, data_root, device)
        rows.append(row)
        print(f"VALIDATION {index}/30 {run.ticker} seed={run.seed} " + " ".join(f"{k}={row[k]:.9g}" for k in MSE_KEYS), flush=True)
    tables = {"per_run": [{k: v for k, v in row.items() if k != "window_evidence"} for row in rows], **aggregate(rows)}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        with (output/f"{name}.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (output/"verification.json").open("x", encoding="utf-8") as handle:
        json.dump({"device": str(device), "lambdas": LAMBDAS, "split": "validation_only",
                   "mse_tolerance": {"rtol": 1e-4, "atol": 1e-6},
                   "runs": [{"ticker": r["ticker"], "seed": r["seed"], "window_evidence": r["window_evidence"]} for r in rows],
                   "interpretation": "Frozen attention5 heads, FiLM and price gate; only inference beta is interpolated. Lambda=0 is not a separately trained mean5 model. Positive improvement_pct means lower MSE; wins use each ticker's three-seed mean. This is validation diagnostics, not test evidence."},
                  handle, indent=2, allow_nan=False)
    for name in ("by_seed", "across_seeds", "per_ticker", "comparisons"):
        print(name.upper())
        for record in tables[name]:
            print(json.dumps(record, allow_nan=False))
    print(f"Reports: {output}")
    return tables


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    analyze(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    cli()
