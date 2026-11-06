"""Verify saved last-date controls and bind new runs to the same replayed windows."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_blend_price_anchor import execution_device, check_scaler
from analyze_persistence_baseline import split_parameters
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset, NOVELTY_POLICY
from forecasting_task.models import FiLM as film_module
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model
from forecasting_task.window_evidence import window_evidence, check_saved_test_windows, check_validation_dates


RUNS = Path("/auto/datasets/kondyurin/FinTexTS/runs")
DATA = Path("/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts")
TICKERS = tuple("AAPL ABBV ABT ACN ADBE ADI AMAT AMD AMGN AMZN".split())
SEEDS = (7, 17, 27)
SETTINGS = {"model_type": "FiLM", "seq_len": 64, "label_len": 16, "pred_len": 3,
            "batch_size": 32, "num_epoch": 50, "patience": 10, "lr": .001,
            "e_layers": 2, "task_name": "short_term_forecast", "output_attention": False,
            "enc_in": 4, "dec_in": 4, "c_out": 4, "used_col_prefixes": list(TEXT_GROUPS),
            "fusion_mode": "blend", "text_num_layers": 1, "text_weight": .1}


def check_settings(args):
    for key, expected in SETTINGS.items():
        if args.get(key) != expected:
            raise ValueError(f"Control configuration mismatch {key}: {args.get(key)} != {expected}")
    if (args.get("group_gate") != "context_price_only" or args.get("fixed_alpha") != .1
            or args.get("text_weight_mode") != "fixed" or args.get("group_weight_mode") != "context_price_only"
            or args.get("text_temporal_pool", "last") != "last" or args.get("text_history_mode", "last") != "last"
            or args.get("price_context_lags", [1, 5]) != [1, 5]):
        raise ValueError("Expected fixed-alpha=0.1 last-date text and original price-only gate")
    if args.get("novelty_policy") != NOVELTY_POLICY or args.get("sampling_policy") != "independent_train_generator":
        raise ValueError("Preprocessing or sampling policy differs from control")
    gate = {"type": "Linear", "in_features": 18, "out_features": 5, "parameters": 95, "initialization": "zeros"}
    if args.get("gate_structure") != gate:
        raise ValueError("Expected original Linear(18,5) gate")


def load_gate_control(manifest_path, args, datasets):
    """Called before new model construction/training; checks both replayed splits."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "gate_text_history_control_v1":
        raise ValueError("Unsupported gate control manifest")
    check_settings(vars(args))
    ticker = Path(str(args.data_path).replace("\\", "/")).stem
    key = f"{ticker}/seed{args.seed}"
    matches = [r for r in manifest["runs"] if r["key"] == key]
    if len(matches) != 1:
        raise ValueError(f"Missing/duplicate verified control {key}")
    reference = matches[0]
    if reference["status"] != "verified_replay" or reference.get("settings") != SETTINGS:
        raise ValueError(f"Unverified control {key}")
    for split in ("val", "test"):
        if args.window_evidence[split] != reference["window_evidence"][split]:
            raise ValueError(f"New {split} windows/targets differ from control {key}")
    for dataset in datasets:
        for name in ("mean", "scale", "var"):
            actual = np.asarray(getattr(dataset.scaler, name+"_"))
            saved = np.asarray(reference["scaler"][name])
            if actual.shape != (4,) or saved.shape != (4,) or not np.allclose(actual, saved, rtol=1e-7, atol=1e-9):
                raise ValueError(f"New scaler {name} differs from control {key}")
    return reference


@torch.no_grad()
def validate_control(output, ticker, seed, data_root):
    path = Path(output)/"best_checkpoint.pt"
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        args = checkpoint["args"]
        summary = json.loads((Path(output)/"summary.json").read_text(encoding="utf-8"))
        check_settings(args)
        split_parameters(args, checkpoint)
        if args.get("gate_text_history_mode", "none") != "none":
            raise ValueError("Reference already uses a text-history gate")
        for source in (summary, args["text_model_config"]):
            if (source.get("group_gate") != "context_price_only"
                    or source.get("text_temporal_pool", "last") != "last"
                    or source.get("text_history_mode", "last") != "last"
                    or source.get("gate_text_history_mode", "none") != "none"):
                raise ValueError("Reference condition mismatch")
        for source in (args, summary):
            name = Path(str(source["data_path"]).replace("\\", "/")).name
            if name != ticker+".parquet" or source["seed"] != seed or "fnspid" in source["data_path"]:
                raise ValueError("Reference ticker/seed/preprocessing mismatch")
        if checkpoint["epoch"] != summary["best_epoch"]:
            raise ValueError("Reference checkpoint epoch differs from summary")
        expected_mse = float(summary["best_val_mse"])
        if not np.isfinite(expected_mse) or expected_mse < 0:
            raise ValueError("Invalid saved reference MSE")
        datasets, evidence = {}, {}
        for split in ("val", "test"):
            base = Dataset_Custom(args["seq_len"], args["label_len"], args["pred_len"], split, list(TEXT_GROUPS),
                                  root_path=str(data_root), data_path=ticker+".parquet")
            datasets[split] = ContextNoveltyDataset(base)
            check_scaler(datasets[split], checkpoint, path)
            evidence[split] = window_evidence(datasets[split])
        historical_hashes = all(source.get("window_evidence") == evidence for source in (args, summary))
        for source in (args, summary):
            if "window_evidence" in source and source["window_evidence"] != evidence:
                raise ValueError("Saved reference fingerprints contradict current windows")
        check_saved_test_windows(Path(output)/"test_io.pt", evidence["test"])
        check_validation_dates(Path(output)/"validation_group_diagnostics.csv", evidence["val"])
        # No test inference: only the saved normalized test arrays are compared.
        device = torch.device("cpu")
        with execution_device(device):
            price = film_module.Model(SimpleNamespace(**args)).cpu().eval()
            text = build_text_model(args["text_model_config"], args["group_weight_mode"]).cpu().eval()
            price.load_state_dict(checkpoint["model_state_dict"], strict=True)
            text.load_state_dict(checkpoint["text_model_state_dict"], strict=True)
            total, count = 0., 0
            for batch in DataLoader(datasets["val"], batch_size=args["batch_size"], shuffle=False, drop_last=False, num_workers=0):
                x, y, e, xm, ym = [v.float() for v in batch[:5]]
                truth = y[:, -args["pred_len"]:]
                decoder = torch.cat((y[:, :args["label_len"]], torch.zeros_like(truth)), 1)
                metadata = {key: value.float() for key, value in batch[6].items()}
                prediction = text(price(x, xm, decoder, ym), x[:, -1], e, price_window=x, **metadata)
                total += (prediction.double()-truth.double()).square().sum().item()
                count += truth.numel()
        actual_mse = total/count if count else float("nan")
        if not np.isclose(actual_mse, expected_mse, rtol=1e-4, atol=1e-6):
            raise ValueError(f"Control validation MSE={actual_mse} != saved best_val_mse={expected_mse}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                digest.update(chunk)
        return {"key": f"{ticker}/seed{seed}", "checkpoint": str(path), "checkpoint_sha256": digest.hexdigest(),
                "status": "verified_replay", "settings": SETTINGS, "best_epoch": checkpoint["epoch"],
                "replayed_val_mse": actual_mse, "saved_val_mse": expected_mse, "window_evidence": evidence,
                "scaler": {name: checkpoint[f"scaler_{name}"] for name in ("mean", "scale", "var")},
                "historical_window_verification": "full_saved_fingerprints" if historical_hashes else "partial_historical_validation",
                "verification_detail": "Settings/scaler, saved validation dates, saved normalized test inputs/targets and replayed best validation MSE agree. Without historical validation fingerprints, equality of all historical validation tensors is not proven; no historical embedding hash is available."}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise ValueError(f"{path}: {error}") from error


def verify_controls(results_root, data_root, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Control report already exists: {output}")
    root = Path(results_root)/"full/context_price_only"
    expected = {root/ticker/f"seed{seed}/output/best_checkpoint.pt" for ticker, seed in itertools.product(TICKERS, SEEDS)}
    paths = list(root.glob("*/seed*/output/best_checkpoint.pt"))
    if len(paths) != 30 or set(paths) != expected:
        raise ValueError("Expected exactly 30 saved last-date controls for the prescribed tickers/seeds")
    seen = set()
    for path in paths:
        stat = path.stat()
        identity = stat.st_dev, stat.st_ino
        if not stat.st_size or identity in seen:
            raise ValueError(f"Empty/duplicate control checkpoint: {path}")
        seen.add(identity)
    records = []
    for ticker, seed in itertools.product(TICKERS, SEEDS):
        record = validate_control(root/ticker/f"seed{seed}/output", ticker, seed, data_root)
        records.append(record)
        print(f"CONTROL OK {ticker} seed={seed} {record['historical_window_verification']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump({"schema": "gate_text_history_control_v1", "runs": records}, handle, indent=2, allow_nan=False)
    return records


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=RUNS/"context_price_only_10_v1")
    parser.add_argument("--data_root", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    verify_controls(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    cli()
