"""Read-only learning-curve and best-validation temporal-weight diagnostics.

Reads history, YAML configuration, summary and validation CSV only. No model,
checkpoint tensors, test predictions or training entry points are loaded.
"""

import argparse
import csv
from datetime import date
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_RUNS = Path("/auto/datasets/kondyurin/FinTexTS/runs")
TICKERS = tuple("AAPL ABBV ABT ACN ADBE ADI AMAT AMD AMGN AMZN".split())
SEEDS = (7, 17, 27)
VARIANTS = ("attention5", "mean5")
GROUPS = ("macro", "sector", "targetCompany", "relatedCompany", "filing")
AGES = (4, 3, 2, 1, 0)
METRICS = ("l1_uniform", "max_beta", "effective_dates", "effective_ratio")
CONFIG_FIELDS = ("seq_len", "label_len", "pred_len", "batch_size", "model_type", "lr",
                 "num_epoch", "patience", "used_col_prefixes", "fusion_mode", "text_num_layers",
                 "text_weight", "gate_structure", "novelty_policy", "sampling_policy", "text_mlp_structure")
ARTIFACTS = ("args.yaml", "summary.json", "history.jsonl", "validation_group_diagnostics.csv")


def finite(value, label):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {label}")
    return value


def beta_metrics(beta, present):
    """No normalization/repair: malformed saved coefficients are rejected."""
    beta = np.asarray(beta, dtype=float)
    mask = np.asarray(present)
    if beta.shape != (5,) or mask.shape != (5,) or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected five beta values and binary daily masks")
    mask = mask.astype(bool)
    if not np.isfinite(beta).all() or np.any(beta < 0) or np.any(beta > 1+1e-6):
        raise ValueError("Invalid beta range/finiteness")
    if np.any(beta[~mask] != 0):
        raise ValueError("Absent dates must have exactly zero beta")
    count = int(mask.sum())
    if not count:
        return {"n_present": 0, "window_kind": "empty", "selectivity_eligible": False,
                **dict.fromkeys(METRICS), "uniform_max_beta": None, "uniform_effective_dates": None}
    if not np.isclose(beta.sum(), 1, rtol=0, atol=1e-6):
        raise ValueError(f"Nonempty beta sum must equal one; found {beta.sum()}")
    effective = 1/float(np.square(beta).sum())
    return {"n_present": count, "window_kind": "single" if count == 1 else "multiple",
            "selectivity_eligible": count >= 2,
            "l1_uniform": float(np.abs(beta[mask]-1/count).sum()), "max_beta": float(beta.max()),
            "effective_dates": effective, "effective_ratio": effective/count,
            "uniform_max_beta": 1/count, "uniform_effective_dates": count}


def history_statistics(history, summary):
    if not history or [r["epoch"] for r in history] != list(range(1, len(history)+1)):
        raise ValueError("History epochs must be nonempty, contiguous and unique from 1")
    for row in history:
        if type(row["epoch"]) is not int:
            raise ValueError("History epoch must be an integer")
        for metric in ("train_mse", "val_mse"):
            row[metric] = finite(row[metric], metric)
            if row[metric] < 0:
                raise ValueError(f"Negative {metric}")
    best = int(np.argmin([row["val_mse"] for row in history]))+1
    if summary["best_epoch"] != best or summary["epochs_completed"] != len(history):
        raise ValueError("Summary best epoch/completed count disagrees with history")
    if not np.isclose(finite(summary["best_val_mse"], "best_val_mse"), history[best-1]["val_mse"], rtol=1e-5, atol=1e-7):
        raise ValueError("Summary best validation MSE disagrees with history")
    result = {"best_epoch": best, "epochs_completed": len(history), "block_size": 5,
              "block_status": "complete" if len(history) >= 10 else "insufficient_epochs",
              "previous_start": None, "previous_end": None, "last_start": None, "last_end": None}
    for metric in ("train_mse", "val_mse"):
        result.update({f"{metric}_{suffix}": None for suffix in ("previous5", "last5", "change", "change_pct")})
    if len(history) >= 10:
        result.update(previous_start=len(history)-9, previous_end=len(history)-5,
                      last_start=len(history)-4, last_end=len(history))
        for metric in ("train_mse", "val_mse"):
            before = float(np.mean([r[metric] for r in history[-10:-5]]))
            after = float(np.mean([r[metric] for r in history[-5:]]))
            result.update({f"{metric}_previous5": before, f"{metric}_last5": after,
                           f"{metric}_change": after-before,
                           f"{metric}_change_pct": 100*(after-before)/before if before else None})
    return result


def validate_configuration(args, summary, variant, ticker, seed):
    if args["seed"] != seed or summary["seed"] != seed:
        raise ValueError("Saved seed disagrees with directory")
    for source in (args, summary):
        if Path(source["data_path"].replace("\\", "/")).name != ticker+".parquet":
            raise ValueError("Ticker disagrees with saved data_path")
        if (source.get("text_temporal_pool") != variant or source.get("group_gate") != "context_price_only"
                or source.get("fixed_alpha") != .1 or source.get("text_weight_mode") != "fixed"):
            raise ValueError("Expected fixed-alpha=0.1 context_price_only and requested pooling")
    expected = {"seq_len": 64, "label_len": 16, "pred_len": 3, "batch_size": 32,
                "num_epoch": 50, "patience": 10, "lr": .001, "model_type": "FiLM",
                "used_col_prefixes": list(GROUPS), "fusion_mode": "blend", "text_num_layers": 1, "text_weight": .1}
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"Unexpected {key}: {args.get(key)}")
    for field in CONFIG_FIELDS:
        if field not in args:
            raise ValueError(f"Missing saved configuration {field}")
    if args.get("price_context_lags", [1, 5]) != [1, 5]:
        raise ValueError("Expected original gate lags [1,5]")


def read_validation(path, variant):
    rows = []
    last_date = None
    with path.open(newline="", encoding="utf-8") as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            window, group = divmod(position, len(GROUPS))
            if int(row["window_index"]) != window or row["group"] != GROUPS[group] or row["text_temporal_pool"] != variant:
                raise ValueError(f"{path}: invalid window/group/pooling order")
            start, end = row["text_window_start_date"], row["text_window_end_date"]
            if date.fromisoformat(start) >= date.fromisoformat(end) or row["input_end_date"] != end:
                raise ValueError(f"{path}: invalid five-row window boundaries")
            if not group:
                if last_date is not None and end <= last_date:
                    raise ValueError(f"{path}: duplicate or unordered validation dates")
                last_date = end
            elif any(row[k] != rows[-1][k] for k in ("input_end_date", "text_window_start_date", "text_window_end_date")):
                raise ValueError(f"{path}: group window dates disagree")
            count, present = int(row["available_dates"]), int(row["present"])
            if count not in range(6) or present != int(count > 0):
                raise ValueError(f"{path}: invalid group availability")
            parsed = {"window_index": window, "group": row["group"], "input_end_date": end,
                      "text_window_start_date": start, "text_window_end_date": end, "n_present": count}
            if variant == "attention5":
                dates = [date.fromisoformat(row[f"date_d{d}"]) for d in range(1, 6)]
                if dates != sorted(set(dates)) or dates[0].isoformat() != start or dates[-1].isoformat() != end:
                    raise ValueError(f"{path}: invalid individual five-row dates")
                parsed["daily_dates"] = "|".join(d.isoformat() for d in dates)
                if group and parsed["daily_dates"] != rows[-1]["daily_dates"]:
                    raise ValueError(f"{path}: individual dates disagree between groups")
                if [int(row[f"age_d{d}"]) for d in range(1, 6)] != list(AGES):
                    raise ValueError(f"{path}: wrong row ages")
                mask = [int(row[f"present_d{d}"]) for d in range(1, 6)]
                beta = [float(row[f"beta_d{d}"]) for d in range(1, 6)]
                try:
                    stats = beta_metrics(beta, mask)
                except ValueError as error:
                    raise ValueError(f"{path}: window={window}, group={row['group']}: {error}") from error
                if stats["n_present"] != count:
                    raise ValueError(f"{path}: daily mask/count mismatch")
                parsed.update(stats)
                for age, weight, available in zip(AGES, beta, mask):
                    parsed[f"beta_age{age}"] = weight
                    parsed[f"present_age{age}"] = available
            rows.append(parsed)
    if not rows or len(rows) % len(GROUPS):
        raise ValueError(f"{path}: incomplete/empty validation windows")
    return rows


def load_runs(runs_root):
    histories, summaries, weights, audit = [], [], [], []
    expected = set(itertools.product(TICKERS, SEEDS))
    physical = set()
    reference = {}
    for variant in VARIANTS:
        root = Path(runs_root)/f"context_price_{variant}_10_v1/full/context_price_only"
        directories = list(root.glob("*/seed*/output"))
        expected_paths = {root/t/f"seed{s}/output" for t, s in expected}
        if set(directories) != expected_paths or len(directories) != 30:
            raise ValueError(f"{root}: expected exactly the same 10 tickers x seeds 7,17,27; found {len(directories)} runs")
        for ticker, seed in itertools.product(TICKERS, SEEDS):
            output = root/ticker/f"seed{seed}/output"
            try:
                for name in ARTIFACTS:
                    stat = (output/name).stat()
                    identity = stat.st_dev, stat.st_ino
                    if not stat.st_size or identity in physical:
                        raise ValueError(f"Empty/duplicate artifact {name}")
                    physical.add(identity)
                args = yaml.safe_load((output/"args.yaml").read_text(encoding="utf-8"))
                summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
                validate_configuration(args, summary, variant, ticker, seed)
                history = [json.loads(line) for line in (output/"history.jsonl").read_text(encoding="utf-8").splitlines()]
                stats = history_statistics(history, summary)
                if len(history) > args["num_epoch"]:
                    raise ValueError("History exceeds configured training budget")
                rows = read_validation(output/"validation_group_diagnostics.csv", variant)
                diagnostic = summary["validation_diagnostics"]
                if (diagnostic["checkpoint"] != "best_checkpoint.pt" or diagnostic["windows"] != len(rows)//5
                        or diagnostic["rows"] != len(rows) or not np.isclose(diagnostic["mse"], summary["best_val_mse"], rtol=1e-5, atol=1e-7)):
                    raise ValueError("Validation diagnostic metadata disagrees with selected checkpoint")
                evidence = args["window_evidence"]["val"]
                if evidence != summary["window_evidence"]["val"] or evidence["windows"] != len(rows)//5:
                    raise ValueError("Saved validation window evidence mismatch")
                if ticker in reference:
                    previous = reference[ticker]
                    for field in CONFIG_FIELDS:
                        if args[field] != previous["args"][field]:
                            raise ValueError(f"Paired configuration mismatch: {field}")
                    if evidence != previous["evidence"]:
                        raise ValueError("Paired validation window fingerprints mismatch")
                    for actual, original in zip(rows, previous["rows"]):
                        for field in ("window_index", "group", "input_end_date", "text_window_start_date", "text_window_end_date", "n_present"):
                            if actual[field] != original[field]:
                                raise ValueError(f"Paired window/count mismatch: {field}")
                        if variant == "attention5":
                            for field in ("daily_dates", *(f"present_age{age}" for age in AGES)):
                                if actual[field] != original[field]:
                                    raise ValueError(f"Attention dates/masks differ across seeds: {field}")
                else:
                    reference[ticker] = {"args": args, "evidence": evidence, "rows": rows}
                identity = {"variant": variant, "ticker": ticker, "seed": seed}
                histories.extend({**identity, "epoch": r["epoch"], "train_mse": r["train_mse"], "val_mse": r["val_mse"],
                                  "selected_best": r["epoch"] == stats["best_epoch"]} for r in history)
                summaries.append({**identity, **stats})
                if variant == "attention5":
                    weights.extend({"ticker": ticker, "seed": seed, **r} for r in rows)
                audit.append({**identity, "windows": len(rows)//5, "best_epoch": stats["best_epoch"],
                              "diagnostic_best_mse_verified": True, "saved_val_fingerprints_match": True,
                              "window_dates_counts_match": True})
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{output}: {error}") from error
    return pd.DataFrame(histories), pd.DataFrame(summaries), pd.DataFrame(weights), pd.DataFrame(audit)


def curve_statistics(history):
    """Available-run and same-epoch paired-cohort curves, with explicit membership."""
    rows = []
    for seed in ("all", *SEEDS):
        scope = history if seed == "all" else history[history.seed == seed]
        for epoch in range(1, int(scope.epoch.max())+1):
            points = scope[scope.epoch == epoch]
            keys = {v: set(zip(points[points.variant == v].ticker, points[points.variant == v].seed)) for v in VARIANTS}
            paired = keys["attention5"] & keys["mean5"]
            for cohort in ("available", "paired"):
                for variant in VARIANTS:
                    group = points[points.variant == variant]
                    if cohort == "paired":
                        group = group.loc[np.array([key in paired for key in zip(group.ticker, group.seed)], dtype=bool)]
                    members = sorted(zip(group.ticker, group.seed))
                    rows.append({"seed_scope": seed, "epoch": epoch, "cohort": cohort, "variant": variant,
                                 "n_runs": len(group), "members": ";".join(f"{t}:seed{s}" for t, s in members),
                                 "train_mse": float(group.train_mse.mean()) if len(group) else None,
                                 "val_mse": float(group.val_mse.mean()) if len(group) else None})
    return pd.DataFrame(rows)


def weight_statistics(weights):
    distributions, ages = [], []
    scopes = [weights.assign(seed_scope=weights.seed.astype(str)), weights.assign(seed_scope="all")]
    for frame in scopes:
        for (seed, group, count), values in frame.groupby(["seed_scope", "group", "n_present"], sort=True):
            common = {"seed_scope": seed, "group": group, "n_present": int(count), "n_windows": len(values),
                      "n_tickers": values.ticker.nunique(), "selectivity_eligible": count >= 2}
            for metric in METRICS:
                sample = values[metric].dropna().to_numpy(dtype=float)
                record = {**common, "metric": metric, "n_defined": len(sample), "mean": None,
                          **dict.fromkeys(("min", "q05", "q25", "median", "q75", "q95", "max"))}
                if len(sample):
                    record.update(mean=float(np.mean(sample)))
                    record.update(zip(("min", "q05", "q25", "median", "q75", "q95", "max"),
                                      map(float, np.quantile(sample, [0, .05, .25, .5, .75, .95, 1]))))
                distributions.append(record)
            for age in AGES:
                present = values[f"present_age{age}"].to_numpy(dtype=bool)
                beta = values[f"beta_age{age}"].to_numpy(dtype=float)
                uniform = present.astype(float)/max(count, 1)
                quantiles = np.quantile(beta[present], [.05, .5, .95]) if present.any() else [None]*3
                ages.append({**common, "age": age, "n_available": int(present.sum()),
                             "mean_beta_all_windows": float(beta.mean()), "uniform_mean_all_windows": float(uniform.mean()),
                             "mean_beta_if_present": float(beta[present].mean()) if present.any() else None,
                             "uniform_if_present": 1/count if present.any() else None,
                             "q05_beta_if_present": quantiles[0], "median_beta_if_present": quantiles[1],
                             "q95_beta_if_present": quantiles[2],
                             "mean_difference_if_present": float((beta-uniform)[present].mean()) if present.any() else None})
    counts = weights.groupby(["seed", "group", "n_present", "window_kind"], as_index=False).size().rename(columns={"size": "n_windows"})
    per_run = weights.groupby(["ticker", "seed", "group", "n_present"], as_index=False)[list(METRICS)].mean()
    return {"weight_quantiles": pd.DataFrame(distributions), "age_weights": pd.DataFrame(ages),
            "window_counts": counts, "weights_by_run": per_run}


def make_plots(history, runs, curves, distributions, ages, output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def save(fig, name):
        FigureCanvasAgg(fig)
        with (output/name).open("xb") as handle:
            fig.savefig(handle, format="png", dpi=130)

    for ticker, seed in itertools.product(TICKERS, SEEDS):
        figure = Figure(figsize=(11, 4), tight_layout=True)
        axes = figure.subplots(1, 2, sharey=True)
        for axis, variant in zip(axes, VARIANTS):
            frame = history[(history.ticker == ticker) & (history.seed == seed) & (history.variant == variant)]
            best = frame[frame.selected_best].iloc[0]
            axis.plot(frame.epoch, frame.train_mse, label="train MSE")
            axis.plot(frame.epoch, frame.val_mse, label="validation MSE")
            axis.axvline(best.epoch, color="black", linestyle=":", label=f"best epoch {best.epoch}")
            axis.scatter([best.epoch], [best.val_mse], color="black", zorder=3)
            axis.set(title=f"{ticker} seed={seed}: {variant}", xlabel="Epoch", ylabel="Normalized OHLC MSE")
            axis.legend(fontsize="small")
        save(figure, f"learning_{ticker}_seed{seed}.png")
    for seed in ("all", *SEEDS):
        figure = Figure(figsize=(11, 9), tight_layout=True)
        axes = figure.subplots(3, 1)
        scope = curves[curves.seed_scope == seed]
        for variant, color in zip(VARIANTS, ("tab:blue", "tab:orange")):
            for cohort, style in (("available", "-"), ("paired", "--")):
                group = scope[(scope.variant == variant) & (scope.cohort == cohort)]
                label = f"{variant} ({cohort})"
                for axis, metric in zip(axes[:2], ("train_mse", "val_mse")):
                    axis.plot(group.epoch, group[metric], style, color=color, label=label)
                    axis.set(ylabel=metric, xlabel="Epoch")
                axes[2].step(group.epoch, group.n_runs, where="mid", linestyle=style, color=color, label=label)
        axes[0].set_title(f"seed={seed}: available and paired cohorts; membership can change")
        axes[2].set(xlabel="Epoch", ylabel="Remaining runs", ylim=(0, None))
        for axis in axes:
            axis.legend(fontsize="small")
            axis.grid(alpha=.2)
        save(figure, f"learning_aggregate_seed{seed}.png")
    figure = Figure(figsize=(10, 4), tight_layout=True)
    axes = figure.subplots(1, 2)
    for axis, variant in zip(axes, VARIANTS):
        group = runs[runs.variant == variant]
        axis.hist(group.best_epoch, bins=np.arange(.5, group.epochs_completed.max()+1.5, 1))
        axis.set(title=variant, xlabel="Selected best epoch", ylabel="Run count")
    save(figure, "best_epochs.png")
    for seed in ("all", *map(str, SEEDS)):
        figure = Figure(figsize=(15, 10), tight_layout=True)
        axes = figure.subplots(4, 5, squeeze=False)
        for column, group in enumerate(GROUPS):
            for axis, metric in zip(axes[:, column], METRICS):
                frame = distributions[(distributions.seed_scope == seed) & (distributions.group == group)
                                      & (distributions.metric == metric) & (distributions.n_present > 0)]
                axis.plot(frame.n_present, frame["median"].to_numpy(dtype=float), marker="o", label="median")
                axis.fill_between(frame.n_present.to_numpy(dtype=float), frame.q05.to_numpy(dtype=float), frame.q95.to_numpy(dtype=float), alpha=.2, label="5–95%")
                axis.plot(frame.n_present, frame["mean"].to_numpy(dtype=float), linestyle="--", label="mean")
                reference = {"l1_uniform": np.zeros(len(frame)), "max_beta": 1/frame.n_present,
                             "effective_dates": frame.n_present, "effective_ratio": np.ones(len(frame))}[metric]
                axis.plot(frame.n_present, reference, linestyle=":", color="black", label="uniform")
                axis.set(title=f"{group}: {metric}", xlabel="Present dates N (N=1 is not selectivity)", xticks=range(1, 6))
                axis.legend(fontsize="x-small")
        figure.suptitle(f"seed={seed}; distributions pool windows within group/N; empty windows in CSV")
        save(figure, f"beta_metrics_seed{seed}.png")
        figure = Figure(figsize=(15, 13), tight_layout=True)
        axes = figure.subplots(5, 5, squeeze=False)
        for row, count in enumerate(range(1, 6)):
            for column, group in enumerate(GROUPS):
                axis = axes[row, column]
                frame = ages[(ages.seed_scope == seed) & (ages.group == group) & (ages.n_present == count)].sort_values("age", ascending=False)
                axis.plot(frame.age, frame.mean_beta_if_present, "o-", label="beta | date present")
                axis.fill_between(frame.age.to_numpy(dtype=float), frame.q05_beta_if_present.to_numpy(dtype=float),
                                  frame.q95_beta_if_present.to_numpy(dtype=float), alpha=.2, label="5–95% | present")
                axis.plot(frame.age, frame.uniform_if_present, ":", color="black", label="uniform 1/N")
                axis.set(title=f"{group}, N={count}", xlabel="Row age", xticks=list(AGES), xlim=(4.2, -.2))
                axis.legend(fontsize="x-small")
        figure.suptitle(f"seed={seed}; conditional age weights (availability counts and all-window means in CSV)")
        save(figure, f"beta_age_seed{seed}.png")


def analyze(runs_root=DEFAULT_RUNS, output_dir=None):
    output = Path(output_dir) if output_dir else Path(runs_root)/"context_price_attention5_10_v1/diagnostics/attention_behavior"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Report directory already exists: {output}")
    history, runs, weights, audit = load_runs(runs_root)
    curves = curve_statistics(history)
    blocks = []
    for (variant, seed), group in runs.groupby(["variant", "seed"]):
        complete = group[group.block_status == "complete"]
        blocks.append({"variant": variant, "seed": seed, "n_runs": len(group), "n_with_two_blocks": len(complete),
                       **{metric: float(complete[metric].mean()) if len(complete) else None
                          for metric in ("train_mse_change", "val_mse_change", "train_mse_change_pct", "val_mse_change_pct")}})
    best_distribution = runs.groupby(["variant", "seed", "best_epoch"], as_index=False).size().rename(columns={"size": "n_runs"})
    tables = {"history": history, "learning_curves": curves, "run_summary": runs,
              "best_epoch_distribution": best_distribution, "block_changes_by_seed": pd.DataFrame(blocks),
              "weights_per_window": weights, "verification": audit, **weight_statistics(weights)}
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output/f"{name}.csv", index=False, mode="x")
    make_plots(history, runs, curves, tables["weight_quantiles"], tables["age_weights"], output)
    interpretation = {
        "runs_per_condition": 30, "validation_weight_checks": "all passed; no repaired/renormalized saved beta",
        "learning_curves": "Arithmetic mean over available runs; n_runs and exact membership saved at every epoch. Paired curves use the current intersection, whose membership can also change.",
        "last_two_blocks": "Each run's own last ten epochs, split into two consecutive blocks of five; after minus before. Short histories are undefined, never padded.",
        "weight_statistics": "Window-weighted distributions stratified by seed, group and N; all-seed rows pool windows. weights_by_run permits ticker-balanced reaggregation.",
        "uniform_reference": "Uniform 1/N on the SAME attention daily presence mask. Mean5 windows/counts/bounds and saved OHLC fingerprints match; legacy mean5 CSV has no individual daily masks. Historical text values are not independently verified.",
        "caution": "Empty windows are separate and have no defined concentration metrics. N=1 beta=1 is forced, not evidence of selectivity. Train/validation trajectories and cohort changes must be inspected; final metrics alone do not establish overfitting.",
        "evidence_limit": "Best-epoch/diagnostic MSE and saved metadata are cross-checked; checkpoint tensors, raw data and test outputs are not read. A/beta are not causal news evidence.",
    }
    with (output/"interpretation.json").open("x", encoding="utf-8") as handle:
        json.dump(interpretation, handle, indent=2, allow_nan=False)
    for name in ("best_epoch_distribution", "block_changes_by_seed", "window_counts", "weight_quantiles", "age_weights"):
        print(name.upper())
        print(tables[name].to_csv(index=False))
    print(json.dumps(interpretation, indent=2))
    print(f"Reports: {output}")
    return tables


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output_dir", type=Path)
    analyze(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    cli()
