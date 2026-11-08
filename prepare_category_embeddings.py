"""Explicit server-only preparation/verification; never overwrite old parquet."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_task.category_data import (CATEGORIES, FILING, GROUPS, PROVENANCE, DEFAULT_DATA, LEGACY_DATA,
                                           file_hash, load_manifest, reproduce_groups)


def clean_texts(frame, columns):
    # Same pandas strip/empty->null policy as prepare_embeddings.ipynb.
    return frame[columns].apply(lambda s: s.str.strip().replace("", None))


def text_digest(metadata, texts):
    import hashlib
    payload = metadata[["date", "ticker"]].astype(str).to_json(orient="split", index=False)
    payload += texts.to_json(orient="split", index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_ticker(metadata, clean, embeddings, legacy, atol):
    if (metadata.duplicated(["ticker", "date"]).any() or not pd.to_datetime(metadata.date).is_monotonic_increasing
            or not np.isfinite(metadata[["open", "high", "low", "close"]].to_numpy(dtype=float)).all()):
        raise ValueError("Invalid source price rows/dates")
    if not metadata[["date", "ticker", "open", "high", "low", "close"]].reset_index(drop=True).equals(
            legacy[["date", "ticker", "open", "high", "low", "close"]].reset_index(drop=True)):
        raise ValueError("Pinned dataset OHLC/date/ticker rows differ from legacy parquet")
    result = legacy.copy()
    for columns in CATEGORIES.values():
        for column in columns:
            present = clean[column].notna().to_numpy()
            matrix = np.asarray(embeddings[column], dtype=np.float32)
            if matrix.shape != (len(legacy), 384) or np.any(matrix[~present] != 0):
                raise ValueError(f"Invalid encoding/mask for {column}")
            result[column+"_present"] = present
            result = pd.concat([result, pd.DataFrame(matrix, columns=[f"{column}_emb{i}" for i in range(384)])], axis=1)
    total = np.zeros((len(legacy), 384), np.float32)
    counts = np.zeros(len(legacy), np.int32)
    for column in FILING:
        present = clean[column].notna().to_numpy()
        total[present] += embeddings[column][present]
        counts += present
    filing = np.zeros_like(total)
    np.divide(total, counts[:, None], out=filing, where=counts[:, None] > 0)
    old = legacy[[f"filing_emb{i}" for i in range(384)]].to_numpy(np.float32)
    filing_diff = float(np.max(np.abs(filing-old), initial=0))
    if filing_diff > atol or not np.isfinite(filing).all():
        raise ValueError(f"Filing reproduction max_abs_diff={filing_diff} exceeds atol={atol}")
    result["filing_present"] = counts > 0
    maximum = reproduce_groups(result, atol)
    return result, {**maximum, "filing": filing_diff}


def verify(data_root=DEFAULT_DATA, legacy_root=LEGACY_DATA, expected_tickers=100):
    manifest = load_manifest(data_root)
    files = sorted(Path(data_root).glob("*.parquet"))
    if len(files) != expected_tickers or len({p.stem.casefold() for p in files}) != expected_tickers or set(manifest["files"]) != {p.stem for p in files}:
        raise ValueError(f"Expected {expected_tickers} distinct category parquet files and matching manifest")
    for path in files:
        load_manifest(data_root, path.stem)
        frame = pd.read_parquet(path)
        old_path = Path(legacy_root)/path.name
        record = manifest["files"][path.stem]
        if file_hash(old_path) != record["legacy_sha256"]:
            raise ValueError(f"Legacy parquet changed: {old_path}")
        old = pd.read_parquet(old_path)
        if not frame[old.columns].equals(old):
            raise ValueError(f"Original grouped vectors/OHLC/filing changed in {path}")
        maximum = reproduce_groups(frame, manifest["atol"])
        maximum["filing_saved_reencoding"] = record["max_abs_diff"]["filing"]
        maximum["filing_copy"] = 0.0  # Exact frame[old.columns].equals(old) above.
        if record["rows"] != len(frame) or not record.get("source_text_sha256"):
            raise ValueError(f"Missing row/text provenance for {path}")
        print(f"VERIFIED {path.stem}: max_abs_diff={maximum}; atol={manifest['atol']}, rtol=0", flush=True)
    return manifest


def prepare(output=DEFAULT_DATA, legacy_root=LEGACY_DATA, reuse_from=None, atol=1e-6):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    # Download/encoding is deliberately lazy and only reached by explicit prepare.
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Explicit preparation needs the notebook's datasets environment; no dependencies were installed") from error
    ds = load_dataset(PROVENANCE["dataset"], revision=PROVENANCE["dataset_revision"])["train"]
    columns = [c for cols in CATEGORIES.values() for c in cols]+FILING
    frame = ds.select_columns(["date", "ticker", "open", "high", "low", "close", *columns]).to_pandas()
    if frame.ticker.isna().any() or frame.ticker.nunique() != 100:
        raise ValueError("Pinned dataset must contain exactly 100 tickers")
    # Notebook sorts per ticker AFTER encoding; sorting here preserves row/text alignment.
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date in source dataset")
    cached = verify(reuse_from, legacy_root) if reuse_from else None
    encoder = None
    if not cached:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Explicit encoding needs the notebook's sentence-transformers environment; no dependencies were installed") from error
        encoder = SentenceTransformer(PROVENANCE["encoder"], revision=PROVENANCE["encoder_revision"])
        if encoder.get_sentence_embedding_dimension() != 384:
            raise ValueError("Encoder dimension differs from notebook")
        versions = {name: importlib.metadata.version(name) for name in ("sentence-transformers", "transformers", "torch", "datasets")}
        max_length = int(encoder.max_seq_length)  # No override, including filing truncation.
    else:
        versions, max_length = cached["encoder_versions"], cached["encoder_max_seq_length"]
    manifest = {"schema": "fintexts_categories_v1", "provenance": PROVENANCE, "categories": CATEGORIES,
                "filing_columns": FILING, "encoder_versions": versions, "encoder_max_seq_length": max_length,
                "atol": atol, "files": {}, "verification": {"status": "pending"}}
    output.mkdir(parents=True, exist_ok=False)
    for ticker, raw in frame.groupby("ticker", sort=True):
        raw = raw.reset_index(drop=True)
        metadata = raw[["date", "ticker", "open", "high", "low", "close"]]
        clean = clean_texts(raw, columns)
        source_hash = text_digest(metadata, clean)
        legacy_path = Path(legacy_root)/(ticker+".parquet")
        legacy = pd.read_parquet(legacy_path)
        if not metadata.equals(legacy[["date", "ticker", "open", "high", "low", "close"]].reset_index(drop=True)):
            raise ValueError(f"Pinned source OHLC/date/ticker differs from legacy: {ticker}")
        if cached:
            if cached["files"][ticker]["source_text_sha256"] != source_hash:
                raise ValueError(f"Cached category source texts differ from pinned dataset: {ticker}")
            result = pd.read_parquet(Path(reuse_from)/(ticker+".parquet"))
            maximum = reproduce_groups(result, atol)
            maximum["filing"] = cached["files"][ticker]["max_abs_diff"]["filing"]
            if maximum["filing"] > atol:
                raise ValueError("Cached filing reproduction exceeds requested tolerance")
        else:
            embeddings = {}
            for column in columns:
                present = clean[column].notna().to_numpy()
                matrix = np.zeros((len(raw), 384), np.float32)
                if present.any():
                    matrix[present] = encoder.encode(clean.loc[present, column].tolist(), batch_size=64,
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
                embeddings[column] = matrix
            result, maximum = build_ticker(metadata, clean, embeddings, legacy, atol)
        target = output/(ticker+".parquet")
        result.to_parquet(target, index=False)
        manifest["files"][ticker] = {"rows": len(result), "sha256": file_hash(target), "legacy_sha256": file_hash(legacy_path),
                                     "source_text_sha256": source_hash, "max_abs_diff": maximum}
        print(f"PREPARED {ticker}: max_abs_diff={maximum}; atol={atol}, rtol=0", flush=True)
    manifest["verification"]["status"] = "passed"
    with (output/"manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    verify(output, legacy_root)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, default=DEFAULT_DATA)
    p.add_argument("--legacy_root", type=Path, default=LEGACY_DATA)
    p.add_argument("--reuse_from", type=Path)
    p.add_argument("--atol", type=float, default=1e-6)
    p = sub.add_parser("verify")
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--legacy_root", type=Path, default=LEGACY_DATA)
    args = vars(parser.parse_args(argv))
    action = args.pop("action")
    (prepare if action == "prepare" else verify)(**args)


if __name__ == "__main__":
    cli()
