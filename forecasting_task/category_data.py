"""Versioned category schema and strict provenance/reproduction checks."""

import hashlib
import json
from pathlib import Path

import numpy as np

GROUPS = ("macro", "sector", "targetCompany", "relatedCompany", "filing")
CATEGORIES = {g: [f"{g}_category{i}" for i in range(1, n+1)] for g, n in zip(GROUPS[:4], (5, 5, 3, 3))}
FILING = ["filing_financialStatement", "filing_governanceRisks", "filing_overviewProduct", "filing_recentEventCatalyst", "filing_strategyMarketOps"]
PROVENANCE = {"dataset": "EXAONE-BI/FinTexTS", "dataset_revision": "6a4ce04ac3f2e8a4ae15a5826e6980b50560eb29",
              "split": "train", "encoder": "sentence-transformers/all-MiniLM-L6-v2",
              "encoder_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "embedding_size": 384,
              "normalize_embeddings": True, "batch_size": 64, "empty_policy": "strip_then_empty_or_null_absent",
              "max_seq_length_policy": "pinned_encoder_default_unmodified", "filing_policy": "mean_present_five_original_categories"}
DEFAULT_DATA = Path("/auto/datasets/kondyurin/FinTexTS/minilm_category_attention_v1/fintexts")
LEGACY_DATA = Path("/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category_arrays(frame):
    values = np.zeros((len(frame), 4, 5, 384), dtype=np.float32)
    mask = np.zeros((len(frame), 4, 5), dtype=bool)
    for g, columns in enumerate(CATEGORIES.values()):
        for k, column in enumerate(columns):
            raw = frame[column+"_present"].to_numpy()
            if not np.isin(raw, [0, 1]).all():
                raise ValueError(f"Invalid explicit mask {column}")
            mask[:, g, k] = raw.astype(bool)
            values[:, g, k] = frame[[f"{column}_emb{i}" for i in range(384)]].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or np.any(values[~mask] != 0):
        raise ValueError("Category vectors must be finite, absent vectors exactly zero")
    if mask.any() and not np.allclose(np.linalg.norm(values[mask], axis=-1), 1, rtol=0, atol=2e-5):
        raise ValueError("Present MiniLM categories must retain unit normalization")
    filing = frame["filing_present"].to_numpy()
    if not np.isin(filing, [0, 1]).all():
        raise ValueError("Invalid filing presence mask")
    return values, mask, filing.astype(bool)


def reproduce_groups(frame, atol):
    values, mask, filing = category_arrays(frame)
    maximum = {}
    for g, group in enumerate(GROUPS[:4]):
        # Match the notebook's sequential float32 summation and division.
        total = np.zeros((len(frame), 384), dtype=np.float32)
        for k in range(len(CATEGORIES[group])):
            total += values[:, g, k]
        counts = mask[:, g].sum(-1)
        mean = np.zeros_like(total)
        np.divide(total, counts[:, None], out=mean, where=counts[:, None] > 0)
        old = frame[[f"{group}_emb{i}" for i in range(384)]].to_numpy(dtype=np.float32)
        maximum[group] = float(np.max(np.abs(old-mean), initial=0))
        if not np.isfinite(old).all() or maximum[group] > atol:
            raise ValueError(f"{group}: grouped mean max_abs_diff={maximum[group]} exceeds atol={atol}, rtol=0")
        if not np.array_equal(counts > 0, np.abs(old).sum(-1) >= 1e-6):
            raise ValueError(f"{group}: explicit presence differs from legacy norm mask; old-path reproduction is incompatible")
    old_filing = frame[[f"filing_emb{i}" for i in range(384)]].to_numpy(dtype=np.float32)
    if not np.isfinite(old_filing).all() or not np.array_equal(filing, np.abs(old_filing).sum(-1) >= 1e-6):
        raise ValueError("Filing legacy vector/mask mismatch")
    return maximum


def load_manifest(root, ticker=None):
    path = Path(root)/"manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "fintexts_categories_v1" or manifest.get("provenance") != PROVENANCE:
        raise ValueError(f"Incompatible category provenance: {path}")
    if manifest.get("categories") != CATEGORIES or manifest.get("filing_columns") != FILING:
        raise ValueError("Category schema or filing categories differ")
    if (not isinstance(manifest.get("encoder_max_seq_length"), int) or manifest["encoder_max_seq_length"] <= 0
            or set(manifest.get("encoder_versions", {})) != {"sentence-transformers", "transformers", "torch", "datasets"}
            or not all(isinstance(v, str) and v for v in manifest["encoder_versions"].values())
            or not np.isfinite(manifest["atol"]) or manifest["atol"] <= 0):
        raise ValueError("Missing encoder defaults/versions or invalid reproduction tolerance")
    if manifest.get("verification", {}).get("status") != "passed":
        raise ValueError("Category preparation reproduction verification did not pass")
    if ticker is not None:
        record = manifest["files"][ticker]
        for name in ("sha256", "legacy_sha256", "source_text_sha256"):
            value = record.get(name, "")
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Missing/invalid {name}: {ticker}")
        maximum = record.get("max_abs_diff", {})
        if set(maximum) != set(GROUPS) or any(not np.isfinite(v) or v < 0 or v > manifest["atol"] for v in maximum.values()):
            raise ValueError(f"Unverified group/filing reproduction: {ticker}")
        if file_hash(Path(root)/(ticker+".parquet")) != record["sha256"]:
            raise ValueError(f"Category parquet fingerprint mismatch: {ticker}")
    return manifest
