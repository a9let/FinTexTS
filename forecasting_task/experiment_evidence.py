"""Evidence for reusing controls across branches with the same common modules."""

import hashlib
import json
from pathlib import Path

import numpy as np


def state_fingerprint(module, exclude_prefixes=()):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def common_initialization(model, text_model):
    return {"film_sha256": state_fingerprint(model),
            "heads_sha256": state_fingerprint(text_model.group_mlps),
            "price_gate_sha256": state_fingerprint(text_model.gate) if text_model.gate is not None else None}


def source_data_evidence(datasets):
    result = {}
    for flag, dataset in datasets.items():
        base = dataset.base
        dates = "\n".join(dataset.row_dates).encode()
        result[flag] = {
            "rows": len(base.data_text_x), "groups": list(base.used_col_prefixes),
            "text_shape": list(base.data_text_x.shape),
            "text_sha256": hashlib.sha256(np.asarray(base.data_text_x, dtype="<f4").tobytes()).hexdigest(),
            "prices_sha256": hashlib.sha256(np.asarray(base.data_price_x, dtype="<f4").tobytes()).hexdigest(),
            "dates_sha256": hashlib.sha256(dates).hexdigest()}
    path = Path(datasets["train"].base.root_path) / datasets["train"].base.data_path
    result["parquet_sha256"] = None
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["parquet_sha256"] = digest.hexdigest()
    return result
