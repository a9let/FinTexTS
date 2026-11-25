"""Opt-in causal features; Dataset_Custom and its temporal/scaling policy stay intact."""

import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.models.GroupText import TEXT_GROUPS


NOVELTY_POLICY = {
    "presence_l1_threshold": 1e-6,
    "previous": "strictly earlier present row of the same ticker and group",
    "cosine": "float64 normalized dot product clipped to [-1,1]",
    "missing_pair": "novelty=0,pair_available=0",
    "history": "all available earlier rows, including previous splits",
    "price_context": "concat(X[-1]-X[-2],X[-1]-X[-6]); normalized input OHLC only",
}


def causal_novelty(embeddings):
    """Compute each row before updating history; missing rows do not erase history."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (5, 384) or not np.isfinite(values).all():
        raise ValueError("Expected finite embeddings [rows, 5, 384] for one ticker")
    # Match the model's float32 presence calculation, including its boundary.
    present = np.abs(values.astype(np.float32)).sum(axis=-1) >= 1e-6
    novelty = np.zeros(values.shape[:2], dtype=np.float32)
    pairs = np.zeros_like(novelty)
    previous = [None] * 5
    for row in range(len(values)):
        for group in range(5):
            if not present[row, group]:
                continue
            current = values[row, group]
            norm = np.linalg.norm(current)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Present embedding must have a finite positive norm")
            unit = current / norm
            if previous[group] is not None:
                cosine = np.clip(np.dot(unit, previous[group]), -1.0, 1.0)
                novelty[row, group] = 1.0 - cosine
                pairs[row, group] = 1.0
            previous[group] = unit
    return novelty, pairs


class ContextNoveltyDataset(Dataset):
    """Append last-input-day novelty/pair features to the original six fields."""

    def __init__(self, base, include_calendar_days=False, extended_text_window=None):
        if base.seq_len < 6 or tuple(base.used_col_prefixes) != TEXT_GROUPS:
            raise ValueError("group_gate requires seq_len>=6 and all five ordered groups")
        self.base = base
        self.include_calendar_days = include_calendar_days
        self.seq_len, self.label_len, self.pred_len = base.seq_len, base.label_len, base.pred_len
        self.scaler = base.scaler
        frame = pd.read_parquet(os.path.join(base.root_path, base.data_path))
        if "fnspid" in base.data_path:
            frame = frame.ffill()
        frame = frame[("2019-01-01" <= frame["date"]) & (frame["date"] < "2023-12-19")].reset_index(drop=True)
        if "ticker" not in frame or frame["ticker"].isna().any() or frame["ticker"].nunique() != 1:
            raise ValueError("group_gate requires exactly one identified ticker per parquet")
        dates = pd.to_datetime(frame["date"])
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("group_gate requires unique chronologically ordered dataset dates")
        embeddings = np.stack([
            frame[[f"{group}_emb{i}" for i in range(384)]].to_numpy(dtype=np.float32)
            for group in TEXT_GROUPS
        ], axis=1)
        embeddings = np.where(np.isnan(embeddings), 0, embeddings)
        novelty, pairs = causal_novelty(embeddings)
        # Reproduce the base dataset's actual row slice, including split overlap.
        train = int((frame["date"] < "2022-01-01").sum())
        validation = int(((frame["date"] >= "2022-01-01") & (frame["date"] < "2023-01-01")).sum())
        test = int(((frame["date"] >= "2023-01-01") & (frame["date"] < "2024-01-01")).sum())
        starts = [0, train - base.seq_len, len(frame) - test - base.seq_len]
        stops = [train, train + validation, len(frame)]
        start, stop = starts[base.set_type], stops[base.set_type]
        if start < 0:
            raise ValueError("Insufficient preceding rows for the requested input window")
        if not np.array_equal(embeddings[start:stop], base.data_text_x):
            raise ValueError("Novelty history does not match Dataset_Custom embeddings")
        prices = base.scaler.transform(frame[["open", "high", "low", "close"]].values)
        if not np.array_equal(prices[start:stop], base.data_price_x):
            raise ValueError("Novelty row indices do not match Dataset_Custom price rows")
        self.row_dates = dates.iloc[start:stop].dt.strftime("%Y-%m-%d").to_numpy()
        self.novelty = novelty[start:stop]
        self.pair_available = pairs[start:stop]
        self.extended_text_window = extended_text_window
        if extended_text_window is not None:
            if extended_text_window <= base.seq_len:
                raise ValueError("Extended text window must be longer than the price window")
            source_start = max(0, start + base.seq_len - extended_text_window)
            self._text_offset = start - source_start
            self._text_source = embeddings[source_start:stop].copy()
            self._text_dates = dates.iloc[source_start:stop].dt.strftime("%Y-%m-%d").to_numpy()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = index + self.seq_len - 1
        metadata = {"novelty": self.novelty[row], "pair_available": self.pair_available[row]}
        if self.include_calendar_days:
            metadata['calendar_days'] = self.row_dates[index:index + self.seq_len].astype('datetime64[D]').astype(np.int64)
        if self.extended_text_window is not None:
            span = self._text_span(index)
            # Collate the prefix only; the original six fields already carry the last seq_len rows.
            history = self._text_source[span.start:span.stop - self.seq_len]
            missing = self.extended_text_window - self.seq_len - len(history)
            if missing:
                history = np.pad(history, ((missing, 0), (0, 0), (0, 0)))
            metadata['text_history'] = history
        return (*self.base[index], metadata)

    def _text_span(self, index):
        stop = int(index) + self.seq_len + self._text_offset
        return slice(max(0, stop - self.extended_text_window), stop)

    def text_window_dates(self, index):
        """Dates of real source rows, excluding collation padding."""
        return self._text_dates[self._text_span(index)]

    def extended_text_evidence(self):
        """Fingerprint the source and the exact causal slice of every text window."""
        bounds = hashlib.sha256()
        lengths = []
        for index in range(len(self)):
            dates = self.text_window_dates(index)
            if dates[-1] != self.window_end_date(index):
                raise ValueError("Text window end does not match the last price date")
            lengths.append(len(dates))
            bounds.update(f'{dates[0]}|{dates[-1]}|{len(dates)}\n'.encode())
        return {
            'requested_rows': self.extended_text_window,
            'windows': len(lengths),
            'min_rows': min(lengths),
            'max_rows': max(lengths),
            'source_text_sha256': hashlib.sha256(self._text_source.astype('<f4', copy=False).tobytes()).hexdigest(),
            'source_dates_sha256': hashlib.sha256('\n'.join(self._text_dates).encode()).hexdigest(),
            'window_bounds_sha256': bounds.hexdigest(),
        }

    def window_end_date(self, index):
        return str(self.row_dates[int(index) + self.seq_len - 1])

    def inverse_transform(self, values):
        return self.base.inverse_transform(values)


def gate_batch_inputs(metadata, text, device, *, use_window):
    features = {key: value.float().to(device) for key, value in metadata.items()}
    history = features.pop('text_history', None)
    if use_window:
        features['text_window'] = text if history is None else torch.cat((history, text), dim=1)
    return features


def get_context_novelty_dataloader(batch_size, **kwargs):
    include_days = kwargs.pop('include_calendar_days', False)
    extended_text_window = kwargs.pop('extended_text_window', None)
    matched_baseline = kwargs.pop('matched_baseline', False)
    factory = ContextNoveltyDataset
    if matched_baseline:
        from forecasting_task.mean_router_protocol import BaselineDataset
        factory = BaselineDataset
    dataset = factory(Dataset_Custom(**kwargs), include_calendar_days=include_days,
                                    extended_text_window=extended_text_window)
    train = kwargs["flag"] == "train"
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=1, drop_last=train)
