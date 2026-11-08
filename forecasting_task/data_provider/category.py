"""Opt-in category data; original Dataset_Custom owns prices, scaling and splits."""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.category_data import load_manifest, category_arrays, reproduce_groups, GROUPS


class CategoryDataset(Dataset):
    def __init__(self, base):
        if tuple(base.used_col_prefixes) != GROUPS or base.seq_len < 6 or "fnspid" in base.data_path:
            raise ValueError("Categories require five ordered groups, seq_len>=6 and no forward fill")
        self.base = base
        self.seq_len, self.label_len, self.pred_len = base.seq_len, base.label_len, base.pred_len
        self.scaler = base.scaler
        ticker = Path(base.data_path).stem
        self.manifest = load_manifest(base.root_path, ticker)
        frame = pd.read_parquet(Path(base.root_path)/base.data_path)
        if self.manifest["files"][ticker]["rows"] != len(frame):
            raise ValueError("Category manifest row count mismatch")
        if frame.ticker.nunique() != 1 or str(frame.ticker.iloc[0]) != ticker:
            raise ValueError("Category parquet must contain exactly the requested ticker")
        dates = pd.to_datetime(frame.date)
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Category dates must be unique and chronological")
        reproduce_groups(frame, self.manifest["atol"])
        frame = frame[(frame.date >= "2019-01-01") & (frame.date < "2023-12-19")].reset_index(drop=True)
        train = int((frame.date < "2022-01-01").sum())
        val = int(((frame.date >= "2022-01-01") & (frame.date < "2023-01-01")).sum())
        test = int((frame.date >= "2023-01-01").sum())
        start = [0, train-base.seq_len, len(frame)-test-base.seq_len][base.set_type]
        stop = [train, train+val, len(frame)][base.set_type]
        if start < 0:
            raise ValueError("Insufficient history for input window")
        selected = frame.iloc[start:stop]
        prices = base.scaler.transform(selected[["open", "high", "low", "close"]].values)
        groups = np.stack([selected[[f"{g}_emb{i}" for i in range(384)]].to_numpy(np.float32) for g in GROUPS], 1)
        if not np.array_equal(prices, base.data_price_x) or not np.array_equal(groups, base.data_text_x):
            raise ValueError("Category rows do not align with Dataset_Custom")
        self.categories, self.masks, self.filing = category_arrays(selected)
        self.row_dates = pd.to_datetime(selected.date).dt.strftime("%Y-%m-%d").to_numpy()
        self.ticker = ticker

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = index+self.seq_len-1
        return (*self.base[index], {"category_embeddings": self.categories[row], "category_present": self.masks[row],
                                    "filing_present": self.filing[row]})

    def window_end_date(self, index):
        return str(self.row_dates[int(index)+self.seq_len-1])

    def inverse_transform(self, values):
        return self.base.inverse_transform(values)

    def category_evidence(self):
        values, masks = hashlib.sha256(), hashlib.sha256()
        for index in range(len(self)):
            row = index+self.seq_len-1
            values.update(self.categories[row].astype("<f4").tobytes())
            masks.update(self.masks[row].astype(np.uint8).tobytes())
            masks.update(np.uint8(self.filing[row]).tobytes())
        return {"windows": len(self), "category_values_sha256": values.hexdigest(), "category_masks_sha256": masks.hexdigest()}


def get_category_dataloader(batch_size, **kwargs):
    dataset = CategoryDataset(Dataset_Custom(**kwargs))
    train = kwargs["flag"] == "train"
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=1, drop_last=train)
