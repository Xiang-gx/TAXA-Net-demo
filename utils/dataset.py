"""Sliding windows from preprocessed 15-minute station observations."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SolarSlidingWindowDataset(Dataset):
    """Return GHI history, weather, calendar features, and future observed GHI."""
    def __init__(self, csv_path: str, seq_len: int = 480, pred_len: int = 96, stride: int = 1,
                 endo_cols: list = None, exo_cols: list = None, timestamp_col: str = "timestamp"):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

        self.endo_cols = endo_cols or ["Global_Horizontal_Radiation"]
        self.exo_cols = exo_cols or [
            "Wind_Speed",
            "Weather_Temperature_Celsius",
            "Weather_Relative_Humidity",
            "Weather_Daily_Rainfall"
        ]

        df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        for col in self.endo_cols + self.exo_cols:
            if col not in df.columns:
                raise ValueError(f"Required feature column '{col}' not found in CSV. Available columns: {list(df.columns)}")

        self.x_en_all = df[self.endo_cols].values.astype(np.float32)
        self.x_ex_all = df[self.exo_cols].values.astype(np.float32)
        self.y_all = self.x_en_all[:, 0]

        # Calendar features with a fixed 366-day annual period.
        timestamps = pd.to_datetime(df[timestamp_col])
        self.timestamps = timestamps.values
        time_seconds = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
        tod_sin = np.sin(2 * np.pi * time_seconds / 86400.0)
        tod_cos = np.cos(2 * np.pi * time_seconds / 86400.0)

        day_of_year = timestamps.dt.dayofyear - 1
        doy_sin = np.sin(2 * np.pi * day_of_year / 366.0)
        doy_cos = np.cos(2 * np.pi * day_of_year / 366.0)

        self.x_time_all = np.stack([tod_sin, tod_cos, doy_sin, doy_cos], axis=-1).astype(np.float32)

        total_window_len = seq_len + pred_len
        self.valid_starts = []

        valid_steps = np.diff(self.timestamps) == np.timedelta64(15, "m")

        for s in range(0, len(df) - total_window_len + 1, stride):
            e = s + seq_len
            f = e + pred_len

            # Require 15-minute intervals throughout history and target.
            if not valid_steps[s:f - 1].all():
                continue

            window_en = self.x_en_all[s:e]
            window_ex = self.x_ex_all[s:e]
            window_time = self.x_time_all[s:e]
            window_y = self.y_all[e:f]

            if not (np.isnan(window_en).any() or np.isnan(window_ex).any() or
                    np.isnan(window_time).any() or np.isnan(window_y).any()):
                self.valid_starts.append(s)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        e = s + self.seq_len
        f = e + self.pred_len

        x_en = torch.from_numpy(self.x_en_all[s:e])      # [seq_len, C_en]
        x_ex = torch.from_numpy(self.x_ex_all[s:e])      # [seq_len, C_ex]
        x_time = torch.from_numpy(self.x_time_all[s:e])  # [seq_len, 4]
        y = torch.from_numpy(self.y_all[e:f])            # [pred_len]

        return x_en, x_ex, x_time, y

    def get_timestamps(self, idx: int = 0):
        """Return timestamps spanning history and forecast."""
        s = self.valid_starts[idx]
        f = s + self.seq_len + self.pred_len
        return self.timestamps[s:f]


def build_demo_dataloader(csv_path: str, seq_len: int = 480, pred_len: int = 96,
                          stride: int = 1, batch_size: int = 32,
                          endo_cols: list = None, exo_cols: list = None):
    """Return the dataset and its inference DataLoader."""
    dataset = SolarSlidingWindowDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        endo_cols=endo_cols,
        exo_cols=exo_cols
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return dataset, dataloader
