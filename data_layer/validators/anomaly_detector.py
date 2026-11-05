"""Anomaly detection beyond basic health checks: volume spikes,
volatility clustering, and OHLC internal inconsistency."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class AnomalyDetector:
    def volume_anomalies(self, df: pd.DataFrame, window: int = 48, k: float = 5.0) -> int:
        if "volume" not in df or len(df) < window:
            return 0
        vol = df["volume"].astype(float)
        med = vol.rolling(window).median().iloc[-1]
        if not med or np.isnan(med) or med == 0:
            return 0
        return int((vol.iloc[-1] / med) > k)

    def ohlc_inconsistent(self, df: pd.DataFrame) -> int:
        bad = (
            (df["high"] < df[["open", "close", "low"]].max(axis=1))
            | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        )
        return int(bad.sum())

    def volatility_cluster(self, df: pd.DataFrame, window: int = 24, ratio: float = 3.0) -> bool:
        """Current realized vol vs long-run vol — flags regime shock."""
        rets = np.log(df["close"].astype(float)).diff().dropna()
        if len(rets) < window * 4:
            return False
        recent = rets.tail(window).std()
        baseline = rets.std()
        return bool(baseline > 0 and recent / baseline > ratio)
