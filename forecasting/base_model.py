"""Forecasting model base contract. Every model — baseline or ML —
implements fit/predict, is evaluated walk-forward, and emits
unified ForecastPoint outputs with uncertainty."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from core.clock import get_clock
from core.interfaces import ForecastPoint

_HORIZON_TD: dict[str, timedelta] = {"1h": timedelta(hours=1), "4h": timedelta(hours=4),
                                      "1d": timedelta(days=1)}


class ForecastModel(ABC):
    """Learns to predict the forward return (and its distribution) for a
    given horizon. Target convention: next-horizon simple return."""

    name: str = "abstract"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit on chronological data. X: feature frame; y: forward returns."""

    @abstractmethod
    def predict_expected_return(self, x: pd.Series) -> Optional[float]:
        """Expected return over the horizon (fractional). None when the
        model cannot produce a forecast."""

    def predict(self, asset_symbol: str, horizon: str, x: pd.Series) -> Optional[ForecastPoint]:
        """Default: point forecast with a bootstrap interval from training
        residuals (subclasses override for native intervals)."""
        expected = self.predict_expected_return(x)
        if expected is None:
            return None
        now = get_clock().now()
        residuals = getattr(self, "train_residuals_", None)
        lower = upper = vol = None
        if residuals is not None and len(residuals) > 30:
            vol = float(np.std(residuals))
            lower = float(np.percentile(residuals, 10))
            upper = float(np.percentile(residuals, 90))
        return ForecastPoint(
            model_id=self.name, asset_symbol=asset_symbol, horizon=horizon,
            generated_ts=now, target_ts=now + _HORIZON_TD[horizon],
            expected_return=float(expected),
            p_up=None, expected_volatility=vol,
            interval_lower=lower, interval_upper=upper,
            uncertainty_score=vol,
        )


def forward_returns(close: pd.Series, horizon: str) -> pd.Series:
    """Point-in-time-safe target: return between t and t+horizon."""
    shift = {"1h": 1, "4h": 1, "1d": 1}[horizon]
    # horizon expressed in bars of the training timeframe
    return close.shift(-shift) / close - 1


def horizon_bars(horizon: str, interval: str) -> int:
    """Number of `interval` bars covered by `horizon` (1h model forecasting
    1h ahead = 1 bar; daily model forecasting 4h ahead = 1 bar of 4h)."""
    h = {"1h": 1, "4h": 4, "1d": 24}[horizon]
    iv = {"1h": 1, "4h": 4, "1d": 24}[interval]
    return max(1, h // iv)
