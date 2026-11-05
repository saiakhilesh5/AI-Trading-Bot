"""Baselines every model must beat (spec: random walk, historical average,
simple trend following, volatility-scaled)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.base_model import ForecastModel


class RandomWalkModel(ForecastModel):
    """Expected return = 0. The bar every claimed alpha must clear."""
    name = "baseline_random_walk"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_residuals_ = y.values  # actual returns = forecast errors

    def predict_expected_return(self, x: pd.Series):
        return 0.0


class HistoricalAverageModel(ForecastModel):
    name = "baseline_historical_average"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        arr = y.dropna().values
        self.mean_ = float(arr.mean()) if len(arr) else 0.0
        self.train_residuals_ = arr - self.mean_

    def predict_expected_return(self, x: pd.Series):
        return getattr(self, "mean_", 0.0)


class TrendFollowingModel(ForecastModel):
    """Momentum sign over lookback, scaled by recent volatility."""
    name = "baseline_trend_following"

    def __init__(self, lookback: int = 24):
        self.lookback = lookback

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_residuals_ = y.values

    def predict_expected_return(self, x: pd.Series):
        close_col = "close" if "close" in x.index else None
        if close_col is None or pd.isna(x[close_col]):
            return None
        mom_key = "momentum_10"
        vol_key = "hist_vol_24"
        mom = x.get(mom_key, np.nan)
        vol = x.get(vol_key, np.nan)
        if pd.isna(mom) or pd.isna(vol) or vol == 0:
            return 0.0
        return float(np.sign(mom) * min(abs(mom), 3 * vol))


class VolatilityScaledModel(ForecastModel):
    """Always 'expects' a 1-sigma move in the direction of recent drift."""
    name = "baseline_volatility_scaled"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_residuals_ = y.values

    def predict_expected_return(self, x: pd.Series):
        vol = x.get("hist_vol_24", np.nan)
        drift = x.get("ret_1", np.nan)
        if pd.isna(vol) or vol == 0:
            return 0.0
        direction = np.sign(drift) if not pd.isna(drift) else 0.0
        return float(direction * vol)


BASELINES = {
    "random_walk": RandomWalkModel,
    "historical_average": HistoricalAverageModel,
    "trend_following": TrendFollowingModel,
    "volatility_scaled": VolatilityScaledModel,
}
