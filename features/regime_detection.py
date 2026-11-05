"""Market regime detection: statistical volatility/trend classification
(deterministic, independently testable). HMM is a later benchmark candidate —
it must beat this classifier before replacing it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import MarketRegime


class RegimeClassifier:
    """Classifies each timestamp from realized vol percentile and trend slope."""

    def __init__(self, vol_window: int = 72, vol_lookback: int = 1000,
                 trend_window: int = 48, vol_low_pct: float = 30.0, vol_high_pct: float = 70.0):
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.trend_window = trend_window
        self.vol_low_pct = vol_low_pct
        self.vol_high_pct = vol_high_pct

    def classify_series(self, df: pd.DataFrame) -> pd.Series:
        """Returns a Series of MarketRegime values aligned to df."""
        close = df["close"].astype(float)
        log_rets = np.log(close).diff()
        vol = log_rets.rolling(self.vol_window).std()
        vol_pct = vol.rolling(self.vol_lookback, min_periods=100).rank(pct=True) * 100

        # trend slope: linear regression of log price over window (normalised)
        logp = np.log(close)
        slope = logp.rolling(self.trend_window).apply(_slope, raw=False)
        atr = (logp.diff().abs().rolling(self.trend_window).mean())

        regimes = []
        for vp, sl, at in zip(vol_pct, slope, atr):
            if pd.isna(vp) or pd.isna(sl) or pd.isna(at) or at == 0:
                regimes.append(None)
                continue
            norm_slope = sl / at  # slope in units of average bar-to-bar movement
            if vp >= self.vol_high_pct:
                # high vol dominates classification when combined with direction
                regimes.append(MarketRegime.HIGH_VOL)
            elif vp <= self.vol_low_pct and abs(norm_slope) < 0.15:
                regimes.append(MarketRegime.LOW_VOL)
            elif norm_slope > 0.15:
                regimes.append(MarketRegime.TRENDING_UP)
            elif norm_slope < -0.15:
                regimes.append(MarketRegime.TRENDING_DOWN)
            else:
                regimes.append(MarketRegime.SIDEWAYS)
        return pd.Series(regimes, index=df.index)

    def current_regime(self, df: pd.DataFrame) -> MarketRegime | None:
        series = self.classify_series(df)
        if series.empty or series.iloc[-1] is None:
            return None
        return series.iloc[-1]

    def transition_risk(self, regimes: pd.Series, window: int = 24) -> float:
        """Fraction of regime changes in the recent window — a proxy for
        'regime transition' instability."""
        recent = regimes.dropna().tail(window)
        if len(recent) < 3:
            return 0.0
        changes = (recent != recent.shift(1)).sum() - 1
        return float(changes / len(recent))


def _slope(window: pd.Series) -> float:
    """OLS slope of the window against time index."""
    if len(window) < 3:
        return np.nan
    y = window.values
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return np.nan
    return float((x * (y - y.mean())).sum() / denom)
