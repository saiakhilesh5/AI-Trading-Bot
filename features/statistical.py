"""Statistical features: volatility, return distributions, drawdowns,
correlations, autocorrelation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    rets = close.pct_change()
    log_rets = np.log(close).diff()
    out = pd.DataFrame(index=df.index)

    out["ret_1"] = rets
    # historical volatility (annualised-ish scaling per horizon omitted; raw std)
    for window in (24, 72, 168):
        out[f"hist_vol_{window}"] = log_rets.rolling(window).std()
    # realized volatility (Parkinson, uses high/low range)
    hl = np.log(df["high"].astype(float) / df["low"].astype(float)) ** 2
    out["realized_vol_24"] = np.sqrt(hl.rolling(24).mean() / (4 * np.log(2)))
    # return distribution shape
    out["skew_72"] = rets.rolling(72).skew()
    out["kurtosis_72"] = rets.rolling(72).kurt()
    # drawdown characteristics
    roll_max = close.cummax()
    dd = close / roll_max - 1
    out["drawdown"] = dd
    out["drawdown_7d"] = close.rolling(168).max().div(close) - 1
    # volatility clustering: std of |returns|'s rolling std
    out["vol_clustering"] = rets.abs().rolling(24).std().rolling(24).mean()
    # autocorrelation of returns (lag 1 and 24)
    for lag in (1, 24):
        out[f"autocorr_{lag}"] = rets.rolling(72).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag + 2 else np.nan, raw=False)

    return out


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 168) -> pd.Series:
    return a.rolling(window).corr(b)
