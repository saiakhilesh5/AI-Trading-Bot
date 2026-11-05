"""Deterministic technical indicators (pandas-ta classic).
No LLM ever performs numerical calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns: ts, open, high, low, close, volume. Returns a DataFrame
    indexed like the input with all technical features."""
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    out["rsi_14"] = ta.rsi(close, length=14)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        out["macd"] = macd.iloc[:, 0]
        out["macd_signal"] = macd.iloc[:, 2]
        out["macd_hist"] = macd.iloc[:, 1]
    for length in (10, 20, 50):
        out[f"ema_{length}"] = ta.ema(close, length=length)
        out[f"sma_{length}"] = ta.sma(close, length=length)
    atr = ta.atr(high, low, close, length=14)
    if atr is not None:
        out["atr_14"] = atr
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        out["bb_lower"], out["bb_mid"], out["bb_upper"] = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]
        out["bb_position"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["momentum_10"] = ta.mom(close, length=10)
    out["roc_10"] = ta.roc(close, length=10)

    # VWAP (session-agnostic rolling version)
    tp = (high + low + close) / 3
    out["vwap_24"] = (tp * volume).rolling(24).sum() / volume.rolling(24).sum().replace(0, np.nan)

    # trend strength: |EMA10 - EMA50| normalised by ATR
    if "atr_14" in out:
        out["trend_strength"] = (out["ema_10"] - out["ema_50"]).abs() / out["atr_14"]

    # support/resistance proxies: rolling extremes and distance to them
    for window in (48, 168):
        roll_max = high.rolling(window).max()
        roll_min = low.rolling(window).min()
        out[f"dist_resistance_{window}"] = (roll_max - close) / close
        out[f"dist_support_{window}"] = (close - roll_min) / close
        # breakout: close crossing the prior rolling extreme
        out[f"breakout_up_{window}"] = (close > roll_max.shift(1)).astype(int)
        out[f"breakout_down_{window}"] = (close < roll_min.shift(1)).astype(int)

    # volume anomalies
    vol_med = volume.rolling(48).median()
    out["volume_z"] = (volume - vol_med) / volume.rolling(48).std().replace(0, np.nan)

    # volatility-adjusted returns
    ret = close.pct_change()
    if "atr_14" in out:
        out["vol_adj_return"] = ret / (out["atr_14"] / close).replace(0, np.nan)

    return out
