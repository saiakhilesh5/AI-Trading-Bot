"""The 8 comparison strategies required by the spec. Each produces a per-bar
position (-1, 0, +1) from features available at that bar (no look-ahead).

1. buy_and_hold          - always long
2. trend_following       - EMA crossover
3. technical_only        - momentum + RSI gate
4. forecast_only         - statistical forecast sign
5. technical_forecast    - TA + forecast agreement
6. technical_news        - TA + news agent bias (news = deterministic proxy in backtest)
7. forecast_news_macro   - forecast + news/macro proxies
8. full_system           - everything with uncertainty gating (closest offline
                           analogue of the live multi-agent system)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

STRATEGIES = ["buy_and_hold", "trend_following", "technical_only", "forecast_only",
              "technical_forecast", "technical_news", "forecast_news_macro",
              "full_system"]


def _features(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    df["ema_fast"] = close.ewm(span=10).mean()
    df["ema_slow"] = close.ewm(span=50).mean()
    df["momentum"] = close.pct_change(10)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=14).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["vol"] = close.pct_change().rolling(24).std()
    return df


def buy_and_hold(close: pd.Series, forecast: Optional[pd.Series] = None,
                 news_bias: Optional[pd.Series] = None) -> pd.Series:
    return pd.Series(1.0, index=close.index)


def trend_following(close: pd.Series, forecast: Optional[pd.Series] = None,
                    news_bias: Optional[pd.Series] = None) -> pd.Series:
    f = _features(close)
    return pd.Series(np.where(f["ema_fast"] > f["ema_slow"], 1.0, -1.0),
                     index=close.index)


def technical_only(close: pd.Series, forecast: Optional[pd.Series] = None,
                   news_bias: Optional[pd.Series] = None) -> pd.Series:
    f = _features(close)
    pos = pd.Series(0.0, index=close.index)
    long_ok = (f["momentum"] > 0) & (f["rsi"] < 70)
    short_ok = (f["momentum"] < 0) & (f["rsi"] > 30)
    pos[long_ok] = 1.0
    pos[short_ok] = -1.0
    return pos


def forecast_only(close: pd.Series, forecast: Optional[pd.Series] = None,
                  news_bias: Optional[pd.Series] = None) -> pd.Series:
    if forecast is None:
        return pd.Series(0.0, index=close.index)
    return pd.Series(np.sign(forecast.reindex(close.index).fillna(0.0)), index=close.index)


def technical_forecast(close: pd.Series, forecast: Optional[pd.Series] = None,
                       news_bias: Optional[pd.Series] = None) -> pd.Series:
    ta = technical_only(close)
    fc = forecast_only(close, forecast)
    return pd.Series(np.where(ta == fc, ta, 0.0), index=close.index)


def technical_news(close: pd.Series, forecast: Optional[pd.Series] = None,
                   news_bias: Optional[pd.Series] = None) -> pd.Series:
    ta = technical_only(close)
    if news_bias is None:
        return ta
    bias = news_bias.reindex(close.index).fillna(0.0)
    agree = ((ta > 0) & (bias >= 0)) | ((ta < 0) & (bias <= 0))
    return pd.Series(np.where(agree, ta, 0.0), index=close.index)


def forecast_news_macro(close: pd.Series, forecast: Optional[pd.Series] = None,
                       news_bias: Optional[pd.Series] = None) -> pd.Series:
    base = forecast_only(close, forecast)
    if news_bias is None:
        return base
    bias = news_bias.reindex(close.index).fillna(0.0)
    agree = ((base > 0) & (bias >= 0)) | ((base < 0) & (bias <= 0))
    return pd.Series(np.where(agree, base, 0.0), index=close.index)


def full_system(close: pd.Series, forecast: Optional[pd.Series] = None,
                news_bias: Optional[pd.Series] = None) -> pd.Series:
    """All signals must agree, plus a volatility gate: stay flat when the
    forecast disagrees with the trend or vol spikes above its 95th pct."""
    f = _features(close)
    ta = technical_only(close)
    fc = forecast_only(close, forecast)
    base = pd.Series(np.where(ta == fc, ta, 0.0), index=close.index)
    if news_bias is not None:
        bias = news_bias.reindex(close.index).fillna(0.0)
        agree = ((base > 0) & (bias >= 0)) | ((base < 0) & (bias <= 0))
        base = pd.Series(np.where(agree, base, 0.0), index=close.index)
    vol_gate = f["vol"] <= f["vol"].rolling(200, min_periods=20).quantile(0.95)
    base[~vol_gate.fillna(False)] = 0.0
    return base


_FUNCS = {name: globals()[name] for name in STRATEGIES}


def run_strategy(name: str, close: pd.Series, forecast: Optional[pd.Series] = None,
                 news_bias: Optional[pd.Series] = None) -> pd.Series:
    return _FUNCS[name](close, forecast, news_bias)
