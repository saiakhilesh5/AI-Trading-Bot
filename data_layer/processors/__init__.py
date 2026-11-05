"""Data processors: deduplication and in-memory normalization helpers."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from core.interfaces import Candle


def deduplicate(candles: Iterable[Candle]) -> list[Candle]:
    """First occurrence wins — point-in-time integrity preserved."""
    seen: set[tuple[str, str, pd.Timestamp]] = set()
    out: list[Candle] = []
    for c in candles:
        key = (c.asset_symbol, c.interval.value, c.ts)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase OHLCV columns, enforce UTC timestamps, sort, drop NaN rows."""
    ren = {c: c.lower() for c in df.columns}
    df = df.rename(columns=ren)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close"])
    return df.sort_values("ts").reset_index(drop=True)
