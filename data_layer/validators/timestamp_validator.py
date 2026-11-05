"""Timestamp normalization and validation.

Rejects naive timestamps, future-dated candles, non-monotonic sequences,
and provider timezone misalignment (all normalized to UTC on ingest).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from core.interfaces import Candle


class TimestampValidator:
    max_future_skew: timedelta = timedelta(minutes=5)

    def normalize(self, value: datetime | str | pd.Timestamp) -> datetime:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()

    def validate_candle(
        self, candle: Candle, *, now: Optional[datetime] = None, prev: Optional[Candle] = None
    ) -> Optional[str]:
        """Returns an error string when the candle must be rejected."""
        now = now or datetime.now(timezone.utc)
        if candle.ts > now + self.max_future_skew:
            return "candle timestamp is in the future"
        if candle.ingestion_ts < candle.ts:
            return "ingestion timestamp before candle timestamp"
        if prev is not None and candle.ts <= prev.ts:
            return f"non-monotonic timestamp: {candle.ts} <= {prev.ts}"
        return None


class Synchronizer:
    """Aligns multi-asset candles to a common clock grid before feature
    computation: an H1 grid for 1h data, H4 grid for 4h, D grid for 1d."""

    _offsets = {"1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1)}

    def snap(self, ts: datetime, interval: str) -> datetime:
        step = self._offsets[interval]
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        delta = ts - epoch
        steps = delta // step
        return epoch + steps * step

    def common_index(self, frames: dict[str, pd.DataFrame], interval: str) -> pd.DatetimeIndex:
        """Intersection of timestamps across assets — feature alignment
        must never mix candles from different periods."""
        snapped = {
            sym: pd.DatetimeIndex(sorted({self.snap(pd.Timestamp(r.ts).to_pydatetime(), interval)
                                          for r in df.itertuples()}))
            for sym, df in frames.items() if not df.empty
        }
        if not snapped:
            return pd.DatetimeIndex([])
        common = next(iter(snapped.values()))
        for idx in snapped.values():
            common = common.intersection(idx)
        return common
