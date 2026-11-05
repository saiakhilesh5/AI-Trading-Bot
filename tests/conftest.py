"""Shared fixtures."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.clock import SimulatedClock, set_clock  # noqa: E402
from core.interfaces import Candle, Interval  # noqa: E402


@pytest.fixture
def sim_clock():
    clock = SimulatedClock(datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc))
    set_clock(clock)
    yield clock
    set_clock(None)


def make_candles(
    asset="BTCUSD",
    interval="1h",
    n=200,
    start=None,
    price=100.0,
    step_hours=1,
    seed=42,
    with_weekends=True,
):
    """Deterministic noisy candles ending just before the simulated clock
    (2026-01-15 12:00 UTC). Weekend-skipping data walks back from the clock."""
    import random

    end = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    rng = random.Random(seed)
    step = timedelta(hours=step_hours)

    if start is None and not with_weekends:
        # walk backwards from the clock, skipping weekends
        stamps = []
        ts = end - step
        while len(stamps) < n:
            if ts.weekday() < 5:
                stamps.append(ts)
            ts = ts - step
        stamps.reverse()
    elif start is None:
        stamps = [end - step * (n - i) for i in range(n)]
    else:
        stamps = []
        ts = start
        while len(stamps) < n:
            if with_weekends or ts.weekday() < 5:
                stamps.append(ts)
            ts = ts + step

    out = []
    cur = price
    for ts in stamps:
        o = cur
        c = o * (1 + rng.gauss(0.001, 0.004))
        out.append(Candle(
            asset_symbol=asset, interval=Interval(interval), ts=ts,
            open=o, high=max(o, c) * 1.002, low=min(o, c) * 0.998, close=c,
            volume=10.0, source="test", ingestion_ts=end))
        cur = c
    return out
