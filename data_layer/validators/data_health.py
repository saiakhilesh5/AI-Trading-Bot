"""Data-health validation (spec Layer 1). Ported and extended from the
original implementation: gaps, duplicates, staleness, price jumps,
invalid values, provider failures, latency, conflicting feeds."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from core.interfaces import (
    AssetHealthReport, Candle, QualityFinding, QualityFlag, QualitySeverity,
)

_INTERVAL_TD: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


@dataclass(frozen=True)
class HealthConfig:
    max_missing_streak: int = 3
    staleness_intervals: int = 3
    jump_sigma_k: float = 8.0
    jump_window: int = 100
    max_ingestion_latency_s: float = 300.0
    conflicting_feed_tolerance: float = 0.02  # 2% disagreement => conflict


def candles_to_frame(candles) -> pd.DataFrame:
    """Accepts pydantic Candle or ORM Candle rows (attribute access)."""
    if not candles:
        return pd.DataFrame()
    rows = [
        {
            "asset_symbol": c.asset_symbol, "interval": c.interval, "ts": c.ts,
            "open": c.open, "high": c.high, "low": c.low, "close": c.close,
            "volume": c.volume, "source": c.source, "ingestion_ts": c.ingestion_ts,
        }
        for c in candles
    ]
    return pd.DataFrame(rows)


class DataHealthChecker:
    def __init__(self, config: HealthConfig = HealthConfig()):
        self.cfg = config

    def check(
        self,
        asset_symbol: str,
        interval: str,
        candles: pd.DataFrame,
        *,
        now: Optional[datetime] = None,
        trades_24h: bool = True,
        provider_failed_recently: bool = False,
        ingestion_latency_s: Optional[float] = None,
        cross_check_prices: Optional[dict[str, float]] = None,
    ) -> AssetHealthReport:
        from core.clock import get_clock
        now = now or get_clock().now()
        step = _INTERVAL_TD.get(interval)
        if step is None:
            raise ValueError(f"Unknown interval '{interval}'")

        findings: list[QualityFinding] = []
        if candles.empty:
            return AssetHealthReport(
                asset_symbol=asset_symbol, interval=interval, checked_at=now,
                last_candle_ts=None,
                findings=[QualityFinding(
                    flag=QualityFlag.MISSING_OBSERVATIONS, severity=QualitySeverity.CRITICAL,
                    detail=f"No candles available for {asset_symbol} {interval}",
                )],
                trade_eligible=False,
            )

        ts = pd.to_datetime(candles["ts"])
        last_ts = ts.iloc[-1].to_pydatetime()

        # for non-24h markets (COMEX metals, DXY): weekends are not data gaps
        if trades_24h:
            gap_ts = ts
        else:
            weekdays = pd.DatetimeIndex([t for t in ts if t.weekday() < 5])
            gap_ts = weekdays if len(weekdays) else ts

        # duplicates
        dup = int(ts.duplicated().sum())
        if dup:
            findings.append(QualityFinding(
                flag=QualityFlag.DUPLICATE_RECORDS, severity=QualitySeverity.WARNING,
                detail=f"{dup} duplicate timestamps detected", observed_ts=last_ts))

        # gaps
        unique_ts = gap_ts.drop_duplicates().sort_values()

        def _first(idx):
            return idx.iloc[0] if isinstance(idx, pd.Series) else idx[0]

        def _last(idx):
            return idx.iloc[-1] if isinstance(idx, pd.Series) else idx[-1]

        if len(unique_ts) > 1:
            if not trades_24h:
                _, max_streak = _session_gap_analysis(unique_ts, step)
                missing = 0
                if max_streak > 0:
                    sev = QualitySeverity.CRITICAL if max_streak > self.cfg.max_missing_streak else QualitySeverity.WARNING
                    findings.append(QualityFinding(
                        flag=QualityFlag.MISSING_OBSERVATIONS, severity=sev,
                        detail=f"missing candles vs learned session, max gap {max_streak}x interval",
                        observed_ts=last_ts))
            else:
                span = (_last(unique_ts) - _first(unique_ts)).total_seconds()
                expected = int(span / step.total_seconds()) + 1
                missing = expected - len(unique_ts)
                if missing > 0:
                    max_streak = _max_gap_streak(unique_ts, step)
                    sev = QualitySeverity.CRITICAL if max_streak > self.cfg.max_missing_streak else QualitySeverity.WARNING
                    findings.append(QualityFinding(
                        flag=QualityFlag.MISSING_OBSERVATIONS, severity=sev,
                        detail=f"{missing} missing candles, max gap {max_streak}x interval",
                        observed_ts=last_ts))

        # staleness (session-aware: weekend now + weekday last candle is fine)
        if trades_24h:
            effective_now = now
        else:
            cur = now
            while cur.weekday() >= 5:  # walk back to Friday
                cur = cur - timedelta(days=1)
            effective_now = cur
        age = (effective_now - last_ts) / step
        if age > self.cfg.staleness_intervals:
            findings.append(QualityFinding(
                flag=QualityFlag.STALE_DATA, severity=QualitySeverity.CRITICAL,
                detail=f"Last candle {age:.1f} intervals old", observed_ts=last_ts))

        # price jumps
        closes = candles["close"].astype(float)
        if len(closes) > self.cfg.jump_window // 2:
            rets = np.log(closes).diff().dropna()
            sigma = rets.tail(self.cfg.jump_window).std()
            if sigma is not None and sigma > 1e-6:
                k = abs(rets.iloc[-1]) / sigma
                if k > self.cfg.jump_sigma_k:
                    findings.append(QualityFinding(
                        flag=QualityFlag.PRICE_JUMP, severity=QualitySeverity.WARNING,
                        detail=f"Last return {rets.iloc[-1]:.4f} is {k:.1f} sigma",
                        observed_ts=last_ts))

        # invalid values
        bad = int((candles[["open", "high", "low", "close"]].isna().any(axis=1) | (closes <= 0)).sum())
        if bad:
            findings.append(QualityFinding(
                flag=QualityFlag.INVALID_VALUES, severity=QualitySeverity.CRITICAL,
                detail=f"{bad} candles with invalid values", observed_ts=last_ts))

        # provider / latency
        if provider_failed_recently:
            findings.append(QualityFinding(
                flag=QualityFlag.PROVIDER_FAILURE, severity=QualitySeverity.WARNING,
                detail="Recent provider failure in ingest log"))
        if ingestion_latency_s is not None and ingestion_latency_s > self.cfg.max_ingestion_latency_s:
            findings.append(QualityFinding(
                flag=QualityFlag.HIGH_LATENCY, severity=QualitySeverity.WARNING,
                detail=f"Ingestion latency {ingestion_latency_s:.0f}s"))

        # conflicting feeds (independent source cross-check)
        if cross_check_prices:
            ref_close = float(closes.iloc[-1])
            for src_name, ref_price in cross_check_prices.items():
                if ref_price and abs(ref_price - ref_close) / ref_close > self.cfg.conflicting_feed_tolerance:
                    findings.append(QualityFinding(
                        flag=QualityFlag.CONFLICTING_FEEDS, severity=QualitySeverity.CRITICAL,
                        detail=f"{src_name} reports {ref_price:.2f} vs stored {ref_close:.2f} (>2% divergence)",
                        observed_ts=last_ts))

        if not findings:
            findings.append(QualityFinding(
                flag=QualityFlag.OK, severity=QualitySeverity.INFO, detail="All checks passed"))

        return AssetHealthReport(
            asset_symbol=asset_symbol, interval=interval, checked_at=now,
            last_candle_ts=last_ts, findings=findings,
            trade_eligible=not any(f.severity == QualitySeverity.CRITICAL for f in findings),
        )


def _session_gap_analysis(unique_ts, step: timedelta) -> tuple[int, int]:
    """Non-24h markets (COMEX metals, DXY): learn the trading schedule from
    observed slot frequencies instead of assuming 24h weekdays.

    A (weekday, hour, minute) slot is 'expected' when seen in >= 30% of the
    weeks in the window. Missing candles and the longest consecutive missing
    run are then evaluated over the most recent 7 days against that schedule.
    """
    series = pd.Series(unique_ts) if isinstance(unique_ts, pd.DatetimeIndex) else unique_ts
    counts: dict[tuple, int] = {}
    for t in series:
        key = (t.weekday(), t.hour, t.minute)
        counts[key] = counts.get(key, 0) + 1
    max_count = max(counts.values()) if counts else 0
    # rule 1: a slot is expected when seen in >= half the weeks of the most common slot
    threshold = max(2, int(max_count * 0.5))
    expected_slots = {k for k, c in counts.items() if c >= threshold}

    # rule 2: cross-check across weekdays — an hour traded on most weekdays is
    # expected on every weekday present in the data. This prevents a data hole
    # from deleting its own expectation (self-concealment).
    weekdays_present = {wd for (wd, _, _) in counts}
    hour_weekdays: dict[int, set] = {}
    for (wd, hour, minute), c in counts.items():
        if c > 0:
            hour_weekdays.setdefault((hour, minute), set()).add(wd)
    for (wd, hour, minute) in counts:
        observers = hour_weekdays.get((hour, minute), set())
        if len(observers) >= max(2, 0.6 * len(weekdays_present)) and wd in weekdays_present:
            expected_slots.add((wd, hour, minute))

    recent_end = series.iloc[-1]
    recent_start = recent_end - timedelta(days=7)
    recent = [t for t in series if t >= recent_start]

    missing_total = 0
    streak = max_streak = 0
    prev = None
    for t in recent:
        if prev is not None:
            gap_steps = int((t - prev).total_seconds() / step.total_seconds())
            for i in range(1, gap_steps):
                candidate = prev + step * i
                if (candidate.weekday(), candidate.hour, candidate.minute) in expected_slots:
                    missing_total += 1
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
        prev = t
    return len(expected_slots), max(max_streak, 0)


def _weekday_slots(start_ts, end_ts, step: timedelta) -> int:
    """Expected candle count between start and end on weekdays only
    (non-24h markets: COMEX metals, dollar index)."""
    count = 0
    cur = start_ts.to_pydatetime() if hasattr(start_ts, "to_pydatetime") else start_ts
    end = end_ts.to_pydatetime() if hasattr(end_ts, "to_pydatetime") else end_ts
    while cur <= end:
        if cur.weekday() < 5:
            count += 1
        cur = cur + step
    return count


def _max_gap_streak(sorted_ts, step: timedelta) -> int:
    if len(sorted_ts) < 2:
        return 0
    series = pd.Series(sorted_ts) if isinstance(sorted_ts, pd.DatetimeIndex) else sorted_ts
    steps = series.diff().dropna() / step
    steps = steps.round().astype(int)
    steps = steps[steps > 0]
    return int(steps.max() - 1) if not steps.empty else 0
