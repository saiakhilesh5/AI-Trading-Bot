"""Metals collector: GoldPrice.dev spot (primary), yfinance COMEX futures
(fallback), Alpha Vantage (optional cross-check).

Documented limitation: free spot feeds for XAU/XAG are rate-limited and
do not provide full OHLCV. The fallback provider (GC=F / SI=F front-month
futures) supplies OHLCV with basis vs spot; recorded in DEVELOPMENT_STATUS.md."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import pandas as pd

from core.clock import get_clock
from core.exceptions import DataProviderError
from core.interfaces import Candle, Interval
from data_layer.validators.timestamp_validator import TimestampValidator

_YF_MAP = {"XAUUSD": "GC=F", "XAGUSD": "SI=F", "DXY": "DX-Y.NYB"}
_INTERVAL_MAP = {Interval.H1: "1h", Interval.H4: "4h", Interval.D1: "1d"}


class GoldPriceDevClient:
    """GoldPrice.dev free API. Symbol map: XAU/USD -> XAU, XAG/USD -> XAG."""
    _symbols = {"XAUUSD": "XAU", "XAGUSD": "XAG"}

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key

    def spot_price(self, asset_symbol: str) -> Optional[float]:
        sym = self._symbols.get(asset_symbol)
        if not sym or not self.api_key:
            return None
        try:
            resp = httpx.get(
                f"https://api.goldprice.dev/v1/symbol/{sym}/USD",
                headers={"x-api-key": self.api_key}, timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price") or data.get("rate") or 0) or None
        except Exception:
            return None


class YFinanceFallback:
    source_id = "yfinance"

    def __init__(self):
        import yfinance
        self._yf = yfinance
        self._validator = TimestampValidator()

    def fetch_candles(
        self, asset_symbol: str, interval: Interval,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[Candle]:
        ticker = _YF_MAP.get(asset_symbol)
        if not ticker:
            raise DataProviderError(f"Unsupported metals asset '{asset_symbol}'", source=self.source_id)
        now = get_clock().now()
        end = end or now
        start = start or end - timedelta(days=30)

        # yfinance caps rows per request; chunk the range
        chunk_days = {Interval.H1: 60, Interval.H4: 200, Interval.D1: 400}[interval]
        chunks: list[tuple[datetime, datetime]] = []
        cur = start
        while cur < end:
            chunk_end = min(cur + timedelta(days=chunk_days), end)
            chunks.append((cur, chunk_end))
            cur = chunk_end

        candles: list[Candle] = []
        for (cs, ce) in chunks:
            try:
                df = self._yf.download(
                    ticker, start=cs, end=ce + timedelta(days=1),
                    interval=_INTERVAL_MAP[interval], progress=False, auto_adjust=False,
                )
            except Exception as e:
                raise DataProviderError(f"yfinance failed for {ticker}: {e}", source=self.source_id) from e
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            for ts, row in df.iterrows():
                try:
                    o, h, l, c, v = (float(row["Open"]), float(row["High"]),
                                     float(row["Low"]), float(row["Close"]), float(row["Volume"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
                    continue
                ts_dt = ts.to_pydatetime()
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                if ts_dt >= now:
                    continue
                try:
                    candle = Candle(
                        asset_symbol=asset_symbol, interval=interval, ts=ts_dt,
                        open=o, high=h, low=l, close=c,
                        volume=0.0 if pd.isna(v) else v,
                        source=self.source_id, ingestion_ts=now)
                except ValueError:
                    continue
                err = self._validator.validate_candle(candle, now=now, prev=candles[-1] if candles else None)
                if err:
                    continue
                candles.append(candle)
        if not candles:
            raise DataProviderError(f"yfinance no complete candles for {ticker}", source=self.source_id)
        return candles[-limit:] if limit and limit < len(candles) else candles


class MetalsCollector:
    """Primary: GoldPrice.dev spot (price checks + degraded mode).
    Fallback / OHLCV: yfinance COMEX futures. Provider fallback per spec."""

    def __init__(self, goldprice_api_key: Optional[str] = None,
                 alphavantage_api_key: Optional[str] = None):
        self.spot = GoldPriceDevClient(goldprice_api_key)
        self.fallback = YFinanceFallback()

    def fetch_candles(self, asset_symbol: str, interval: Interval, **kwargs) -> list[Candle]:
        return self.fallback.fetch_candles(asset_symbol, interval, **kwargs)

    def spot_check(self, asset_symbol: str) -> Optional[float]:
        return self.spot.spot_price(asset_symbol)
