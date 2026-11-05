"""Crypto collector: Binance OHLCV via CCXT (primary), CoinGecko (fallback),
plus funding rates / open interest from Binance Futures (public)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import httpx

from core.clock import get_clock
from core.exceptions import DataProviderError
from core.interfaces import Candle, CryptoDerivativeStats, Interval
from data_layer.validators.timestamp_validator import TimestampValidator

_INTERVAL_MAP = {Interval.H1: "1h", Interval.H4: "4h", Interval.D1: "1d"}
_SYMBOL_MAP = {"BTCUSD": "BTC/USDT", "ETHUSD": "ETH/USDT"}
_COINGECKO = {"BTCUSD": "bitcoin", "ETHUSD": "ethereum"}


class CryptoCollector:
    source_id = "ccxt:binance"
    fallback_id = "coingecko"

    def __init__(self):
        self._exchange = ccxt.binance({"enableRateLimit": True})
        self._validator = TimestampValidator()
        self._http = httpx.Client(timeout=15)

    def fetch_candles(
        self, asset_symbol: str, interval: Interval,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[Candle]:
        symbol = _SYMBOL_MAP.get(asset_symbol)
        if not symbol:
            raise DataProviderError(f"Unsupported crypto asset '{asset_symbol}'", source=self.source_id)
        now = get_clock().now()
        candles: list[Candle] = []
        page_start = int(start.timestamp() * 1000) if start else None
        max_pages = 10
        for _ in range(max_pages):
            try:
                params: dict = {"limit": 1000}
                if page_start:
                    params["since"] = page_start
                raw = self._exchange.fetch_ohlcv(symbol, _INTERVAL_MAP[interval], **params)
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                raise DataProviderError(f"Binance OHLCV failed: {e}", source=self.source_id) from e
            if not raw:
                break
            if end:
                end_ms = int(end.timestamp() * 1000)
                raw = [r for r in raw if r[0] <= end_ms]

            for ts_ms, o, h, l, c, v in raw:
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                if ts >= now:  # skip still-open candle
                    continue
                try:
                    candle = Candle(
                        asset_symbol=asset_symbol, interval=interval, ts=ts,
                        open=float(o), high=float(h), low=float(l), close=float(c),
                        volume=float(v), source=self.source_id, ingestion_ts=now,
                    )
                except ValueError:
                    continue  # invalid rows are dropped, never fabricated
                err = self._validator.validate_candle(candle, now=now, prev=candles[-1] if candles else None)
                if err:
                    continue
                candles.append(candle)
            if len(candles) >= limit or (end and raw and raw[-1][0] >= int(end.timestamp() * 1000)):
                break
            page_start = raw[-1][0] + 1
            if not start and not raw:
                break
        return candles

    def fetch_derivatives(self, asset_symbol: str) -> list[CryptoDerivativeStats]:
        symbol = _SYMBOL_MAP.get(asset_symbol)
        if not symbol:
            raise DataProviderError(f"Unsupported asset '{asset_symbol}'", source=self.source_id)
        base = symbol.split("/")[0]
        now = get_clock().now()
        out: list[CryptoDerivativeStats] = []
        try:
            funding = self._exchange.fapiPublicGetFundingRate({"symbol": f"{base}USDT", "limit": 100})
            for f in funding:
                ts = datetime.fromtimestamp(int(f["fundingTime"]) / 1000, tz=timezone.utc)
                if ts >= now:
                    continue
                out.append(CryptoDerivativeStats(
                    asset_symbol=asset_symbol, ts=ts, funding_rate=float(f["fundingRate"]),
                    source=f"{self.source_id}:funding", ingestion_ts=now))
            oi = self._exchange.fapiPublicGetOpenInterest({"symbol": f"{base}USDT"})
            out.append(CryptoDerivativeStats(
                asset_symbol=asset_symbol,
                ts=datetime.fromtimestamp(oi["time"] / 1000, tz=timezone.utc),
                open_interest=float(oi["openInterest"]),
                source=f"{self.source_id}:oi", ingestion_ts=now))
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise DataProviderError(f"Binance derivatives failed: {e}", source=self.source_id) from e
        return out

    def fetch_fallback_price(self, asset_symbol: str) -> Optional[float]:
        """CoinGecko spot for cross-checking / degraded mode."""
        coin = _COINGECKO.get(asset_symbol)
        if not coin:
            return None
        try:
            resp = self._http.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": coin, "vs_currencies": "usd"},
            )
            resp.raise_for_status()
            return float(resp.json()[coin]["usd"])
        except Exception:
            return None

    def fetch_fallback_history(self, asset_symbol: str, days: int = 365) -> list[Candle]:
        """Daily-only fallback history from CoinGecko (degraded mode)."""
        coin = _COINGECKO.get(asset_symbol)
        if not coin:
            return []
        try:
            resp = self._http.get(
                f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
            )
            resp.raise_for_status()
            prices = resp.json()["prices"]
        except Exception as e:
            raise DataProviderError(f"CoinGecko history failed: {e}", source=self.fallback_id) from e
        now = get_clock().now()
        out = []
        for point in prices:
            ts = datetime.fromtimestamp(point[0] / 1000, tz=timezone.utc)
            if ts >= now:
                continue
            # CoinGecko gives a price series only: build synthetic OHLC per point
            price = float(point[1])
            try:
                out.append(Candle(
                    asset_symbol=asset_symbol, interval=Interval.D1, ts=ts,
                    open=price, high=price, low=price, close=price, volume=0.0,
                    source=self.fallback_id, ingestion_ts=now))
            except ValueError:
                continue
        return out
