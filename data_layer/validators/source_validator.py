"""Source verification: provider fallback chain and independent
price cross-checks between sources."""
from __future__ import annotations

import hashlib
from typing import Optional

import httpx

from core.exceptions import DataProviderError
from core.interfaces import SourceReliability

KNOWN_RELIABILITY: dict[str, SourceReliability] = {
    "binance": SourceReliability.HIGH,
    "coingecko": SourceReliability.MEDIUM,
    "goldprice_dev": SourceReliability.MEDIUM,
    "alphavantage": SourceReliability.MEDIUM,
    "yfinance": SourceReliability.MEDIUM,
    "fred": SourceReliability.HIGH,
    "wsj": SourceReliability.HIGH,
    "cnbc": SourceReliability.MEDIUM,
    "cointelegraph": SourceReliability.MEDIUM,
    "coindesk": SourceReliability.MEDIUM,
    "gdelt": SourceReliability.LOW,
    "google-news": SourceReliability.LOW,
}


def reliability_of(source: str) -> SourceReliability:
    return KNOWN_RELIABILITY.get(source, SourceReliability.UNKNOWN)


def dedup_key(source: str, url: str) -> str:
    return hashlib.sha256(f"{source}|{url}".encode()).hexdigest()[:24]


class CoinGeckoCrossCheck:
    """Independent crypto price check for conflicting-feed detection."""

    api = "https://api.coingecko.com/api/v3/simple/price"

    def __init__(self, client: Optional[httpx.Client] = None):
        self._client = client or httpx.Client(timeout=15)

    def price(self, coin: str) -> Optional[float]:
        try:
            resp = self._client.get(self.api, params={"ids": coin, "vs_currencies": "usd"})
            resp.raise_for_status()
            data = resp.json()
            return float(data[coin]["usd"])
        except Exception:
            return None  # cross-check unavailable is not an error — health stays clean
