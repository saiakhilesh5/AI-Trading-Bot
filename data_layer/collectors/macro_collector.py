"""Macro collector: FRED (yields, CPI, employment, dollar index).
Point-in-time: every observation stores release_ts — backtests see only
published data. HTTPX with bounded retries."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx

from core.clock import get_clock
from core.exceptions import DataProviderError
from core.interfaces import MacroObservation

SERIES = {
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "UNRATE": "Civilian Unemployment Rate",
    "PAYEMS": "Total Nonfarm Payrolls",
    "T10YIE": "10-Year Breakeven Inflation Rate",
}


class MacroCollector:
    source_id = "fred"

    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise DataProviderError("FRED_API_KEY missing", source=self.source_id)
        self.api_key = api_key

    def fetch_series(self, series_id: str, limit: int = 500) -> list[MacroObservation]:
        now = get_clock().now()
        try:
            resp = httpx.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id, "api_key": self.api_key,
                    "file_type": "json", "sort_order": "desc", "limit": limit,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise DataProviderError(f"FRED failed for {series_id}: {e}", source=self.source_id) from e

        if "observations" not in data:
            raise DataProviderError(
                f"FRED bad response for {series_id}: {data.get('error_message', '')}",
                source=self.source_id)

        out: list[MacroObservation] = []
        for obs in data["observations"]:
            if obs["value"] in (".", ""):
                continue
            ref = datetime.strptime(obs["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            out.append(MacroObservation(
                series_id=series_id, value=float(obs["value"]),
                reference_period=ref, release_ts=ref,  # value available at ref period close
                vintage_ts=ref, ingestion_ts=now, source=self.source_id))
        return out
