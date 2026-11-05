"""Contract tests: malformed market data must be rejected at the boundary."""
from datetime import datetime, timezone

import pytest

from core.interfaces import Candle, Interval, NewsArticle

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def base_candle(**overrides):
    params = dict(
        asset_symbol="BTCUSD", interval=Interval.H1, ts=TS,
        open=100.0, high=101.0, low=99.0, close=100.5,
        volume=10.0, source="test", ingestion_ts=TS,
    )
    params.update(overrides)
    return params


class TestCandleContract:
    def test_valid_candle(self):
        Candle(**base_candle())

    def test_high_below_open_rejected(self):
        with pytest.raises(Exception):
            Candle(**base_candle(high=99.9))

    def test_low_above_close_rejected(self):
        with pytest.raises(Exception):
            Candle(**base_candle(low=100.6))

    def test_zero_price_rejected(self):
        with pytest.raises(Exception):
            Candle(**base_candle(open=0))

    def test_negative_volume_rejected(self):
        with pytest.raises(Exception):
            Candle(**base_candle(volume=-1))

    def test_naive_timestamp_rejected(self):
        with pytest.raises(Exception):
            Candle(**base_candle(ts=datetime(2026, 1, 1)))


class TestNewsContract:
    def test_valid_article(self):
        NewsArticle(
            external_id="x" * 24, url="https://example.com/a",
            title="Fed holds rates", source="cnbc",
            published_ts=TS, ingested_ts=TS,
        )

    def test_naive_published_ts_rejected(self):
        with pytest.raises(Exception):
            NewsArticle(
                external_id="x" * 24, url="https://example.com/a",
                title="t", source="cnbc",
                published_ts=datetime(2026, 1, 1), ingested_ts=TS,
            )
