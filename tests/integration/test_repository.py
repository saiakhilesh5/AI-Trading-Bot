"""Repository integration tests against the live TimescaleDB container.

These are integration tests: they require docker compose to be running.
Each test cleans its own rows to stay idempotent.
"""
from datetime import datetime, timedelta, timezone

import pytest

from core.interfaces import Candle, Interval, NewsArticle
from data_layer.storage import repository as repo
from data_layer.storage.data_access import get_session, init_database
from data_layer.storage.schemas import Candle as CandleORM, NewsArticle as NewsORM

from tests.conftest import make_candles

pytestmark = pytest.mark.skipif(__import__("os").environ.get("SKIP_DB_TESTS") == "1", reason="DB tests skipped")


@pytest.fixture(scope="module", autouse=True)
def _db():
    init_database()
    yield


@pytest.fixture(autouse=True)
def _clean():
    session = get_session()
    session.query(CandleORM).filter(CandleORM.asset_symbol == "TESTASSET").delete()
    session.query(NewsORM).filter(NewsORM.external_id == "testarticle01").delete()
    session.commit()
    session.close()
    yield
    session = get_session()
    session.query(CandleORM).filter(CandleORM.asset_symbol == "TESTASSET").delete()
    session.query(NewsORM).filter(NewsORM.external_id == "testarticle01").delete()
    session.commit()
    session.close()


def _orm(candles):
    return [CandleORM(
        asset_symbol=c.asset_symbol, interval=c.interval.value, ts=c.ts,
        open=c.open, high=c.high, low=c.low, close=c.close, volume=c.volume,
        source=c.source, ingestion_ts=c.ingestion_ts) for c in candles]


class TestRepository:
    def test_upsert_and_read_roundtrip(self, sim_clock):
        session = get_session()
        try:
            candles = make_candles(asset="TESTASSET", n=20)
            repo.upsert_candles(session, _orm(candles))
            session.commit()
            out = repo.get_candles(session, "TESTASSET", "1h")
            assert len(out) == 20
            assert out[0].close == candles[0].close
        finally:
            session.close()

    def test_upsert_is_idempotent(self, sim_clock):
        session = get_session()
        try:
            candles = make_candles(asset="TESTASSET", n=20)
            repo.upsert_candles(session, _orm(candles))
            session.commit()
            repo.upsert_candles(session, _orm(candles))
            session.commit()
            out = repo.get_candles(session, "TESTASSET", "1h")
            assert len(out) == 20  # no duplicates
        finally:
            session.close()

    def test_first_ingested_version_wins(self, sim_clock):
        session = get_session()
        try:
            candles = make_candles(asset="TESTASSET", n=20)
            repo.upsert_candles(session, _orm(candles))
            session.commit()
            revised = _orm([candles[10].model_copy(update={"close": 999.0, "high": 1000.0})])
            repo.upsert_candles(session, revised)
            session.commit()
            out = repo.get_candles(session, "TESTASSET", "1h")
            target = [c for c in out if c.ts == candles[10].ts][0]
            assert target.close == candles[10].close  # history never rewritten
        finally:
            session.close()

    def test_as_of_hides_future_ingestions(self, sim_clock):
        session = get_session()
        try:
            early = make_candles(asset="TESTASSET", n=10)
            repo.upsert_candles(session, _orm(early))
            session.commit()
            later_ts = datetime.now(timezone.utc) + timedelta(days=365)
            later = make_candles(asset="TESTASSET", n=5, start=early[-1].ts + timedelta(hours=1))
            later_orm = _orm(later)
            for r in later_orm:
                r.ingestion_ts = later_ts
            repo.upsert_candles(session, later_orm)
            session.commit()
            visible = repo.get_candles(session, "TESTASSET", "1h",
                                       as_of=datetime(2026, 1, 16, tzinfo=timezone.utc))
            assert len(visible) == 10  # backtest cannot see future-ingested rows
        finally:
            session.close()

    def test_news_deduplication(self, sim_clock):
        session = get_session()
        try:
            art = NewsORM(
                external_id="testarticle01", url="https://example.com/x",
                title="t", summary="", source="cnbc", source_reliability="medium",
                published_ts=datetime(2026, 1, 14, tzinfo=timezone.utc),
                ingested_ts=datetime(2026, 1, 15, tzinfo=timezone.utc),
                affected_assets="[]")
            assert repo.insert_news_articles(session, [art]) == 1
            assert repo.insert_news_articles(session, [art]) == 0  # duplicate rejected
        finally:
            session.close()

    def test_last_candle_ts(self, sim_clock):
        session = get_session()
        try:
            candles = make_candles(asset="TESTASSET", n=10)
            repo.upsert_candles(session, _orm(candles))
            session.commit()
            assert repo.last_candle_ts(session, "TESTASSET", "1h") == candles[-1].ts
        finally:
            session.close()
