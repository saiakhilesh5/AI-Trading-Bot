"""Ingestion orchestrator: collect -> validate -> store -> health-check.
Provider fallback: primary source failure triggers the configured fallback.
Failures are recorded in ingest_log; data is never fabricated."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from config.settings import get_settings
from core.clock import get_clock
from core.exceptions import DataProviderError
from core.logging_setup import get_logger
from core.interfaces import Asset, AssetHealthReport, Candle, Interval, NewsArticle
from data_layer.collectors.crypto_collector import CryptoCollector
from data_layer.collectors.metals_collector import MetalsCollector
from data_layer.storage import repository as repo
from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import AssetConfig, CryptoDerivativeStat, MarketEvent, NewsArticle as NewsArticleORM
from data_layer.validators.data_health import DataHealthChecker, HealthConfig, candles_to_frame

log = get_logger()


class CollectorBundle:
    """Lazily-built collectors per asset class with fallback chains."""

    def __init__(self):
        self.settings = get_settings()
        self._crypto = CryptoCollector()
        self._metals = MetalsCollector(
            goldprice_api_key=self.settings.goldprice_api_key,
            alphavantage_api_key=self.settings.alphavantage_api_key,
        )

    def candles(self, asset: Asset, interval: Interval,
                start: Optional[datetime], limit: int) -> tuple[list[Candle], str]:
        """Returns (candles, source_used). Falls back on primary failure."""
        if asset.asset_class.value == "crypto":
            try:
                return self._crypto.fetch_candles(asset.symbol, interval, start=start, limit=limit), "primary"
            except DataProviderError:
                log.warning("crypto_primary_failed", asset=asset.symbol)
                return self._crypto.fetch_fallback_history(asset.symbol, days=365), "fallback"
        else:
            # metals/DXY: yfinance is the OHLCV workhorse (only OHLCV source)
            return self._metals.fetch_candles(asset.symbol, interval, start=start, limit=limit), "primary"

    def spot_check(self, asset: Asset) -> Optional[float]:
        """Independent price verification where a free source exists."""
        if asset.asset_class.value == "crypto":
            return self._crypto.fetch_fallback_price(asset.symbol)
        if asset.asset_class.value == "precious_metal":
            return self._metals.spot_check(asset.symbol)
        return None

    def derivatives(self, asset: Asset):
        if asset.asset_class.value == "crypto":
            return self._crypto.fetch_derivatives(asset.symbol)
        return []


class IngestionOrchestrator:
    def __init__(self, health_checker: Optional[DataHealthChecker] = None):
        self.collectors = CollectorBundle()
        self.health = health_checker or DataHealthChecker()

    def ingest_candles(self, asset: Asset, interval: Interval,
                       start: Optional[datetime] = None, limit: int = 1000) -> AssetHealthReport:
        session = get_session()
        try:
            log_row = repo.begin_ingest(session, "orchestrator", asset.symbol, "candles")
            session.commit()
            t0 = time.monotonic()
            failed_recently = self._failed_recently(session, asset.price_source)
            try:
                candles, used = self.collectors.candles(asset, interval, start, limit)
            except DataProviderError as e:
                repo.finish_ingest(session, log_row.id, "failed", 0, str(e))
                repo.audit(session, "ingestion", "provider_failure", asset.symbol, {"error": str(e)})
                session.commit()
                log.warning("ingest_failed", asset=asset.symbol, interval=interval.value)
                return self._health(session, asset, interval, failed_recently=True)

            repo.upsert_candles(session, candles)
            repo.finish_ingest(session, log_row.id, "ok", len(candles), None)
            repo.audit(session, "ingestion", "candles_stored", asset.symbol,
                       {"count": len(candles), "source_used": used, "interval": interval.value})
            session.commit()
            latency = time.monotonic() - t0
            return self._health(session, asset, interval, failed_recently=failed_recently,
                                latency_s=latency)
        finally:
            session.close()

    def ingest_derivatives(self, asset: Asset) -> int:
        session = get_session()
        try:
            stats = self.collectors.derivatives(asset)
            if not stats:
                return 0
            for s in stats:
                session.add(CryptoDerivativeStat(
                    asset_symbol=s.asset_symbol, ts=s.ts,
                    funding_rate=s.funding_rate, open_interest=s.open_interest,
                    source=s.source, ingestion_ts=s.ingestion_ts))
            session.commit()
            return len(stats)
        except DataProviderError as e:
            session.rollback()
            log.warning("derivatives_ingest_failed", asset=asset.symbol, error=str(e))
            return 0
        finally:
            session.close()

    def ingest_news(self, articles: list[NewsArticle]) -> int:
        if not articles:
            return 0
        session = get_session()
        try:
            rows = [NewsArticleORM(
                external_id=a.external_id, url=a.url, title=a.title, summary=a.summary,
                source=a.source, source_reliability=a.source_reliability.value,
                published_ts=a.published_ts, ingested_ts=a.ingested_ts,
                affected_assets=",".join(a.affected_assets)) for a in articles]
            return repo.insert_news_articles(session, rows)
        finally:
            session.close()

    def run_all(self, interval: Interval) -> list[AssetHealthReport]:
        session = get_session()
        try:
            assets = [Asset(**{
                "symbol": a.symbol, "name": a.name,
                "asset_class": a.asset_class, "price_source": a.price_source,
                "trades_24h": a.trades_24h, "notes": a.notes,
            }) for a in repo.load_assets(session)]
        finally:
            session.close()

        reports: list[AssetHealthReport] = []
        for asset in assets:
            try:
                report = self.ingest_candles(asset, interval)
                if asset.asset_class.value == "crypto":
                    self.ingest_derivatives(asset)
            except Exception as e:  # orchestration failure: log, continue others
                log.error("ingest_orchestration_failed", asset=asset.symbol, error=str(e))
                continue
            reports.append(report)
        return reports

    def _health(self, session: Session, asset: Asset, interval: Interval, *,
                failed_recently: bool, latency_s: Optional[float] = None) -> AssetHealthReport:
        now = get_clock().now()
        from datetime import timedelta
        start = now - timedelta(days=30) if interval != Interval.D1 else None
        candles = repo.get_candles(session, asset.symbol, interval.value, start=start)
        frame = candles_to_frame(candles).tail(500)
        cross = self.collectors.spot_check(asset)
        cross_map = {"independent_source": cross} if cross else None
        report = self.health.check(
            asset.symbol, interval.value, frame,
            now=now, trades_24h=asset.trades_24h,
            provider_failed_recently=failed_recently,
            ingestion_latency_s=latency_s, cross_check_prices=cross_map)
        repo.record_health(session, asset.symbol, interval.value, report.trade_eligible, report.findings)
        session.commit()
        return report

    def _failed_recently(self, session: Session, price_source: str, window_minutes: int = 30) -> bool:
        from datetime import timedelta

        from data_layer.storage.schemas import IngestLog
        cutoff = get_clock().now() - timedelta(minutes=window_minutes)
        count = (
            session.query(IngestLog)
            .filter(IngestLog.status == "failed", IngestLog.started_at >= cutoff)
            .count()
        )
        return count > 0
