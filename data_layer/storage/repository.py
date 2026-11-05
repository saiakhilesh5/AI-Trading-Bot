"""Repository pattern over the ORM. Subsystems go through these functions;
nothing outside data_layer touches the Session directly."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from core.clock import get_clock
from data_layer.storage.schemas import (
    AssetConfig, AuditLog, Candle, DataQuality, IngestLog, MacroObservation,
    NewsArticle,
)


def upsert_candles(session: Session, rows: Sequence[Candle]) -> int:
    """Bulk upsert; on conflict keep the first-ingested version (point-in-time)."""
    if not rows:
        return 0
    payload = [
        {
            "asset_symbol": r.asset_symbol, "interval": r.interval, "ts": r.ts,
            "open": r.open, "high": r.high, "low": r.low, "close": r.close,
            "volume": r.volume, "source": r.source, "ingestion_ts": r.ingestion_ts,
        }
        for r in rows
    ]
    stmt = pg_insert(Candle).values(payload)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_candle")
    session.execute(stmt)
    return len(payload)


def get_candles(
    session: Session,
    asset_symbol: str,
    interval: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    as_of: Optional[datetime] = None,
) -> list[Candle]:
    stmt = (
        select(Candle)
        .where(Candle.asset_symbol == asset_symbol, Candle.interval == interval)
        .order_by(Candle.ts)
    )
    if start:
        stmt = stmt.where(Candle.ts >= start)
    if end:
        stmt = stmt.where(Candle.ts <= end)
    if as_of:
        # point-in-time read: only data the system knew about by as_of
        stmt = stmt.where(Candle.ingestion_ts <= as_of)
    return list(session.scalars(stmt))


def last_candle_ts(session: Session, asset_symbol: str, interval: str) -> Optional[datetime]:
    stmt = (
        select(Candle.ts)
        .where(Candle.asset_symbol == asset_symbol, Candle.interval == interval)
        .order_by(Candle.ts.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def upsert_macro(session: Session, rows: Sequence[MacroObservation]) -> int:
    if not rows:
        return 0
    payload = [
        {
            "series_id": r.series_id, "value": r.value,
            "reference_period": r.reference_period, "release_ts": r.release_ts,
            "vintage_ts": r.vintage_ts, "ingestion_ts": r.ingestion_ts,
            "source": r.source,
        }
        for r in rows
    ]
    stmt = pg_insert(MacroObservation).values(payload)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_macro")
    session.execute(stmt)
    return len(payload)


def get_macro_as_of(
    session: Session, series_id: str, as_of: datetime
) -> list[MacroObservation]:
    """Point-in-time macro: only releases published by `as_of`."""
    stmt = (
        select(MacroObservation)
        .where(MacroObservation.series_id == series_id, MacroObservation.release_ts <= as_of)
        .order_by(MacroObservation.reference_period)
    )
    return list(session.scalars(stmt))


def insert_news_articles(session: Session, rows: Sequence[NewsArticle]) -> int:
    """Insert with application-level duplicate protection (external_id seen before)."""
    if not rows:
        return 0
    existing_ids = {
        r.external_id
        for r in session.query(NewsArticle.external_id).filter(
            NewsArticle.external_id.in_([r.external_id for r in rows])
        ).all()
    }
    new = 0
    for r in rows:
        if r.external_id in existing_ids:
            continue
        session.add(_news_row(r))
        new += 1
    return new


def _news_row(r: NewsArticle) -> NewsArticle:
    """Interface contract -> ORM row (external content stored verbatim but
    never trusted; see agents.llm_client.wrap_untrusted)."""
    return NewsArticle(
        external_id=r.external_id, url=r.url, title=r.title, summary=r.summary,
        source=r.source, source_reliability=r.source_reliability.value
        if hasattr(r.source_reliability, "value") else str(r.source_reliability),
        published_ts=r.published_ts, ingested_ts=r.ingested_ts,
        affected_assets=json.dumps(r.affected_assets),
    )


def begin_ingest(session: Session, provider: str, asset_symbol: Optional[str], data_kind: str) -> IngestLog:
    log = IngestLog(
        started_at=get_clock().now(), provider=provider,
        asset_symbol=asset_symbol, data_kind=data_kind, status="running",
    )
    session.add(log)
    session.flush()
    return log


def finish_ingest(session: Session, log_id: int, status: str, count: int, error: Optional[str] = None) -> None:
    log = session.get(IngestLog, log_id)
    if log is None:
        return
    log.finished_at = get_clock().now()
    log.status = status
    log.records_fetched = count
    log.error = error


def record_health(session: Session, asset_symbol: str, interval: str, trade_eligible: bool, findings: list) -> None:
    session.add(
        DataQuality(
            checked_at=get_clock().now(), asset_symbol=asset_symbol, interval=interval,
            trade_eligible=trade_eligible,
            findings_json=json.dumps([f.model_dump(mode="json") for f in findings]),
        )
    )


def audit(session: Session, actor: str, action: str, subject: Optional[str] = None, detail: Optional[dict] = None) -> None:
    session.add(
        AuditLog(
            ts=get_clock().now(), actor=actor, action=action, subject=subject,
            detail_json=json.dumps(detail) if detail else None,
        )
    )


def seed_assets(session: Session, assets: Sequence[AssetConfig]) -> None:
    for a in assets:
        stmt = pg_insert(AssetConfig).values(
            symbol=a.symbol, name=a.name, asset_class=a.asset_class,
            price_source=a.price_source, trades_24h=a.trades_24h,
            notes=a.notes, enabled=True,
        ).on_conflict_do_nothing(index_elements=["symbol"])
        session.execute(stmt)


def load_assets(session: Session, enabled_only: bool = True) -> list[AssetConfig]:
    stmt = select(AssetConfig).order_by(AssetConfig.symbol)
    if enabled_only:
        stmt = stmt.where(AssetConfig.enabled)
    return list(session.scalars(stmt))


def clear_candles(session: Session, asset_symbol: str) -> None:
    session.execute(delete(Candle).where(Candle.asset_symbol == asset_symbol))
