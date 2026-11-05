"""SQLAlchemy models. Hypertables (TimescaleDB) are created for
time-series tables in migration; plain tables for registry/state.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, PrimaryKeyConstraint,
    String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    from core.clock import get_clock
    return get_clock().now()


# ---------- market data (hypertables) ----------

class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("asset_symbol", "interval", "ts", name="uq_candle"),
        PrimaryKeyConstraint("id", "ts", name="pk_candles"),
    )

    id: Mapped[int] = mapped_column(Integer, autoincrement=True)
    asset_symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    interval: Mapped[str] = mapped_column(String(4), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    ingestion_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CryptoDerivativeStat(Base):
    __tablename__ = "crypto_derivatives"
    __table_args__ = (
        UniqueConstraint("asset_symbol", "ts", "source", name="uq_deriv"),
        PrimaryKeyConstraint("id", "ts", name="pk_crypto_derivatives"),
    )

    id: Mapped[int] = mapped_column(Integer, autoincrement=True)
    asset_symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    funding_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    ingestion_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MacroObservation(Base):
    """Point-in-time: a revised value is a NEW row keyed by vintage."""
    __tablename__ = "macro_observations"
    __table_args__ = (
        UniqueConstraint("series_id", "reference_period", "vintage_ts", "release_ts", name="uq_macro"),
        PrimaryKeyConstraint("id", "release_ts", name="pk_macro_observations"),
    )

    id: Mapped[int] = mapped_column(Integer, autoincrement=True)
    series_id: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    reference_period: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    release_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    vintage_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingestion_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="fred")


class NewsArticle(Base):
    # composite PK: TimescaleDB requires the partition column in unique indexes
    __tablename__ = "news_articles"
    __table_args__ = (PrimaryKeyConstraint("internal_id", "published_ts", name="pk_news_articles"),)

    internal_id: Mapped[int] = mapped_column(Integer, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_reliability: Mapped[str] = mapped_column(String(16), nullable=False, default="unknown")
    published_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    ingested_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    affected_assets: Mapped[str] = mapped_column(Text, nullable=False, default="[]")


class MarketEvent(Base):
    __tablename__ = "news_events"
    __table_args__ = (PrimaryKeyConstraint("internal_id", "published_ts", name="pk_news_events"),)

    internal_id: Mapped[int] = mapped_column(Integer, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    affected_assets: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_reliability: Mapped[str] = mapped_column(String(16), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(16), nullable=False)
    occurred_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    ingested_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


# ---------- system state ----------

class AssetConfig(Base):
    __tablename__ = "assets"
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(16), nullable=False)
    price_source: Mapped[str] = mapped_column(String(64), nullable=False)
    trades_24h: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class IngestLog(Base):
    __tablename__ = "ingest_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_symbol: Mapped[str | None] = mapped_column(String(16), nullable=True)
    data_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    records_fetched: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DataQuality(Base):
    __tablename__ = "data_quality"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    asset_symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    interval: Mapped[str] = mapped_column(String(4), nullable=False)
    trade_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    findings_json: Mapped[str] = mapped_column(Text, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    detail_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class ConfigVersion(Base):
    __tablename__ = "config_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")


# ---------- decision / risk / execution (populated by later milestones, schema fixed now) ----------

class Decision(Base):
    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    asset_symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    no_trade_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    proposal_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_verdict_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    asset_symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    instrument_id: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    order_type: Mapped[str] = mapped_column(String(8), nullable=False)
    requested_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    requested_size: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    decision_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class Fill(Base):
    __tablename__ = "fills"

    fill_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(32), ForeignKey("orders.order_id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class Position(Base):
    __tablename__ = "positions"

    position_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    instrument_id: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(8), nullable=False, default="open")
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    fees_paid: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    funding_paid: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    decision_id: Mapped[str | None] = mapped_column(String(32), nullable=True)


class PortfolioState(Base):
    """Singleton row (id=1). Survives restarts: the paper portfolio and all
    counters must never reset or duplicate on an unexpected restart."""
    __tablename__ = "portfolio_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    starting_capital: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    updated_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    peak_equity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    daily_realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    daily_date: Mapped[str] = mapped_column(String(10), nullable=False, default="")
    consecutive_losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trading_halted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class ModelRegistration(Base):
    __tablename__ = "model_registry"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    asset_symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    horizon: Mapped[str] = mapped_column(String(4), nullable=False)
    feature_config: Mapped[str] = mapped_column(Text, nullable=False)
    hyperparameters: Mapped[str] = mapped_column(Text, nullable=False)
    training_data_range: Mapped[str] = mapped_column(Text, nullable=False)
    validation_metrics: Mapped[str] = mapped_column(Text, nullable=False)
    registered_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="registered")


class ForecastRecord(Base):
    __tablename__ = "forecasts"
    __table_args__ = (PrimaryKeyConstraint("id", "ts", name="pk_forecasts"),)

    id: Mapped[int] = mapped_column(Integer, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(64), ForeignKey("model_registry.model_id"), nullable=False)
    asset_symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    horizon: Mapped[str] = mapped_column(String(4), nullable=False)
    target_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)


class AgentOutputRecord(Base):
    __tablename__ = "agent_outputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class CostLedgerEntry(Base):
    __tablename__ = "cost_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    category: Mapped[str] = mapped_column(String(16), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
