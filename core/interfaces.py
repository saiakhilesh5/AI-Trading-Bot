"""Shared typed contracts (Pydantic). Every layer boundary validates against these."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Interval(str, Enum):
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    PRECIOUS_METAL = "precious_metal"
    MACRO = "macro"


class Asset(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    name: str
    asset_class: AssetClass
    price_source: str
    trades_24h: bool = True
    notes: str = ""


def _utc(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (UTC)")
    return v


class Candle(BaseModel):
    model_config = ConfigDict(frozen=True)
    asset_symbol: str
    interval: Interval
    ts: datetime                      # candle open time (UTC)
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    source: str
    ingestion_ts: datetime

    @field_validator("high")
    @classmethod
    def _high(cls, v, info):
        o, c = info.data.get("open"), info.data.get("close")
        if o is not None and v < o:
            raise ValueError(f"high {v} < open {o}")
        if c is not None and v < c:
            raise ValueError(f"high {v} < close {c}")
        return v

    @field_validator("low")
    @classmethod
    def _low(cls, v, info):
        o, c = info.data.get("open"), info.data.get("close")
        if o is not None and v > o:
            raise ValueError(f"low {v} > open {o}")
        if c is not None and v > c:
            raise ValueError(f"low {v} > close {c}")
        return v

    @field_validator("ts", "ingestion_ts")
    @classmethod
    def _candle_ts_utc(cls, v: datetime) -> datetime:
        return _utc(v)


class CryptoDerivativeStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    asset_symbol: str
    ts: datetime
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    source: str
    ingestion_ts: datetime


class MacroObservation(BaseModel):
    """Point-in-time: revised values are new records, never overwrites."""
    model_config = ConfigDict(frozen=True)
    series_id: str
    value: float
    reference_period: datetime
    release_ts: datetime      # the ONLY time a backtest may see this value
    vintage_ts: datetime
    ingestion_ts: datetime
    source: str = "fred"


class SourceReliability(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    SINGLE_SOURCE = "single_source"
    CORROBORATED = "corroborated"
    CONFLICTING = "conflicting"


class NewsArticle(BaseModel):
    model_config = ConfigDict(frozen=True)
    external_id: str
    url: str
    title: str
    summary: str = ""
    source: str
    source_reliability: SourceReliability = SourceReliability.UNKNOWN
    published_ts: datetime
    ingested_ts: datetime
    affected_assets: List[str] = Field(default_factory=list)

    @field_validator("published_ts", "ingested_ts")
    @classmethod
    def _news_ts_utc(cls, v: datetime) -> datetime:
        return _utc(v)


class MarketEvent(BaseModel):
    """Structured event extracted by the News Intelligence Agent."""
    model_config = ConfigDict(frozen=True)
    event_id: str
    title: str
    description: str
    category: str
    affected_assets: List[str]
    source_url: str
    source_reliability: SourceReliability
    verification_status: VerificationStatus
    occurred_ts: Optional[datetime] = None
    published_ts: datetime
    ingested_ts: datetime
    facts: List[str]
    potential_implications: List[str]
    expected_impact_horizon: str
    already_priced_in: Optional[bool] = None


class QualityFlag(str, Enum):
    OK = "ok"
    MISSING_OBSERVATIONS = "missing_observations"
    DUPLICATE_RECORDS = "duplicate_records"
    STALE_DATA = "stale_data"
    INVALID_VALUES = "invalid_values"
    PRICE_JUMP = "unexpected_price_jump"
    INCOMPLETE_CANDLE = "incomplete_candle"
    TIMESTAMP_ERROR = "timestamp_error"
    PROVIDER_FAILURE = "provider_failure"
    HIGH_LATENCY = "abnormal_latency"
    CONFLICTING_FEEDS = "conflicting_feeds"


class QualitySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class QualityFinding(BaseModel):
    model_config = ConfigDict(frozen=True)
    flag: QualityFlag
    severity: QualitySeverity
    detail: str
    observed_ts: Optional[datetime] = None


class AssetHealthReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    asset_symbol: str
    interval: str
    checked_at: datetime
    last_candle_ts: Optional[datetime] = None
    findings: List[QualityFinding]
    trade_eligible: bool

    @property
    def has_critical(self) -> bool:
        return any(f.severity == QualitySeverity.CRITICAL for f in self.findings)


class ForecastPoint(BaseModel):
    """Unified forecast output across all model families."""
    model_config = ConfigDict(frozen=True)
    model_id: str
    asset_symbol: str
    horizon: str
    generated_ts: datetime
    target_ts: datetime
    expected_return: Optional[float] = None       # fractional return over horizon
    p_up: Optional[float] = None                  # directional probability
    expected_volatility: Optional[float] = None   # predicted std of returns
    interval_lower: Optional[float] = None       # e.g. 10th pct of return dist
    interval_upper: Optional[float] = None       # e.g. 90th pct
    interval_level: float = 0.8
    uncertainty_score: Optional[float] = None


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"


class DecisionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"
    WAIT = "WAIT_FOR_CONFIRMATION"


class ProposedTrade(BaseModel):
    model_config = ConfigDict(frozen=True)
    asset_symbol: str
    instrument_id: str
    direction: str                     # "long" | "short"
    proposed_entry: float
    stop_loss: float
    take_profit: float
    expected_holding_hours: float
    expected_return: float
    expected_uncertainty: float
    supporting_signals: List[str]
    conflicting_signals: List[str]
    relevant_news: List[str]
    market_regime: str
    invalidation_conditions: List[str]


class PositionSnapshot(BaseModel):
    """Read-only view of an open position as consumed by the risk engine."""
    model_config = ConfigDict(frozen=True)
    asset_symbol: str
    direction: str                       # "long" | "short"
    notional: float
    entry_price: float
    unrealized_pnl: float


class PortfolioSnapshot(BaseModel):
    """Deterministic inputs the risk engine needs. Maintained by the portfolio
    accountant (Phase 5); the risk engine never trusts agent output here."""
    model_config = ConfigDict(frozen=True)
    equity: float
    available_balance: float
    open_positions: List[PositionSnapshot] = Field(default_factory=list)
    daily_realized_pnl: float = 0.0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    orders_today_per_asset: Dict[str, int] = Field(default_factory=dict)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0 or self.equity >= self.peak_equity:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100.0

    @property
    def daily_loss_pct(self) -> float:
        if self.equity + max(0.0, -self.daily_realized_pnl) <= 0:
            return 0.0
        return max(0.0, -self.daily_realized_pnl) / self.equity * 100.0

    def group_notional(self, symbols: list[str]) -> float:
        return sum(p.notional for p in self.open_positions if p.asset_symbol in symbols)


class TradeDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    decision_id: str
    ts: datetime
    asset_symbol: str
    action: DecisionAction
    no_trade_reason: Optional[str] = None
    proposed_trade: Optional[ProposedTrade] = None


class AgentOutput(BaseModel):
    """Strict agent contract. Malformed LLM responses are rejected, never patched."""
    model_config = ConfigDict(frozen=True)
    agent_id: str
    ts: datetime
    input_references: List[str] = Field(default_factory=list)
    findings: List[str]
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    conflicting_evidence: List[str] = Field(default_factory=list)
    processing_status: str = "ok"
