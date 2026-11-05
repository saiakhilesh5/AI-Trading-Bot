"""News intelligence agent output contract."""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from agents.schemas.macro_schema import ImpactDirection, ImpactHorizon


class EventCategory(str, Enum):
    CENTRAL_BANK = "central_bank"
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    GEOPOLITICAL = "geopolitical"
    REGULATION = "regulation"
    EXCHANGE = "exchange"
    COMMODITY_SUPPLY = "commodity_supply"
    MACRO_POLICY = "macro_policy"
    CRYPTO_MARKET = "crypto_market"
    OTHER = "other"


class ExtractedEvent(BaseModel):
    title: str
    category: EventCategory = EventCategory.OTHER
    affected_assets: List[str] = Field(default_factory=list)  # subset of XAUUSD,XAGUSD,BTCUSD,ETHUSD
    impact_direction: ImpactDirection = ImpactDirection.UNCERTAIN
    impact_horizon: ImpactHorizon = ImpactHorizon.UNCERTAIN
    already_priced_in: bool = False
    is_rumour: bool = False
    key_facts: List[str] = Field(default_factory=list)
    source_references: List[str] = Field(default_factory=list)  # external_ids of articles
    market_implication: str = ""


class NewsAnalysis(BaseModel):
    events: List[ExtractedEvent] = Field(default_factory=list)
    duplicates_removed: int = 0
    conflicts_detected: List[str] = Field(default_factory=list)
    stale_or_recycled: List[str] = Field(default_factory=list)
    overall_summary: str = ""
    warnings: List[str] = Field(default_factory=list)
