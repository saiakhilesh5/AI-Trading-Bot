"""Macroeconomic interpretation agent output contract."""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ImpactDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class ImpactHorizon(str, Enum):
    IMMEDIATE = "immediate"     # minutes-hours
    SHORT = "short"             # hours-2 days
    MEDIUM = "medium"           # days-2 weeks
    LONG = "long"               # weeks+
    UNCERTAIN = "uncertain"


class AssetMacroAssessment(BaseModel):
    asset_symbol: str
    bias: ImpactDirection = ImpactDirection.UNCERTAIN
    reasoning: str = ""
    key_drivers: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)


class MacroAnalysis(BaseModel):
    regime_summary: str = ""
    surprises_detected: List[str] = Field(default_factory=list)
    upcoming_risks: List[str] = Field(default_factory=list)
    asset_assessments: List[AssetMacroAssessment] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
