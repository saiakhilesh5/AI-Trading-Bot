"""Decision reviewer agent output contract.

risk_multiplier is constrained to [0, 1]: the reviewer may only reduce risk,
never increase it, and never execute anything itself."""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    REDUCE_RISK = "reduce_risk"
    WAIT_FOR_CONFIRMATION = "wait_for_confirmation"
    VETO = "veto"


class TradeReview(BaseModel):
    verdict: ReviewVerdict = ReviewVerdict.VETO
    risk_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""
    conflicting_evidence: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    unsupported_assumptions: List[str] = Field(default_factory=list)
    news_risks: List[str] = Field(default_factory=list)
    uncertainty_concern: str = ""
    warnings: List[str] = Field(default_factory=list)

    def effective_multiplier(self) -> float:
        """APPROVE keeps 1.0; explicit reduce/wait scale down; veto forces 0."""
        if self.verdict == ReviewVerdict.VETO:
            return 0.0
        if self.verdict == ReviewVerdict.APPROVE:
            return 1.0
        return min(0.5, self.risk_multiplier)
