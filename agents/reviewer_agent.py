"""Decision Reviewer Agent.

Reviews a ProposedTrade against forecasts, news, macro, regime and risk info.
It may only reduce risk (multiplier in [0,1]); it never approves execution and
never touches the risk engine. Verdict VETO forces multiplier 0. On LLM failure
the reviewer FAILS CLOSED (veto) - a trade cannot pass review without it."""
from __future__ import annotations

import json
from typing import Optional

from core.exceptions import LLMError
from core.interfaces import ProposedTrade
from core.logging_setup import get_logger
from agents.llm_client import LLMClient, wrap_untrusted
from agents.prompts.reviewer_prompt import REVIEWER_SYSTEM_PROMPT, reviewer_user_prompt
from agents.schemas.macro_schema import MacroAnalysis
from agents.schemas.news_schema import NewsAnalysis
from agents.schemas.review_schema import TradeReview

log = get_logger()


class ReviewerAgent:
    agent_id = "decision_reviewer"

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client

    def review(self, proposal: ProposedTrade,
               forecast_summary: Optional[dict] = None,
               news: Optional[NewsAnalysis] = None,
               macro: Optional[MacroAnalysis] = None,
               regime: Optional[dict] = None,
               risk_summary: Optional[dict] = None) -> TradeReview:
        proposal_block = wrap_untrusted(json.dumps({
            "asset": proposal.asset_symbol, "direction": proposal.direction,
            "entry": proposal.proposed_entry, "stop_loss": proposal.stop_loss,
            "take_profit": proposal.take_profit,
            "expected_return": proposal.expected_return,
            "expected_uncertainty": proposal.expected_uncertainty,
            "expected_holding_hours": proposal.expected_holding_hours,
            "market_regime": proposal.market_regime,
            "supporting_signals": proposal.supporting_signals,
            "conflicting_signals": proposal.conflicting_signals,
            "invalidation_conditions": proposal.invalidation_conditions,
        }, default=str))
        f_block = json.dumps(forecast_summary or {}, default=str)
        n_block = json.dumps(news.model_dump(mode="json") if news else {}, default=str)[:6000]
        m_block = json.dumps(macro.model_dump(mode="json") if macro else {}, default=str)[:6000]
        r_block = json.dumps(regime or {}, default=str)
        k_block = json.dumps(risk_summary or {}, default=str)

        try:
            client = self.client or LLMClient()
            return client.structured(REVIEWER_SYSTEM_PROMPT,
                                      reviewer_user_prompt(proposal_block, f_block,
                                                           wrap_untrusted(n_block),
                                                           wrap_untrusted(m_block),
                                                           r_block, k_block),
                                      TradeReview)
        except LLMError as e:
            log.warning("reviewer_fails_closed", error=str(e))
            return TradeReview(
                verdict="veto", risk_multiplier=0.0,
                rationale="Reviewer LLM unavailable - failing closed.",
                warnings=[f"llm_unavailable:{e}"])
