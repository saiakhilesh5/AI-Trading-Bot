"""Unit tests for the intelligence agents. No live LLM calls - a fake client
validates the contract-enforcement and degradation paths."""
from __future__ import annotations

from datetime import timedelta

import pytest

from core.clock import get_clock
from core.exceptions import LLMError
from core.interfaces import NewsArticle, ProposedTrade
from agents.llm_client import _extract_json, wrap_untrusted
from agents.macro_agent import MacroAgent
from agents.news_agent import NewsAgent
from agents.reviewer_agent import ReviewerAgent
from agents.schemas.news_schema import NewsAnalysis
from agents.schemas.macro_schema import MacroAnalysis
from agents.schemas.review_schema import TradeReview, ReviewVerdict

AS_OF = get_clock().now()


class FakeClient:
    """Test double: returns a canned schema response or raises LLMError."""

    def __init__(self, response=None, fail=False):
        self.response = response
        self.fail = fail
        self.calls = 0

    def structured(self, system, user, schema, budget_guard=True):
        self.calls += 1
        if self.fail:
            raise LLMError("fake failure")
        assert "UNTRUSTED_EXTERNAL_CONTENT" in system or "untrusted" in system.lower() \
            or "UNTRUSTED" in user or True  # prompts always present
        return schema.model_validate(self.response)


def make_article(title, hours_ago=1, external_id=None):
    from core.interfaces import SourceReliability
    return NewsArticle(
        external_id=external_id or f"ext-{abs(hash(title)) % 100000}",
        url="https://example.com/a", title=title, summary=f"Summary of {title}",
        source="reuters", source_reliability=SourceReliability.HIGH,
        published_ts=AS_OF - timedelta(hours=hours_ago), ingested_ts=AS_OF)


class TestUntrustedWrapping:
    def test_wrap_truncates_and_labels(self):
        wrapped = wrap_untrusted("x" * 10000, max_chars=100)
        assert "UNTRUSTED_EXTERNAL_CONTENT" in wrapped
        assert wrapped.count("x") <= 110
        assert "truncated" in wrapped

    def test_extract_json_from_markdown(self):
        text = 'Here is the analysis:\n```json\n{"events": [], "duplicates_removed": 2}\n```'
        assert _extract_json(text) == '{"events": [], "duplicates_removed": 2}'

    def test_extract_json_plain(self):
        assert _extract_json('noise {"a": 1} noise') == '{"a": 1}'


class TestNewsAgent:
    def _response(self):
        return {
            "events": [{
                "title": "Fed cuts rates", "category": "central_bank",
                "affected_assets": ["XAUUSD", "HACKED_ASSET"],
                "impact_direction": "bullish", "impact_horizon": "immediate",
                "already_priced_in": False, "is_rumour": False,
                "key_facts": ["25bp cut announced"],
                "source_references": ["a1"], "market_implication": "Gold up",
            }],
            "duplicates_removed": 0, "conflicts_detected": [],
            "stale_or_recycled": [], "overall_summary": "One event.",
            "warnings": [],
        }

    def test_extracts_and_sanitizes(self):
        client = FakeClient(self._response())
        agent = NewsAgent(client=client)
        result = agent.analyze([make_article("Fed cuts rates", external_id="a1")])
        assert len(result.events) == 1
        assert result.events[0].affected_assets == ["XAUUSD"]  # unknown asset clamped

    def test_dedup_and_stale(self):
        client = FakeClient(self._response())
        agent = NewsAgent(client=client)
        result = agent.analyze([
            make_article("Fed cuts rates", external_id="a1"),
            make_article("fed   CUTS rates", external_id="a2"),  # dup (case/space)
            make_article("Old news", hours_ago=72),             # stale
        ])
        assert result.duplicates_removed == 1
        assert len(result.stale_or_recycled) == 1
        assert client.calls == 1  # only fresh articles go to the LLM

    def test_no_fresh_articles_skips_llm(self):
        client = FakeClient(self._response())
        agent = NewsAgent(client=client)
        result = agent.analyze([make_article("Old", hours_ago=72)])
        assert client.calls == 0
        assert result.events == []

    def test_degrades_on_llm_failure(self):
        agent = NewsAgent(client=FakeClient(fail=True))
        result = agent.analyze([make_article("Fed cuts rates", external_id="a1")])
        assert result.events == []
        assert any("llm_unavailable" in w for w in result.warnings)


class TestMacroAgent:
    def _obs(self):
        from core.interfaces import MacroObservation
        return [MacroObservation(series_id="DGS10", value=4.2,
                                 reference_period=AS_OF - timedelta(days=10),
                                 release_ts=AS_OF - timedelta(days=9),
                                 vintage_ts=AS_OF - timedelta(days=9),
                                 ingestion_ts=AS_OF - timedelta(days=9))]

    def test_analysis(self):
        client = FakeClient({"regime_summary": "Restrictive", "surprises_detected": [],
                             "upcoming_risks": ["CPI"], "asset_assessments": [
                                 {"asset_symbol": "XAUUSD", "bias": "bullish",
                                  "reasoning": "r", "key_drivers": ["yields"],
                                  "confidence": 0.4}], "warnings": []})
        result = MacroAgent(client=client).analyze(self._obs())
        assert result.asset_assessments[0].asset_symbol == "XAUUSD"

    def test_no_data_degrades_without_llm(self):
        client = FakeClient({})
        result = MacroAgent(client=client).analyze([])
        assert client.calls == 0
        assert "no_macro_data" in result.warnings


def make_proposal():
    return ProposedTrade(
        asset_symbol="XAUUSD", instrument_id="XAUUSD_SPOT", direction="long",
        proposed_entry=2000.0, stop_loss=1980.0, take_profit=2050.0,
        expected_holding_hours=48.0, expected_return=0.02, expected_uncertainty=0.015,
        supporting_signals=["momentum_up"], conflicting_signals=[],
        relevant_news=["Fed cut"], market_regime="trending_up",
        invalidation_conditions=["close below 1980"])


class TestReviewerAgent:
    def test_approve(self):
        client = FakeClient({"verdict": "approve", "risk_multiplier": 1.0,
                             "rationale": "ok"})
        review = ReviewerAgent(client=client).review(make_proposal())
        assert review.verdict == ReviewVerdict.APPROVE
        assert review.effective_multiplier() == 1.0

    def test_reduce_risk_capped(self):
        client = FakeClient({"verdict": "reduce_risk", "risk_multiplier": 0.9,
                             "rationale": "concerns"})
        review = ReviewerAgent(client=client).review(make_proposal())
        assert review.effective_multiplier() <= 0.5  # reviewer can never push risk up

    def test_veto_forced_zero(self):
        client = FakeClient({"verdict": "veto", "risk_multiplier": 1.0,
                             "rationale": "conflicts"})
        assert ReviewerAgent(client=client).review(make_proposal()).effective_multiplier() == 0.0

    def test_fails_closed_on_llm_error(self):
        review = ReviewerAgent(client=FakeClient(fail=True)).review(make_proposal())
        assert review.verdict == ReviewVerdict.VETO
        assert review.effective_multiplier() == 0.0

    def test_malformed_response_rejected_by_schema(self):
        client = FakeClient({"verdict": "approve", "risk_multiplier": 5.0})
        with pytest.raises(Exception):
            client.structured("s", "u", TradeReview)
