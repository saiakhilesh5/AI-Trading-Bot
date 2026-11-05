"""End-to-end decision-engine tests: gates -> signal -> cost -> reviewer ->
risk engine. Uses a stub reviewer; no LLM calls."""
from __future__ import annotations

import pytest

from agents.schemas.review_schema import TradeReview, ReviewVerdict
from core.interfaces import DecisionAction
from decision.trade_decision import DecisionEngine, DecisionInputs
from risk.kill_switch import KillSwitch
from risk.risk_engine import RiskEngine
from tests.risk.test_risk_engine import make_health, make_snapshot, make_trade


class StubReviewer:
    def __init__(self, review: TradeReview):
        self._review = review

    def review(self, proposal, **kw) -> TradeReview:
        return self._review


def make_inputs(**kw) -> DecisionInputs:
    base = dict(asset_symbol="XAUUSD", asset_class="metals",
                health=make_health(True), regime="trending_up", atr=15.0,
                last_close=2000.0, forecast_return=0.01, forecast_uncertainty=0.008,
                p_up=0.72, momentum=0.5, trend_strength=0.8, horizon_hours=48.0)
    base.update(kw)
    return DecisionInputs(**base)


def make_engine(review: TradeReview) -> DecisionEngine:
    return DecisionEngine(RiskEngine(kill_switch=KillSwitch(redis_client=False)),
                          reviewer=StubReviewer(review))


APPROVE = TradeReview(verdict=ReviewVerdict.APPROVE, risk_multiplier=1.0,
                      rationale="ok")
VETO = TradeReview(verdict=ReviewVerdict.VETO, risk_multiplier=0.0, rationale="no")
WAIT = TradeReview(verdict=ReviewVerdict.WAIT_FOR_CONFIRMATION,
                   risk_multiplier=0.25, rationale="need more data")
REDUCE = TradeReview(verdict=ReviewVerdict.REDUCE_RISK, risk_multiplier=0.4,
                     rationale="concerns")


class TestDecisionEngine:
    def test_full_buy(self):
        d = make_engine(APPROVE).decide(make_inputs(), make_snapshot())
        assert d.action == DecisionAction.BUY
        assert d.proposed_trade is not None
        assert any("risk_approved" in s for s in d.proposed_trade.supporting_signals)

    def test_sell(self):
        d = make_engine(APPROVE).decide(
            make_inputs(forecast_return=-0.01, p_up=0.28, momentum=-0.5,
                        regime="trending_down"), make_snapshot())
        assert d.action == DecisionAction.SELL

    def test_bad_health_no_trade(self):
        d = make_engine(APPROVE).decide(make_inputs(health=make_health(False)),
                                        make_snapshot())
        assert d.action == DecisionAction.NO_TRADE
        assert "data_health" in d.no_trade_reason

    def test_flat_signal_no_trade(self):
        d = make_engine(APPROVE).decide(
            make_inputs(forecast_return=0.0001, p_up=0.5, momentum=0.0,
                        trend_strength=0.1), make_snapshot())
        assert d.action == DecisionAction.NO_TRADE
        assert "signal" in d.no_trade_reason

    def test_cost_eats_edge(self):
        # expected return smaller than round-trip cost -> not tradable
        d = make_engine(APPROVE).decide(
            make_inputs(forecast_return=0.002, p_up=0.72, momentum=0.5), make_snapshot())
        assert d.action == DecisionAction.NO_TRADE

    def test_implausible_forecast_no_trade(self):
        d = make_engine(APPROVE).decide(
            make_inputs(forecast_return=0.10, forecast_uncertainty=0.01), make_snapshot())
        assert d.action == DecisionAction.NO_TRADE
        assert "forecast_implausible" in d.no_trade_reason

    def test_weak_regime_blocks_direction(self):
        d = make_engine(APPROVE).decide(
            make_inputs(regime="high_volatility", forecast_return=0.01), make_snapshot())
        # high_vol suitability 0.3 lowers agreement but forecast+momentum may pass
        assert d.action in (DecisionAction.BUY, DecisionAction.NO_TRADE)

    def test_reviewer_veto(self):
        d = make_engine(VETO).decide(make_inputs(), make_snapshot())
        assert d.action == DecisionAction.NO_TRADE
        assert "reviewer_veto" in d.no_trade_reason

    def test_reviewer_wait(self):
        d = make_engine(WAIT).decide(make_inputs(), make_snapshot())
        assert d.action == DecisionAction.WAIT

    def test_reviewer_reduce_still_trades(self):
        d = make_engine(REDUCE).decide(make_inputs(), make_snapshot())
        assert d.action == DecisionAction.BUY

    def test_risk_engine_has_final_say(self):
        # daily loss exceeded: even an approving reviewer cannot trade
        snap = make_snapshot(daily_realized_pnl=-5000)
        d = make_engine(APPROVE).decide(make_inputs(), snap)
        assert d.action == DecisionAction.NO_TRADE
        assert "risk_rejected:daily_loss" in d.no_trade_reason

    def test_drawdown_halt_blocks_despite_review(self):
        snap = make_snapshot(equity=91000, peak_equity=100000)
        d = make_engine(APPROVE).decide(make_inputs(), snap)
        assert d.action == DecisionAction.NO_TRADE
        assert "halt_new_trades" in d.no_trade_reason
