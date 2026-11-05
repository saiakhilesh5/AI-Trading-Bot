"""Decision engine: gates -> signal combination -> cost adjustment -> proposal
-> reviewer agent -> risk engine (final authority). Every decision, including
NO TRADE, is recorded with its reason."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from config.settings import get_trading_config
from core.clock import get_clock
from core.interfaces import (AssetHealthReport, PortfolioSnapshot, ProposedTrade,
                             TradeDecision, DecisionAction)
from core.logging_setup import get_logger
from agents.reviewer_agent import ReviewerAgent
from agents.schemas.macro_schema import MacroAnalysis
from agents.schemas.news_schema import NewsAnalysis
from agents.schemas.review_schema import ReviewVerdict
from decision.cost_estimator import estimate_costs
from decision.gates import (GateResult, check_data_health, check_forecast_uncertainty,
                            check_news_conflicts, check_regime_suitability,
                            check_upcoming_releases)
from decision.signal_combiner import combine
from risk.risk_engine import RiskEngine, RiskVerdict

log = get_logger()


@dataclass
class DecisionInputs:
    """Everything the engine needs for one asset at one decision point."""
    asset_symbol: str
    asset_class: str
    health: AssetHealthReport
    regime: str
    atr: float
    last_close: float
    forecast_return: float
    forecast_uncertainty: float
    p_up: float
    momentum: float
    trend_strength: float
    horizon_hours: float
    news: Optional[NewsAnalysis] = None
    macro: Optional[MacroAnalysis] = None


class DecisionEngine:
    def __init__(self, risk_engine: RiskEngine, reviewer: Optional[ReviewerAgent] = None):
        self.risk = risk_engine
        self.reviewer = reviewer or ReviewerAgent()

    def decide(self, inputs: DecisionInputs, snapshot: PortfolioSnapshot,
               now=None) -> TradeDecision:
        now = now or get_clock().now()
        decision_id = f"d-{uuid.uuid4().hex[:12]}"

        no_trade = self._gates_or_no_trade(inputs, decision_id, now)
        if no_trade is not None:
            return no_trade

        # costs, net expected return
        cost = estimate_costs(inputs.asset_class, inputs.horizon_hours)
        net_expected = inputs.forecast_return - cost.total_return_cost

        suit = get_trading_config()["decision"]["regime_suitability"].get(
            inputs.regime, {"long": 0.0, "short": 0.0})

        fc_dir = "long" if net_expected > 0 else "short"
        signal = combine(inputs.forecast_return, inputs.forecast_uncertainty,
                         inputs.p_up, inputs.momentum, inputs.trend_strength,
                         inputs.regime, suit.get(fc_dir, 0.0), net_expected)

        if not signal.tradable:
            reason = (f"signal:{signal.direction}:agreement={signal.agreement_score} "
                      f"net_ret={net_expected:+.5f}")
            return self._record(decision_id, now, inputs.asset_symbol,
                                DecisionAction.NO_TRADE, reason=reason)

        proposal = self._build_proposal(inputs, signal, now)
        return self._review_and_risk(inputs, snapshot, proposal, decision_id, now)

    # ---- gates ----------------------------------------------------------
    def _gates_or_no_trade(self, inputs: DecisionInputs, decision_id, now):
        checks: list[GateResult] = [
            check_data_health(inputs.health),
            check_upcoming_releases(now, inputs.macro.upcoming_risks if inputs.macro else None),
            check_news_conflicts(inputs.news.conflicts_detected if inputs.news else None),
        ]
        # uncertainty plausibility (needs a direction first - check both signs)
        u = check_forecast_uncertainty(inputs.forecast_return, inputs.forecast_uncertainty)
        if not u.passed:
            return self._record(decision_id, now, inputs.asset_symbol,
                                DecisionAction.NO_TRADE, reason=u.reason)
        for g in checks:
            if not g.passed:
                return self._record(decision_id, now, inputs.asset_symbol,
                                    DecisionAction.NO_TRADE, reason=g.reason)
        return None

    # ---- proposal ---------------------------------------------------------
    def _build_proposal(self, inputs: DecisionInputs, signal, now) -> ProposedTrade:
        tcfg = get_trading_config()
        stop_mult = tcfg["decision"]["stop_atr_multiple"]
        rr = tcfg["decision"]["take_profit_rr"]
        entry = inputs.last_close
        if signal.direction == "long":
            stop = entry - stop_mult * inputs.atr
            tp = entry + rr * stop_mult * inputs.atr
        else:
            stop = entry + stop_mult * inputs.atr
            tp = entry - rr * stop_mult * inputs.atr
        relevant_news = [e.title for e in (inputs.news.events if inputs.news else [])
                         if inputs.asset_symbol in e.affected_assets][:5]
        macro_bias = ""
        if inputs.macro:
            for a in inputs.macro.asset_assessments:
                if a.asset_symbol == inputs.asset_symbol:
                    macro_bias = f"macro_bias:{a.bias}"
                    break
        return ProposedTrade(
            asset_symbol=inputs.asset_symbol,
            instrument_id=f"{inputs.asset_symbol}_PAPER",
            direction=signal.direction,
            proposed_entry=round(entry, 4), stop_loss=round(stop, 4),
            take_profit=round(tp, 4),
            expected_holding_hours=inputs.horizon_hours,
            expected_return=signal.expected_return,
            expected_uncertainty=signal.uncertainty,
            supporting_signals=signal.supporting + [macro_bias],
            conflicting_signals=signal.conflicting,
            relevant_news=relevant_news,
            market_regime=inputs.regime,
            invalidation_conditions=[
                f"close beyond stop {stop:.4f}",
                "data quality degraded",
                "news conflict emerges",
            ])

    # ---- reviewer + risk ---------------------------------------------------
    def _review_and_risk(self, inputs: DecisionInputs, snapshot: PortfolioSnapshot,
                         proposal: ProposedTrade, decision_id, now) -> TradeDecision:
        forecast_summary = {"expected_return": proposal.expected_return,
                            "uncertainty": proposal.expected_uncertainty,
                            "p_up": inputs.p_up}
        risk_summary = {"equity": snapshot.equity,
                        "open_positions": len(snapshot.open_positions),
                        "drawdown_pct": snapshot.drawdown_pct}
        review = self.reviewer.review(proposal, forecast_summary=forecast_summary,
                                      news=inputs.news, macro=inputs.macro,
                                      regime={"regime": inputs.regime},
                                      risk_summary=risk_summary)

        if review.verdict == ReviewVerdict.WAIT_FOR_CONFIRMATION:
            return self._record(decision_id, now, inputs.asset_symbol,
                                DecisionAction.WAIT,
                                proposed_trade=proposal,
                                reason=f"reviewer_wait:{review.rationale[:200]}")
        if review.effective_multiplier() <= 0.0:
            return self._record(decision_id, now, inputs.asset_symbol,
                                DecisionAction.NO_TRADE,
                                proposed_trade=proposal,
                                reason=f"reviewer_veto:{review.rationale[:200]}")

        verdict: RiskVerdict = self.risk.size_with_review(
            proposal, snapshot, inputs.atr, review.effective_multiplier())
        if not verdict.approved:
            return self._record(decision_id, now, inputs.asset_symbol,
                                DecisionAction.NO_TRADE,
                                proposed_trade=proposal,
                                reason=f"risk_rejected:{verdict.reason}")

        # approved: attach final notional in supporting signals for the executor
        final = proposal.model_copy(update={
            "supporting_signals": proposal.supporting_signals +
                                  [f"risk_approved:notional={verdict.sized_notional:.2f}",
                                   f"risk_amount={verdict.risk_amount:.2f}"]})
        return self._record(decision_id, now, inputs.asset_symbol,
                            DecisionAction.BUY if final.direction == "long" else DecisionAction.SELL,
                            proposed_trade=final)

    # ---- persistence ------------------------------------------------------
    def _record(self, decision_id, now, asset, action: DecisionAction,
                proposed_trade=None, reason=None) -> TradeDecision:
        decision = TradeDecision(decision_id=decision_id, ts=now,
                                  asset_symbol=asset, action=action,
                                  no_trade_reason=reason,
                                  proposed_trade=proposed_trade)
        try:
            from data_layer.storage.data_access import get_session
            from data_layer.storage.schemas import Decision as DecisionORM
            with get_session() as session:
                session.add(DecisionORM(
                    decision_id=decision_id, ts=now, asset_symbol=asset,
                    decision=action.value, no_trade_reason=reason,
                    proposal_json=proposed_trade.model_dump_json() if proposed_trade else None))
                session.commit()
        except Exception:
            log.warning("decision_persist_failed", decision_id=decision_id)
        return decision
