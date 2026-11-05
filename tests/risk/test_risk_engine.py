"""Risk engine and decision engine tests. Deterministic controls must reject
every violating trade; the reviewer can only shrink risk."""
from __future__ import annotations

import pytest

from core.interfaces import (AssetHealthReport, PositionSnapshot, PortfolioSnapshot,
                             ProposedTrade, QualityFlag, QualitySeverity)
from decision.cost_estimator import estimate_costs
from decision.gates import (check_data_health, check_forecast_uncertainty,
                            check_news_conflicts, check_regime_suitability)
from decision.signal_combiner import combine
from risk.drawdown import drawdown_state
from risk.exposure import check_exposure
from risk.kill_switch import KillSwitch
from risk.position_sizing import size_position
from risk.risk_engine import RiskEngine


def make_snapshot(**kw) -> PortfolioSnapshot:
    base = dict(equity=100000.0, available_balance=100000.0, peak_equity=100000.0)
    base.update(kw)
    return PortfolioSnapshot(**base)


def make_trade(direction="long", entry=2000.0, stop=1960.0, tp=2072.0,
               asset="XAUUSD") -> ProposedTrade:
    return ProposedTrade(
        asset_symbol=asset, instrument_id=f"{asset}_PAPER", direction=direction,
        proposed_entry=entry, stop_loss=stop, take_profit=tp,
        expected_holding_hours=48.0, expected_return=0.01, expected_uncertainty=0.008,
        supporting_signals=["s"], conflicting_signals=[],
        relevant_news=[], market_regime="trending_up",
        invalidation_conditions=["c"])


def make_health(eligible=True):
    from core.clock import get_clock
    from core.interfaces import Interval
    return AssetHealthReport(
        asset_symbol="XAUUSD", interval=Interval.D1, trade_eligible=eligible,
        findings=[], last_candle_ts=get_clock().now(), checked_at=get_clock().now())


class TestCostEstimator:
    def test_crypto_costs(self):
        est = estimate_costs("crypto", 48.0)
        assert est.round_trip_bps == 38  # (10+4+5)*2
        assert est.funding_bps == 6.0     # 3 bps/day * 2 days
        assert est.total_return_cost == pytest.approx(0.0044)

    def test_metals_cheaper(self):
        est = estimate_costs("metals", 48.0)
        assert est.round_trip_bps < 30


class TestGates:
    def test_bad_health_blocks(self):
        assert not check_data_health(make_health(False)).passed

    def test_good_health_passes(self):
        assert check_data_health(make_health(True)).passed

    def test_regime_suitability(self):
        assert check_regime_suitability("trending_up", "long").passed
        assert check_regime_suitability("trending_up", "short").warnings
        assert not check_regime_suitability("unknown_regime", "long").passed

    def test_implausible_forecast_blocked(self):
        ok = check_forecast_uncertainty(0.005, 0.02)
        assert ok.passed
        bad = check_forecast_uncertainty(0.05, 0.01)  # 5x its own uncertainty
        assert not bad.passed

    def test_news_conflict_blocks(self):
        assert not check_news_conflicts(["source A says X, source B says Y"]).passed
        assert check_news_conflicts([]).passed


class TestSignalCombiner:
    def test_strong_long(self):
        sig = combine(forecast_return=0.01, forecast_uncertainty=0.008, p_up=0.72,
                      momentum=0.5, trend_strength=0.8, regime="trending_up",
                      direction_suitability=1.0, net_expected_return=0.008)
        assert sig.direction == "long"
        assert sig.tradable
        assert sig.agreement_score >= 0.6

    def test_conflicting_momentum_reduces_agreement(self):
        strong = combine(0.01, 0.008, 0.72, 0.5, 0.8, "trending_up", 1.0, 0.008)
        weak = combine(0.01, 0.008, 0.72, -0.5, 0.8, "trending_up", 1.0, 0.008)
        assert weak.agreement_score < strong.agreement_score
        assert "momentum:short" in weak.conflicting[0]

    def test_flat_forecast_not_tradable(self):
        sig = combine(0.0, 0.008, 0.5, 0.5, 0.8, "trending_up", 1.0, 0.0)
        assert sig.direction is None and not sig.tradable

    def test_below_min_return_not_tradable(self):
        sig = combine(0.001, 0.008, 0.55, 0.5, 0.8, "trending_up", 1.0, 0.0001)
        assert not sig.tradable


class TestPositionSizing:
    def test_risk_defined_sizing(self):
        r = size_position(equity=100000, entry=2000, stop_loss=1960, atr=15.0)
        assert r.notional > 0
        assert r.risk_amount == pytest.approx(1000.0)  # 1% of equity
        # notional capped at 20% of equity
        assert r.notional == pytest.approx(20000.0)

    def test_stop_too_tight_rejected(self):
        r = size_position(100000, 2000, 1999, atr=15.0)  # 1 << 1.5*15
        assert r.notional == 0 and "stop_too_tight" in r.note

    def test_reviewer_can_only_shrink(self):
        full = size_position(100000, 2000, 1960, atr=15.0, reviewer_multiplier=1.0)
        shrunk = size_position(100000, 2000, 1960, atr=15.0, reviewer_multiplier=0.25)
        grown = size_position(100000, 2000, 1960, atr=15.0, reviewer_multiplier=3.0)
        assert shrunk.risk_amount < full.risk_amount
        assert grown.risk_amount == full.risk_amount  # clamped at 1.0

    def test_notional_capped_at_20pct_equity(self):
        r = size_position(100000, 2000, 1990, atr=5.0)
        assert r.notional <= 20000


class TestExposure:
    def test_group_cap(self):
        snap = make_snapshot(open_positions=[
            PositionSnapshot(asset_symbol="XAUUSD", direction="long",
                             notional=30000.0, entry_price=2000, unrealized_pnl=0)])
        v = check_exposure(snap, "XAGUSD", added_notional=20000)
        assert not v.ok and v.reason == "group_exposure_cap"

    def test_low_correlation_relaxes_group(self):
        snap = make_snapshot(open_positions=[
            PositionSnapshot(asset_symbol="XAUUSD", direction="long",
                             notional=30000.0, entry_price=2000, unrealized_pnl=0)])
        v = check_exposure(snap, "XAGUSD", added_notional=20000,
                           measured_correlation=0.2)
        assert v.ok

    def test_asset_cap(self):
        snap = make_snapshot(open_positions=[
            PositionSnapshot(asset_symbol="XAUUSD", direction="long",
                             notional=24000.0, entry_price=2000, unrealized_pnl=0)])
        v = check_exposure(snap, "XAUUSD", added_notional=5000)
        assert not v.ok and v.reason == "asset_exposure_cap"


class TestDrawdownLadder:
    def test_levels(self):
        assert drawdown_state(make_snapshot(equity=100000, peak_equity=100000)).level == "ok"
        assert drawdown_state(make_snapshot(equity=95000, peak_equity=100000)).level == "reduce_risk"
        assert drawdown_state(make_snapshot(equity=91000, peak_equity=100000)).level == "halt_new_trades"
        assert drawdown_state(make_snapshot(equity=87000, peak_equity=100000)).level == "emergency_shutdown"


class TestKillSwitch:
    def test_local_fallback(self):
        ks = KillSwitch(redis_client=False)
        assert not ks.is_halted("XAUUSD")
        ks.trigger("test", asset="XAUUSD")
        assert ks.is_halted("XAUUSD")
        assert not ks.is_halted("ETHUSD")
        ks.clear(asset="XAUUSD")
        assert not ks.is_halted("XAUUSD")

    def test_global_halt(self):
        ks = KillSwitch(redis_client=False)
        ks.trigger("global emergency")
        assert ks.is_halted("XAUUSD") and ks.is_halted("ETHUSD")
        assert ks.reason() == "global emergency"


class TestRiskEngine:
    def _engine(self):
        return RiskEngine(kill_switch=KillSwitch(redis_client=False))

    def test_approves_valid_trade(self):
        v = self._engine().approve(make_trade(), make_snapshot(), atr=15.0)
        assert v.approved and v.sized_notional > 0

    def test_rejects_bad_bracket(self):
        v = self._engine().approve(make_trade(stop=2010, tp=2072), make_snapshot(), atr=15.0)
        assert not v.approved and "bracket_invalid" in v.reason

    def test_rejects_low_rr(self):
        # valid bracket but TP/SL ratio 0.5 < 1.5
        v = self._engine().approve(make_trade(stop=1980, tp=2010), make_snapshot(), atr=15.0)
        assert not v.approved and "reward_risk" in v.reason

    def test_daily_loss_blocks(self):
        snap = make_snapshot(daily_realized_pnl=-4000)  # >3%
        v = self._engine().approve(make_trade(), snap, atr=15.0)
        assert not v.approved and "daily_loss" in v.reason

    def test_drawdown_halt_blocks(self):
        snap = make_snapshot(equity=91000, peak_equity=100000)
        v = self._engine().approve(make_trade(), snap, atr=15.0)
        assert not v.approved and "halt_new_trades" in v.reason

    def test_emergency_shutdown_blocks(self):
        snap = make_snapshot(equity=85000, peak_equity=100000)
        v = self._engine().approve(make_trade(), snap, atr=15.0)
        assert not v.approved and "emergency_shutdown" in v.reason

    def test_loss_streak_blocks(self):
        snap = make_snapshot(consecutive_losses=3)
        v = self._engine().approve(make_trade(), snap, atr=15.0)
        assert not v.approved and "loss_streak" in v.reason

    def test_max_positions_blocks(self):
        snap = make_snapshot(open_positions=[
            PositionSnapshot(asset_symbol=f"A{i}", direction="long", notional=1000.0, entry_price=100.0, unrealized_pnl=0.0) for i in range(4)])
        v = self._engine().approve(make_trade(), snap, atr=15.0)
        assert not v.approved and "max_open_positions" in v.reason

    def test_duplicate_direction_blocks(self):
        snap = make_snapshot(open_positions=[
            PositionSnapshot(asset_symbol="XAUUSD", direction="long", notional=1000.0, entry_price=100.0, unrealized_pnl=0.0)])
        v = self._engine().approve(make_trade(), snap, atr=15.0)
        assert not v.approved and "duplicate_position" in v.reason

    def test_order_frequency_blocks(self):
        snap = make_snapshot(orders_today_per_asset={"XAUUSD": 3})
        v = self._engine().approve(make_trade(), snap, atr=15.0)
        assert not v.approved and "order_frequency" in v.reason

    def test_kill_switch_blocks(self):
        e = self._engine()
        e.kill_switch.trigger("manual stop")
        v = e.approve(make_trade(), make_snapshot(), atr=15.0)
        assert not v.approved and "kill_switch" in v.reason

    def test_reviewer_multiplier_shrinks_size(self):
        e = self._engine()
        trade = make_trade(stop=1775.0, tp=2400.0)  # wide stop -> below notional caps
        full = e.approve(trade, make_snapshot(), atr=150.0)
        small = e.size_with_review(trade, make_snapshot(), atr=150.0,
                                   reviewer_multiplier=0.5)
        assert small.approved and small.sized_notional < full.sized_notional
