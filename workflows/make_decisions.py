"""One decision cycle: for each asset, gather point-in-time inputs, forecast
with the active (validated) model, run the agents, and produce a TradeDecision
through the decision engine. No fabrication: if data is missing or no validated
model exists, the outcome is NO TRADE with a reason."""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from core.clock import get_clock
from core.interfaces import DecisionAction, Interval, TradeDecision
from core.logging_setup import get_logger
from data_layer.storage.data_access import get_session
from data_layer.storage import repository as repo
from features import assemble
from features.regime_detection import RegimeClassifier
from decision.trade_decision import DecisionEngine, DecisionInputs
from risk.risk_engine import RiskEngine

log = get_logger()

_ASSETS = {
    "XAUUSD": ("metals", Interval.H1), "XAGUSD": ("metals", Interval.H1),
    "BTCUSD": ("crypto", Interval.H1), "ETHUSD": ("crypto", Interval.H1),
}


def _forecast_for(session, symbol: str, horizon: str = "1d"
                  ) -> tuple[float, float, float, str | None]:
    """Returns (expected_return, uncertainty, p_up, blocker). A blocker of
    'no_validated_model' means the asset has no registry-validated model, so
    no trade is permitted (no fabrication of forecasts)."""
    from forecasting import candidate_models, load_feature_frame
    from forecasting.base_model import forward_returns
    from forecasting.model_registry import active_models, record_forecast

    regs = active_models(session, symbol)
    if not regs:
        return 0.0, 0.0, 0.5, "no_validated_model"
    reg = regs[0]
    try:
        frame = load_feature_frame(session, symbol, horizon)
        if frame.empty or len(frame) < 300:
            return 0.0, 0.0, 0.5, "insufficient_history_for_forecast"
        y = forward_returns(frame["close"], horizon)
        model_cls = candidate_models().get(reg.model_type)
        if model_cls is None:
            return 0.0, 0.0, 0.5, f"unknown_model_type:{reg.model_type}"
        model = model_cls()
        X_train, y_train = frame.iloc[:-1], y.iloc[:-1]
        model.fit(X_train, y_train)
        fp = model.predict(symbol, horizon, frame.iloc[-1])
        if fp is None:
            return 0.0, 0.0, 0.5, "forecast_unavailable"
        record_forecast(session, reg.model_id, symbol, reg.horizon,
                        fp.model_dump(mode="json"), fp.target_ts)
        session.commit()
        unc = fp.uncertainty_score
        if unc is None or unc <= 0:
            half_width = None
            if fp.interval_lower is not None and fp.interval_upper is not None:
                half_width = (fp.interval_upper - fp.interval_lower) / 2
            unc = half_width or 0.01
        return (float(fp.expected_return or 0.0), float(unc),
                float(fp.p_up) if fp.p_up is not None else 0.5, None)
    except Exception as e:
        log.warning("forecast_failed", asset=symbol, error=str(e))
        return 0.0, 0.0, 0.5, f"forecast_error:{e}"


def run_decision_cycle(news_analysis=None, macro_analysis=None,
                       reviewer=None) -> list[TradeDecision]:
    """Produce one TradeDecision per asset. Agents' latest analyses are passed
    in from the orchestrator (or None -> degraded to no-news mode)."""
    from data_layer.collectors.orchestrator import IngestionOrchestrator
    from execution.order_manager import OrderManager
    from execution.portfolio import Portfolio

    decisions: list[TradeDecision] = []
    session = get_session()
    orch: Optional[IngestionOrchestrator] = None
    try:
        portfolio = Portfolio()
        om = OrderManager()
        engine = DecisionEngine(RiskEngine(), reviewer=reviewer)
        marks: dict[str, float] = {}
        health_map: dict[str, object] = {}

        for symbol, (asset_class, interval) in _ASSETS.items():
            try:
                if orch is None:
                    orch = IngestionOrchestrator()
                asset = _asset_obj(symbol)
                report = orch.ingest_candles(asset, interval, start=None, limit=200)
                health_map[symbol] = report
                candles = repo.get_candles(session, symbol, interval.value)
                if candles:
                    marks[symbol] = candles[-1].close
            except Exception as e:
                log.warning("ingest_failed_during_cycle", asset=symbol, error=str(e))

        snapshot = portfolio.snapshot(marks, orders_today=om.all_orders_today(
            [s for s in _ASSETS]))
        portfolio.update_peak(snapshot.equity)

        for symbol, (asset_class, interval) in _ASSETS.items():
            try:
                report = health_map.get(symbol)
                if report is None or not marks.get(symbol):
                    decisions.append(_no_trade(symbol, "ingest_failed"))
                    continue
                from forecasting import load_feature_frame
                feats = load_feature_frame(session, symbol, interval.value)
                if feats.empty or len(feats) < 100:
                    decisions.append(_no_trade(symbol, "insufficient_history"))
                    continue

                regime = str(RegimeClassifier().current_regime(feats))
                last = feats.iloc[-1]
                atr = float(last.get("atr_14") or float(last["close"]) * 0.01)

                exp_ret, unc, p_up, blocker = _forecast_for(session, symbol)
                if blocker:
                    decisions.append(_no_trade(symbol, blocker))
                    continue

                inputs = DecisionInputs(
                    asset_symbol=symbol, asset_class=asset_class,
                    health=report, regime=str(regime), atr=atr,
                    last_close=float(last["close"]),
                    forecast_return=exp_ret, forecast_uncertainty=unc, p_up=p_up,
                    momentum=float(last.get("momentum_10") or 0.0),
                    trend_strength=float(last.get("trend_strength") or 0.0),
                    horizon_hours=24.0,
                    news=news_analysis, macro=macro_analysis)
                decisions.append(engine.decide(inputs, snapshot))
            except Exception as e:
                log.error("decision_cycle_failed", asset=symbol, error=str(e))
                decisions.append(_no_trade(symbol, f"cycle_error:{e}"))
        return decisions
    finally:
        session.close()


def _asset_obj(symbol: str):
    from core.interfaces import Asset, AssetClass
    cls = AssetClass.CRYPTO if symbol in ("BTCUSD", "ETHUSD") else AssetClass.PRECIOUS_METAL
    return Asset(symbol=symbol, name=symbol, asset_class=cls,
                 price_source="orchestrator", trades_24h=cls == AssetClass.CRYPTO)


def _no_trade(symbol: str, reason: str) -> TradeDecision:
    import uuid
    engine = DecisionEngine(RiskEngine())
    return engine._record(f"d-{uuid.uuid4().hex[:12]}", get_clock().now(), symbol,
                          DecisionAction.NO_TRADE, reason=reason)
