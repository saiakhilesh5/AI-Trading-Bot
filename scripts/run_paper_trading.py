"""Continuous paper-trading supervisor. This is the 24/7 entry point:

    python -m scripts.run_paper_trading [--once] [--cycle-seconds 900]

Startup: lifecycle recovery (DB retry, Redis, kill-switch check, reconcile).
Tasks on the scheduler:
- ingest_candles:  fresh 1h candles + health validation (every 15 min)
- ingest_news:     RSS/GDELT news (every 30 min)
- ingest_macro:    FRED macro series (every 6 h)
- agent_cycle:     news + macro agents via LangGraph (every 30 min)
- trade_cycle:     forecast -> decisions -> paper execution -> SL/TP (15 min)
- daily_report:    portfolio report (daily)

Safe against restarts: decisions/orders are deduped by id, portfolio state is
in the DB, and the kill switch halts new trades until manually cleared."""
from __future__ import annotations

import argparse
from datetime import timedelta

from config.settings import get_settings
from core.clock import get_clock
from core.lifecycle import shutdown, startup
from core.logging_setup import get_logger, setup_logging
from core.scheduler import Scheduler

log = get_logger()
_ASSETS = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"]


def _ingest_candles() -> None:
    from core.interfaces import Asset, AssetClass, Interval
    from data_layer.collectors.orchestrator import IngestionOrchestrator
    orch = IngestionOrchestrator()
    for sym in _ASSETS:
        cls = AssetClass.CRYPTO if sym in ("BTCUSD", "ETHUSD") else AssetClass.PRECIOUS_METAL
        asset = Asset(symbol=sym, name=sym, asset_class=cls,
                      price_source="orchestrator", trades_24h=cls == AssetClass.CRYPTO)
        for iv in [Interval.H1]:
            try:
                report = orch.ingest_candles(asset, iv, start=None, limit=200)
                if not report.trade_eligible:
                    log.warning("asset_not_trade_eligible", asset=sym,
                                flags=[f.flag.value for f in report.findings])
            except Exception as e:
                log.warning("candle_ingest_failed", asset=sym, error=str(e))


def _ingest_news() -> None:
    from data_layer.collectors.news_collector import NewsCollector
    from data_layer.storage.data_access import get_session
    from data_layer.storage import repository as repo
    articles = NewsCollector().fetch_articles()
    with get_session() as session:
        n = repo.insert_news_articles(session, articles)
        if n:
            log.info("news_ingested", count=n)


def _ingest_macro() -> None:
    from config.settings import get_settings
    s = get_settings()
    if not s.fred_api_key:
        log.warning("macro_ingest_skipped_no_api_key")
        return
    from data_layer.collectors.macro_collector import MacroCollector
    from data_layer.storage.data_access import get_session
    from data_layer.storage import repository as repo
    collector = MacroCollector(api_key=s.fred_api_key)
    obs = []
    for series in ["DGS10", "DGS2", "T10YIE", "CPIAUCSL", "UNRATE", "PAYEMS"]:
        try:
            obs.extend(collector.fetch_series(series))
        except Exception as e:
            log.warning("macro_series_failed", series=series, error=str(e))
    with get_session() as session:
        n = repo.upsert_macro(session, obs)
        if n:
            log.info("macro_ingested", count=n)


def _agent_cycle(context: dict) -> None:
    from agents.agent_orchestrator import AgentOrchestrator, load_recent_articles
    from data_layer.storage.data_access import get_session
    from data_layer.storage import repository as repo
    session = get_session()
    try:
        arts = load_recent_articles(hours=36)
        now = get_clock().now()
        obs = []
        for series in ["DGS10", "DGS2", "T10YIE", "CPIAUCSL", "UNRATE", "PAYEMS", "DXY"]:
            obs.extend(repo.get_macro_as_of(session, series, as_of=now))
        news, macro = AgentOrchestrator().run(arts, obs)
        context["news"] = news
        context["macro"] = macro
    finally:
        session.close()


def _trade_cycle(context: dict) -> None:
    from workflows.make_decisions import run_decision_cycle
    from workflows.execute_paper_trades import execute_decisions, process_brackets
    from core.interfaces import Interval
    from data_layer.storage.data_access import get_session
    from data_layer.storage import repository as repo

    decisions = run_decision_cycle(news_analysis=context.get("news"),
                                   macro_analysis=context.get("macro"))
    for d in decisions:
        log.info("decision", asset=d.asset_symbol, action=d.action.value,
                 reason=d.no_trade_reason or "")
    session = get_session()
    try:
        marks = {}
        for sym in _ASSETS:
            candles = repo.get_candles(session, sym, Interval.H1.value)
            if candles:
                marks[sym] = candles[-1].close
    finally:
        session.close()
    opened = execute_decisions(decisions, marks)
    if opened:
        log.info("positions_opened", ids=opened)
    closed = process_brackets(marks)
    if closed:
        log.info("positions_closed", ids=closed)


def _daily_report() -> None:
    from workflows.daily_report import generate
    path = generate()
    log.info("daily_report_written", path=path)


def _model_refresh() -> None:
    from workflows.run_forecasting import refresh_models
    summary = refresh_models()
    log.info("model_refresh", promoted=len(summary["promoted"]),
             demoted=len(summary["demoted"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-trading supervisor")
    parser.add_argument("--once", action="store_true",
                        help="run one cycle of every task then exit")
    parser.add_argument("--cycle-seconds", type=int, default=900)
    args = parser.parse_args()

    setup_logging(get_settings())
    report = startup()
    if not report.db_ok:
        return 1
    if report.halted:
        log.error("startup_halted", reason=report.halt_reason)
        return 2  # requires manual kill-switch clear

    context: dict = {}
    sched = Scheduler()
    sched.every("ingest_candles", timedelta(seconds=args.cycle_seconds),
                _ingest_candles)
    sched.every("ingest_news", timedelta(minutes=30), _ingest_news)
    sched.every("ingest_macro", timedelta(hours=6), _ingest_macro)
    sched.every("agent_cycle", timedelta(minutes=30), lambda: _agent_cycle(context))
    sched.every("trade_cycle", timedelta(seconds=args.cycle_seconds),
               lambda: _trade_cycle(context))
    sched.every("daily_report", timedelta(days=1), _daily_report)
    sched.every("model_refresh", timedelta(days=1), _model_refresh)

    log.info("paper_trading_loop_started", mode=get_settings().trading_mode.value)
    if args.once:
        sched.run_pending()
    else:
        sched.run_forever(poll_seconds=10)
    shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
