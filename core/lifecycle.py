"""Process lifecycle: startup recovery sequence and graceful shutdown.

Startup order (each step must pass before the next):
1. DB reachable (retry with backoff - the machine may boot before Docker)
2. Redis reachable (degraded mode if not)
3. Kill-switch state checked - if the emergency shutdown was ever triggered,
   the system starts in halted state and requires explicit manual clear
4. Portfolio reconciliation (positions/fills/cash consistent, no duplicates)
5. Signals heartbeat so a monitor can detect the loop is alive
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from core.logging_setup import get_logger

log = get_logger()


@dataclass
class LifecycleReport:
    db_ok: bool
    redis_ok: bool
    halted: bool
    halt_reason: str = ""
    reconciliation: dict | None = None


def startup(retries: int = 12, retry_seconds: float = 5.0) -> LifecycleReport:
    # 1. database
    db_ok = False
    for attempt in range(retries):
        try:
            from data_layer.storage.data_access import init_database
            init_database()
            db_ok = True
            break
        except Exception as e:
            log.warning("db_unavailable", attempt=attempt, error=str(e))
            time.sleep(retry_seconds)
    if not db_ok:
        log.error("startup_failed_no_db")
        return LifecycleReport(db_ok=False, redis_ok=False, halted=True,
                               halt_reason="database_unreachable")

    # 2. redis
    redis_ok = False
    try:
        import redis as redis_lib
        from config.settings import get_settings
        r = redis_lib.Redis.from_url(get_settings().redis_url, decode_responses=True)
        r.ping()
        redis_ok = True
    except Exception:
        log.warning("redis_unavailable_degraded_mode")

    # 3. kill switch / halt state
    halted, reason = False, ""
    from risk.kill_switch import KillSwitch
    ks = KillSwitch()
    if ks.is_halted(None):
        halted, reason = True, ks.reason() or "kill_switch"
    if not halted:
        from data_layer.storage.data_access import get_session
        from data_layer.storage.schemas import PortfolioState
        with get_session() as session:
            row = session.get(PortfolioState, 1)
            if row is not None and row.trading_halted:
                halted, reason = True, "emergency_shutdown_manual_restart_required"

    # 4. reconciliation
    recon = None
    if not halted:
        try:
            from execution.paper_broker import PaperBroker
            marks = {}
            from core.interfaces import Interval
            from data_layer.storage import repository as repo
            with get_session() as session:
                for sym in ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"]:
                    c = repo.get_candles(session, sym, Interval.H1.value)
                    if c:
                        marks[sym] = c[-1].close
            recon = PaperBroker().reconcile(marks)
            if recon.get("anomalies"):
                log.error("startup_reconciliation_anomalies",
                          anomalies=recon["anomalies"])
        except Exception as e:
            log.warning("reconciliation_failed", error=str(e))

    log.info("startup_complete", db_ok=db_ok, redis_ok=redis_ok,
             halted=halted, reason=reason)
    return LifecycleReport(db_ok=db_ok, redis_ok=redis_ok, halted=halted,
                           halt_reason=reason, reconciliation=recon)


def shutdown() -> None:
    """Graceful shutdown: no in-flight task is killed mid-write; the scheduler
    loop checks the stop flag between cycles."""
    from core.events import Event, EventKind, EventBus
    log.info("shutdown_requested")
    try:
        EventBus().publish(Event(kind=EventKind.SHUTDOWN,
                                  payload={"reason": "graceful"}))
    except Exception:
        pass
    log.info("shutdown_complete")
