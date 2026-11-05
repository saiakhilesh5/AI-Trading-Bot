"""System health check: infra reachability, data freshness, quality, portfolio,
kill switches. Exit code 0 = healthy, 1 = degraded."""
from __future__ import annotations

import sys
from datetime import timedelta

from core.clock import get_clock
from core.interfaces import Interval
from data_layer.storage.data_access import get_session
from data_layer.storage import repository as repo
from data_layer.storage.schemas import PortfolioState
from risk.kill_switch import KillSwitch

_ASSETS = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"]
failures: list[str] = []


def check_db() -> bool:
    try:
        from data_layer.storage.data_access import init_database
        init_database()
        print("[ok] database reachable")
        return True
    except Exception as e:
        print(f"[FAIL] database unreachable: {e}")
        return False


def check_redis() -> bool:
    try:
        import redis
        from config.settings import get_settings
        r = redis.Redis.from_url(get_settings().redis_url)
        r.ping()
        print("[ok] redis reachable")
        return True
    except Exception as e:
        print(f"[WARN] redis unreachable (degraded mode): {e}")
        return False


def check_data_freshness() -> None:
    now = get_clock().now()
    with get_session() as session:
        for sym in _ASSETS:
            candles = repo.get_candles(session, sym, Interval.H1.value)
            if not candles:
                failures.append(f"{sym}:no_data")
                print(f"[FAIL] {sym}: no candles stored")
                continue
            age = now - candles[-1].ts
            if age > timedelta(hours=3):
                failures.append(f"{sym}:stale")
                print(f"[WARN] {sym}: last candle {age} old")
            else:
                print(f"[ok] {sym}: fresh ({age})")


def check_portfolio() -> None:
    with get_session() as session:
        state = session.get(PortfolioState, 1)
        if state is None:
            print("[WARN] portfolio not initialized (created on first run)")
        else:
            print(f"[ok] portfolio: cash={state.cash:,.2f} "
                  f"peak={state.peak_equity:,.2f} halted={state.trading_halted}")
            if state.trading_halted:
                failures.append("portfolio_halted")


def check_kill_switch() -> None:
    ks = KillSwitch()
    if ks.is_halted(None):
        failures.append("kill_switch")
        print(f"[FAIL] kill switch active: {ks.reason()}")
    else:
        print("[ok] no kill switches active")


def main() -> int:
    if not check_db():
        return 1
    check_redis()
    check_data_freshness()
    check_portfolio()
    check_kill_switch()
    if failures:
        print(f"\nDEGRADED: {failures}")
        return 1
    print("\nHEALTHY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
