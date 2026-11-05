"""Compare the 8 spec-required strategy configurations on stored history.
Output: reports/strategy_comparison_<asset>.json + console table."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from backtesting.engine import backtest_all, walk_forward_backtest
from backtesting.strategies import STRATEGIES
from config.settings import get_settings
from core.interfaces import Interval
from core.logging_setup import get_logger, setup_logging
from data_layer.storage.data_access import get_session
from data_layer.storage import repository as repo

log = get_logger()


def load_close(symbol: str, interval: Interval) -> pd.Series:
    session = get_session()
    try:
        candles = repo.get_candles(session, symbol, interval.value)
        if not candles:
            raise SystemExit(f"No data for {symbol} {interval.value}. "
                             "Run scripts/download_historical_data.py first.")
        df = pd.DataFrame({"ts": [c.ts for c in candles],
                           "close": [c.close for c in candles]}).set_index("ts")
        return df["close"].astype(float).tail(2000)
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTCUSD")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--cost-bps", type=float, default=15.0)
    args = parser.parse_args()

    setup_logging(get_settings())
    interval = Interval(args.interval)
    close = load_close(args.asset, interval)
    ppy = {"1h": 24 * 365, "4h": 6 * 365, "1d": 252}[args.interval]

    print(f"\n=== {args.asset} {args.interval} (n={len(close)}) ===")
    print("\n-- Full-sample (in-sample, illustrative only) --")
    full = backtest_all(close, cost_bps_per_change=args.cost_bps, periods_per_year=ppy)
    print(full.to_string())

    print("\n-- Walk-forward (out-of-sample mean per window) --")
    wf = walk_forward_backtest(close, train_bars=300, test_bars=60,
                               cost_bps_per_change=args.cost_bps)
    print(wf.to_string() if not wf.empty else "(insufficient history for walk-forward)")

    os.makedirs("reports", exist_ok=True)
    out = {
        "asset": args.asset, "interval": args.interval,
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_bars": len(close),
        "full_sample": json.loads(full.to_json(orient="index")),
        "walk_forward": json.loads(wf.to_json(orient="index")) if not wf.empty else {},
        "note": "Full-sample numbers are illustrative; walk-forward is the "
                "valid out-of-sample result. Neither is live performance.",
    }
    path = f"reports/strategy_comparison_{args.asset}_{args.interval}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
