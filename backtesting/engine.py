"""Backtest engine: strategy positions on close -> per-bar returns, net of a
per-change transaction cost. Chronological only; positions are decided at bar
t-1 close and earn bar t's return (no look-ahead)."""
from __future__ import annotations

import pandas as pd

from backtesting.metrics import evaluate
from backtesting.strategies import STRATEGIES, run_strategy


def backtest(name: str, close: pd.Series, forecast: pd.Series | None = None,
             news_bias: pd.Series | None = None,
             cost_bps_per_change: float = 15.0,
             periods_per_year: int = 252) -> dict:
    if len(close) < 60:
        return {"strategy": name, "error": "insufficient_data"}

    position = run_strategy(name, close, forecast, news_bias)
    # shift: decision made on bar t-1 applies to bar t's return
    pos_lag = position.shift(1).fillna(0.0)
    bar_ret = close.pct_change().fillna(0.0)
    gross = pos_lag * bar_ret
    changes = pos_lag.diff().abs().fillna(pos_lag.abs())
    costs = changes * cost_bps_per_change / 10000.0
    net = gross - costs
    result = evaluate(net, periods_per_year=periods_per_year,
                      transaction_cost_total=float(costs.sum()))
    result["strategy"] = name
    return result


def backtest_all(close: pd.Series, forecast: pd.Series | None = None,
                 news_bias: pd.Series | None = None,
                 cost_bps_per_change: float = 15.0,
                 periods_per_year: int = 252) -> pd.DataFrame:
    rows = [backtest(name, close, forecast, news_bias, cost_bps_per_change,
                     periods_per_year) for name in STRATEGIES]
    rows = [r for r in rows if "error" not in r]
    df = pd.DataFrame(rows).set_index("strategy")
    return df[["total_return", "sharpe", "sortino", "max_drawdown", "calmar",
               "profit_factor", "win_rate", "n_trades", "transaction_costs"]]


def walk_forward_backtest(close: pd.Series, train_bars: int = 300,
                          test_bars: int = 60, gap_bars: int = 3,
                          forecast_fn=None, cost_bps_per_change: float = 15.0) -> pd.DataFrame:
    """Rolling out-of-sample evaluation: on each window, a simple momentum
    forecast is fit in-sample and all strategies are scored out-of-sample."""
    from backtesting.strategies import STRATEGIES
    n = len(close)
    if n < train_bars + test_bars + gap_bars:
        return pd.DataFrame()
    out_rows: list[dict] = []
    start = 0
    while start + train_bars + gap_bars + test_bars <= n:
        train = close.iloc[start:start + train_bars]
        test = close.iloc[start + train_bars + gap_bars:start + train_bars + gap_bars + test_bars]
        # expanding momentum forecast: drift of last 30 train bars
        drift = float(train.pct_change().tail(30).mean())
        fc = pd.Series(drift, index=test.index)
        for name in STRATEGIES:
            r = backtest(name, test, forecast=fc, cost_bps_per_change=cost_bps_per_change)
            if "error" not in r:
                r["window_start"] = str(test.index[0])
                out_rows.append(r)
        start += test_bars
    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    return (df.groupby("strategy")[["total_return", "sharpe", "max_drawdown",
                                   "profit_factor", "win_rate", "n_trades"]]
            .mean().round(4))
