"""Backtesting engine tests: no look-ahead, costs applied, all 8 strategies run."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import backtest, backtest_all, walk_forward_backtest
from backtesting.metrics import evaluate
from backtesting.strategies import STRATEGIES


def make_close(n=300, seed=42, drift=0.0005) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.02, n)
    return pd.Series(100 * np.cumprod(1 + rets),
                     index=pd.date_range("2025-01-01", periods=n, freq="D"))


class TestMetrics:
    def test_evaluate_basic(self):
        rets = pd.Series([0.01, -0.005, 0.02, 0.001])
        m = evaluate(rets)
        assert m["win_rate"] == 0.75
        assert m["n_trades"] == 4
        assert m["total_return"] == pytest.approx((1.01 * 0.995 * 1.02 * 1.001) - 1,
                                                  abs=1e-5)

    def test_max_drawdown(self):
        rets = pd.Series([0.1, -0.1, 0.05])
        assert evaluate(rets)["max_drawdown"] > 0.05

    def test_flat_series(self):
        m = evaluate(pd.Series([0.0] * 10))
        assert m["sharpe"] == 0.0


class TestStrategies:
    def test_all_strategies_run(self):
        close = make_close()
        for name in STRATEGIES:
            from backtesting.strategies import run_strategy
            pos = run_strategy(name, close, forecast=pd.Series(0.001, index=close.index))
            assert len(pos) == len(close)
            assert set(pos.unique()) <= {-1.0, 0.0, 1.0}

    def test_buy_and_hold_constant(self):
        close = make_close(100)
        pos = from_backtest = None
        from backtesting.strategies import buy_and_hold
        assert (buy_and_hold(close) == 1.0).all()

    def test_insufficient_data(self):
        r = backtest("buy_and_hold", make_close(30))
        assert "error" in r


class TestBacktestEngine:
    def test_costs_reduce_returns(self):
        close = make_close()
        free = backtest("buy_and_hold", close, cost_bps_per_change=0.0)
        paid = backtest("buy_and_hold", close, cost_bps_per_change=50.0)
        assert paid["total_return"] < free["total_return"]

    def test_no_lookahead(self):
        """Corrupting FUTURE prices must not change positions decided earlier."""
        close = make_close(200)
        from backtesting.strategies import technical_only
        base = technical_only(close)
        tampered = close.copy()
        tampered.iloc[-1] *= 10.0  # absurd future spike
        tampered_pos = technical_only(tampered)
        # position at bar t must be identical up to the last bar
        assert (base.iloc[:-1] == tampered_pos.iloc[:-1]).all()

    def test_backtest_all_shapes(self):
        df = backtest_all(make_close(200))
        assert len(df) == 8
        assert set(df.columns) >= {"total_return", "sharpe", "max_drawdown"}

    def test_walk_forward_windows(self):
        df = walk_forward_backtest(make_close(600), train_bars=150, test_bars=80)
        assert len(df) == 8  # all strategies averaged across windows
        assert df.index.isin(STRATEGIES).all()
