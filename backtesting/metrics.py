"""Performance metrics. All computed from a return series (fractions per bar)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if len(downside) < 1 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / downside.std())


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (peak - equity) / peak
    return float(dd.max() or 0.0)


def calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if mdd <= 0:
        return 0.0
    annual = (1 + total_return(returns)) ** (periods_per_year / max(len(returns), 1)) - 1
    return float(annual / mdd)


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def evaluate(returns: pd.Series, periods_per_year: int = 252,
             transaction_cost_total: float = 0.0) -> dict:
    return {
        "total_return": round(total_return(returns), 5),
        "sharpe": round(sharpe(returns, periods_per_year), 3),
        "sortino": round(sortino(returns, periods_per_year), 3),
        "max_drawdown": round(max_drawdown(returns), 4),
        "calmar": round(calmar(returns, periods_per_year), 3),
        "profit_factor": round(profit_factor(returns), 3),
        "win_rate": round(win_rate(returns), 4),
        "n_trades": int((returns != 0).sum()),
        "transaction_costs": round(transaction_cost_total, 5),
    }
