"""Daily summary report: portfolio state, positions, decisions, costs,
data health. Written to reports/daily/<date>.md. Metrics are paper-trading
results, never live performance."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from core.interfaces import Interval
from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import (CostLedgerEntry, Decision as DecisionORM,
                                         PortfolioState,
                                         Position as PositionORM)
from data_layer.storage import repository as repo

_ASSETS = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"]


def generate(as_of: datetime | None = None) -> str:
    as_of = as_of or datetime.now(timezone.utc)
    session = get_session()
    lines = [f"# Daily Report - {as_of.date().isoformat()}"]
    try:
        marks = {}
        for sym in _ASSETS:
            candles = repo.get_candles(session, sym, Interval.H1.value)
            if candles:
                marks[sym] = candles[-1].close

        state = session.get(PortfolioState, 1)
        if state:
            open_pos = (session.query(PositionORM)
                        .filter(PositionORM.status == "open").all())
            equity = state.cash + sum(p.size * marks.get(p.asset_symbol, p.entry_price)
                                     for p in open_pos)
            lines += [
                "## Portfolio",
                f"- Equity: {equity:,.2f} USD (cash {state.cash:,.2f})",
                f"- Open positions: {len(open_pos)}",
                f"- Daily realized P&L: {state.daily_realized_pnl:,.2f}",
                f"- Consecutive losses: {state.consecutive_losses}",
                f"- Drawdown from peak: "
                f"{(max(0.0, state.peak_equity - equity) / state.peak_equity * 100 if state.peak_equity else 0):.2f}%",
                f"- Trading halted: {state.trading_halted}",
            ]

        yesterday = as_of.replace(hour=0, minute=0, second=0, microsecond=0)
        decisions = (session.query(DecisionORM)
                    .filter(DecisionORM.ts >= yesterday).all())
        by_action: dict[str, int] = {}
        reasons: list[str] = []
        for d in decisions:
            by_action[d.decision] = by_action.get(d.decision, 0) + 1
            if d.no_trade_reason:
                reasons.append(d.no_trade_reason)
        lines += ["## Decisions (today)"]
        if by_action:
            lines += [f"- {k}: {v}" for k, v in sorted(by_action.items())]
            from collections import Counter
            top = Counter(r.split(":")[0] for r in reasons).most_common(5)
            lines += [f"- Top no-trade causes: {top}"]
        else:
            lines += ["- No decisions recorded today."]

        costs = (session.query(CostLedgerEntry)
                 .filter(CostLedgerEntry.ts >= yesterday).all())
        lines += ["## Costs (today)",
                  f"- LLM spend: {sum(c.cost_usd for c in costs if c.category == 'llm'):.4f} USD"]

        lines += ["## Marks"]
        for sym, px in marks.items():
            lines.append(f"- {sym}: {px:,.2f}")
    finally:
        session.close()

    lines += ["", "_Paper-trading results. Not live performance._"]
    os.makedirs("reports/daily", exist_ok=True)
    path = f"reports/daily/{as_of.date().isoformat()}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
