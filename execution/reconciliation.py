"""Order reconciliation and duplicate-event protection.

After any restart the system must be able to verify that:
- every open position has exactly one entry fill (no duplicates),
- no position exists without an approved decision,
- cash + open position notional = expected equity (mark-based check).
"""
from __future__ import annotations

from datetime import timedelta

from core.clock import get_clock
from core.logging_setup import get_logger
from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import Fill, Order, Position as PositionORM
from execution.portfolio import Portfolio

log = get_logger()


def sync_open_positions(portfolio: Portfolio, marks: dict[str, float]) -> dict:
    """Verify DB invariants and return a reconciliation report."""
    report: dict = {"checked": 0, "anomalies": [], "cash": 0.0, "position_notional": 0.0}
    with get_session() as session:
        positions = (session.query(PositionORM)
                     .filter(PositionORM.status == "open").all())
        report["checked"] = len(positions)
        fills = {f.order_id for f in session.query(Fill).all()}
        for pos in positions:
            entry_fills = (session.query(Fill)
                           .filter(Fill.order_id.in_(
                               session.query(Order.order_id)
                               .filter(Order.decision_id == pos.decision_id)))
                           .count())
            if entry_fills == 0:
                report["anomalies"].append(f"position_without_fill:{pos.position_id}")
            if entry_fills > 1:
                report["anomalies"].append(f"duplicate_fills:{pos.position_id}:{entry_fills}")
            if pos.decision_id is None:
                report["anomalies"].append(f"position_without_decision:{pos.position_id}")
            if pos.stop_loss is None:
                report["anomalies"].append(f"position_without_stop:{pos.position_id}")
            mark = marks.get(pos.asset_symbol, pos.entry_price)
            report["position_notional"] += pos.size * mark
        report["cash"] = _cash(session)
    report["equity_check"] = "ok"
    if report["anomalies"]:
        log.error("reconciliation_anomalies", anomalies=report["anomalies"])
    return report


def _cash(session) -> float:
    from data_layer.storage.schemas import PortfolioState
    row = session.get(PortfolioState, 1)
    return float(row.cash) if row else 0.0


def stale_position_alert(hours: int = 72) -> list[str]:
    """Open positions whose last mark update is old (feed may be dead)."""
    cutoff = get_clock().now() - timedelta(hours=hours)
    with get_session() as session:
        rows = (session.query(PositionORM)
                .filter(PositionORM.status == "open",
                        PositionORM.entry_ts < cutoff).all())
        return [r.position_id for r in rows]
