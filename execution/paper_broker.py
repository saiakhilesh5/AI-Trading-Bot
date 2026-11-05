"""Simulated-only broker. There is no code path from this module to any real
exchange; real-money execution is not implemented (spec requirement)."""
from __future__ import annotations

from typing import Optional

from core.clock import get_clock
from core.exceptions import ExecutionError
from core.interfaces import TradeDecision, DecisionAction
from core.logging_setup import get_logger
from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import Fill, Position as PositionORM
from execution.fill_simulator import simulate_fill
from execution.order_manager import OrderManager
from execution.portfolio import Portfolio

log = get_logger()


class PaperBroker:
    def __init__(self, portfolio: Optional[Portfolio] = None,
                 orders: Optional[OrderManager] = None):
        self.portfolio = portfolio or Portfolio()
        self.orders = orders or OrderManager()

    # ---- entry -------------------------------------------------------------
    def execute_decision(self, decision: TradeDecision, ref_price: float,
                         asset_class: str, atr: float) -> Optional[PositionORM]:
        if decision.action not in (DecisionAction.BUY, DecisionAction.SELL):
            return None
        t = decision.proposed_trade
        if t is None:
            raise ExecutionError("decision_without_proposal")

        side = "buy" if t.direction == "long" else "sell"
        approved_notional = _approved_notional(t)
        if approved_notional <= 0:
            raise ExecutionError("no_approved_notional")
        size = approved_notional / ref_price

        order = self.orders.create_order(
            asset_symbol=t.asset_symbol, instrument_id=t.instrument_id,
            side=side, order_type="market", requested_size=size,
            decision_id=decision.decision_id)

        vol = _annualized_vol(atr, ref_price)
        fill = simulate_fill(asset_class, side, size, ref_price,
                             realized_vol_annualized=vol)
        if not self.portfolio.reserve(fill.size * fill.price):
            self.orders.mark_rejected(order.order_id, "insufficient_balance")
            raise ExecutionError("insufficient_balance")

        with get_session() as session:
            pos = PositionORM(
                position_id=f"p-{decision.decision_id}", instrument_id=t.instrument_id,
                asset_symbol=t.asset_symbol, side=side, size=fill.size,
                entry_price=fill.price, entry_ts=get_clock().now(),
                stop_loss=t.stop_loss, take_profit=t.take_profit,
                status="open", fees_paid=fill.fee, decision_id=decision.decision_id)
            session.add(pos)
            session.add(Fill(fill_id=f"f-{decision.decision_id}", order_id=order.order_id,
                             ts=fill.ts, price=fill.price, size=fill.size,
                             fee=fill.fee, slippage=fill.slippage))
            session.commit()
        self.orders.mark_filled(order.order_id)
        log.info("paper_position_opened", asset=t.asset_symbol, side=side,
                 size=fill.size, price=fill.price)
        return pos

    # ---- SL/TP processing on each new mark ---------------------------------
    def process_marks(self, marks: dict[str, float],
                      asset_classes: Optional[dict[str, str]] = None) -> list[str]:
        """marks: {symbol: mark_price}. Closes positions whose SL/TP is hit.
        Returns closed position_ids."""
        closed: list[str] = []
        with get_session() as session:
            positions = (session.query(PositionORM)
                         .filter(PositionORM.status == "open").all())
            for pos in positions:
                if pos.asset_symbol not in marks:
                    continue
                if _bracket_hit(pos, marks[pos.asset_symbol]):
                    self._close_position(
                        session, pos, marks[pos.asset_symbol],
                        (asset_classes or {}).get(pos.asset_symbol, "metals"))
                    closed.append(pos.position_id)
        return closed

    def _close_position(self, session, pos: PositionORM, mark: float,
                        asset_class: str) -> None:
        exit_side = "sell" if pos.side == "buy" else "buy"
        fill = simulate_fill(asset_class, exit_side, pos.size, mark)
        if pos.side == "buy":
            pnl = (fill.price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - fill.price) * pos.size
        total_fees = (pos.fees_paid or 0.0) + fill.fee
        pnl -= total_fees
        pos.exit_price = fill.price
        pos.exit_ts = get_clock().now()
        pos.status = "closed"
        pos.realized_pnl = pnl
        pos.fees_paid = total_fees
        session.commit()
        self.portfolio.release(pos.size * pos.entry_price)
        self.portfolio.record_realized(pnl, fee=fill.fee)
        log.info("paper_position_closed", asset=pos.asset_symbol,
                 exit=fill.price, pnl=round(pnl, 2))

    # ---- housekeeping --------------------------------------------------------
    def reconcile(self, marks: dict[str, float]) -> dict:
        from execution.reconciliation import sync_open_positions
        return sync_open_positions(self.portfolio, marks)


def _approved_notional(t) -> float:
    for s in t.supporting_signals:
        if s.startswith("risk_approved:notional="):
            return float(s.split("=")[1])
    return 0.0


def _annualized_vol(atr: float, price: float) -> float:
    if price <= 0:
        return 0.3
    return min(3.0, (atr / price) * (252 ** 0.5))


def _bracket_hit(pos: PositionORM, mark: float) -> bool:
    sl, tp = pos.stop_loss, pos.take_profit
    if pos.side == "buy":
        return (sl is not None and mark <= sl) or (tp is not None and mark >= tp)
    return (sl is not None and mark >= sl) or (tp is not None and mark <= tp)
