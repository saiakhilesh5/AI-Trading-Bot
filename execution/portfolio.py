"""Portfolio accounting: cash, positions, realized/unrealized P&L, drawdown,
daily P&L reset, loss streaks. All state persisted in the DB so an unexpected
restart loses nothing and duplicates nothing."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from config.settings import get_trading_config
from core.clock import get_clock
from core.interfaces import PortfolioSnapshot, PositionSnapshot
from core.logging_setup import get_logger
from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import PortfolioState, Position as PositionORM

log = get_logger()


class Portfolio:
    def __init__(self, starting_capital: Optional[float] = None):
        self._starting = (starting_capital
                          or float(get_trading_config()["paper_trading"]["starting_capital_usd"]))
        self._ensure_row()

    # ---- persistence -----------------------------------------------------
    def _ensure_row(self) -> None:
        with get_session() as session:
            row = session.get(PortfolioState, 1)
            if row is None:
                now = get_clock().now()
                session.add(PortfolioState(
                    id=1, starting_capital=self._starting, cash=self._starting,
                    updated_ts=now, peak_equity=self._starting,
                    daily_date=now.date().isoformat()))
                session.commit()

    def _row(self, session) -> PortfolioState:
        row = session.get(PortfolioState, 1)
        assert row is not None
        # daily reset at UTC date rollover
        today = get_clock().now().date().isoformat()
        if row.daily_date != today:
            row.daily_date = today
            row.daily_realized_pnl = 0.0
        return row

    # ---- queries ----------------------------------------------------------
    def open_positions(self) -> list[PositionORM]:
        with get_session() as session:
            return (session.query(PositionORM)
                    .filter(PositionORM.status == "open").all())

    def position_value(self, pos: PositionORM, mark_price: float) -> float:
        """Notional at current mark."""
        return pos.size * mark_price

    def unrealized_pnl(self, pos: PositionORM, mark_price: float) -> float:
        if pos.side == "buy":
            return (mark_price - pos.entry_price) * pos.size
        return (pos.entry_price - mark_price) * pos.size

    def equity(self, marks: dict[str, float]) -> float:
        """cash + sum(position_size * mark)."""
        with get_session() as session:
            row = self._row(session)
            equity = row.cash
            for pos in session.query(PositionORM).filter(PositionORM.status == "open").all():
                mark = marks.get(pos.asset_symbol, pos.entry_price)
                equity += pos.size * mark
            return float(equity)

    def snapshot(self, marks: dict[str, float],
                 orders_today: Optional[dict[str, int]] = None) -> PortfolioSnapshot:
        """Risk-engine view: equity, open positions, daily P&L, streaks."""
        with get_session() as session:
            row = self._row(session)
            positions = (session.query(PositionORM)
                         .filter(PositionORM.status == "open").all())
            equity = row.cash + sum(
                p.size * marks.get(p.asset_symbol, p.entry_price) for p in positions)
            open_snap = [PositionSnapshot(
                asset_symbol=p.asset_symbol, direction="long" if p.side == "buy" else "short",
                notional=p.size * marks.get(p.asset_symbol, p.entry_price),
                entry_price=p.entry_price,
                unrealized_pnl=self.unrealized_pnl(p, marks.get(p.asset_symbol, p.entry_price)))
                for p in positions]
            today = get_clock().now().date().isoformat()
            orders_today = dict(orders_today or {})
            return PortfolioSnapshot(
                equity=float(equity),
                available_balance=float(row.cash),
                open_positions=open_snap,
                daily_realized_pnl=float(row.daily_realized_pnl) if row.daily_date == today else 0.0,
                consecutive_losses=int(row.consecutive_losses),
                peak_equity=float(row.peak_equity),
                orders_today_per_asset=orders_today)

    # ---- mutations ---------------------------------------------------------
    def reserve(self, notional: float) -> bool:
        """Deduct cash for an entry fill. Returns False if insufficient."""
        with get_session() as session:
            row = self._row(session)
            if row.cash < notional:
                return False
            row.cash -= notional
            row.updated_ts = get_clock().now()
            session.commit()
            return True

    def release(self, notional: float) -> None:
        """Return cash when a position closes."""
        with get_session() as session:
            row = self._row(session)
            row.cash += notional
            row.updated_ts = get_clock().now()
            session.commit()

    def record_realized(self, pnl: float, fee: float = 0.0) -> None:
        """Update daily P&L, loss streak, peak equity, emergency halt flag."""
        with get_session() as session:
            row = self._row(session)
            row.daily_realized_pnl += pnl
            if pnl < 0:
                row.consecutive_losses += 1
            elif pnl > 0:
                row.consecutive_losses = 0
            row.cash -= fee  # fees leave the account
            row.updated_ts = get_clock().now()
            session.commit()
        # peak equity updated lazily from snapshots via update_peak

    def update_peak(self, equity: float) -> None:
        with get_session() as session:
            row = self._row(session)
            if equity > row.peak_equity:
                row.peak_equity = float(equity)
                row.updated_ts = get_clock().now()
                session.commit()

    def set_halted(self, halted: bool) -> None:
        with get_session() as session:
            row = self._row(session)
            row.trading_halted = halted
            row.updated_ts = get_clock().now()
            session.commit()

    # ---- risk numbers -------------------------------------------------------
    def daily_loss_pct(self, marks: dict[str, float]) -> float:
        snap = self.snapshot(marks)
        return snap.daily_loss_pct

    def drawdown_pct(self, marks: dict[str, float]) -> float:
        snap = self.snapshot(marks)
        return snap.drawdown_pct
