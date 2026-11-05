"""Paper portfolio: equity, cash, open positions, closed trade history."""
import streamlit as st
import pandas as pd

from core.interfaces import Interval
from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import (Fill, Order, PortfolioState,
                                        Position as PositionORM)
from data_layer.storage import repository as repo

st.title("Paper Portfolio")

session = get_session()
try:
    state = session.get(PortfolioState, 1)
    if state is None:
        st.info("Portfolio not initialized. It is created on first broker run.")
        st.stop()

    marks = {}
    for sym in ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"]:
        candles = repo.get_candles(session, sym, Interval.H1, limit=1)
        if candles:
            marks[sym] = candles[-1].close

    open_pos = (session.query(PositionORM)
                .filter(PositionORM.status == "open").all())
    pos_val = sum(p.size * marks.get(p.asset_symbol, p.entry_price) for p in open_pos)
    equity = state.cash + pos_val
    peak = max(state.peak_equity, equity)
    dd = 0.0 if equity >= peak else (peak - equity) / peak * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity (USD)", f"{equity:,.2f}")
    c2.metric("Cash", f"{state.cash:,.2f}")
    c3.metric("Open positions", len(open_pos))
    c4.metric("Drawdown", f"{dd:.2f}%")

    st.subheader("Open positions")
    if open_pos:
        st.dataframe(pd.DataFrame([{
            "asset": p.asset_symbol, "side": p.side, "size": p.size,
            "entry": p.entry_price, "entry_ts": p.entry_ts,
            "stop": p.stop_loss, "target": p.take_profit,
            "unrealized_pnl": (marks.get(p.asset_symbol, p.entry_price) - p.entry_price)
                               * p.size * (1 if p.side == "buy" else -1),
        } for p in open_pos]), use_container_width=True)
    else:
        st.info("No open positions.")

    st.subheader("Closed trades")
    closed = (session.query(PositionORM)
              .filter(PositionORM.status == "closed")
              .order_by(PositionORM.exit_ts.desc()).limit(100).all())
    if closed:
        df = pd.DataFrame([{
            "asset": p.asset_symbol, "side": p.side, "size": p.size,
            "entry": p.entry_price, "exit": p.exit_price, "exit_ts": p.exit_ts,
            "realized_pnl": p.realized_pnl, "fees": p.fees_paid,
        } for p in closed])
        st.dataframe(df, use_container_width=True)
        wins = (df.realized_pnl > 0).sum()
        st.metric("Win rate", f"{wins / len(df) * 100:.1f}%"
                  if len(df) else "-")
        st.metric("Total realized P&L", f"{df.realized_pnl.sum():,.2f}")
    else:
        st.info("No closed trades yet.")
finally:
    session.close()
