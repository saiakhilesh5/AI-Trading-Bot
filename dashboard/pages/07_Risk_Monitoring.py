"""Risk monitoring: kill switches, drawdown ladder position, data health."""
import streamlit as st
import pandas as pd

from core.interfaces import Interval
from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import DataQuality, PortfolioState
from data_layer.storage import repository as repo
from risk.kill_switch import KillSwitch
from execution.portfolio import Portfolio

st.title("Risk Monitoring")

st.subheader("Kill switches")
ks = KillSwitch()
status = ks.status()
if status:
    for k, v in status.items():
        st.error(f"HALTED: {k}") if v else None
else:
    st.success("No kill switches triggered.")

st.subheader("Portfolio risk state")
session = get_session()
try:
    state = session.get(PortfolioState, 1)
    if state:
        from config.settings import get_risk_config
        ladder = get_risk_config()["drawdown_ladder"]
        marks = {}
        for sym in ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"]:
            candles = repo.get_candles(session, sym, Interval.H1, limit=1)
            if candles:
                marks[sym] = candles[-1].close
        snap = Portfolio().snapshot(marks)
        c1, c2, c3 = st.columns(3)
        c1.metric("Drawdown", f"{snap.drawdown_pct:.2f}%")
        c2.metric("Daily P&L", f"{snap.daily_realized_pnl:,.2f}")
        c3.metric("Consecutive losses", snap.consecutive_losses)
        for rung in ladder:
            hit = snap.drawdown_pct >= rung["threshold_pct"]
            label = (f"{rung['action']} @ {rung['threshold_pct']}% drawdown "
                     f"(x{rung['risk_multiplier']} risk)")
            (st.error if hit else st.caption)(f"{'ACTIVE ' if hit else ''}{label}")
    else:
        st.info("Portfolio not initialized yet.")

    st.subheader("Data quality (latest checks)")
    checks = (session.query(DataQuality)
              .order_by(DataQuality.checked_at.desc()).limit(12).all())
    if checks:
        st.dataframe(pd.DataFrame([{
            "ts": c.checked_at, "asset": c.asset_symbol, "interval": c.interval,
            "eligible": c.trade_eligible, "findings": c.findings_json,
        } for c in checks]), use_container_width=True)
    else:
        st.info("No health checks recorded.")
finally:
    session.close()
