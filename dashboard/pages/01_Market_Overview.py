"""Market overview: latest prices, data health, regime per asset."""
import streamlit as st
import pandas as pd

from core.interfaces import Interval
from data_layer.storage.data_access import get_session
from data_layer.storage import repository as repo

st.title("Market Overview")

session = get_session()
try:
    assets = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"]
    rows = []
    for sym in assets:
        candles = repo.get_candles(session, sym, Interval.H1, limit=24)
        if not candles:
            rows.append({"asset": sym, "last": None, "change_24h": None,
                         "status": "no data"})
            continue
        last = candles[-1]
        first = candles[0]
        chg = (last.close - first.close) / first.close * 100 if first.close else 0.0
        rows.append({"asset": sym, "last": round(last.close, 2),
                     "change_24h_pct": round(chg, 2),
                     "last_candle": last.ts, "source": last.source})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Price history (daily, 180 days)")
    asset = st.selectbox("Asset", assets)
    candles = repo.get_candles(session, asset, Interval.D1, limit=180)
    if candles:
        df = pd.DataFrame([{"date": c.ts.date(), "close": c.close} for c in candles])
        st.line_chart(df.set_index("date"))
    else:
        st.warning("No candles stored for this asset. Run scripts/download_historical_data.py")
finally:
    session.close()
