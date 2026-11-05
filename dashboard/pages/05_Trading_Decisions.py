"""Trading decisions log, including every NO TRADE and its reason."""
import json
import streamlit as st
import pandas as pd

from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import Decision as DecisionORM

st.title("Trading Decisions")

session = get_session()
try:
    rows = (session.query(DecisionORM)
            .order_by(DecisionORM.ts.desc()).limit(200).all())
    if not rows:
        st.info("No decisions recorded yet. Run workflows/make_decisions.py.")
    else:
        st.dataframe(pd.DataFrame([{
            "ts": r.ts, "asset": r.asset_symbol, "action": r.decision,
            "reason": r.no_trade_reason or "",
        } for r in rows]), use_container_width=True)

        trades = [r for r in rows if r.proposal_json]
        st.subheader(f"Proposed trades ({len(trades)})")
        for r in trades[:10]:
            t = r.proposal_json if isinstance(r.proposal_json, dict) else json.loads(r.proposal_json)
            with st.expander(f"{r.ts} {r.asset_symbol} {r.decision}"):
                st.json(t)
finally:
    session.close()
