"""Agent monitoring: outputs, LLM cost ledger."""
import json
import datetime as dt
import streamlit as st
import pandas as pd

from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import AgentOutputRecord, CostLedgerEntry

st.title("Agent Monitoring")

session = get_session()
try:
    st.subheader("Recent agent outputs")
    recs = (session.query(AgentOutputRecord)
            .order_by(AgentOutputRecord.ts.desc()).limit(20).all())
    if recs:
        st.dataframe(pd.DataFrame([{
            "ts": r.ts, "agent": r.agent_id, "run": r.run_id, "status": r.status,
        } for r in recs]), use_container_width=True)
        for r in recs[:5]:
            with st.expander(f"{r.agent_id} @ {r.ts}"):
                st.json(json.loads(r.payload_json))
    else:
        st.info("No agent runs recorded yet.")

    st.subheader("LLM cost ledger (current month)")
    month_start = dt.datetime.now(dt.timezone.utc).replace(day=1, hour=0, minute=0)
    entries = (session.query(CostLedgerEntry)
               .filter(CostLedgerEntry.ts >= month_start).all())
    if entries:
        df = pd.DataFrame([{"ts": e.ts, "category": e.category, "provider": e.provider,
                            "cost_usd": e.cost_usd, "tokens": e.tokens_used}
                           for e in entries])
        st.dataframe(df, use_container_width=True)
        st.metric("Total spend this month (USD)",
                  f"{df[df.category=='llm'].cost_usd.sum():.4f}")
    else:
        st.info("No LLM costs recorded.")
finally:
    session.close()
