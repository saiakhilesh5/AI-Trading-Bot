"""News intelligence: latest agent analyses and raw articles."""
import json
import streamlit as st
import pandas as pd

from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import AgentOutputRecord, NewsArticle as NewsArticleORM

st.title("News Intelligence")

session = get_session()
try:
    st.subheader("Latest agent analysis")
    recs = (session.query(AgentOutputRecord)
            .filter(AgentOutputRecord.agent_id == "news_intelligence")
            .order_by(AgentOutputRecord.ts.desc()).limit(10).all())
    if not recs:
        st.info("No agent runs recorded yet. The news agent runs in the trading loop.")
    for r in recs:
        with st.expander(f"{r.ts} - {r.run_id}"):
            st.json(json.loads(r.payload_json))

    st.subheader("Recent articles (last 48h)")
    import datetime as dt
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48)
    arts = (session.query(NewsArticleORM)
            .filter(NewsArticleORM.published_ts >= cutoff)
            .order_by(NewsArticleORM.published_ts.desc()).limit(50).all())
    if arts:
        st.dataframe(pd.DataFrame([{
            "published": a.published_ts, "source": a.source,
            "reliability": a.source_reliability, "title": a.title,
        } for a in arts]), use_container_width=True)
    else:
        st.info("No recent articles ingested. Run the news collector workflow.")
finally:
    session.close()
