"""Forecasting: registered models, benchmark results, calibration."""
import json
import streamlit as st
import pandas as pd

from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import Experiment, ForecastRecord, ModelRegistration

st.title("Forecasting")

session = get_session()
try:
    st.subheader("Model registry")
    regs = (session.query(ModelRegistration)
            .order_by(ModelRegistration.registered_ts.desc()).limit(50).all())
    if regs:
        st.dataframe(pd.DataFrame([{
            "model_id": r.model_id, "type": r.model_type, "version": r.version,
            "asset": r.asset_symbol, "horizon": r.horizon, "status": r.status,
            "metrics": json.loads(r.validation_metrics) if r.validation_metrics else {},
        } for r in regs]), use_container_width=True)
    else:
        st.info("No models registered. Run scripts/run_backtest.py to benchmark.")

    st.subheader("Recent forecasts")
    f = (session.query(ForecastRecord)
         .order_by(ForecastRecord.ts.desc()).limit(100).all())
    if f:
        st.dataframe(pd.DataFrame([{
            "ts": r.ts, "asset": r.asset_symbol, "horizon": r.horizon,
            "model": r.model_id, "target": r.target_ts,
        } for r in f]), use_container_width=True)
    else:
        st.info("No forecasts recorded yet.")

    st.subheader("Experiments")
    exps = session.query(Experiment).limit(50).all()
    for e in exps:
        with st.expander(f"{e.experiment_id} - {e.status} - outcome: {e.outcome}"):
            st.json(json.loads(e.result_json) if e.result_json else {})
finally:
    session.close()
