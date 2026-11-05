"""Model performance: benchmark reports, calibration, experiment outcomes."""
import json
import os
import streamlit as st
import pandas as pd

from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import Experiment, ModelRegistration

st.title("Model Performance")

report_path = os.path.join("reports", "benchmark_latest.json")
st.subheader("Latest benchmark (walk-forward)")
if os.path.exists(report_path):
    with open(report_path) as f:
        report = json.load(f)
    rows = []
    for (asset, interval, horizon), results in report.items():
        for name, m in results.items():
            rows.append({"asset": asset, "interval": interval, "horizon": horizon,
                         "model": name, **{k: v for k, v in m.items()
                                           if isinstance(v, (int, float))}})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.json(report)
else:
    st.info("No benchmark report. Run scripts/run_backtest.py")

session = get_session()
try:
    st.subheader("Registered models by status")
    regs = (session.query(ModelRegistration)
            .order_by(ModelRegistration.registered_ts.desc()).all())
    if regs:
        st.dataframe(pd.DataFrame([{
            "model_id": r.model_id, "asset": r.asset_symbol,
            "horizon": r.horizon, "status": r.status,
        } for r in regs]), use_container_width=True)
    else:
        st.info("No models registered.")

    st.subheader("Experiments")
    exps = session.query(Experiment).limit(30).all()
    if exps:
        st.dataframe(pd.DataFrame([{
            "experiment": e.experiment_id, "status": e.status, "outcome": e.outcome,
        } for e in exps]), use_container_width=True)
finally:
    session.close()

st.caption("Performance figures are walk-forward backtest results. "
           "They are not live trading results and must never be reported as such.")
