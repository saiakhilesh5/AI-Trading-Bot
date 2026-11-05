"""Trading system control centre (paper trading only). Run:
    streamlit run dashboard/app.py"""
import streamlit as st

st.set_page_config(page_title="AI Trading System", page_icon="📊", layout="wide")

from core.logging_setup import setup_logging
from config.settings import get_settings
from data_layer.storage.data_access import init_database

try:
    setup_logging(get_settings())
    init_database()
except Exception as e:
    st.error(f"Database unavailable: {e}")
    st.stop()

mode = get_settings().trading_mode.value.upper()
st.title("Autonomous AI Trading Intelligence System")
st.sidebar.success(f"Operating mode: {mode} - PAPER ONLY")
st.sidebar.caption("Real-money execution is disabled by design.")
st.sidebar.info("Navigate using the pages in the sidebar.")
st.markdown("""
**Assets under analysis:** XAU/USD (Gold), XAG/USD (Silver), BTC (Bitcoin), ETH (Ethereum)

**Architecture:** 8 layers - data foundation, features, forecasting, AI agents,
decision engine, deterministic risk engine, paper execution, monitoring.

Use the pages in the sidebar to inspect markets, intelligence, forecasts,
decisions, portfolio, risk and model performance.
""")

col1, col2, col3, col4 = st.columns(4)
for c, name in zip([col1, col2, col3, col4],
                   ["Gold", "Silver", "Bitcoin", "Ethereum"]):
    c.metric(name, "-", "-")
st.caption("Detailed metrics on the Market Overview page.")
