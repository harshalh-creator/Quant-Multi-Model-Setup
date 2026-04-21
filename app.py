"""
Quant Finance Platform — Fixed for Streamlit Cloud
"""

import streamlit as st
import sys
import os

# Fix path so Python can find pages/ and utils/ folders
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="Quant Finance Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 Quant Platform")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "⚖️ Portfolio Optimizer",
            "🎲 Monte Carlo Simulator",
            "🔭 Regime Detection",
            "🔬 Factor Screener",
            "⚠️ Risk Analytics",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Data Settings")
    data_period = st.selectbox("Historical period", ["1y", "2y", "3y", "5y"], index=1)
    st.session_state["data_period"] = data_period
    rf_rate = st.number_input("Risk-free rate (%)", value=6.5, min_value=0.0, max_value=15.0, step=0.1)
    st.session_state["rf_rate"] = rf_rate / 100
    st.markdown("---")
    st.caption("Data: Yahoo Finance (yfinance)")
    st.caption("© 2025 Quant Finance Platform")


def load_page(folder, module_name):
    """Load a page by directly running its file."""
    try:
        import importlib.util
        file_path = os.path.join(ROOT, folder, f"{module_name}.py")
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            st.write("Files in root:", os.listdir(ROOT))
            return
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.show()
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)


if page == "🏠 Dashboard":
    load_page("pages", "dashboard")
elif page == "⚖️ Portfolio Optimizer":
    load_page("pages", "optimizer")
elif page == "🎲 Monte Carlo Simulator":
    load_page("pages", "montecarlo")
elif page == "🔭 Regime Detection":
    load_page("pages", "regime")
elif page == "🔬 Factor Screener":
    load_page("pages", "factor_screener")
elif page == "⚠️ Risk Analytics":
    load_page("pages", "risk_analytics")

