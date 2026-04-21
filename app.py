"""
Quant Finance Platform — Main Entry Point (Fixed for Streamlit Cloud)
"""

import streamlit as st
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Quant Finance Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/quant-platform",
        "Report a bug": "https://github.com/yourusername/quant-platform/issues",
        "About": "Advanced Quant Finance Platform — Portfolio Optimizer, Monte Carlo, Regime Detection",
    },
)

# --- Custom CSS ---
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0e1117; }
    [data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    .banner {
        background: linear-gradient(90deg, #0f3460, #533483);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .signal-bull { color: #00d4a4; font-weight: 600; }
    .signal-bear { color: #ff4b4b; font-weight: 600; }
    .signal-neutral { color: #ffa500; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar navigation ---
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

def load_page(module_path):
    """Dynamically load a page module and call show()."""
    try:
        mod = importlib.import_module(module_path)
        mod.show()
    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.exception(e)


if page == "🏠 Dashboard":
    load_page("pages.dashboard")
elif page == "⚖️ Portfolio Optimizer":
    load_page("pages.optimizer")
elif page == "🎲 Monte Carlo Simulator":
    load_page("pages.montecarlo")
elif page == "🔭 Regime Detection":
    load_page("pages.regime")
elif page == "🔬 Factor Screener":
    load_page("pages.factor_screener")
elif page == "⚠️ Risk Analytics":
    load_page("pages.risk_analytics")
