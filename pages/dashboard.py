"""pages/dashboard.py — Live Market Dashboard"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from utils.data import fetch_prices, fetch_index, INDICES, NSE_LARGE_CAP, compute_returns
from utils.regime import detect_regime, REGIME_COLORS


def show():
    st.markdown("## 🏠 Live Market Dashboard")
    st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y %H:%M')}")

    period = st.session_state.get("data_period", "2y")

    # ── Index tiles ──────────────────────────────────────────────────────────
    index_tickers = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "NIFTY Bank": "^NSEBANK", "Gold": "GOLDBEES.NS"}
    cols = st.columns(len(index_tickers))

    for col, (name, ticker) in zip(cols, index_tickers.items()):
        with st.spinner(f"Loading {name}..."):
            series = fetch_index(ticker, "5d")
        with col:
            if len(series) >= 2:
                cur, prev = series.iloc[-1], series.iloc[-2]
                chg = (cur - prev) / prev * 100
                st.metric(name, f"{cur:,.0f}", f"{chg:+.2f}%")
            else:
                st.metric(name, "—", "—")

    st.markdown("---")

    # ── Main chart ───────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### NIFTY 50 — Price & Regime")
        with st.spinner("Fetching NIFTY 50..."):
            nifty = fetch_index("^NSEI", period)

        if len(nifty) > 200:
            result = detect_regime(nifty, method="composite")
            regimes = result.regimes

            fig = go.Figure()

            # Regime background bands
            prev_r, start_i = regimes.iloc[0], nifty.index[0]
            for i, (dt, r) in enumerate(regimes.items()):
                if r != prev_r or i == len(regimes) - 1:
                    color_map = {"Bull": "rgba(0,212,164,0.08)", "Bear": "rgba(255,75,75,0.08)",
                                 "Sideways": "rgba(59,130,246,0.06)", "High Volatility": "rgba(245,158,11,0.08)"}
                    fig.add_vrect(x0=start_i, x1=dt, fillcolor=color_map.get(prev_r, "rgba(0,0,0,0.03)"),
                                  opacity=1, line_width=0)
                    prev_r, start_i = r, dt

            fig.add_trace(go.Scatter(x=nifty.index, y=nifty.values, mode="lines",
                                     line=dict(color="#3b82f6", width=1.5), name="NIFTY 50"))
            # SMAs
            sma50 = nifty.rolling(50).mean()
            sma200 = nifty.rolling(200).mean()
            fig.add_trace(go.Scatter(x=sma50.index, y=sma50.values, mode="lines",
                                     line=dict(color="#f59e0b", width=1, dash="dot"), name="SMA 50"))
            fig.add_trace(go.Scatter(x=sma200.index, y=sma200.values, mode="lines",
                                     line=dict(color="#ef4444", width=1, dash="dash"), name="SMA 200"))

            fig.update_layout(height=320, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", y=-0.15), xaxis=dict(showgrid=False),
                               yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig, use_container_width=True)

            reg = result.current_regime
            color = REGIME_COLORS.get(reg, "#888")
            st.markdown(
                f"**Current Regime:** <span style='color:{color}; font-weight:700'>{reg}</span> "
                f"(confidence: {result.current_prob*100:.0f}%)",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### Sector Heatmap")
        sector_tickers = {
            "IT": "^CNXIT", "Bank": "^NSEBANK", "Pharma": "NIFTYPHARMA.NS",
            "Auto": "NIFTYAUTO.NS", "FMCG": "NIFTYFMCG.NS",
        }
        rows = []
        for sec, t in sector_tickers.items():
            try:
                s = fetch_index(t, "5d")
                if len(s) >= 2:
                    chg = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100
                    rows.append({"Sector": sec, "Change": round(chg, 2)})
            except Exception:
                pass

        if rows:
            df_sec = pd.DataFrame(rows).sort_values("Change", ascending=False)
            fig_s = go.Figure(go.Bar(
                x=df_sec["Change"], y=df_sec["Sector"], orientation="h",
                marker_color=["#00d4a4" if v >= 0 else "#ff4b4b" for v in df_sec["Change"]],
                text=[f"{v:+.2f}%" for v in df_sec["Change"]], textposition="outside",
            ))
            fig_s.update_layout(height=280, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0),
                                 xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig_s, use_container_width=True)

    # ── Top movers ──────────────────────────────────────────────────────────
    st.markdown("### Top NSE Movers (Large Cap)")
    sample = NSE_LARGE_CAP[:15]
    with st.spinner("Fetching movers..."):
        prices = fetch_prices(sample, period="5d")

    if not prices.empty and len(prices) >= 2:
        chg = ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100).dropna().sort_values(ascending=False)
        col_g, col_l = st.columns(2)
        with col_g:
            st.markdown("**Gainers**")
            for t, v in chg.head(5).items():
                st.markdown(f"<span style='color:#00d4a4'>▲ {t.replace('.NS','')} &nbsp; +{v:.2f}%</span>", unsafe_allow_html=True)
        with col_l:
            st.markdown("**Losers**")
            for t, v in chg.tail(5).items():
                st.markdown(f"<span style='color:#ff4b4b'>▼ {t.replace('.NS','')} &nbsp; {v:.2f}%</span>", unsafe_allow_html=True)
