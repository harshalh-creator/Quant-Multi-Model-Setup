"""pages/factor_screener.py — Multi-Factor Stock Screener"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.data import (fetch_prices, fetch_ticker_info, compute_returns,
                         annualised_return, annualised_vol, NSE_LARGE_CAP, NSE_MID_CAP)


UNIVERSE = NSE_LARGE_CAP + NSE_MID_CAP[:10]


def score_assets(tickers, returns_df, infos, weights):
    """Score each asset on 6 factors and compute composite score."""
    rows = []
    for t in tickers:
        if t not in returns_df.columns:
            continue
        ret_series = returns_df[t].dropna()
        if len(ret_series) < 60:
            continue
        info = infos.get(t, {})
        name = t.replace(".NS","").replace(".BO","")

        # ── Value factor ──────────────────────────────────────────────────
        pe = info.get("pe")
        pb = info.get("pb")
        v_score = 0.5
        if pe and pe > 0:
            v_score = max(0, min(1, 1 - (pe - 5) / 70))   # lower PE → higher score
        if pb and pb > 0:
            v_score = (v_score + max(0, min(1, 1 - (pb - 0.5) / 10))) / 2

        # ── Momentum factor ──────────────────────────────────────────────
        if len(ret_series) >= 252:
            mom12 = (1 + ret_series).prod() ** (252 / len(ret_series)) - 1
        elif len(ret_series) >= 126:
            mom12 = (1 + ret_series.iloc[-126:]).prod() ** (252 / 126) - 1
        else:
            mom12 = 0
        m_score = max(0, min(1, (mom12 + 0.3) / 0.6))

        # ── Quality factor ────────────────────────────────────────────────
        roe = info.get("roe", 0) or 0
        de  = info.get("debt_equity", 1) or 1
        q_score = max(0, min(1, roe / 0.25)) * 0.6 + max(0, min(1, 1 - de / 150)) * 0.4

        # ── Growth factor ─────────────────────────────────────────────────
        eg = info.get("eps_growth", 0) or 0
        rg = info.get("rev_growth", 0) or 0
        g_score = max(0, min(1, (eg + 0.1) / 0.5)) * 0.5 + max(0, min(1, (rg + 0.1) / 0.4)) * 0.5

        # ── Low-volatility factor ─────────────────────────────────────────
        vol = ret_series.std() * np.sqrt(252)
        lv_score = max(0, min(1, 1 - (vol - 0.1) / 0.5))

        # ── Size factor (small cap = higher score in small-cap tilt) ─────
        mcap = info.get("market_cap", 1e12) or 1e12
        sz_score = max(0, min(1, 1 - np.log10(mcap / 1e9) / 4))

        composite = (
            v_score * weights["value"] +
            m_score * weights["momentum"] +
            q_score * weights["quality"] +
            g_score * weights["growth"] +
            lv_score * weights["lowvol"] +
            sz_score * weights["size"]
        )

        rows.append({
            "Ticker": name, "_full": t,
            "Score": round(composite * 100, 1),
            "Value": round(v_score * 100, 0),
            "Momentum": round(m_score * 100, 0),
            "Quality": round(q_score * 100, 0),
            "Growth": round(g_score * 100, 0),
            "Low Vol": round(lv_score * 100, 0),
            "P/E": round(pe, 1) if pe else "—",
            "ROE %": f"{roe*100:.1f}" if roe else "—",
            "Ann. Return": f"{annualised_return(ret_series)*100:.1f}%",
            "Volatility": f"{vol*100:.1f}%",
        })

    return pd.DataFrame(rows).sort_values("Score", ascending=False)


def show():
    st.markdown("## 🔬 Multi-Factor Stock Screener")

    with st.sidebar:
        st.markdown("### Factor Weights")
        w_val  = st.slider("Value",       0, 50, 25, step=5) / 100
        w_mom  = st.slider("Momentum",    0, 50, 25, step=5) / 100
        w_qual = st.slider("Quality",     0, 50, 20, step=5) / 100
        w_grow = st.slider("Growth",      0, 50, 15, step=5) / 100
        w_lv   = st.slider("Low Volatility", 0, 50, 10, step=5) / 100
        w_size = st.slider("Size",        0, 50,  5, step=5) / 100
        total_w = w_val + w_mom + w_qual + w_grow + w_lv + w_size
        if total_w == 0:
            st.error("Weights sum to 0."); return
        weights = {k: v / total_w for k, v in {
            "value": w_val, "momentum": w_mom, "quality": w_qual,
            "growth": w_grow, "lowvol": w_lv, "size": w_size
        }.items()}
        st.markdown(f"*(normalised — sum = 100%)*")
        st.markdown("### Universe")
        universe_choice = st.selectbox("Universe", ["NIFTY Large Cap (Top 25)", "NIFTY Mid Cap (Top 10)", "Combined"])
        n_top = st.slider("Show top N stocks", 5, 30, 15)
        period = st.session_state.get("data_period", "2y")

    if universe_choice == "NIFTY Large Cap (Top 25)":
        tickers = NSE_LARGE_CAP[:25]
    elif universe_choice == "NIFTY Mid Cap (Top 10)":
        tickers = NSE_MID_CAP[:10]
    else:
        tickers = NSE_LARGE_CAP[:20] + NSE_MID_CAP[:10]

    with st.spinner("Fetching prices and fundamentals..."):
        prices = fetch_prices(tickers, period=period)
        infos = {}
        prog = st.progress(0, "Loading fundamentals...")
        for i, t in enumerate(tickers):
            infos[t] = fetch_ticker_info(t)
            prog.progress((i + 1) / len(tickers))
        prog.empty()

    if prices.empty:
        st.error("No data available."); return

    returns = compute_returns(prices, log=False)
    scored = score_assets(tickers, returns, infos, weights)
    top_n = scored.head(n_top)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe size", len(scored))
    c2.metric("Avg composite score", f"{scored['Score'].mean():.1f}")
    c3.metric("Top stock", top_n.iloc[0]["Ticker"] if len(top_n) else "—")
    c4.metric("Top score", f"{top_n.iloc[0]['Score']:.1f}" if len(top_n) else "—")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Top Factor Stocks")
        display_cols = ["Ticker", "Score", "Value", "Momentum", "Quality", "Growth", "P/E", "ROE %", "Ann. Return", "Volatility"]
        def color_score(val):
            if isinstance(val, (int, float)):
                if val >= 70: return "color: #00d4a4"
                if val >= 50: return "color: #f59e0b"
                return "color: #ef4444"
            return ""
        st.dataframe(
            top_n[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart of scores
        fig_bar = go.Figure(go.Bar(
            x=top_n["Ticker"], y=top_n["Score"],
            marker_color=top_n["Score"].apply(
                lambda v: "#00d4a4" if v >= 70 else ("#f59e0b" if v >= 50 else "#ef4444")
            ),
            text=top_n["Score"].round(1), textposition="outside",
        ))
        fig_bar.update_layout(height=250, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
                               xaxis=dict(tickangle=-30), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown("### Factor Weight Radar")
        fig_radar = go.Figure(go.Scatterpolar(
            r=[weights["value"]*100, weights["momentum"]*100, weights["quality"]*100,
               weights["growth"]*100, weights["lowvol"]*100, weights["size"]*100],
            theta=["Value", "Momentum", "Quality", "Growth", "Low Vol", "Size"],
            fill="toself", fillcolor="rgba(59,130,246,0.2)",
            line=dict(color="#3b82f6", width=2),
        ))
        fig_radar.update_layout(height=280, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 margin=dict(l=30,r=30,t=30,b=30),
                                 polar=dict(radialaxis=dict(visible=True, range=[0, 50])))
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### Score Distribution")
        fig_hist = go.Figure(go.Histogram(
            x=scored["Score"], nbinsx=20,
            marker_color="#3b82f6", opacity=0.8,
        ))
        fig_hist.update_layout(height=200, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
                                xaxis_title="Composite Score", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Factor scatter ────────────────────────────────────────────────────────
    st.markdown("### Momentum vs Value (bubble = Score)")
    fig_sc = go.Figure(go.Scatter(
        x=scored["Value"], y=scored["Momentum"],
        mode="markers+text",
        marker=dict(size=scored["Score"] / 5, color=scored["Score"],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Score", thickness=12)),
        text=scored["Ticker"], textposition="top center",
        textfont=dict(size=10),
    ))
    fig_sc.update_layout(height=320, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
                          xaxis_title="Value Score", yaxis_title="Momentum Score")
    st.plotly_chart(fig_sc, use_container_width=True)
