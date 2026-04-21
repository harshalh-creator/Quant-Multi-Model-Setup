"""pages/regime.py — Market Regime Detection"""
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
 
from utils.data import fetch_index, INDICES
from utils.regime import (detect_regime, REGIME_COLORS, REGIME_ALLOC,
                           compute_transition_matrix, regime_statistics)
 
 
def hex_to_rgba(hex_color, alpha=0.12):
    """Convert hex color to rgba string for Plotly."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
 
 
def show():
    st.markdown("## 🔭 Market Regime Detection")
 
    with st.sidebar:
        st.markdown("### Regime Settings")
        index_name = st.selectbox("Index", list(INDICES.keys()))
        ticker = INDICES[index_name]
        method = st.selectbox("Detection model", {
            "composite": "Composite Multi-Signal",
            "hmm": "Hidden Markov Model (HMM)",
            "zscore": "Z-Score + Momentum",
            "trend": "Trend + Vol Filter",
        }.keys(), format_func=lambda k: {"composite":"Composite","hmm":"HMM","zscore":"Z-Score","trend":"Trend+Vol"}[k])
        n_states = st.slider("HMM states (HMM only)", 2, 4, 3) if method == "hmm" else 3
        period = st.session_state.get("data_period", "2y")
 
    with st.spinner(f"Fetching {index_name}..."):
        prices = fetch_index(ticker, period)
 
    if len(prices) < 200:
        st.warning("Need at least 200 trading days. Increase the period.")
        return
 
    with st.spinner("Detecting regimes..."):
        result = detect_regime(prices, method=method, n_states=n_states)
 
    reg = result.current_regime
    prob = result.current_prob
    color = REGIME_COLORS.get(reg, "#888888")
 
    # ── Current regime banner ────────────────────────────────────────────────
    alloc = REGIME_ALLOC.get(reg, {})
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-left: 4px solid {color};
                border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom:1rem;">
        <div style="font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:2px;">Current Regime</div>
        <div style="font-size:1.8rem; font-weight:700; color:{color};">{reg}</div>
        <div style="font-size:0.9rem; color:#aaa; margin-top:4px;">
            Confidence: {prob*100:.0f}% &nbsp;|&nbsp; Model: {method.upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Price + regime chart ─────────────────────────────────────────────────
    st.markdown("### Price History with Regime Overlay")
    regimes = result.regimes
    probs = result.probabilities
 
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.04)
 
    # Background bands — using rgba() strings (Plotly compatible)
    prev_r = regimes.iloc[0]
    start_dt = regimes.index[0]
    for dt, r in regimes.items():
        if r != prev_r:
            fc = hex_to_rgba(REGIME_COLORS.get(prev_r, "#888888"), alpha=0.15)
            fig.add_vrect(x0=start_dt, x1=dt,
                          fillcolor=fc,
                          opacity=1, line_width=0, row=1, col=1)
            prev_r, start_dt = r, dt
    # Last band
    fc = hex_to_rgba(REGIME_COLORS.get(prev_r, "#888888"), alpha=0.15)
    fig.add_vrect(x0=start_dt, x1=regimes.index[-1],
                  fillcolor=fc,
                  opacity=1, line_width=0, row=1, col=1)
 
    # Price
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode="lines",
                             line=dict(color="#3b82f6", width=1.5), name="Price"), row=1, col=1)
 
    # SMAs from signals
    sigs = result.signals
    if "sma50_series" in sigs:
        fig.add_trace(go.Scatter(x=sigs["sma50_series"].index, y=sigs["sma50_series"],
                                 line=dict(color="#f59e0b", width=1, dash="dot"), name="SMA 50"), row=1, col=1)
    if "sma200_series" in sigs:
        fig.add_trace(go.Scatter(x=sigs["sma200_series"].index, y=sigs["sma200_series"],
                                 line=dict(color="#ef4444", width=1, dash="dash"), name="SMA 200"), row=1, col=1)
 
    # RSI
    if "rsi_series" in sigs:
        fig.add_trace(go.Scatter(x=sigs["rsi_series"].index, y=sigs["rsi_series"],
                                 line=dict(color="#a78bfa", width=1.2), name="RSI(14)"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00d4a4", row=2, col=1)
 
    # Rolling vol
    if "roll_vol_series" in sigs:
        fig.add_trace(go.Scatter(x=sigs["roll_vol_series"].index, y=sigs["roll_vol_series"] * 100,
                                 line=dict(color="#f59e0b", width=1), name="Vol (21d Ann.)"), row=3, col=1)
 
    fig.update_layout(height=520, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.06, font=dict(size=10)))
    fig.update_yaxes(row=2, col=1, title_text="RSI", range=[0, 100])
    fig.update_yaxes(row=3, col=1, title_text="Vol %")
    st.plotly_chart(fig, use_container_width=True)
 
    # ── Columns ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
 
    with col1:
        st.markdown("### Signals Dashboard")
        s = result.signals
        signal_data = [
            ("Price vs SMA 200", "Bullish" if s.get("above_sma200") else "Bearish"),
            ("Price vs SMA 50",  "Bullish" if s.get("above_sma50")  else "Bearish"),
            ("Golden/Death Cross","Golden ✓" if s.get("golden_cross") else "Death ✗"),
            ("MACD Histogram",   "Positive" if s.get("macd_bullish") else "Negative"),
            (f"RSI(14) = {s.get('rsi', 0):.1f}",
             "Overbought" if s.get("rsi_overbought") else ("Oversold" if s.get("rsi_oversold") else "Neutral")),
            (f"Momentum 3M = {s.get('mom3m', 0)*100:.1f}%",
             "Positive" if (s.get("mom3m") or 0) > 0 else "Negative"),
            (f"Momentum 6M = {s.get('mom6m', 0)*100:.1f}%",
             "Positive" if (s.get("mom6m") or 0) > 0 else "Negative"),
            (f"Ann. Vol = {s.get('roll_vol', 0)*100:.1f}%",
             "Low" if s.get("roll_vol", 0.2) < 0.15 else ("High" if s.get("roll_vol", 0.2) > 0.25 else "Moderate")),
        ]
        for sig, status in signal_data:
            bull = status in ("Bullish", "Golden ✓", "Positive", "Low", "Neutral")
            icon = "🟢" if bull else ("🟡" if status in ("Neutral", "Moderate") else "🔴")
            st.markdown(f"{icon} **{sig}** — {status}")
 
    with col2:
        st.markdown("### Transition Matrix")
        trans = result.transition_matrix
        cols_t = trans.columns.tolist()
        z = trans.values
        fig_t = go.Figure(go.Heatmap(
            z=z * 100, x=cols_t, y=cols_t,
            colorscale="Blues",
            text=np.round(z * 100, 1),
            texttemplate="%{text:.0f}%",
            textfont=dict(size=11),
            colorbar=dict(title="%", thickness=10),
        ))
        fig_t.update_layout(height=280, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_t, use_container_width=True)
 
    with col3:
        st.markdown("### Regime-Based Allocation")
        alloc_df = pd.DataFrame([
            {"Asset Class": k, "Weight": f"{v}%"}
            for k, v in alloc.items()
        ])
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
 
        fig_a = go.Figure(go.Pie(
            labels=list(alloc.keys()), values=list(alloc.values()),
            hole=0.5, textinfo="percent", textfont_size=11,
            marker=dict(colors=["#3b82f6","#00d4a4","#f59e0b","#ef4444","#a78bfa","#ec4899"]),
        ))
        fig_a.update_layout(height=200, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig_a, use_container_width=True)
 
    # ── Regime statistics table ──────────────────────────────────────────────
    st.markdown("### Historical Regime Statistics")
    stats_df = result.regime_stats
    if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
 
    # ── Regime probability time series ──────────────────────────────────────
    if not probs.empty and probs.shape[1] > 1:
        st.markdown("### Regime Probability Over Time")
        fig_prob = go.Figure()
        for col in probs.columns:
            c = REGIME_COLORS.get(col, "#888888")
            fig_prob.add_trace(go.Scatter(
                x=probs.index, y=probs[col].values,
                mode="lines",
                stackgroup="one",
                name=col,
                line=dict(width=0.5, color=c),
                fillcolor=hex_to_rgba(c, alpha=0.6),
            ))
        fig_prob.update_layout(
            height=200, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_prob, use_container_width=True)
