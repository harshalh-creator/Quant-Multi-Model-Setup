"""pages/risk_analytics.py — Advanced Risk Analytics"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data import (fetch_prices, fetch_index, compute_returns, NSE_LARGE_CAP,
                         annualised_return, annualised_vol, max_drawdown, sortino, calmar)
from utils.risk import (compute_var_all_methods, compute_all_ratios,
                         stress_test, rolling_risk_metrics, var_historical, cvar_historical)
from utils.optimizer import sample_cov


POPULAR = NSE_LARGE_CAP[:10]


def show():
    st.markdown("## ⚠️ Risk Analytics")

    with st.sidebar:
        st.markdown("### Portfolio")
        sel = st.multiselect("Assets", NSE_LARGE_CAP, default=POPULAR[:5],
                             format_func=lambda x: x.replace(".NS",""))
        pv = st.number_input("Portfolio value (₹)", min_value=10_000, value=1_000_000,
                              step=50_000, format="%d")
        conf = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1,
                             format_func=lambda v: f"{v*100:.0f}%")
        holding = st.slider("Holding period (days)", 1, 30, 1)
        period = st.session_state.get("data_period", "2y")
        rf = st.session_state.get("rf_rate", 0.065)

    if len(sel) < 1:
        st.info("Select at least 1 asset."); return

    with st.spinner("Loading data..."):
        prices = fetch_prices(sel, period=period)
        bench = fetch_index("^NSEI", period)

    if prices.empty:
        st.error("No price data."); return

    returns = compute_returns(prices, log=False)
    # Equal-weight portfolio returns
    w = np.ones(len(prices.columns)) / len(prices.columns)
    port_ret = (returns * w).sum(axis=1)
    port_prices_cum = (port_ret + 1).cumprod()

    bench_ret = bench.pct_change().dropna()
    bench_aligned = bench_ret.reindex(port_ret.index).dropna()
    port_aligned = port_ret.reindex(bench_aligned.index)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    ratios = compute_all_ratios(port_ret, port_prices_cum, bench_aligned, rf)
    var95 = var_historical(port_ret, conf, holding)
    cvar95 = cvar_historical(port_ret, conf, holding)

    row1 = st.columns(5)
    metrics = [
        ("Ann. Return", f"{ratios['Ann. Return']*100:.1f}%"),
        ("Ann. Volatility", f"{ratios['Ann. Volatility']*100:.1f}%"),
        ("Sharpe Ratio", f"{ratios['Sharpe Ratio']:.2f}"),
        ("Max Drawdown", f"{ratios['Max Drawdown']*100:.1f}%"),
        (f"VaR ({conf*100:.0f}%, {holding}d)", f"₹{var95 * pv:,.0f}"),
    ]
    for col, (label, val) in zip(row1, metrics):
        col.metric(label, val)

    row2 = st.columns(5)
    metrics2 = [
        ("CVaR (ES)", f"₹{cvar95 * pv:,.0f}"),
        ("Sortino Ratio", f"{ratios['Sortino Ratio']:.2f}"),
        ("Calmar Ratio", f"{ratios['Calmar Ratio']:.2f}"),
        ("Beta (NIFTY)", f"{ratios['Beta']:.2f}"),
        ("Omega Ratio", f"{min(ratios['Omega Ratio'], 99):.2f}"),
    ]
    for col, (label, val) in zip(row2, metrics2):
        col.metric(label, val)

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── Return distribution with VaR/CVaR ────────────────────────────────────
    with col1:
        st.markdown("### Return Distribution")
        daily_ret = port_ret.dropna()
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=daily_ret * 100, nbinsx=60,
            marker_color=["#ef4444" if v < -var95 * 100 else "#3b82f6" for v in daily_ret],
            name="Returns", opacity=0.75,
        ))

        # KDE overlay
        from scipy.stats import gaussian_kde, norm
        kde_x = np.linspace(daily_ret.min(), daily_ret.max(), 200)
        kde_y = gaussian_kde(daily_ret)(kde_x)
        scale = len(daily_ret) * (daily_ret.max() - daily_ret.min()) / 60
        fig_dist.add_trace(go.Scatter(x=kde_x * 100, y=kde_y * scale,
                                      mode="lines", line=dict(color="#00d4a4", width=2), name="KDE"))
        # Normal reference
        norm_y = norm.pdf(kde_x, daily_ret.mean(), daily_ret.std()) * scale
        fig_dist.add_trace(go.Scatter(x=kde_x * 100, y=norm_y,
                                      mode="lines", line=dict(color="#888", width=1, dash="dot"), name="Normal"))

        fig_dist.add_vline(x=-var95 * 100, line_color="#ef4444", line_dash="dash",
                           annotation_text=f"VaR {conf*100:.0f}%", annotation_font_color="#ef4444")
        fig_dist.add_vline(x=-cvar95 * 100, line_color="#f97316", line_dash="dot",
                           annotation_text="CVaR", annotation_font_color="#f97316")

        fig_dist.update_layout(height=280, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
                                xaxis_title="Daily Return (%)", legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_dist, use_container_width=True)

        # Distribution stats
        c1a, c1b, c1c, c1d = st.columns(4)
        c1a.metric("Skewness", f"{daily_ret.skew():.2f}")
        c1b.metric("Excess Kurtosis", f"{daily_ret.kurtosis():.2f}")
        c1c.metric("Ann. VaR", f"{var95*np.sqrt(252)*100:.1f}%")
        c1d.metric("Ann. CVaR", f"{cvar95*np.sqrt(252)*100:.1f}%")

    # ── VaR comparison table ──────────────────────────────────────────────────
    with col2:
        st.markdown("### VaR Comparison (All Methods)")
        var_df = compute_var_all_methods(port_ret, pv, conf, holding)
        st.dataframe(var_df, use_container_width=True, hide_index=True)

        st.markdown("### Performance Ratios")
        ratio_names = ["Sharpe Ratio","Sortino Ratio","Calmar Ratio","Treynor Ratio","Omega Ratio","Alpha","Beta","Information Ratio"]
        ratio_df = pd.DataFrame([
            {"Metric": k, "Value": f"{ratios[k]:.3f}" if abs(ratios.get(k, 0)) < 1000 else "—"}
            for k in ratio_names if k in ratios
        ])
        st.dataframe(ratio_df, use_container_width=True, hide_index=True)

    # ── Rolling metrics ──────────────────────────────────────────────────────
    st.markdown("### Rolling Risk Metrics (63-day window)")
    roll = rolling_risk_metrics(port_ret, window=63)
    roll = roll.dropna()

    fig_r = make_subplots(rows=2, cols=2, shared_xaxes=True,
                          subplot_titles=["Rolling Volatility", "Rolling Sharpe",
                                          "Rolling VaR (95%)", "Rolling Max Drawdown"],
                          vertical_spacing=0.12, horizontal_spacing=0.08)

    pairs = [
        ("Rolling Vol", 1, 1, "#3b82f6"),
        ("Rolling Sharpe", 1, 2, "#00d4a4"),
        ("Rolling VaR (95%)", 2, 1, "#ef4444"),
        ("Rolling Max DD", 2, 2, "#f59e0b"),
    ]
    for col_name, row, c, color in pairs:
        if col_name in roll.columns:
            fig_r.add_trace(
                go.Scatter(x=roll.index, y=roll[col_name], mode="lines",
                           line=dict(color=color, width=1.2), showlegend=False),
                row=row, col=c,
            )

    fig_r.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_r, use_container_width=True)

    # ── Drawdown chart ───────────────────────────────────────────────────────
    from utils.data import compute_drawdown
    dd = compute_drawdown(port_prices_cum)
    st.markdown("### Drawdown History")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, mode="lines",
                                fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
                                line=dict(color="#ef4444", width=1.2), name="Drawdown"))
    fig_dd.update_layout(height=200, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
                          yaxis_title="Drawdown %", showlegend=False)
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Stress tests ─────────────────────────────────────────────────────────
    st.markdown("### Stress Test Scenarios")
    beta = ratios.get("Beta", 1.0)
    stress_df = stress_test(pv, beta)
    st.dataframe(stress_df, use_container_width=True, hide_index=True)
