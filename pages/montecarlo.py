"""pages/montecarlo.py — Monte Carlo Simulation"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data import fetch_prices, compute_returns, annualised_return, annualised_vol, NSE_LARGE_CAP
from utils.monte_carlo import MCParams, run_simulation, terminal_distribution_bins


def fmt_inr(v: float) -> str:
    if v >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    if v >= 1e5:
        return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"


def show():
    st.markdown("## 🎲 Monte Carlo Portfolio Simulator")

    col_ctrl, col_out = st.columns([1, 2])

    with col_ctrl:
        st.markdown("### Simulation Parameters")
        source = st.radio("Return source", ["Manual input", "Live ticker data"], horizontal=True)

        if source == "Live ticker data":
            ticker = st.selectbox("Ticker", NSE_LARGE_CAP[:15],
                                  format_func=lambda x: x.replace(".NS",""))
            period = st.session_state.get("data_period", "2y")
            with st.spinner(f"Fetching {ticker}..."):
                prices = fetch_prices([ticker], period=period)
            if not prices.empty:
                ret = compute_returns(prices.squeeze(), log=False)
                mu = annualised_return(ret)
                sigma = annualised_vol(ret)
                st.success(f"Live: μ={mu*100:.1f}%, σ={sigma*100:.1f}%")
            else:
                mu, sigma = 0.12, 0.18
        else:
            mu    = st.slider("Annual return (%)", 1, 35, 12) / 100
            sigma = st.slider("Annual volatility (%)", 1, 60, 18) / 100

        S0 = st.number_input("Starting value (₹)", min_value=10_000, value=1_000_000, step=50_000, format="%d")
        T  = st.slider("Time horizon (years)", 1, 30, 10)
        n_sims = st.select_slider("Simulations", [100, 500, 1000, 2000, 5000], value=1000)
        contrib = st.number_input("Annual contribution (₹)", min_value=0, value=120_000, step=10_000, format="%d")
        model = st.selectbox("Stochastic model", {
            "gbm": "Geometric Brownian Motion",
            "jump": "Merton Jump-Diffusion",
            "heston": "Heston Stochastic Vol",
            "garch": "GARCH(1,1)",
        }.keys(), format_func=lambda k: {"gbm":"GBM","jump":"Merton Jump-Diffusion","heston":"Heston SV","garch":"GARCH(1,1)"}[k])
        vr = st.selectbox("Variance reduction", ["antithetic", "none"],
                          format_func=lambda x: {"antithetic":"Antithetic Variates","none":"None"}[x])

        run = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    with col_out:
        if not run:
            st.info("Configure parameters on the left and click Run Simulation.")
            return

        params = MCParams(
            S0=S0, mu=mu, sigma=sigma, T=T, n_sims=n_sims,
            annual_contrib=contrib, model=model, variance_reduction=vr,
        )

        with st.spinner(f"Running {n_sims:,} simulations ({model.upper()})..."):
            result = run_simulation(params)

        s = result.stats
        # ── KPIs ─────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median End Value", fmt_inr(s["median"]))
        k2.metric("95th Percentile", fmt_inr(s["p95"]))
        k3.metric("5th Percentile",  fmt_inr(s["p5"]))
        k4.metric("Expected CAGR",   f"{s['cagr']*100:.1f}%")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("P(Profit)",       f"{s['p_profit']*100:.1f}%")
        k6.metric("P(2× Capital)",   f"{s['p_2x']*100:.1f}%")
        k7.metric("P(5× Capital)",   f"{s['p_5x']*100:.1f}%")
        k8.metric("P(50% Drawdown)", f"{s['p_ruin']*100:.1f}%", delta=None)

        # ── Paths chart ──────────────────────────────────────────────────
        st.markdown("### Simulation Paths")
        fig_p = go.Figure()
        n_show = min(150, len(result.paths))
        t_axis = result.time_axis
        idx = np.linspace(0, len(t_axis)-1, min(len(t_axis), T*12+1), dtype=int)
        t_dec = t_axis[idx]

        for i in range(n_show):
            path = result.paths[i, idx]
            fig_p.add_trace(go.Scatter(x=t_dec, y=path, mode="lines",
                                       line=dict(color="rgba(59,130,246,0.07)", width=1),
                                       showlegend=False, hoverinfo="skip"))

        # Percentile bands
        for pct, color, dash, label in [
            (95, "#00d4a4", "dot", "95th pct"),
            (50, "#3b82f6", "solid", "Median"),
            (5,  "#ef4444", "dot", "5th pct"),
        ]:
            pct_path = np.percentile(result.paths[:, idx], pct, axis=0)
            fig_p.add_trace(go.Scatter(x=t_dec, y=pct_path, mode="lines",
                                       line=dict(color=color, width=2, dash=dash), name=label))

        fig_p.update_layout(height=320, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
                             xaxis_title="Years", yaxis_title="Portfolio Value (₹)",
                             yaxis_tickformat=".2s",
                             legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_p, use_container_width=True)

        # ── Terminal distribution ────────────────────────────────────────
        col_d, col_t = st.columns([3, 2])
        with col_d:
            st.markdown("### Terminal Value Distribution")
            edges, counts = terminal_distribution_bins(result, 50)
            mids = (edges[:-1] + edges[1:]) / 2
            colors = ["#ef4444" if m < S0 else "#00d4a4" if m >= s["median"] else "#3b82f6" for m in mids]
            fig_d = go.Figure(go.Bar(x=mids, y=counts, marker_color=colors, marker_line_width=0))
            fig_d.add_vline(x=s["median"], line_color="#f59e0b", line_dash="dash",
                            annotation_text=f"Median: {fmt_inr(s['median'])}", annotation_position="top right",
                            annotation_font_color="#f59e0b")
            fig_d.add_vline(x=S0, line_color="#888", line_dash="dot",
                            annotation_text="Initial", annotation_position="top left")
            fig_d.update_layout(height=260, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
                                 xaxis_tickformat=".2s", showlegend=False)
            st.plotly_chart(fig_d, use_container_width=True)

        with col_t:
            st.markdown("### Percentile Table")
            pct_rows = []
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                v = result.percentiles.get(p, np.percentile(result.terminal_values, p))
                cagr_p = (v / S0) ** (1 / T) - 1
                pct_rows.append({
                    "Pct": f"{p}th",
                    "End Value": fmt_inr(v),
                    "CAGR": f"{cagr_p*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)
