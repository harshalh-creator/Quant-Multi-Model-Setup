"""pages/optimizer.py — Portfolio Optimization with live data"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data import (fetch_prices, compute_returns, NSE_LARGE_CAP, NSE_MID_CAP,
                         annualised_return, annualised_vol, sharpe, max_drawdown)
from utils.optimizer import (max_sharpe as opt_max_sharpe, min_variance, mean_variance,
                               equal_risk_contribution, hierarchical_risk_parity,
                               black_litterman, efficient_frontier, random_portfolios,
                               sample_cov, ledoit_wolf_cov, ewma_cov,
                               portfolio_return, portfolio_vol)


POPULAR = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
           "BAJFINANCE.NS","HINDUNILVR.NS","TITAN.NS","NESTLEIND.NS","WIPRO.NS"]


def show():
    st.markdown("## ⚖️ Portfolio Optimizer")

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Tickers")
        default_sel = POPULAR[:5]
        all_tickers = NSE_LARGE_CAP + NSE_MID_CAP
        selected = st.multiselect("Select assets", all_tickers, default=default_sel,
                                  format_func=lambda x: x.replace(".NS","").replace(".BO",""))
        st.markdown("### Method")
        method = st.selectbox("Optimization method", [
            "Maximum Sharpe Ratio", "Minimum Variance", "Mean-Variance (MVO)",
            "Risk Parity (ERC)", "Hierarchical Risk Parity", "Black-Litterman"])
        st.markdown("### Covariance Estimator")
        cov_est = st.selectbox("Estimator", ["Sample", "Ledoit-Wolf Shrinkage", "EWMA (RiskMetrics)"])
        st.markdown("### Constraints")
        min_w = st.slider("Min weight (%)", 0, 15, 2) / 100
        max_w = st.slider("Max weight (%)", 10, 100, 40) / 100
        period = st.session_state.get("data_period", "2y")
        rf = st.session_state.get("rf_rate", 0.065)

    if len(selected) < 2:
        st.info("Select at least 2 assets in the sidebar.")
        return

    with st.spinner("Fetching live prices..."):
        prices = fetch_prices(selected, period=period)

    if prices.empty or len(prices) < 60:
        st.error("Not enough data. Try increasing the period or selecting different tickers.")
        return

    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    tickers = list(prices.columns)
    returns = compute_returns(prices, log=False)

    # Covariance
    if cov_est == "Ledoit-Wolf Shrinkage":
        cov = ledoit_wolf_cov(returns)
    elif cov_est == "EWMA (RiskMetrics)":
        cov = ewma_cov(returns)
    else:
        cov = sample_cov(returns)

    mu = np.array([annualised_return(returns[t]) for t in tickers])

    # ── Run optimizer ────────────────────────────────────────────────────────
    try:
        if method == "Maximum Sharpe Ratio":
            w = opt_max_sharpe(mu, cov, rf, min_w, max_w)
        elif method == "Minimum Variance":
            w = min_variance(cov, min_w, max_w)
        elif method == "Mean-Variance (MVO)":
            w = mean_variance(mu, cov, min_w=min_w, max_w=max_w)
        elif method == "Risk Parity (ERC)":
            w = equal_risk_contribution(cov, min_w, max_w)
        elif method == "Hierarchical Risk Parity":
            w = hierarchical_risk_parity(returns[tickers])
        elif method == "Black-Litterman":
            cap_w = np.ones(len(mu)) / len(mu)
            mu_eq = 2.5 * cov @ cap_w
            P = np.eye(len(mu))[:2]
            Q = mu[:2] + 0.01
            _, w = black_litterman(mu_eq, cov, P, Q, rf=rf, min_w=min_w, max_w=max_w)
        else:
            w = opt_max_sharpe(mu, cov, rf, min_w, max_w)
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        w = np.ones(len(tickers)) / len(tickers)

    port_ret = portfolio_return(w, mu)
    port_vol = portfolio_vol(w, cov)
    port_sharpe = (port_ret - rf) / port_vol
    port_prices = (returns * w).sum(axis=1) + 1
    port_prices = port_prices.cumprod()
    mdd = max_drawdown(port_prices)

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Expected Return", f"{port_ret*100:.1f}%")
    c2.metric("Portfolio Vol", f"{port_vol*100:.1f}%")
    c3.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
    c4.metric("Max Drawdown", f"{mdd*100:.1f}%")
    c5.metric("Assets", str(len(tickers)))

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    # ── Weights donut ────────────────────────────────────────────────────────
    with col1:
        st.markdown("### Optimal Weights")
        names = [t.replace(".NS","") for t in tickers]
        fig_w = go.Figure(go.Pie(
            labels=names, values=(w * 100).round(1),
            hole=0.55, textinfo="label+percent",
            textfont_size=12,
            marker=dict(colors=px.colors.qualitative.Set2[:len(tickers)]),
        ))
        fig_w.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=0, r=0, t=10, b=0),
                             legend=dict(orientation="h", y=-0.1, font=dict(size=10)))
        st.plotly_chart(fig_w, use_container_width=True)

    # ── Efficient frontier ───────────────────────────────────────────────────
    with col2:
        st.markdown("### Efficient Frontier")
        with st.spinner("Computing frontier..."):
            ef = efficient_frontier(mu, cov, n_points=40, min_w=0, max_w=1)
            rp = random_portfolios(mu, cov, n=1000)

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=rp["vol"] * 100, y=rp["return"] * 100, mode="markers",
            marker=dict(color=rp["sharpe"], colorscale="Viridis", size=4, opacity=0.5,
                        colorbar=dict(title="Sharpe", thickness=10)),
            name="Random portfolios", hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%",
        ))
        if not ef.empty:
            fig_ef.add_trace(go.Scatter(
                x=ef["vol"] * 100, y=ef["return"] * 100, mode="lines",
                line=dict(color="#f59e0b", width=2), name="Efficient Frontier",
            ))
        fig_ef.add_trace(go.Scatter(
            x=[port_vol * 100], y=[port_ret * 100], mode="markers",
            marker=dict(color="#00d4a4", size=14, symbol="star"),
            name="Optimal Portfolio",
        ))
        fig_ef.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="Risk (Vol %)", yaxis_title="Return (%)",
                              legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_ef, use_container_width=True)

    # ── Weights table ────────────────────────────────────────────────────────
    st.markdown("### Asset Details")
    rows = []
    for i, t in enumerate(tickers):
        ret_i = annualised_return(returns[t])
        vol_i = annualised_vol(returns[t])
        sh_i = sharpe(returns[t], rf)
        mdd_i = max_drawdown(prices[t])
        rows.append({
            "Ticker": t.replace(".NS",""), "Weight": f"{w[i]*100:.1f}%",
            "Exp. Return": f"{ret_i*100:.1f}%", "Volatility": f"{vol_i*100:.1f}%",
            "Sharpe": f"{sh_i:.2f}", "Max DD": f"{mdd_i*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Correlation heatmap ──────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Correlation Matrix")
        corr = returns[tickers].corr()
        short_names = [t.replace(".NS","") for t in tickers]
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=short_names, y=short_names,
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=10),
        ))
        fig_corr.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

    with col4:
        st.markdown("### Portfolio Cumulative Return")
        bench = fetch_index("^NSEI", period)
        bench_ret = bench.pct_change().dropna()
        bench_cum = (bench_ret + 1).cumprod()
        bench_cum = bench_cum / bench_cum.iloc[0]
        port_prices_norm = port_prices / port_prices.iloc[0]

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=port_prices_norm.index, y=port_prices_norm.values,
                                      mode="lines", name="Portfolio", line=dict(color="#00d4a4", width=2)))
        common_idx = port_prices_norm.index.intersection(bench_cum.index)
        fig_perf.add_trace(go.Scatter(x=common_idx, y=bench_cum.reindex(common_idx).values,
                                      mode="lines", name="NIFTY 50", line=dict(color="#888", width=1.5, dash="dot")))
        fig_perf.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
                                legend=dict(font=dict(size=11)), yaxis_tickformat=".2f")
        st.plotly_chart(fig_perf, use_container_width=True)
