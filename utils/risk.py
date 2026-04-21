"""
utils/risk.py — Risk Analytics Engine
VaR/CVaR: Historical, Parametric, Cornish-Fisher, EVT
Factor decomposition, stress testing, Tail Risk
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple


# ─── VaR / CVaR ─────────────────────────────────────────────────────────────

def var_historical(returns: pd.Series, confidence: float = 0.95, holding: int = 1) -> float:
    """Historical simulation VaR (sign: positive = loss)."""
    scaled = returns * np.sqrt(holding)
    return float(-np.percentile(scaled.dropna(), (1 - confidence) * 100))


def cvar_historical(returns: pd.Series, confidence: float = 0.95, holding: int = 1) -> float:
    """Historical CVaR / Expected Shortfall."""
    scaled = returns.dropna() * np.sqrt(holding)
    threshold = np.percentile(scaled, (1 - confidence) * 100)
    tail = scaled[scaled <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


def var_parametric(
    returns: pd.Series, confidence: float = 0.95, holding: int = 1
) -> Tuple[float, float]:
    """Parametric (Gaussian) VaR and CVaR."""
    mu = returns.mean() * holding
    sigma = returns.std() * np.sqrt(holding)
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    # CVaR = -mu + sigma * phi(z) / (1 - confidence)
    cvar_val = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
    return float(var), float(cvar_val)


def var_cornish_fisher(
    returns: pd.Series, confidence: float = 0.95, holding: int = 1
) -> Tuple[float, float]:
    """Cornish-Fisher expansion adjusted for skewness and excess kurtosis."""
    mu = returns.mean() * holding
    sigma = returns.std() * np.sqrt(holding)
    s = returns.skew()
    k = returns.kurtosis()  # excess kurtosis
    z = stats.norm.ppf(1 - confidence)
    z_cf = (z
            + (z ** 2 - 1) * s / 6
            + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    var = -(mu + z_cf * sigma)
    cvar_val = var * 1.10  # approximate
    return float(var), float(cvar_val)


def var_evt(
    returns: pd.Series, confidence: float = 0.95, threshold_pct: float = 0.10
) -> Tuple[float, float]:
    """
    Extreme Value Theory (Peaks-Over-Threshold / GPD fit).
    Uses scipy.stats.genpareto for tail fitting.
    """
    losses = -returns.dropna()
    threshold = np.percentile(losses, (1 - threshold_pct) * 100)
    excesses = losses[losses > threshold] - threshold

    if len(excesses) < 10:
        return var_parametric(returns, confidence)

    try:
        c, loc, scale = stats.genpareto.fit(excesses, floc=0)
        n_total = len(losses)
        n_tail = len(excesses)
        u = threshold
        # VaR from GPD
        var_evt_val = u + (scale / c) * ((n_total / n_tail * (1 - confidence)) ** (-c) - 1)
        # CVaR from GPD
        cvar_evt_val = var_evt_val / (1 - c) + (scale - c * u) / (1 - c)
        return float(var_evt_val), float(cvar_evt_val)
    except Exception:
        return var_parametric(returns, confidence)


def compute_var_all_methods(
    returns: pd.Series,
    pv: float,
    confidence: float = 0.95,
    holding: int = 1,
) -> pd.DataFrame:
    """Compute VaR and CVaR by all methods."""
    rows = []
    for name, fn in [
        ("Historical", lambda: (var_historical(returns, confidence, holding), cvar_historical(returns, confidence, holding))),
        ("Parametric", lambda: var_parametric(returns, confidence, holding)),
        ("Cornish-Fisher", lambda: var_cornish_fisher(returns, confidence, holding)),
        ("EVT (GPD)", lambda: var_evt(returns, confidence)),
    ]:
        try:
            v, cv = fn()
        except Exception:
            v, cv = 0.0, 0.0
        rows.append({
            "Method": name,
            "VaR (%)": f"{v * 100:.2f}%",
            "CVaR (%)": f"{cv * 100:.2f}%",
            "VaR (₹)": f"₹{v * pv:,.0f}",
            "CVaR (₹)": f"₹{cv * pv:,.0f}",
        })
    return pd.DataFrame(rows)


# ─── Performance metrics ─────────────────────────────────────────────────────

def compute_all_ratios(
    returns: pd.Series,
    prices: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    rf: float = 0.065,
) -> Dict:
    daily_rf = rf / 252
    excess = returns - daily_rf
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    mdd = float(drawdown.min())

    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    downside = returns[returns < daily_rf].std() * np.sqrt(252)
    sortino = (ann_ret - rf) / downside if downside > 0 else 0
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

    # Omega ratio
    gains = returns[returns > daily_rf] - daily_rf
    losses = daily_rf - returns[returns < daily_rf]
    omega = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf

    # Treynor + Information ratio
    beta, alpha, ir = 1.0, 0.0, 0.0
    if benchmark_returns is not None:
        aligned = pd.concat([excess, benchmark_returns - daily_rf], axis=1).dropna()
        if len(aligned) > 30:
            cov_matrix = aligned.cov()
            beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]
            alpha = float((ann_ret - rf) - beta * (annualised_bm(benchmark_returns) - rf))
            te = (returns - benchmark_returns).std() * np.sqrt(252)
            ir = alpha / te if te > 0 else 0

    treynor = (ann_ret - rf) / beta if beta != 0 else 0

    # Tail risk
    var95 = var_historical(returns, 0.95)
    cvar95 = cvar_historical(returns, 0.95)

    return {
        "Ann. Return": ann_ret, "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe, "Sortino Ratio": sortino,
        "Calmar Ratio": calmar, "Omega Ratio": omega,
        "Treynor Ratio": treynor, "Beta": beta, "Alpha": alpha,
        "Information Ratio": ir, "Max Drawdown": mdd,
        "VaR (95%)": var95, "CVaR (95%)": cvar95,
        "Skewness": returns.skew(), "Kurtosis": returns.kurtosis(),
    }


def annualised_bm(bm_returns: pd.Series) -> float:
    return float((1 + bm_returns).prod() ** (252 / len(bm_returns)) - 1)


# ─── Stress testing ──────────────────────────────────────────────────────────

STRESS_SCENARIOS = [
    {"name": "2008 Global Financial Crisis", "market_shock": -0.55, "vol_mult": 3.5},
    {"name": "2020 COVID Crash (Feb-Mar)", "market_shock": -0.38, "vol_mult": 2.5},
    {"name": "Dot-com Bust (2000-02)",      "market_shock": -0.50, "vol_mult": 2.0},
    {"name": "NIFTY 10% Correction",        "market_shock": -0.10, "vol_mult": 1.5},
    {"name": "RBI Rate Hike +200 bps",      "market_shock": -0.12, "vol_mult": 1.3},
    {"name": "INR Depreciation -15%",       "market_shock": -0.07, "vol_mult": 1.2},
    {"name": "Oil Shock +60%",              "market_shock": -0.09, "vol_mult": 1.4},
    {"name": "Russia-Ukraine Style Event",  "market_shock": -0.18, "vol_mult": 1.8},
    {"name": "China Slowdown",              "market_shock": -0.15, "vol_mult": 1.5},
    {"name": "Flash Crash (-10% in a day)", "market_shock": -0.10, "vol_mult": 4.0},
]


def stress_test(pv: float, beta: float = 1.0) -> pd.DataFrame:
    """Apply market stress scenarios to a portfolio."""
    rows = []
    for s in STRESS_SCENARIOS:
        port_shock = s["market_shock"] * beta
        loss = pv * abs(port_shock)
        rows.append({
            "Scenario": s["name"],
            "Market Shock": f"{s['market_shock']*100:.0f}%",
            "Portfolio Shock": f"{port_shock*100:.1f}%",
            "Loss (₹)": f"₹{loss:,.0f}",
            "Remaining (₹)": f"₹{pv + pv*port_shock:,.0f}",
        })
    return pd.DataFrame(rows)


# ─── Rolling risk ────────────────────────────────────────────────────────────

def rolling_risk_metrics(
    returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    roll_vol = returns.rolling(window).std() * np.sqrt(252)
    roll_var = returns.rolling(window).apply(lambda x: var_historical(pd.Series(x), 0.95), raw=False)
    roll_corr_with_self = pd.Series(1.0, index=returns.index)  # placeholder
    roll_sharpe = (returns.rolling(window).mean() * 252 - 0.065) / (returns.rolling(window).std() * np.sqrt(252) + 1e-10)
    roll_dd = returns.rolling(window).apply(lambda x: (pd.Series(x).add(1).cumprod() / pd.Series(x).add(1).cumprod().cummax() - 1).min(), raw=False)

    return pd.DataFrame({
        "Rolling Vol": roll_vol,
        "Rolling VaR (95%)": roll_var,
        "Rolling Sharpe": roll_sharpe,
        "Rolling Max DD": roll_dd,
    })
