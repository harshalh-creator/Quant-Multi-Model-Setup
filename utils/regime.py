"""
utils/regime.py — Market Regime Detection Engine
Models: HMM (hmmlearn), Z-Score + Momentum, Composite Multi-Signal
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RegimeResult:
    regimes: pd.Series              # date-indexed regime labels
    probabilities: pd.DataFrame     # date-indexed regime probabilities
    current_regime: str
    current_prob: float
    transition_matrix: np.ndarray
    regime_stats: Dict
    signals: Dict


REGIME_LABELS = {0: "Bull", 1: "Sideways", 2: "Bear", 3: "High Volatility"}
REGIME_COLORS = {"Bull": "#00d4a4", "Bear": "#ff4b4b", "Sideways": "#3b82f6", "High Volatility": "#f59e0b"}
REGIME_ALLOC = {
    "Bull":           {"Large Cap": 45, "Mid/Small Cap": 25, "Debt": 10, "Gold": 5,  "Cash": 5,  "International": 10},
    "Bear":           {"Large Cap": 15, "Mid/Small Cap": 5,  "Debt": 45, "Gold": 20, "Cash": 15, "International": 0},
    "Sideways":       {"Large Cap": 30, "Mid/Small Cap": 15, "Debt": 30, "Gold": 10, "Cash": 15, "International": 0},
    "High Volatility":{"Large Cap": 20, "Mid/Small Cap": 10, "Debt": 35, "Gold": 20, "Cash": 15, "International": 0},
}


def compute_technical_signals(prices: pd.Series) -> Dict:
    """Compute a rich set of technical signals."""
    returns = prices.pct_change().dropna()
    close = prices

    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()

    # MACD
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # Rolling volatility (21-day annualised)
    roll_vol = returns.rolling(21).std() * np.sqrt(252)

    # ATR (proxy)
    atr = ((close.rolling(14).max() - close.rolling(14).min()) / close.rolling(14).mean())

    # Momentum (3M, 6M, 12M)
    mom1 = close.pct_change(21)
    mom3 = close.pct_change(63)
    mom6 = close.pct_change(126)
    mom12 = close.pct_change(252)

    latest = close.iloc[-1]
    return {
        "price": latest,
        "sma20": sma20.iloc[-1],
        "sma50": sma50.iloc[-1],
        "sma200": sma200.iloc[-1],
        "above_sma20": latest > sma20.iloc[-1],
        "above_sma50": latest > sma50.iloc[-1],
        "above_sma200": latest > sma200.iloc[-1],
        "golden_cross": sma50.iloc[-1] > sma200.iloc[-1],
        "death_cross": sma50.iloc[-1] < sma200.iloc[-1],
        "macd_hist": macd_hist.iloc[-1],
        "macd_bullish": macd_hist.iloc[-1] > 0,
        "rsi": rsi.iloc[-1],
        "rsi_overbought": rsi.iloc[-1] > 70,
        "rsi_oversold": rsi.iloc[-1] < 30,
        "bb_pct": bb_pct.iloc[-1],
        "roll_vol": roll_vol.iloc[-1],
        "mom1m": mom1.iloc[-1],
        "mom3m": mom3.iloc[-1],
        "mom6m": mom6.iloc[-1],
        "mom12m": mom12.iloc[-1],
        "atr_pct": atr.iloc[-1],
        # Time series for charts
        "rsi_series": rsi,
        "macd_series": macd_hist,
        "roll_vol_series": roll_vol,
        "sma50_series": sma50,
        "sma200_series": sma200,
    }


def z_score_regime(prices: pd.Series, window: int = 252) -> pd.Series:
    """Z-score based regime classification."""
    returns = prices.pct_change().dropna()
    roll_ret = returns.rolling(window).mean() * 252
    roll_vol = returns.rolling(window).std() * np.sqrt(252)

    z_ret = (roll_ret - roll_ret.rolling(window).mean()) / roll_ret.rolling(window).std()
    z_vol = (roll_vol - roll_vol.rolling(window).mean()) / roll_vol.rolling(window).std()

    def classify(row):
        zr, zv = row["zret"], row["zvol"]
        if pd.isna(zr) or pd.isna(zv):
            return "Sideways"
        if zv > 1.5:
            return "High Volatility"
        if zr > 0.5:
            return "Bull"
        if zr < -0.5:
            return "Bear"
        return "Sideways"

    df = pd.DataFrame({"zret": z_ret, "zvol": z_vol})
    regimes = df.apply(classify, axis=1)
    return regimes


def trend_vol_regime(prices: pd.Series) -> pd.Series:
    """Trend + volatility filter regime (simple, interpretable)."""
    returns = prices.pct_change().dropna()
    sma50  = prices.rolling(50).mean()
    sma200 = prices.rolling(200).mean()
    roll_vol = returns.rolling(21).std() * np.sqrt(252)
    long_vol = returns.rolling(63).std() * np.sqrt(252)

    def classify(i):
        if i < 200:
            return "Sideways"
        v_ratio = roll_vol.iloc[i] / (long_vol.iloc[i] + 1e-10)
        above200 = prices.iloc[i] > sma200.iloc[i]
        above50  = prices.iloc[i] > sma50.iloc[i]
        golden   = sma50.iloc[i] > sma200.iloc[i]

        if v_ratio > 1.5:
            return "High Volatility"
        if above200 and above50 and golden:
            return "Bull"
        if not above200 and not above50 and not golden:
            return "Bear"
        return "Sideways"

    idx = range(len(prices))
    labels = [classify(i) for i in idx]
    return pd.Series(labels, index=prices.index)


def composite_regime(prices: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Multi-signal composite regime with probability estimates.
    Combines trend, momentum, volatility, and RSI signals.
    """
    returns = prices.pct_change().dropna()
    sma50  = prices.rolling(50).mean()
    sma200 = prices.rolling(200).mean()
    ema12  = prices.ewm(span=12).mean()
    ema26  = prices.ewm(span=26).mean()
    macd   = ema12 - ema26

    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi  = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    roll_vol   = returns.rolling(21).std() * np.sqrt(252)
    long_vol   = returns.rolling(63).std() * np.sqrt(252)
    mom3m      = prices.pct_change(63)
    mom6m      = prices.pct_change(126)

    bull_score = pd.Series(0.0, index=prices.index)
    bear_score = pd.Series(0.0, index=prices.index)
    hv_score   = pd.Series(0.0, index=prices.index)

    # Trend signals
    bull_score += (prices > sma200).astype(float) * 0.20
    bull_score += (prices > sma50).astype(float) * 0.15
    bull_score += (sma50 > sma200).astype(float) * 0.15
    bear_score += (prices < sma200).astype(float) * 0.20
    bear_score += (prices < sma50).astype(float) * 0.15

    # MACD
    bull_score += (macd > 0).astype(float) * 0.10
    bear_score += (macd < 0).astype(float) * 0.10

    # RSI
    bull_score += (rsi > 50).astype(float) * 0.10
    bear_score += (rsi < 50).astype(float) * 0.10
    hv_score   += (rsi > 75).astype(float) * 0.10
    hv_score   += (rsi < 25).astype(float) * 0.10

    # Momentum
    bull_score += (mom3m > 0).astype(float) * 0.10
    bull_score += (mom6m > 0).astype(float) * 0.10
    bear_score += (mom3m < 0).astype(float) * 0.10
    bear_score += (mom6m < 0).astype(float) * 0.10

    # Volatility
    v_ratio = roll_vol / (long_vol + 1e-10)
    hv_score += (v_ratio > 1.3).astype(float) * 0.30

    side_score = 1 - bull_score - bear_score - hv_score
    side_score = side_score.clip(lower=0)

    scores = pd.DataFrame({
        "Bull": bull_score, "Bear": bear_score,
        "High Volatility": hv_score, "Sideways": side_score
    })
    total = scores.sum(axis=1).replace(0, 1)
    probs = scores.div(total, axis=0)

    regimes = probs.idxmax(axis=1)
    regimes.iloc[:200] = "Sideways"

    return regimes, probs


def hmm_regime(prices: pd.Series, n_states: int = 3) -> Tuple[pd.Series, pd.DataFrame]:
    """
    HMM-based regime detection using hmmlearn.
    Falls back to composite method if hmmlearn unavailable.
    """
    returns = prices.pct_change().dropna()
    roll_vol = returns.rolling(5).std()
    features = pd.concat([returns, roll_vol], axis=1).dropna()
    features.columns = ["ret", "vol"]

    try:
        from hmmlearn import hmm
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        X = features.values
        model.fit(X)
        states = model.predict(X)
        proba = model.predict_proba(X)

        # Map HMM states to regime labels by mean return
        state_means = [features["ret"].values[states == s].mean() for s in range(n_states)]
        rank = np.argsort(state_means)[::-1]
        label_map_arr = ["Sideways"] * n_states
        if n_states >= 2:
            label_map_arr[rank[0]] = "Bull"
            label_map_arr[rank[-1]] = "Bear"
        if n_states >= 3:
            vols = [features["vol"].values[states == s].mean() for s in range(n_states)]
            remaining = [i for i in range(n_states) if i not in [rank[0], rank[-1]]]
            if remaining:
                high_v = max(remaining, key=lambda i: vols[i])
                label_map_arr[high_v] = "High Volatility"

        raw_labels = [label_map_arr[s] for s in states]
        regime_series = pd.Series(raw_labels, index=features.index)
        regime_series = regime_series.reindex(prices.index).ffill().bfill()

        prob_cols = [label_map_arr[i] for i in range(n_states)]
        prob_df = pd.DataFrame(proba, index=features.index, columns=prob_cols)
        prob_df = prob_df.reindex(prices.index).ffill().bfill()
        return regime_series, prob_df

    except ImportError:
        # Fallback: composite
        reg, prob = composite_regime(prices)
        return reg, prob


def compute_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Compute empirical transition probability matrix."""
    states = ["Bull", "Bear", "Sideways", "High Volatility"]
    n = len(states)
    counts = pd.DataFrame(0, index=states, columns=states)
    reg_list = regimes.tolist()
    for i in range(len(reg_list) - 1):
        s1, s2 = reg_list[i], reg_list[i + 1]
        if s1 in states and s2 in states:
            counts.loc[s1, s2] += 1
    row_sums = counts.sum(axis=1).replace(0, 1)
    return counts.div(row_sums, axis=0)


def regime_statistics(prices: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Compute per-regime return statistics."""
    returns = prices.pct_change().dropna()
    df = returns.to_frame("ret")
    df["regime"] = regimes.reindex(df.index)
    df = df.dropna()

    stats = []
    for regime in ["Bull", "Bear", "Sideways", "High Volatility"]:
        sub = df[df["regime"] == regime]["ret"]
        if len(sub) < 5:
            continue
        ann_ret = sub.mean() * 252
        ann_vol = sub.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        days = len(sub)
        stats.append({
            "Regime": regime,
            "Days": days,
            "% Time": f"{days / len(df) * 100:.1f}%",
            "Ann. Return": f"{ann_ret * 100:.1f}%",
            "Ann. Vol": f"{ann_vol * 100:.1f}%",
            "Sharpe": f"{sharpe:.2f}",
        })
    return pd.DataFrame(stats)


def detect_regime(
    prices: pd.Series,
    method: str = "composite",
    n_states: int = 3,
) -> RegimeResult:
    """Master regime detection function."""
    if method == "hmm":
        regimes, probs = hmm_regime(prices, n_states)
    elif method == "zscore":
        regimes = z_score_regime(prices)
        probs = pd.get_dummies(regimes).reindex(columns=["Bull", "Bear", "Sideways", "High Volatility"], fill_value=0)
    elif method == "trend":
        regimes = trend_vol_regime(prices)
        probs = pd.get_dummies(regimes).reindex(columns=["Bull", "Bear", "Sideways", "High Volatility"], fill_value=0)
    else:  # composite
        regimes, probs = composite_regime(prices)

    current = regimes.iloc[-1]
    current_p = float(probs.iloc[-1].get(current, 0.5)) if current in probs.columns else 0.5

    trans = compute_transition_matrix(regimes)
    stats = regime_statistics(prices, regimes)
    signals = compute_technical_signals(prices)

    return RegimeResult(
        regimes=regimes,
        probabilities=probs,
        current_regime=current,
        current_prob=current_p,
        transition_matrix=trans,
        regime_stats=stats,
        signals=signals,
    )
