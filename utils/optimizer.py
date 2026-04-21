"""
utils/optimizer.py — Portfolio Optimization Engine
Implements: MVO, HRP, Max Sharpe, Min Variance, Risk Parity, Black-Litterman
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from typing import Optional, Dict, Tuple


# ─── Covariance estimators ──────────────────────────────────────────────────

def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values * 252


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage estimator (analytical formula)."""
    X = returns.values
    n, p = X.shape
    sample = np.cov(X.T, ddof=1) * 252
    mu = np.trace(sample) / p
    delta = np.eye(p) * mu
    # Oracle shrinkage intensity
    beta2 = (np.sum(X**2)**2) / (n**2 * (np.sum(X**2 * np.sum(X**2, axis=0))**0.5 or 1))
    rho = min(beta2, 1.0)
    return (1 - rho) * sample + rho * delta


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance (RiskMetrics lambda)."""
    vals = returns.values
    n, p = vals.shape
    cov = np.cov(vals[-60:].T, ddof=1) if n >= 60 else np.cov(vals.T, ddof=1)
    for row in vals:
        cov = lam * cov + (1 - lam) * np.outer(row, row)
    return cov * 252


# ─── Helpers ────────────────────────────────────────────────────────────────

def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))


def portfolio_sharpe(w, mu, cov, rf=0.065):
    r = portfolio_return(w, mu)
    v = portfolio_vol(w, cov)
    return (r - rf) / v if v > 0 else 0.0


def get_constraints_bounds(n: int, min_w: float = 0.02, max_w: float = 0.40, long_only: bool = True):
    bounds = [(min_w, max_w)] * n if long_only else [(-max_w, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    return bounds, constraints


# ─── Optimization methods ───────────────────────────────────────────────────

def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.065,
               min_w: float = 0.02, max_w: float = 0.40) -> np.ndarray:
    n = len(mu)
    bounds, constraints = get_constraints_bounds(n, min_w, max_w)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        return -(r - rf) / v if v > 0 else 0.0

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})
    return res.x / res.x.sum()


def min_variance(cov: np.ndarray, min_w: float = 0.02, max_w: float = 0.40) -> np.ndarray:
    n = cov.shape[0]
    bounds, constraints = get_constraints_bounds(n, min_w, max_w)
    w0 = np.ones(n) / n

    def port_var(w):
        return w @ cov @ w

    res = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})
    return res.x / res.x.sum()


def mean_variance(mu: np.ndarray, cov: np.ndarray, target_ret: Optional[float] = None,
                  min_w: float = 0.02, max_w: float = 0.40) -> np.ndarray:
    n = len(mu)
    bounds, constraints = get_constraints_bounds(n, min_w, max_w)
    if target_ret is not None:
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target_ret})
    w0 = np.ones(n) / n

    def port_var(w):
        return w @ cov @ w

    res = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})
    return res.x / res.x.sum()


def equal_risk_contribution(cov: np.ndarray, min_w: float = 0.02, max_w: float = 0.40) -> np.ndarray:
    """Equal risk contribution (risk parity) via convex optimization."""
    n = cov.shape[0]
    w0 = np.ones(n) / n

    def erc_objective(w):
        sigma = np.sqrt(w @ cov @ w)
        mrc = cov @ w / sigma          # marginal risk contributions
        rc = w * mrc                   # risk contributions
        target = sigma / n
        return np.sum((rc - target) ** 2)

    bounds = [(max(0.001, min_w), max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(erc_objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 2000, "ftol": 1e-10})
    return res.x / res.x.sum()


def hierarchical_risk_parity(returns: pd.DataFrame) -> np.ndarray:
    """
    Lopez de Prado's HRP algorithm:
    1. Compute correlation / distance matrix
    2. Hierarchical clustering
    3. Quasi-diagonalisation
    4. Recursive bisection allocation
    """
    cov = returns.cov().values
    corr = returns.corr().values
    n = len(corr)

    # Distance matrix
    dist = np.sqrt((1 - corr) / 2)
    dist = np.clip(dist, 0, None)
    np.fill_diagonal(dist, 0)

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # Quasi-diagonalisation (sort order from dendrogram)
    from scipy.cluster.hierarchy import leaves_list
    sort_ix = leaves_list(link)

    # Recursive bisection
    w = np.ones(n)
    clusters = [list(range(n))]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) == 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]

            def cluster_var(idx):
                sub_cov = cov[np.ix_(idx, idx)]
                sub_w = min_variance(sub_cov, min_w=0.0, max_w=1.0)
                return float(sub_w @ sub_cov @ sub_w)

            alpha = 1 - cluster_var(left) / (cluster_var(left) + cluster_var(right) + 1e-10)
            w[left] *= alpha
            w[right] *= (1 - alpha)
            new_clusters += [left, right]
        clusters = new_clusters

    return w / w.sum()


def black_litterman(
    mu_market: np.ndarray,
    cov: np.ndarray,
    P: np.ndarray,          # K×N views matrix
    Q: np.ndarray,          # K views vector
    Omega: Optional[np.ndarray] = None,
    tau: float = 0.05,
    rf: float = 0.065,
    delta: float = 2.5,
    min_w: float = 0.02,
    max_w: float = 0.40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Black-Litterman model.
    Returns (posterior_mu, optimal_weights).
    """
    n = len(mu_market)
    if Omega is None:
        Omega = np.diag(np.diag(tau * P @ cov @ P.T))

    tau_cov = tau * cov
    A = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P)
    b = np.linalg.inv(tau_cov) @ mu_market + P.T @ np.linalg.inv(Omega) @ Q
    mu_bl = A @ b

    w = max_sharpe(mu_bl, cov, rf, min_w, max_w)
    return mu_bl, w


# ─── Efficient frontier ──────────────────────────────────────────────────────

def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 50,
    min_w: float = 0.0,
    max_w: float = 1.0,
) -> pd.DataFrame:
    """Sweep target returns to build the efficient frontier."""
    ret_min = mu.min() * 1.05
    ret_max = mu.max() * 0.95
    target_rets = np.linspace(ret_min, ret_max, n_points)
    records = []
    for tr in target_rets:
        try:
            w = mean_variance(mu, cov, target_ret=tr, min_w=min_w, max_w=max_w)
            records.append({
                "return": portfolio_return(w, mu),
                "vol": portfolio_vol(w, cov),
                "sharpe": (portfolio_return(w, mu) - 0.065) / portfolio_vol(w, cov),
                "weights": w,
            })
        except Exception:
            pass
    return pd.DataFrame(records)


# ─── Monte-Carlo random portfolios ──────────────────────────────────────────

def random_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    n: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    p = len(mu)
    for _ in range(n):
        w = rng.dirichlet(np.ones(p))
        records.append({
            "return": portfolio_return(w, mu),
            "vol": portfolio_vol(w, cov),
            "sharpe": (portfolio_return(w, mu) - 0.065) / portfolio_vol(w, cov),
        })
    return pd.DataFrame(records)
