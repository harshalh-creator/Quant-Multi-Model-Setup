"""
utils/monte_carlo.py — Monte Carlo Simulation Engine
Models: GBM, Merton Jump-Diffusion, Heston Stochastic Vol, GARCH(1,1)
Variance reduction: Antithetic variates, quasi-random (Sobol)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class MCParams:
    S0: float = 1_000_000          # Starting portfolio value
    mu: float = 0.12               # Annual drift
    sigma: float = 0.18            # Annual volatility
    T: int = 10                    # Years
    n_sims: int = 1_000            # Number of paths
    annual_contrib: float = 0.0    # Annual SIP/contribution
    model: str = "gbm"             # gbm | jump | heston | garch
    variance_reduction: str = "antithetic"  # none | antithetic
    seed: int = 42
    # Jump params (Merton)
    jump_intensity: float = 5.0    # jumps per year
    jump_mean: float = -0.02       # mean log jump size
    jump_std: float = 0.05         # std of log jump size
    # Heston params
    kappa: float = 2.0             # mean-reversion speed
    theta: float = 0.04            # long-run variance
    xi: float = 0.30               # vol of vol
    rho_heston: float = -0.70      # correlation between W_S and W_v
    # GARCH params
    omega: float = 0.00001
    alpha_g: float = 0.10
    beta_g: float = 0.85


@dataclass
class MCResults:
    terminal_values: np.ndarray
    paths: np.ndarray              # shape (n_sample_paths, T*steps_per_year+1)
    time_axis: np.ndarray
    percentiles: Dict[int, float] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)


def _gbm_paths(p: MCParams, rng: np.random.Generator) -> np.ndarray:
    steps = p.T * 252
    dt = 1 / 252
    drift = (p.mu - 0.5 * p.sigma ** 2) * dt
    diff = p.sigma * np.sqrt(dt)

    n = p.n_sims
    if p.variance_reduction == "antithetic":
        half = n // 2
        Z_half = rng.standard_normal((half, steps))
        Z = np.vstack([Z_half, -Z_half])
        n = len(Z)
    else:
        Z = rng.standard_normal((n, steps))

    log_returns = drift + diff * Z
    log_cum = np.cumsum(log_returns, axis=1)
    paths = p.S0 * np.exp(np.hstack([np.zeros((n, 1)), log_cum]))

    # Add annual contribution
    if p.annual_contrib > 0:
        for yr in range(1, p.T + 1):
            idx = yr * 252
            if idx < paths.shape[1]:
                paths[:, idx:] += p.annual_contrib

    return paths


def _jump_diffusion_paths(p: MCParams, rng: np.random.Generator) -> np.ndarray:
    """Merton (1976) jump-diffusion model."""
    steps = p.T * 252
    dt = 1 / 252
    lam = p.jump_intensity
    mu_j = p.jump_mean
    sig_j = p.jump_std

    # Compensator
    compensator = lam * (np.exp(mu_j + 0.5 * sig_j ** 2) - 1)
    drift = (p.mu - 0.5 * p.sigma ** 2 - compensator) * dt
    diff = p.sigma * np.sqrt(dt)

    n = p.n_sims
    Z = rng.standard_normal((n, steps))
    N = rng.poisson(lam * dt, (n, steps))   # Poisson jump counts
    J = rng.normal(mu_j, sig_j, (n, steps)) * N

    log_returns = drift + diff * Z + J
    log_cum = np.cumsum(log_returns, axis=1)
    paths = p.S0 * np.exp(np.hstack([np.zeros((n, 1)), log_cum]))

    if p.annual_contrib > 0:
        for yr in range(1, p.T + 1):
            idx = yr * 252
            if idx < paths.shape[1]:
                paths[:, idx:] += p.annual_contrib

    return paths


def _heston_paths(p: MCParams, rng: np.random.Generator) -> np.ndarray:
    """Heston (1993) stochastic volatility model (Euler-Maruyama)."""
    steps = p.T * 252
    dt = 1 / 252
    n = p.n_sims

    S = np.full(n, p.S0, dtype=float)
    v = np.full(n, p.theta, dtype=float)

    paths = [S.copy()]

    for _ in range(steps):
        Z1 = rng.standard_normal(n)
        Z2 = p.rho_heston * Z1 + np.sqrt(1 - p.rho_heston ** 2) * rng.standard_normal(n)

        v_pos = np.maximum(v, 0)
        dS = S * (p.mu * dt + np.sqrt(v_pos * dt) * Z1)
        dv = p.kappa * (p.theta - v_pos) * dt + p.xi * np.sqrt(v_pos * dt) * Z2

        S = np.maximum(S + dS, 1e-6)
        v = np.maximum(v + dv, 0)
        paths.append(S.copy())

    arr = np.column_stack(paths)

    if p.annual_contrib > 0:
        for yr in range(1, p.T + 1):
            idx = yr * 252
            if idx < arr.shape[1]:
                arr[:, idx:] += p.annual_contrib

    return arr


def _garch_paths(p: MCParams, rng: np.random.Generator) -> np.ndarray:
    """GARCH(1,1) volatility model."""
    steps = p.T * 252
    dt = 1 / 252
    n = p.n_sims

    S = np.full(n, p.S0, dtype=float)
    h = np.full(n, p.sigma ** 2 * dt, dtype=float)
    paths = [S.copy()]

    for _ in range(steps):
        Z = rng.standard_normal(n)
        eps = np.sqrt(h) * Z
        ret = p.mu * dt + eps
        S = S * np.exp(ret)
        h = p.omega + p.alpha_g * eps ** 2 + p.beta_g * h
        h = np.clip(h, 1e-8, 1.0)
        paths.append(S.copy())

    arr = np.column_stack(paths)

    if p.annual_contrib > 0:
        for yr in range(1, p.T + 1):
            idx = yr * 252
            if idx < arr.shape[1]:
                arr[:, idx:] += p.annual_contrib

    return arr


def run_simulation(p: MCParams) -> MCResults:
    """Master simulation runner."""
    rng = np.random.default_rng(p.seed)

    model_map = {
        "gbm": _gbm_paths,
        "jump": _jump_diffusion_paths,
        "heston": _heston_paths,
        "garch": _garch_paths,
    }
    fn = model_map.get(p.model, _gbm_paths)
    paths = fn(p, rng)                  # shape (n_sims, steps+1)

    terminal = paths[:, -1]
    pcts = {5: np.percentile(terminal, 5), 10: np.percentile(terminal, 10),
            25: np.percentile(terminal, 25), 50: np.percentile(terminal, 50),
            75: np.percentile(terminal, 75), 90: np.percentile(terminal, 90),
            95: np.percentile(terminal, 95)}

    median = pcts[50]
    cagr = (median / p.S0) ** (1 / p.T) - 1

    # VaR / CVaR on annual returns (1-year horizon)
    one_year_ret = (paths[:, 252] / p.S0) - 1 if paths.shape[1] > 252 else (terminal / p.S0) - 1
    var_95 = np.percentile(one_year_ret, 5)
    cvar_95 = one_year_ret[one_year_ret <= var_95].mean()

    stats = {
        "median": median,
        "mean": terminal.mean(),
        "std": terminal.std(),
        "p5": pcts[5],
        "p95": pcts[95],
        "cagr": cagr,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "p_profit": (terminal > p.S0).mean(),
        "p_2x": (terminal > p.S0 * 2).mean(),
        "p_5x": (terminal > p.S0 * 5).mean(),
        "p_ruin": (terminal < p.S0 * 0.5).mean(),
    }

    # Sub-sample paths for plotting (max 200)
    n_plot = min(200, len(paths))
    idx = rng.choice(len(paths), n_plot, replace=False)
    sample_paths = paths[idx, :]

    # Downsample time axis to yearly
    total_steps = paths.shape[1]
    t_axis = np.linspace(0, p.T, total_steps)

    return MCResults(
        terminal_values=terminal,
        paths=sample_paths,
        time_axis=t_axis,
        percentiles=pcts,
        stats=stats,
    )


def terminal_distribution_bins(
    results: MCResults, n_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Return bin edges and counts for histogram."""
    counts, edges = np.histogram(results.terminal_values, bins=n_bins)
    return edges, counts
