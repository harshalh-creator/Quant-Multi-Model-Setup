"""
utils/data.py — Live data fetching from Yahoo Finance
Handles NSE (.NS), BSE (.BO), and global tickers
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict


# ─── Popular NSE tickers for quick selection ───────────────────────────────
NSE_LARGE_CAP = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "KOTAKBANK.NS", "ITC.NS", "LTIM.NS", "AXISBANK.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "TITAN.NS", "NESTLEIND.NS", "WIPRO.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
    "TECHM.NS", "ULTRACEMCO.NS", "TATAMOTORS.NS", "ADANIENT.NS",
]

NSE_MID_CAP = [
    "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS", "KPITTECH.NS",
    "POLYCAB.NS", "SCHAEFFLER.NS", "ABFRL.NS", "GLENMARK.NS", "LALPATHLAB.NS",
    "METROPOLIS.NS", "DIVI.NS", "ALKEM.NS", "TORNTPHARM.NS", "PAGEIND.NS",
]

INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY Midcap 100": "NIFTY_MIDCAP_100.NS",
    "S&P 500": "^GSPC",
    "Gold (INR)": "GOLDBEES.NS",
    "USD/INR": "USDINR=X",
}

SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking",
    "INFY.NS": "IT", "ICICIBANK.NS": "Banking", "HINDUNILVR.NS": "FMCG",
    "SBIN.NS": "Banking", "BAJFINANCE.NS": "NBFC", "BHARTIARTL.NS": "Telecom",
    "KOTAKBANK.NS": "Banking", "ITC.NS": "FMCG", "AXISBANK.NS": "Banking",
    "ASIANPAINT.NS": "Consumer", "MARUTI.NS": "Auto", "TITAN.NS": "Consumer",
    "NESTLEIND.NS": "FMCG", "WIPRO.NS": "IT", "HCLTECH.NS": "IT",
    "SUNPHARMA.NS": "Pharma", "TATASTEEL.NS": "Metals", "TATAMOTORS.NS": "Auto",
}


@st.cache_data(ttl=900, show_spinner=False)   # 15-min cache
def fetch_prices(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers."""
    try:
        raw = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = tickers
        prices.dropna(how="all", inplace=True)
        # Drop columns with >20% missing
        threshold = len(prices) * 0.8
        prices = prices.dropna(axis=1, thresh=int(threshold))
        return prices
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_info(ticker: str) -> Dict:
    """Fetch fundamental info for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", SECTOR_MAP.get(ticker, "Unknown")),
            "pe": info.get("trailingPE", None),
            "pb": info.get("priceToBook", None),
            "roe": info.get("returnOnEquity", None),
            "eps_growth": info.get("earningsGrowth", None),
            "rev_growth": info.get("revenueGrowth", None),
            "debt_equity": info.get("debtToEquity", None),
            "market_cap": info.get("marketCap", None),
            "beta": info.get("beta", None),
            "div_yield": info.get("dividendYield", None),
            "current_price": info.get("currentPrice", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
        }
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_index(index_ticker: str, period: str = "2y") -> pd.Series:
    """Fetch a single index series."""
    try:
        data = yf.download(index_ticker, period=period, auto_adjust=True, progress=False)
        return data["Close"].squeeze()
    except Exception:
        return pd.Series(dtype=float)


def compute_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Compute log or simple returns."""
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def compute_rolling_stats(
    returns: pd.Series, window: int = 21
) -> Tuple[pd.Series, pd.Series]:
    """Rolling mean return and volatility (annualised)."""
    roll_mean = returns.rolling(window).mean() * 252
    roll_vol = returns.rolling(window).std() * np.sqrt(252)
    return roll_mean, roll_vol


def compute_drawdown(prices: pd.Series) -> pd.Series:
    """Compute drawdown series from price series."""
    peak = prices.cummax()
    return (prices - peak) / peak


def annualised_return(returns: pd.Series) -> float:
    total = (1 + returns).prod()
    n_years = len(returns) / 252
    return float(total ** (1 / n_years) - 1) if n_years > 0 else 0.0


def annualised_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))


def sharpe(returns: pd.Series, rf: float = 0.065) -> float:
    er = annualised_return(returns)
    vol = annualised_vol(returns)
    return (er - rf) / vol if vol > 0 else 0.0


def max_drawdown(prices: pd.Series) -> float:
    return float(compute_drawdown(prices).min())


def calmar(returns: pd.Series, prices: pd.Series) -> float:
    ar = annualised_return(returns)
    mdd = abs(max_drawdown(prices))
    return ar / mdd if mdd > 0 else 0.0


def sortino(returns: pd.Series, rf: float = 0.065) -> float:
    daily_rf = rf / 252
    excess = returns - daily_rf
    downside = excess[excess < 0].std() * np.sqrt(252)
    er = annualised_return(returns)
    return (er - rf) / downside if downside > 0 else 0.0
