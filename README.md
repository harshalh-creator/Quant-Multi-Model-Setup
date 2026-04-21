# 📊 Quant Finance Platform

**Advanced quantitative finance platform for Indian equity markets (NSE/BSE) with live data.**

> Portfolio Optimizer · Monte Carlo Simulator · Market Regime Detection · Factor Screener · Risk Analytics

---

## Features

| Module | What it does |
|---|---|
| **Portfolio Optimizer** | MVO, HRP, Max Sharpe, Min Variance, Risk Parity, Black-Litterman with Ledoit-Wolf / EWMA covariance |
| **Monte Carlo Simulator** | GBM, Merton Jump-Diffusion, Heston Stochastic Vol, GARCH(1,1) with antithetic variates |
| **Regime Detection** | HMM, Composite multi-signal (MA, MACD, RSI, momentum, volatility), Z-score |
| **Factor Screener** | Fama-French style 6-factor scoring: Value, Momentum, Quality, Growth, Low Vol, Size |
| **Risk Analytics** | VaR/CVaR (Historical, Parametric, Cornish-Fisher, EVT), stress tests, rolling metrics |

All data is fetched live from **Yahoo Finance** via `yfinance`.

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/quant-finance-platform.git
cd quant-finance-platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## Project Structure

```
quant-finance-platform/
├── app.py                      # Main entry point & navigation
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml             # Dark theme + server config
├── pages/
│   ├── __init__.py
│   ├── dashboard.py            # Live market dashboard
│   ├── optimizer.py            # Portfolio optimizer
│   ├── montecarlo.py           # Monte Carlo simulator
│   ├── regime.py               # Regime detection
│   ├── factor_screener.py      # Factor screener
│   └── risk_analytics.py       # Risk analytics
└── utils/
    ├── __init__.py
    ├── data.py                 # Live data fetching & metrics
    ├── optimizer.py            # Optimization engine (MVO, HRP, BL...)
    ├── monte_carlo.py          # Simulation engine
    ├── regime.py               # Regime detection engine
    └── risk.py                 # Risk metrics (VaR, CVaR, stress...)
```

---

## Deploying to GitHub

```bash
# 1. Initialize git (if not done)
git init
git add .
git commit -m "Initial commit: Quant Finance Platform"

# 2. Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/quant-finance-platform.git
git branch -M main
git push -u origin main
```

**Important:** Add a `.gitignore` file:

```
venv/
__pycache__/
*.pyc
.env
.DS_Store
*.egg-info/
dist/
```

---

## Deploying to Streamlit Cloud

1. Push your code to GitHub (above)
2. Go to **https://share.streamlit.io**
3. Click **"New app"**
4. Select:
   - Repository: `your-username/quant-finance-platform`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**

Streamlit Cloud will automatically install `requirements.txt`.

**Free tier:** 1 GB RAM, always-on, public URL like:
`https://your-username-quant-finance-platform-app-XXXXX.streamlit.app`

---

## Configuration

Edit `.streamlit/config.toml` to change the theme or server settings.

**Risk-free rate** and **historical period** can be adjusted in the sidebar at runtime.

---

## Data Sources

- **Prices**: Yahoo Finance via `yfinance` — NSE tickers end in `.NS`, BSE in `.BO`
- **Indices**: `^NSEI` (NIFTY 50), `^BSESN` (SENSEX), `^NSEBANK` (NIFTY Bank)
- **Fundamentals**: Yahoo Finance `ticker.info` (P/E, P/B, ROE, etc.)
- **Cache**: 15-minute Streamlit cache (`@st.cache_data(ttl=900)`)

---

## Adding Your Own Tickers

In `utils/data.py`, add tickers to `NSE_LARGE_CAP` or `NSE_MID_CAP` lists:

```python
NSE_LARGE_CAP = [
    "RELIANCE.NS", "TCS.NS", ...
    "YOURTICKER.NS",   # ← add here
]
```

---

## Strategy Notes

### Optimizer methods
- **MVO (Markowitz)**: Classic mean-variance — sensitive to return estimates
- **HRP**: Hierarchical Risk Parity — robust, no matrix inversion, handles ill-conditioning
- **Max Sharpe**: Best risk-adjusted return — recommended for momentum portfolios
- **Min Variance**: Lowest risk — best in bearish/sideways regimes
- **Risk Parity (ERC)**: Equal risk contribution — diversified, regime-robust
- **Black-Litterman**: Blends CAPM equilibrium with analyst views

### Regime signals (composite model weights)
| Signal | Weight |
|---|---|
| Price vs SMA 200/50 | 35% |
| Golden/Death cross | 15% |
| MACD histogram | 10% |
| RSI | 10% |
| 3M + 6M Momentum | 20% |
| Volatility ratio | 10% |

### Factor weights (defaults)
| Factor | Default |
|---|---|
| Value (P/E, P/B) | 25% |
| Momentum (6-12M) | 25% |
| Quality (ROE, D/E) | 20% |
| Growth (EPS, Rev) | 15% |
| Low Volatility | 10% |
| Size | 5% |

---

## License

MIT License — free to use, modify, and deploy.
