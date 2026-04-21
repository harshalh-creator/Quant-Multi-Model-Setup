"""
India Pro Quant Engine — core module used by all Streamlit pages.
All bugs from v1 fixed. Import this everywhere.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ─────────────────────────────────────────────────────────────
class Config:
    NSE_UNIVERSE = [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK",
        "HINDUNILVR","ITC","KOTAKBANK","LT","SBIN",
        "AXISBANK","BAJFINANCE","BHARTIARTL","MARUTI","TITAN",
        "WIPRO","ULTRACEMCO","ASIANPAINT","TECHM","SUNPHARMA",
        "NTPC","POWERGRID","ADANIENT","ONGC","COALINDIA",
        "HINDALCO","TATASTEEL","JSWSTEEL","TATAMOTOR","BAJAJFINSV",
        "DIVISLAB","DRREDDY","CIPLA","PIDILITIND","MUTHOOTFIN",
        "NAUKRI","LTIM","HCLTECH","PERSISTENT","MPHASIS",
    ]
    SECTOR_MAP = {
        "RELIANCE":"Energy","TCS":"IT","HDFCBANK":"Banking","INFY":"IT",
        "ICICIBANK":"Banking","HINDUNILVR":"FMCG","ITC":"FMCG","KOTAKBANK":"Banking",
        "LT":"Infra","SBIN":"PSU Bank","AXISBANK":"Banking","BAJFINANCE":"Finance",
        "BHARTIARTL":"Telecom","MARUTI":"Auto","TITAN":"Consumer","WIPRO":"IT",
        "ULTRACEMCO":"Cement","ASIANPAINT":"Consumer","TECHM":"IT","SUNPHARMA":"Pharma",
        "NTPC":"PSU Energy","POWERGRID":"PSU Energy","ADANIENT":"Conglomerate",
        "ONGC":"Energy","COALINDIA":"Energy","HINDALCO":"Metals","TATASTEEL":"Metals",
        "JSWSTEEL":"Metals","TATAMOTOR":"Auto","BAJAJFINSV":"Finance",
        "DIVISLAB":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","PIDILITIND":"Chemicals",
        "MUTHOOTFIN":"Finance","NAUKRI":"IT","LTIM":"IT","HCLTECH":"IT",
        "PERSISTENT":"IT","MPHASIS":"IT",
    }
    NIFTY50_INDEX = "^NSEI"
    BANKNIFTY     = "^NSEBANK"

    EMA_FAST, EMA_MID, EMA_SLOW, EMA_200 = 9, 21, 50, 200
    RSI_LEN    = 14
    RSI_OB     = 70
    RSI_OS     = 30
    MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
    ATR_LEN    = 14
    BB_LEN     = 20
    BB_STD     = 2.0
    ADX_LEN    = 14
    ADX_THRESH = 25
    VOL_MA     = 20
    BREAKOUT_LOOKBACK = 20

    PORTFOLIO_SIZE  = 1_000_000
    RISK_PER_TRADE  = 0.015
    SWING_ATR_SL    = 2.0
    SWING_ATR_TGT   = 3.0
    COMMISSION      = 0.0003
    STT_DELIVERY    = 0.001
    TOTAL_COST      = 0.0003 + 0.001

    MC_SIMULATIONS  = 10_000
    MC_HORIZON_DAYS = 60


# ─────────────────────────────────────────────────────────────
class DataFetcher:

    @staticmethod
    def get_stock(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        if not YF_OK:
            return pd.DataFrame()
        sym = ticker if (ticker.endswith(".NS") or ticker.startswith("^")) else ticker + ".NS"
        try:
            df = yf.download(sym, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def generate_synthetic(n_bars: int = 600, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        dates = pd.bdate_range("2023-01-01", periods=n_bars)
        mu    = np.where(np.arange(n_bars) < n_bars//2, 0.0005, -0.0001)
        log_r = np.random.normal(mu, 0.012)
        close = 24000 * np.exp(np.cumsum(log_r))
        noise = np.random.uniform(0.003, 0.015, n_bars)
        high  = close * (1 + noise)
        low   = close * (1 - noise)
        open_ = np.roll(close, 1) * (1 + np.random.uniform(-0.004, 0.004, n_bars))
        open_[0] = close[0]
        vol   = np.random.lognormal(13.5, 0.6, n_bars).astype(int)
        return pd.DataFrame({"Open":open_,"High":high,"Low":low,"Close":close,"Volume":vol}, index=dates)


# ─────────────────────────────────────────────────────────────
class Indicators:

    @staticmethod
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def rsi(close, n=14):
        d    = close.diff()
        gain = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    @staticmethod
    def macd(close, fast=12, slow=26, sig=9):
        ml = Indicators.ema(close, fast) - Indicators.ema(close, slow)
        sl = Indicators.ema(ml, sig)
        return ml, sl, ml - sl

    @staticmethod
    def bollinger(close, n=20, k=2.0):
        mid = close.rolling(n).mean()
        std = close.rolling(n).std(ddof=1)
        u, l = mid + k*std, mid - k*std
        return u, mid, l, (u-l)/mid*100

    @staticmethod
    def atr(high, low, close, n=14):
        tr = pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
        return tr.ewm(alpha=1/n, adjust=False).mean()

    @staticmethod
    def adx(high, low, close, n=14):
        up, down = high.diff(), -low.diff()
        pdm = np.where((up>down)&(up>0), up.values, 0.0)
        mdm = np.where((down>up)&(down>0), down.values, 0.0)
        tr  = pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
        atr_w = tr.ewm(alpha=1/n, adjust=False).mean()
        pdi = 100*pd.Series(pdm,index=high.index).ewm(alpha=1/n,adjust=False).mean()/atr_w.replace(0,np.nan)
        mdi = 100*pd.Series(mdm,index=high.index).ewm(alpha=1/n,adjust=False).mean()/atr_w.replace(0,np.nan)
        dx  = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
        return dx.ewm(alpha=1/n,adjust=False).mean(), pdi, mdi

    @staticmethod
    def stochastic(high, low, close, k=14, d=3, smooth=3):
        lo, hi = low.rolling(k).min(), high.rolling(k).max()
        pk = 100*(close-lo)/(hi-lo+1e-9)
        pk = pk.rolling(smooth).mean()
        return pk, pk.rolling(d).mean()

    @staticmethod
    def obv(close, volume):
        return (np.sign(close.diff()).fillna(0)*volume).cumsum()

    @staticmethod
    def supertrend(high, low, close, mult=3.0, n=10):
        atr_v = Indicators.atr(high, low, close, n).values
        hl2   = ((high+low)/2).values
        c_arr = close.values
        upper, lower = hl2+mult*atr_v, hl2-mult*atr_v
        st, dirn = np.full(len(c_arr), np.nan), np.ones(len(c_arr), dtype=int)
        for i in range(1, len(c_arr)):
            if np.isnan(atr_v[i]): continue
            prev = st[i-1] if not np.isnan(st[i-1]) else lower[i]
            if dirn[i-1] == 1:
                st[i]   = max(lower[i], prev)
                dirn[i] = 1 if c_arr[i] >= st[i] else -1
            else:
                st[i]   = min(upper[i], prev)
                dirn[i] = -1 if c_arr[i] <= st[i] else 1
        return pd.Series(st, index=close.index), pd.Series(dirn, index=close.index)

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        for n in [Config.EMA_FAST, Config.EMA_MID, Config.EMA_SLOW, Config.EMA_200]:
            df[f"EMA{n}"] = Indicators.ema(c, n)
        df["RSI"] = Indicators.rsi(c, Config.RSI_LEN)
        df["MACD"], df["MACDsig"], df["MACDhist"] = Indicators.macd(c)
        df["BBupper"], df["BBmid"], df["BBlower"], df["BBwidth"] = Indicators.bollinger(c)
        df["BBsqueeze"] = df["BBwidth"] < df["BBwidth"].rolling(20).min().shift(1)*1.1
        df["ATR"]    = Indicators.atr(h, l, c, Config.ATR_LEN)
        df["ATRpct"] = df["ATR"]/c*100
        df["ADX"], df["PlusDI"], df["MinusDI"] = Indicators.adx(h, l, c, Config.ADX_LEN)
        df["StochK"], df["StochD"] = Indicators.stochastic(h, l, c)
        df["VolMA"]    = v.rolling(Config.VOL_MA).mean()
        df["VolRatio"] = v/df["VolMA"].replace(0, np.nan)
        df["VolSpike"] = df["VolRatio"] > 1.5
        df["OBV"]      = Indicators.obv(c, v)
        df["Supertrend"], df["StDir"] = Indicators.supertrend(h, l, c)
        df["VWAP"]     = (((h+l+c)/3)*v).cumsum()/v.cumsum()
        df["High20"]   = h.rolling(Config.BREAKOUT_LOOKBACK).max()
        df["Low20"]    = l.rolling(Config.BREAKOUT_LOOKBACK).min()
        df["BullRegime"] = (c > df["EMA200"]) & (df["EMA50"] > df["EMA200"])
        df["BearRegime"] = (c < df["EMA200"]) & (df["EMA50"] < df["EMA200"])
        return df.dropna(subset=["EMA200","ADX","ATR","RSI","MACDhist"])


# ─────────────────────────────────────────────────────────────
class Strategies:

    @staticmethod
    def swing(df):
        c = df["Close"]
        buy = ((df["EMA9"]>df["EMA21"])&(df["EMA21"]>df["EMA50"])&(c>df["EMA50"])&
               (df["RSI"]>40)&(df["RSI"]<Config.RSI_OB)&
               (df["MACDhist"]>0)&(df["MACDhist"].shift(1)<0)&
               df["VolSpike"]&(c>df["BBmid"])).astype(int)
        sell= ((df["EMA9"]<df["EMA21"])&(df["EMA21"]<df["EMA50"])&(c<df["EMA50"])&
               (df["RSI"]<60)&(df["RSI"]>Config.RSI_OS)&
               (df["MACDhist"]<0)&(df["MACDhist"].shift(1)>0)&
               df["VolSpike"]&(c<df["BBmid"])).astype(int)*-1
        return (buy+sell).clip(-1,1)

    @staticmethod
    def trend_following(df):
        c = df["Close"]
        buy = ((c>df["EMA200"])&(df["EMA50"]>df["EMA200"])&
               (df["EMA9"].shift(1)<df["EMA21"].shift(1))&(df["EMA9"]>df["EMA21"])&
               (df["ADX"]>Config.ADX_THRESH)&(df["RSI"]>50)&(df["RSI"]<75)).astype(int)
        sell= ((c<df["EMA200"])&(df["EMA50"]<df["EMA200"])&
               (df["EMA9"].shift(1)>df["EMA21"].shift(1))&(df["EMA9"]<df["EMA21"])&
               (df["ADX"]>Config.ADX_THRESH)).astype(int)*-1
        return (buy+sell).clip(-1,1)

    @staticmethod
    def momentum(df):
        c = df["Close"]
        buy = ((df["RSI"].shift(1)<50)&(df["RSI"]>=50)&
               (df["MACDhist"]>df["MACDhist"].shift(1))&(df["MACDhist"]>0)&
               (df["StochK"]>df["StochD"])&(df["StochK"]<80)&
               (c>df["EMA21"])&(df["VolRatio"]>1.2)).astype(int)
        sell= ((df["RSI"].shift(1)>50)&(df["RSI"]<=50)&
               (df["MACDhist"]<df["MACDhist"].shift(1))&(df["MACDhist"]<0)&
               (df["StochK"]<df["StochD"])&(df["StochK"]>20)).astype(int)*-1
        return (buy+sell).clip(-1,1)

    @staticmethod
    def breakout(df):
        c = df["Close"]
        buy = ((c>df["High20"].shift(1))&df["VolSpike"]&df["BBsqueeze"].shift(1)&
               (df["RSI"]>50)&(df["RSI"]<75)&(c>df["EMA50"])).astype(int)
        sell= ((c<df["Low20"].shift(1))&df["VolSpike"]&
               (df["RSI"]<50)&(df["RSI"]>25)&(c<df["EMA50"])).astype(int)*-1
        return (buy+sell).clip(-1,1)

    @staticmethod
    def mean_reversion(df):
        c = df["Close"]
        buy = ((df["RSI"]<Config.RSI_OS)&(c<df["BBlower"])&(df["StochK"]<20)&
               df["BullRegime"]&(df["MACDhist"]>df["MACDhist"].shift(1))).astype(int)
        sell= ((df["RSI"]>Config.RSI_OB)&(c>df["BBupper"])&(df["StochK"]>80)&
               (df["MACDhist"]<df["MACDhist"].shift(1))).astype(int)*-1
        return (buy+sell).clip(-1,1)

    @staticmethod
    def longterm_quality(df):
        c = df["Close"]
        buy = (df["BullRegime"]&(df["StDir"]==1)&
               (df["RSI"]>35)&(df["RSI"]<55)&(c>df["EMA50"])&df["VolSpike"]).astype(int)
        sell= (df["BearRegime"]&(df["StDir"]==-1)&(df["RSI"]>60)).astype(int)*-1
        return (buy+sell).clip(-1,1)

    ALL = {
        "Swing"         : swing.__func__ if hasattr(swing, '__func__') else swing,
        "Trend"         : trend_following.__func__ if hasattr(trend_following, '__func__') else trend_following,
        "Momentum"      : momentum.__func__ if hasattr(momentum, '__func__') else momentum,
        "Breakout"      : breakout.__func__ if hasattr(breakout, '__func__') else breakout,
        "Mean Reversion": mean_reversion.__func__ if hasattr(mean_reversion, '__func__') else mean_reversion,
        "Long-term"     : longterm_quality.__func__ if hasattr(longterm_quality, '__func__') else longterm_quality,
    }


# Fix ALL dict to use static methods properly
Strategies.ALL = {
    "Swing"         : Strategies.swing,
    "Trend"         : Strategies.trend_following,
    "Momentum"      : Strategies.momentum,
    "Breakout"      : Strategies.breakout,
    "Mean Reversion": Strategies.mean_reversion,
    "Long-term"     : Strategies.longterm_quality,
}


# ─────────────────────────────────────────────────────────────
class Backtester:

    def __init__(self, df, signals, atr_sl=Config.SWING_ATR_SL,
                 atr_tgt=Config.SWING_ATR_TGT, portfolio=Config.PORTFOLIO_SIZE):
        self.df        = df.copy()
        self.signals   = signals.reindex(df.index).fillna(0)
        self.atr_sl    = atr_sl
        self.atr_tgt   = atr_tgt
        self.portfolio = portfolio
        self.trades    = []

    def run(self) -> dict:
        df, sigs = self.df, self.signals
        cap = float(self.portfolio)
        pos = 0
        entry_price = sl = target = qty = 0.0
        fee = Config.TOTAL_COST
        for i in range(len(df)):
            bar   = df.iloc[i]
            sig   = int(sigs.iloc[i])
            price = float(bar["Close"])
            atr   = float(bar["ATR"])
            if pos == 0 and sig == 1 and atr > 0:
                sl     = price - self.atr_sl * atr
                target = price + self.atr_tgt * atr
                if price <= sl: continue
                qty = int((cap * Config.RISK_PER_TRADE) / (price - sl))
                if qty <= 0: continue
                cost = qty * price * (1 + fee)
                if cost > cap:
                    qty = int(cap / (price * (1 + fee)))
                    cost = qty * price * (1 + fee)
                if qty <= 0: continue
                entry_price = price; pos = qty; cap -= cost
                self.trades.append(dict(date_in=df.index[i], entry=entry_price,
                    sl=sl, target=target, qty=qty, date_out=None,
                    exit=None, pnl=None, pnl_pct=None, reason=None))
            elif pos > 0:
                ep, reason = None, None
                if price <= sl:       ep, reason = sl, "SL"
                elif price >= target: ep, reason = target, "TARGET"
                elif sig == -1:       ep, reason = price, "SIGNAL"
                if ep is not None:
                    proceeds = qty * ep * (1 - fee)
                    cost_in  = qty * entry_price * (1 + fee)
                    pnl      = proceeds - cost_in
                    cap += proceeds; pos = 0
                    self.trades[-1].update(dict(date_out=df.index[i], exit=ep,
                        pnl=pnl, pnl_pct=pnl/cost_in*100, reason=reason))
        if pos > 0:
            lp = float(df["Close"].iloc[-1])
            proceeds = pos * lp * (1 - fee)
            cost_in  = pos * entry_price * (1 + fee)
            pnl      = proceeds - cost_in
            cap += proceeds
            self.trades[-1].update(dict(date_out=df.index[-1], exit=lp,
                pnl=pnl, pnl_pct=pnl/cost_in*100, reason="MTM"))
        return self._metrics(cap)

    def _metrics(self, final_equity):
        done = [t for t in self.trades if t["pnl"] is not None]
        if not done:
            return {"error": "No completed trades"}
        pnls  = np.array([t["pnl"]     for t in done])
        pcts  = np.array([t["pnl_pct"] for t in done])
        wins  = pnls[pnls > 0]; losses = pnls[pnls <= 0]
        total_ret = (final_equity - self.portfolio) / self.portfolio * 100
        win_rate  = len(wins) / len(pnls) * 100
        avg_win   = float(np.mean(wins))   if len(wins)   > 0 else 0.0
        avg_loss  = float(np.mean(losses)) if len(losses) > 0 else 0.0
        pf  = abs(wins.sum()/losses.sum()) if len(losses)>0 and losses.sum()!=0 else np.inf
        rr  = abs(avg_win/avg_loss)         if avg_loss != 0 else np.inf
        n_years = max((done[-1]["date_out"] - done[0]["date_in"]).days / 365.25, 0.01)
        tpy     = len(done) / n_years
        mu      = pcts.mean() / 100; sig = pcts.std(ddof=1) / 100
        sharpe  = (mu/sig)*np.sqrt(tpy) if sig > 0 else 0.0
        neg     = pcts[pcts < 0] / 100
        sortino = (mu/neg.std(ddof=1))*np.sqrt(tpy) if len(neg) > 1 else 0.0
        eq      = self.portfolio + np.cumsum(pnls)
        peak    = np.maximum.accumulate(eq)
        max_dd  = float(((eq-peak)/peak*100).min())
        return dict(
            total_trades    = len(done),
            win_rate        = round(win_rate, 2),
            total_return_pct= round(total_ret, 2),
            profit_factor   = round(pf, 2) if pf != np.inf else "∞",
            risk_reward     = round(rr, 2)  if rr != np.inf else "∞",
            avg_win         = round(avg_win, 2),
            avg_loss        = round(avg_loss, 2),
            avg_win_pct     = round(float(pcts[pcts>0].mean()), 2) if len(wins)>0 else 0,
            avg_loss_pct    = round(float(pcts[pcts<=0].mean()), 2) if len(losses)>0 else 0,
            best_trade_pct  = round(float(pcts.max()), 2),
            worst_trade_pct = round(float(pcts.min()), 2),
            sharpe_ratio    = round(sharpe, 3),
            sortino_ratio   = round(sortino, 3),
            max_drawdown    = round(max_dd, 2),
            final_equity    = round(final_equity, 2),
            net_pnl         = round(final_equity - self.portfolio, 2),
            equity_curve    = eq.tolist(),
            dates           = [t["date_in"] for t in done],
            trades          = done,
        )


# ─────────────────────────────────────────────────────────────
class MonteCarlo:

    def __init__(self, returns, horizon=Config.MC_HORIZON_DAYS,
                 n_sims=Config.MC_SIMULATIONS, portfolio=Config.PORTFOLIO_SIZE):
        self.ret = returns.dropna().values.astype(float)
        self.H, self.N, self.P = horizon, n_sims, portfolio

    def bootstrap(self):
        paths = np.empty((self.N, self.H+1)); paths[:,0] = self.P
        for t in range(1, self.H+1):
            r = np.random.choice(self.ret, size=self.N, replace=True)
            paths[:,t] = paths[:,t-1] * (1.0 + r)
        return paths[:,-1]

    def gbm(self):
        mu, sigma = self.ret.mean(), self.ret.std(ddof=1)
        Z = np.random.standard_normal((self.N, self.H))
        return self.P * np.exp(((mu-0.5*sigma**2) + sigma*Z).sum(axis=1))

    def fat_tail(self):
        if not SCIPY_OK: return self.bootstrap()
        df_t, loc, scale = stats.t.fit(self.ret)
        curr = np.full(self.N, float(self.P))
        for _ in range(self.H):
            curr *= (1.0 + stats.t.rvs(df=df_t, loc=loc, scale=scale, size=self.N))
        return curr

    def _rm(self, arr):
        pct   = (arr - self.P) / self.P * 100
        var95 = float(np.percentile(pct, 5))
        var99 = float(np.percentile(pct, 1))
        cvar95= float(pct[pct<=var95].mean()) if (pct<=var95).any() else var95
        cvar99= float(pct[pct<=var99].mean()) if (pct<=var99).any() else var99
        return dict(
            mean_return   = round(float(pct.mean()), 2),
            median_return = round(float(np.median(pct)), 2),
            std           = round(float(pct.std()), 2),
            min=round(float(pct.min()),2), max=round(float(pct.max()),2),
            p5 =round(float(np.percentile(pct,5)),2),
            p25=round(float(np.percentile(pct,25)),2),
            p75=round(float(np.percentile(pct,75)),2),
            p95=round(float(np.percentile(pct,95)),2),
            VaR_95=round(var95,2), VaR_99=round(var99,2),
            CVaR_95=round(cvar95,2), CVaR_99=round(cvar99,2),
            prob_profit  = round(float((pct>0).mean()*100),1),
            prob_gt_10   = round(float((pct>10).mean()*100),1),
            prob_loss_10 = round(float((pct<-10).mean()*100),1),
            final_values = arr, pct_values = pct,
        )

    def run_all(self):
        return dict(
            bootstrap = self._rm(self.bootstrap()),
            gbm       = self._rm(self.gbm()),
            fat_tail  = self._rm(self.fat_tail()),
            horizon=self.H, n_sims=self.N,
        )


# ─────────────────────────────────────────────────────────────
class PositionSizer:

    @staticmethod
    def atr_based(price, atr, portfolio=Config.PORTFOLIO_SIZE,
                  risk_pct=Config.RISK_PER_TRADE, sl_mult=Config.SWING_ATR_SL):
        risk_amt = portfolio * risk_pct
        sl_pts   = atr * sl_mult
        qty      = int(risk_amt / sl_pts) if sl_pts > 0 else 0
        exposure = qty * price
        return dict(qty=qty, exposure=round(exposure,2),
                    exposure_pct=round(exposure/portfolio*100,2),
                    risk_amt=round(risk_amt,2), sl_pts=round(sl_pts,2))

    @staticmethod
    def kelly(win_rate, avg_win_pct, avg_loss_pct):
        if avg_loss_pct == 0: return 0.0
        b = abs(avg_win_pct / avg_loss_pct)
        p = win_rate / 100.0
        return max(0.0, round(((b*p-(1-p))/b)*0.5, 4))


# ─────────────────────────────────────────────────────────────
class Screener:

    def __init__(self, universe=None, period="2y"):
        self.universe = universe or Config.NSE_UNIVERSE[:20]
        self.period   = period

    def scan(self, strategy="swing") -> pd.DataFrame:
        fn_map = {
            "swing": Strategies.swing, "trend": Strategies.trend_following,
            "momentum": Strategies.momentum, "breakout": Strategies.breakout,
            "mean_reversion": Strategies.mean_reversion, "longterm": Strategies.longterm_quality,
        }
        fn   = fn_map.get(strategy, Strategies.swing)
        rows = []
        for ticker in self.universe:
            df = DataFetcher.get_stock(ticker, self.period)
            if df.empty or len(df) < 210: continue
            try:
                df   = Indicators.add_all(df)
                sigs = fn(df)
            except Exception: continue
            last = int(sigs.iloc[-1])
            if last == 0: continue
            row = df.iloc[-1]
            ps  = PositionSizer.atr_based(float(row["Close"]), float(row["ATR"]))
            sgn = np.sign(last)
            sl  = float(row["Close"]) - sgn * Config.SWING_ATR_SL * float(row["ATR"])
            tgt = float(row["Close"]) + sgn * Config.SWING_ATR_TGT * float(row["ATR"])
            rows.append(dict(
                Ticker    = ticker,
                Sector    = Config.SECTOR_MAP.get(ticker, "—"),
                Signal    = "BUY" if last==1 else "SELL",
                Price     = round(float(row["Close"]),2),
                RSI       = round(float(row["RSI"]),1),
                MACD_hist = round(float(row["MACDhist"]),3),
                ADX       = round(float(row["ADX"]),1),
                Vol_Ratio = round(float(row["VolRatio"]),2),
                ATR       = round(float(row["ATR"]),2),
                ATR_pct   = round(float(row["ATRpct"]),2),
                SL        = round(sl,2), Target=round(tgt,2),
                RR        = round(Config.SWING_ATR_TGT/Config.SWING_ATR_SL,2),
                Qty       = ps["qty"],
                Regime    = ("BULL" if row["BullRegime"] else "BEAR" if row["BearRegime"] else "NEUTRAL"),
            ))
        if not rows: return pd.DataFrame()
        out = pd.DataFrame(rows)
        out["Score"] = (
            (1-((out["RSI"]-50).abs()/50))*30 +
            out["ADX"].clip(15,50).sub(15).div(35)*25 +
            out["Vol_Ratio"].clip(1,4).sub(1).div(3)*25 +
            out["RR"].clip(1.5,4).sub(1.5).div(2.5)*20
        ).round(1)
        return out.sort_values("Score", ascending=False).reset_index(drop=True)
