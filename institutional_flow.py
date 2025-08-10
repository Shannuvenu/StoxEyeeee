import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
DEFAULT_UNIVERSE = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","LT.NS",
    "SBIN.NS","BHARTIARTL.NS","ITC.NS","AXISBANK.NS","KOTAKBANK.NS","HCLTECH.NS",
    "MARUTI.NS","TITAN.NS","BAJFINANCE.NS","SUNPHARMA.NS","POWERGRID.NS",
    "ULTRACEMCO.NS","NTPC.NS","ONGC.NS"
]
def _safe_last(series):
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return np.nan

def compute_flow_for_symbol(symbol: str) -> dict:
    """
    Heuristic Institutional Flow:
      - VolumeX = today's (or last bar) volume vs 30D average daily volume
      - DayChange% = (LastClose - TodayOpen) / TodayOpen * 100 (intraday direction)
      - Score = log(1+VolumeX) * sign(DayChange)
      - Flow tag:
           Strong Buy: VolumeX >= 2.5 and DayChange >= +2%
           Buy:        VolumeX >= 1.5 and DayChange >= +0.5%
           Strong Sell:VolumeX >= 2.5 and DayChange <= -2%
           Sell:       VolumeX >= 1.5 and DayChange <= -0.5%
           Neutral:    otherwise
    """
    try:
        daily = yf.download(symbol, period="60d", interval="1d", auto_adjust=False, progress=False)
        if daily is None or daily.empty or "Volume" not in daily.columns:
            return {"Symbol": symbol, "Price": np.nan, "DayChange%": np.nan,
                    "VolXAvg": np.nan, "Flow": "No Data", "Score": 0.0}

        avg_vol_30 = float(daily["Volume"].tail(30).mean())
        intraday = None
        for iv in ["5m", "15m", "1h"]:
            intraday = yf.download(symbol, period="1d", interval=iv, auto_adjust=False, progress=False)
            if intraday is not None and not intraday.empty:
                break

        if intraday is None or intraday.empty:
            last_close = _safe_last(daily["Close"])
            today_open = _safe_last(daily["Open"])
            today_vol = float(daily["Volume"].tail(1).iloc[0]) if not daily["Volume"].tail(1).isna().all() else np.nan
        else:
            last_close = _safe_last(intraday["Close"])
            today_open = float(intraday["Open"].iloc[0]) if "Open" in intraday.columns else np.nan
            today_vol = float(intraday["Volume"].sum()) if "Volume" in intraday.columns else np.nan

        volx = float(today_vol / avg_vol_30) if (avg_vol_30 and not np.isnan(today_vol)) else np.nan
        daychg = float(((last_close - today_open) / today_open) * 100.0) if (today_open and not np.isnan(last_close)) else np.nan
        sign = 0
        if not np.isnan(daychg):
            sign = 1 if daychg > 0 else (-1 if daychg < 0 else 0)
        base = 1.0 + (volx if not np.isnan(volx) and volx > 0 else 0)
        score = float(np.log(base) * sign)
        flow = "Neutral"
        if not (np.isnan(volx) or np.isnan(daychg)):
            if volx >= 2.5 and daychg >= 2.0:
                flow = "Strong Buy"
            elif volx >= 1.5 and daychg >= 0.5:
                flow = "Buy"
            elif volx >= 2.5 and daychg <= -2.0:
                flow = "Strong Sell"
            elif volx >= 1.5 and daychg <= -0.5:
                flow = "Sell"

        return {
            "Symbol": symbol,
            "Price": round(last_close, 2) if not np.isnan(last_close) else np.nan,
            "DayChange%": round(daychg, 2) if not np.isnan(daychg) else np.nan,
            "VolXAvg": round(volx, 2) if not np.isnan(volx) else np.nan,
            "Flow": flow,
            "Score": round(score, 3),
        }
    except Exception:
        return {"Symbol": symbol, "Price": np.nan, "DayChange%": np.nan,
                "VolXAvg": np.nan, "Flow": "Error", "Score": 0.0}

def get_flow_dashboard(symbols: list) -> pd.DataFrame:
    rows = [compute_flow_for_symbol(sym) for sym in symbols]
    df = pd.DataFrame(rows)
    df["AbsScore"] = df["Score"].abs()
    df = df.sort_values(["AbsScore", "VolXAvg"], ascending=[False, False]).drop(columns=["AbsScore"])
    return df

def default_universe():
    return DEFAULT_UNIVERSE.copy()