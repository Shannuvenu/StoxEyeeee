# risk_utils.py
from __future__ import annotations
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# --- Helpers -----------------------------------------------------------------

@st.cache_data(ttl=24 * 60 * 60)
def get_sector_map(symbols: List[str]) -> Dict[str, str]:
    """
    Returns {symbol: sector}. Falls back to 'Unknown' if sector is not available.
    Cached for 24h to avoid rate limits.
    """
    sector_map: Dict[str, str] = {}
    for s in symbols:
        sec = "Unknown"
        try:
            t = yf.Ticker(s)
            # Some versions expose .info lazily. Guard it carefully.
            info = {}
            try:
                info = t.get_info()
            except Exception:
                # older yfinance: fallback
                info = getattr(t, "info", {}) or {}
            if isinstance(info, dict):
                sec = (info.get("sector") or info.get("industry") or "Unknown") or "Unknown"
        except Exception:
            pass
        sector_map[s] = sec or "Unknown"
    return sector_map


def _ensure_values(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Current Value column exists."""
    out = df.copy()
    if "Current Value" not in out.columns and {"Quantity", "Live Price"}.issubset(out.columns):
        out["Current Value"] = out["Quantity"] * out["Live Price"]
    return out


def sector_exposure(df: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    """
    Returns a dataframe with columns: Sector, Value, WeightPct
    """
    d = _ensure_values(df)
    d["Sector"] = d["Symbol"].map(sector_map).fillna("Unknown")
    grouped = (
        d.groupby("Sector", dropna=False)["Current Value"]
        .sum()
        .reset_index()
        .rename(columns={"Current Value": "Value"})
        .sort_values("Value", ascending=False)
    )
    total = grouped["Value"].sum() or 1.0
    grouped["WeightPct"] = (grouped["Value"] / total) * 100.0
    return grouped


def position_weights(df: pd.DataFrame) -> pd.Series:
    """Returns a series {symbol: weight %} sorted desc."""
    d = _ensure_values(df)
    total = d["Current Value"].sum() or 1.0
    w = (d.set_index("Symbol")["Current Value"] / total * 100.0).sort_values(ascending=False)
    return w


@st.cache_data(ttl=6 * 60 * 60)
def estimate_volatility(symbol: str, lookback: str = "6mo", interval: str = "1d") -> float:
    """
    Annualized volatility (stdev of daily returns * sqrt(252)) as percentage.
    Uses yfinance directly to stay module-local.
    """
    try:
        df = yf.download(symbol, period=lookback, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return float("nan")
        rets = df["Close"].pct_change().dropna()
        vol = float(rets.std() * math.sqrt(252) * 100.0)
        return vol
    except Exception:
        return float("nan")


def build_risk_summary(
    weights: pd.Series,
    sectors: pd.DataFrame,
    vol_map: Dict[str, float],
) -> List[Tuple[str, str]]:
    """
    Returns a list of (level, message).
    level ∈ {'info','warning','error'}
    """
    messages: List[Tuple[str, str]] = []

    # Single-stock concentration
    if not weights.empty:
        top1_symbol = weights.index[0]
        top1 = float(weights.iloc[0])
        if top1 >= 25:
            messages.append(("warning", f"High single-stock concentration: **{top1_symbol} {top1:.1f}%**."))

        top3 = float(weights.iloc[:3].sum())
        if top3 >= 60:
            messages.append(("warning", f"Top 3 positions together are **{top3:.1f}%** of your portfolio."))

    # Sector concentration
    for _, row in sectors.iterrows():
        if float(row["WeightPct"]) >= 40:
            messages.append(
                ("warning", f"Sector concentration risk: **{row['Sector']} {row['WeightPct']:.1f}%**.")
            )

    # Volatility risk: flag positions with high vol & meaningful weight
    for sym, w in weights.items():
        vol = vol_map.get(sym, float("nan"))
        if np.isfinite(vol) and w >= 10 and vol >= 35:
            messages.append(
                ("warning", f"High volatility exposure: **{sym}** vol **{vol:.1f}%** with weight **{w:.1f}%**.")
            )

    if not messages:
        messages.append(("info", "Risk looks balanced. No major concentration or volatility flags."))
    return messages
