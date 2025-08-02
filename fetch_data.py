import yfinance as yf
import pandas as pd
import streamlit as st

def get_stock_data(symbol, period="5d", interval="1d"):
    try:
        data = yf.download(symbol, period=period, interval=interval, group_by="ticker", auto_adjust=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data[symbol]
                data.dropna(inplace=True)
            except KeyError:
                data.columns = data.columns.get_level_values(-1) 
        else:
            data = data.dropna()
        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(set(data.columns)):
            st.error(f"⚠️ Required columns not found in data: {data.columns.tolist()}")
            return None
        return data
    except Exception as e:
        st.error(f"❌ Error fetching data for {symbol}: {e}")
        return None
def get_realtime_price(symbol):
    data = yf.download(symbol, period="1d", interval="1m")
    if not data.empty and "Close" in data.columns:
        return float(data["Close"].iloc[-1]) 
    return None
