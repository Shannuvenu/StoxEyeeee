import yfinance as yf
import pandas as pd
def get_stock_data(symbol,period="5d",interval="1h"):
    try:
        df=yf.download(symbol,period=period,interval=interval)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"error fetching data:{e}")
        return None