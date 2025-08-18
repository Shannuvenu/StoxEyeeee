import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

def fetch_data(symbols):
    """
    Download adjusted close prices for given symbols (1y daily).
    Returns a wide DataFrame of Close prices.
    """
    df = yf.download(
        symbols, period="1y", interval="1d",
        auto_adjust=True, progress=False
    )["Close"]

    if isinstance(df, pd.Series):  
        df = df.to_frame()

    df = df.dropna(how="all").ffill().dropna()
    return df

def _annualized_stats(price_df: pd.DataFrame):
    """
    Compute annualized mean returns and covariance from daily returns.
    """
    ret = price_df.pct_change().dropna()
    mu = ret.mean().values * 252.0               
    cov = ret.cov().values * 252.0               
    tickers = list(price_df.columns)
    return mu, cov, tickers

def optimize_portfolio(price_df: pd.DataFrame, target_return=None):
    """
    Mean-variance optimization using SciPy (SLSQP):
      - weights >= 0 (no short)
      - sum(weights) = 1
      - if target_return is provided, enforce w @ mu >= target_return
    Returns dict with weights, expected_return, expected_risk, tickers.
    """
    mu, cov, tickers = _annualized_stats(price_df)
    n = len(tickers)

    def variance(w):
        return float(w @ cov @ w)


    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        cons.append({"type": "ineq", "fun": lambda w, mu=mu: w @ mu - target_return})


    bounds = [(0.0, 1.0)] * n

    w0 = np.ones(n) / n

    res = minimize(
        variance, w0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 200}
    )

    w = res.x if res.success else w0
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))

    return {
        "weights": w,
        "expected_return": port_ret,
        "expected_risk": port_vol,
        "tickers": tickers,
    }