import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
def fetch_data(symbols, period="1y"):
    data = yf.download(symbols, period=period, group_by='ticker', auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.loc[:, (slice(None), 'Adj Close')]
            data.columns = data.columns.droplevel(1)
        except KeyError:
            data = data.loc[:, (slice(None), 'Close')]
            data.columns = data.columns.droplevel(1)
    else:
        if "Adj Close" in data.columns:
            data = data["Adj Close"].to_frame()
        elif "Close" in data.columns:
            data = data["Close"].to_frame()
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in the data")
    return data.dropna()
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, risk
def optimize_portfolio(price_data):
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    n = len(mean_returns)
    weights = cp.Variable(n)
    risk = cp.quad_form(weights, cov_matrix)
    objective = cp.Minimize(risk)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    optimal_weights = weights.value
    expected_return, expected_risk = calculate_portfolio_performance(optimal_weights, mean_returns, cov_matrix)

    return {
        "weights": optimal_weights,
        "expected_return": expected_return,
        "expected_risk": expected_risk
    }
