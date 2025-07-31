import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp

def fetch_data(symbols, period="1y"):
    try:
        data = yf.download(symbols, period=period, group_by='ticker', auto_adjust=False, threads=True)

        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[1]:
                data = data.loc[:, (slice(None), 'Adj Close')]
            elif 'Close' in data.columns.levels[1]:
                data = data.loc[:, (slice(None), 'Close')]
            else:
                raise ValueError("Neither 'Adj Close' nor 'Close' found in data")

            data.columns = data.columns.droplevel(1)
        else:
            if "Adj Close" in data.columns:
                data = data[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
            elif "Close" in data.columns:
                data = data[["Close"]].rename(columns={"Close": symbols[0]})
            else:
                raise ValueError("No valid price columns found")
        return data.dropna()
    except Exception as e:
        print(f"[fetch_data] Error: {e}")
        return pd.DataFrame()
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    expected_return = np.dot(weights, mean_returns)
    expected_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return expected_return, expected_risk
def optimize_portfolio(price_data):
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)
    weights = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
    constraints = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if weights.value is None:
        raise ValueError("Optimization failed to produce a valid solution.")
    optimal_weights = weights.value
    expected_return, expected_risk = calculate_portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    return {
        "weights": optimal_weights,
        "expected_return": expected_return,
        "expected_risk": expected_risk
    }
