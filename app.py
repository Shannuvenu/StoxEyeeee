import streamlit as st
import pandas as pd
from fetch_data import get_stock_data
from alerts import check_price_alert, check_volume_alert
from portfolio_optimizer import fetch_data, optimize_portfolio
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="StoxEye", layout="wide")
st.title("📈 StoxEye – Real-Time Stock Dashboard")
st.markdown("Track. Analyze. Grow your investments like a pro.")
st.info("Enter a stock symbol and period to get started.")
symbol = st.text_input("Enter a stock symbol to get started:", "TCS.NS")
period = st.selectbox("Select period:", ["5d", "15d", "1mo", "3mo", "6mo", "1y"], index=1)
interval = st.selectbox("Select interval:", ["1h", "1d", "1wk"], index=1)
if symbol:
    data = get_stock_data(symbol, period=period, interval=interval)
    if data is not None and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

    if data is not None and not data.empty:
        st.subheader(f"📊 Stock chart for {symbol}")
        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(set(data.columns)):
            st.error("❌ Required columns are missing in the data. Cannot plot candlestick chart.")
            st.dataframe(data.head())
        else:
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            fig.update_layout(title=f"{symbol} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(data.tail())
        price_alert = check_price_alert(data)
        volume_alert = check_volume_alert(data)
        if price_alert:
            st.warning(price_alert)
        if volume_alert:
            st.info(volume_alert)
    else:
        st.error("⚠️ No data found for this symbol. Try one like INFY.NS or RELIANCE.NS.")
st.subheader("📈 PORTFOLIO OPTIMIZER")
file = st.file_uploader("📁 Upload a CSV file with stock symbols:", type=["csv"])
if file:
    df = pd.read_csv(file)
else:
    try:
        df = pd.read_csv("data/sample_stocks.csv")
        st.caption("Using default `sample_stocks.csv` from the data folder.")
    except FileNotFoundError:
        df = None
        st.warning("⚠️ Sample file not found. Please upload a CSV.")
if df is not None:
    symbols = df["Symbol"].tolist()
    portfolio_data = fetch_data(symbols)
    if not portfolio_data.empty:
        result = optimize_portfolio(portfolio_data)
        st.success("✅ Optimal portfolio calculated!")
        st.write("### 🧮 Optimal Weights:")
        for sym, w in zip(symbols, result["weights"]):
            st.write(f"- **{sym}**: `{w * 100:.2f}%`")
        fig, ax = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax.pie(
            result["weights"], labels=symbols, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10},
            labeldistance=1.2, pctdistance=0.7
        )
        for text in texts:
            text.set_fontweight('normal')
        for autotext in autotexts:
            autotext.set_fontsize(8)
        ax.axis('equal')
        st.subheader("📊 Portfolio Allocation Pie Chart")
        st.pyplot(fig)

        st.write(f"📈 Expected Return: `{result['expected_return']:.2%}`")
        st.write(f"📉 Expected Risk: `{result['expected_risk']:.2%}`")
    else:
        st.error("❌ No data found for the given symbols.")
if df is not None:
    st.subheader("🕯️ HISTORICAL PRICE COMPARISON")
    selected_symbol = st.selectbox("Pick a stock to view historical candlestick chart:", symbols)
    hist_data = get_stock_data(selected_symbol, period=period, interval=interval)

    if hist_data is not None and isinstance(hist_data.columns, pd.MultiIndex):
        hist_data.columns = hist_data.columns.get_level_values(-1)

    if hist_data is not None and not hist_data.empty:
        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(set(hist_data.columns)):
            st.error("❌ Historical data missing required columns. Cannot plot chart.")
            st.dataframe(hist_data.head())
        else:
            fig2 = go.Figure(data=[go.Candlestick(
                x=hist_data.index,
                open=hist_data["Open"],
                high=hist_data["High"],
                low=hist_data["Low"],
                close=hist_data["Close"],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            fig2.update_layout(title=f"{selected_symbol} Historical Candlestick Chart",
                               xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ No valid historical data found for this stock.")
