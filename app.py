import streamlit as st
import pandas as pd
from fetch_data import get_stock_data
from alerts import check_price_alert, check_volume_alert
from portfolio_optimizer import fetch_data, optimize_portfolio
import matplotlib.pyplot as plt 
st.set_page_config(page_title="StoxEye", layout="wide")
st.title("📈 StoxEye – Real-Time Stock Dashboard")
st.markdown("Track. Analyze. Grow your investments like a pro.")
st.info("Enter a stock symbol to get started.")
symbol = st.text_input("Enter a stock symbol to get started :", "TCS.NS")
if symbol:
    data = get_stock_data(symbol)
    if data is not None:
        st.subheader(f"Stock chart for {symbol}")
        st.line_chart(data["Close"])
        st.dataframe(data.tail())

        price_alert = check_price_alert(data)
        volume_alert = check_volume_alert(data)
        if price_alert:
            st.warning(price_alert)
        if volume_alert:
            st.info(volume_alert)
    else:
        st.error("No data found for this symbol. Try a valid one like INFY.NS or RELIANCE.NS.")
st.subheader("PORTFOLIO OPTIMIZER")
file = st.file_uploader("Upload a CSV file with stock symbols:", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    try:
        df = pd.read_csv("data/sample_stocks.csv")
        st.caption("Using default sample_stocks.csv file from data folder.")
    except FileNotFoundError:
        df = None
        st.warning("Sample file not found. Please upload a CSV file.")
if df is not None:
    symbols = df["Symbol"].tolist()
    data = fetch_data(symbols)
    if not data.empty:
        result = optimize_portfolio(data)
        st.success(" Optimal portfolio calculated!")
        st.write(" Optimal Weights:")
        for sym, w in zip(symbols, result["weights"]):
            st.write(f"{sym}: {w * 100:.2f}%")
        fig,ax=plt.subplots(figsize=(2,1))
        wedges,texts,autotexts=ax.pie(result["weights"],labels=symbols,autopct='%1.1f%%',startangle=90,textprops={'fontsize':10},labeldistance=1.2,pctdistance=0.7)
        for text in texts:
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_fontsize(9)    
        ax.axis('equal')
        st.subheader("portfolio allocation pie chart")
        st.pyplot(fig)
        st.write(f"Expected Return: `{result['expected_return']:.2%}`")
        st.write(f"Expected Risk: `{result['expected_risk']:.2%}`")
    else:
        st.error("No data found for these symbols.")

