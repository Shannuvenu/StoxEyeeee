import streamlit as st
from fetch_data import get_stock_data
st.set_page_config(page_title="StoxEye", layout="wide")
st.title("📈 StoxEye – Real-Time Stock Dashboard")
st.markdown("Track. Analyze. Grow your investments like a pro.")
st.info("Enter a stock symbol to get started.")
symbol= st.text_input("Enter a stock symbol to get started :","TCS.NS")
if symbol:
    data=get_stock_data(symbol)
    if data is not None:
        st.subheader(f"stock chart for {symbol}")
        st.line_chart(data["Close"])
        st.dataframe(data.tail())
    else:
        st.error("no data found for this symbol.try a valid one like INFY.NS or RELIANCE.NS ")    