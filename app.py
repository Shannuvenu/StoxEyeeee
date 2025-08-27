# app.py
import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import date

from fetch_data import get_stock_data, get_realtime_price
from alerts import check_price_alert, check_volume_alert
from portfolio_optimizer import fetch_data, optimize_portfolio
from news_feed import get_news_feed
from institutional_flow import get_flow_dashboard, default_universe

# NEW: Goals
from goals import load_goals, add_goal, remove_goal, evaluate_goal

# ---------------- Watchlist Setup ----------------
WATCHLIST_FILE = "data/watchlist.json"
os.makedirs("data", exist_ok=True)
if not os.path.exists(WATCHLIST_FILE):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump({"watchlist": []}, f)

def load_watchlist():
    with open(WATCHLIST_FILE, "r") as f:
        return json.load(f)["watchlist"]

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump({"watchlist": watchlist}, f, indent=2)

def add_to_watchlist(symbol):
    watchlist = load_watchlist()
    if symbol and symbol not in watchlist:
        watchlist.append(symbol.upper())
        save_watchlist(watchlist)

def remove_from_watchlist(symbol):
    watchlist = load_watchlist()
    if symbol in watchlist:
        watchlist.remove(symbol)
        save_watchlist(watchlist)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="StoxEye", layout="wide")
st.title("📈 StoxEye – Real-Time Stock Dashboard")

# -------- Top Indices --------
st.markdown("## 📊 Market Indices (Real-Time)")
col1, col2, col3 = st.columns(3)
with col1:
    nifty_price = get_realtime_price("^NSEI")
    st.metric("🇮🇳 NIFTY 50", f"₹{nifty_price:.2f}" if nifty_price else "N/A")
with col2:
    sensex_price = get_realtime_price("^BSESN")
    st.metric("📈 SENSEX", f"₹{sensex_price:.2f}" if sensex_price else "N/A")
with col3:
    banknifty_price = get_realtime_price("^NSEBANK")
    st.metric("🏦 BANK NIFTY", f"₹{banknifty_price:.2f}" if banknifty_price else "N/A")

st.caption("⚠️ Educational analysis only. Markets are risky. This is **not** financial advice.")

st.info("Enter a stock symbol and period to get started.")
symbol = st.text_input("Enter a stock symbol:", "TCS.NS").upper()
period = st.selectbox("Select period:", ["5d", "15d", "1mo", "3mo", "6mo", "1y"], index=1)
interval = st.selectbox("Select interval:", ["1h", "1d", "1wk"], index=1)

# -------- Chart + Alerts --------
if symbol:
    data = get_stock_data(symbol, period=period, interval=interval)
    if data is not None and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

    if data is not None and not data.empty:
        st.subheader(f"📊 Stock chart for {symbol}")
        if {"Open", "High", "Low", "Close"}.issubset(data.columns):
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
                increasing_line_color='green', decreasing_line_color='red'
            )])
            fig.update_layout(title=f"{symbol} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Required columns missing for chart.")

        st.dataframe(data.tail())
        p_alert = check_price_alert(data)
        v_alert = check_volume_alert(data)
        if p_alert: st.warning(p_alert)
        if v_alert: st.info(v_alert)
    else:
        st.error("No data found for this symbol.")

# -------- Watchlist Controls --------
c1, c2 = st.columns(2)
with c1:
    if st.button("🔖 Add to Watchlist"):
        add_to_watchlist(symbol)
        st.success(f"{symbol} added to watchlist!")
with c2:
    if st.button("🗑️ Remove from Watchlist"):
        remove_from_watchlist(symbol)
        st.warning(f"{symbol} removed from watchlist.")

# -------- News --------
st.subheader("📰 Latest News Headlines")
news_items = get_news_feed(symbol)
if news_items:
    for article in news_items:
        st.markdown(f"🔹 [{article['headline']}]({article['url']})", unsafe_allow_html=True)
else:
    st.info("No recent news found.")

# -------- Institutional Flow --------
st.subheader("⚡ Institutional Power Tracker (Live)")
universe_choice = st.radio("Universe:", ["My Watchlist", "Default Largecaps", "Custom"], horizontal=True)
if universe_choice == "My Watchlist":
    universe = load_watchlist() or []
elif universe_choice == "Default Largecaps":
    universe = default_universe()
else:
    custom_input = st.text_input("Enter comma-separated symbols (e.g., TCS.NS, RELIANCE.NS)")
    universe = [s.strip().upper() for s in custom_input.split(",") if s.strip()]

colA, colB = st.columns([1,1])
with colA:
    top_n = st.slider("How many top signals to show?", 5, 50, 15, step=5)
with colB:
    refresh = st.button("🔄 Refresh Signals")

if refresh:
    if not universe:
        st.warning("Pick at least one symbol (watchlist is empty or custom not provided).")
    else:
        flow_df = get_flow_dashboard(universe)
        st.dataframe(flow_df.head(top_n), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🟢 Today’s Strong Buys")
            buys = flow_df[flow_df["Flow"].isin(["Strong Buy","Buy"])].head(top_n)[
                ["Symbol","Price","DayChange%","VolXAvg","Flow","Score"]
            ]
            st.dataframe(buys, use_container_width=True) if not buys.empty else st.caption("No buy signals right now.")
        with c2:
            st.markdown("### 🔴 Today’s Strong Sells")
            sells = flow_df[flow_df["Flow"].isin(["Strong Sell","Sell"])].head(top_n)[
                ["Symbol","Price","DayChange%","VolXAvg","Flow","Score"]
            ]
            st.dataframe(sells, use_container_width=True) if not sells.empty else st.caption("No sell signals right now.")

        strong = flow_df[flow_df["Flow"].isin(["Strong Buy","Strong Sell"])].head(3)
        for _, r in strong.iterrows():
            direction = "📈" if "Buy" in r["Flow"] else "📉"
            st.toast(f"{direction} {r['Symbol']} | {r['Flow']} | Δ {r['DayChange%']}% | Vol× {r['VolXAvg']}", icon="⚡")

# -------- Watchlist View --------
st.subheader("📌 Your Watchlist")
watchlist = load_watchlist()
search_term = st.text_input("🔍 Search Watchlist:")
filtered_watchlist = [s for s in watchlist if search_term.upper() in s.upper()] if search_term else watchlist
if filtered_watchlist:
    st.write(", ".join(filtered_watchlist))
else:
    st.info("No matching stocks found.")

# -------- Portfolio Optimizer --------
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
        st.write("### 🧻 Optimal Weights:")
        for sym, w in zip(symbols, result["weights"]):
            st.write(f"- **{sym}**: `{w * 100:.2f}%`")

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(result["weights"], labels=symbols, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.subheader("📊 Portfolio Allocation Pie Chart")
        st.pyplot(fig)

        st.write(f"📈 Expected Return: `{result['expected_return']:.2%}`")
        st.write(f"📉 Expected Risk: `{result['expected_risk']:.2%}`")
    else:
        st.error("❌ No data found for the given symbols.")

    # Comparison
    st.subheader("📊 STOCK COMPARISON GRAPH")
    compare_symbols = st.multiselect("Select stocks to compare price trends:", symbols, default=symbols[:2])
    if compare_symbols:
        price_data = {}
        for sym in compare_symbols:
            df_comp = get_stock_data(sym, period="6mo", interval="1d")
            if df_comp is not None and not df_comp.empty and "Adj Close" in df_comp.columns:
                price_data[sym] = df_comp["Adj Close"].rename(sym)
        if price_data:
            comparison_df = pd.concat(price_data.values(), axis=1)
            fig = go.Figure()
            for sym in comparison_df.columns:
                fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[sym], mode="lines", name=sym))
            fig.update_layout(title="📈 Price Comparison Over Time", xaxis_title="Date", yaxis_title="Adjusted Close")
            st.plotly_chart(fig, use_container_width=True)

# -------- Real-time Portfolio Value + Insights --------
st.subheader("💰 REAL-TIME PORTFOLIO VALUE")
total_investment = 0.0
total_value = 0.0
total_profit = 0.0

if df is not None and all(col in df.columns for col in ["Symbol", "Quantity", "Buy Price"]):
    df["Live Price"] = 0.0
    df["Current Value"] = 0.0
    df["Investment"] = df["Quantity"] * df["Buy Price"]
    df["P&L"] = 0.0

    for idx, row in df.iterrows():
        sym = str(row["Symbol"]).upper()
        qty = float(row["Quantity"])
        live_data = get_stock_data(sym, period="1d", interval="1h")
        if live_data is not None and not live_data.empty:
            latest_price = float(live_data["Close"].iloc[-1])
            df.at[idx, "Live Price"] = latest_price
            df.at[idx, "Current Value"] = qty * latest_price
            df.at[idx, "P&L"] = df.at[idx, "Current Value"] - df.at[idx, "Investment"]

    total_investment = float(df["Investment"].sum())
    total_value = float(df["Current Value"].sum())
    total_profit = total_value - total_investment

    st.dataframe(df[["Symbol", "Quantity", "Buy Price", "Live Price", "Investment", "Current Value", "P&L"]])
    st.success(f"📊 Total Investment: ₹{total_investment:,.2f}")
    st.info(f"💼 Current Portfolio Value: ₹{total_value:,.2f}")
    st.markdown(f"🔺 Profit / Loss: `{total_profit:+,.2f}` ₹")

    st.subheader("🧠 Smart Portfolio Insights")
    df["Return %"] = (df["P&L"] / df["Investment"]).replace([float("inf"), -float("inf")], 0).fillna(0) * 100
    best_stock = df.loc[df["Return %"].idxmax()]
    worst_stock = df.loc[df["Return %"].idxmin()]
    total_return_pct = (total_profit / total_investment) * 100 if total_investment else 0

    st.markdown(f"🔝 **Best Performer**: `{best_stock['Symbol']}` with `{best_stock['Return %']:.2f}%` return.")
    st.markdown(f"🔻 **Worst Performer**: `{worst_stock['Symbol']}` with `{worst_stock['Return %']:.2f}%` return.")
    st.markdown(f"📈 **Total Portfolio Return**: `{total_return_pct:.2f}%`")
else:
    st.warning("Portfolio CSV must include columns: `Symbol`, `Quantity`, and `Buy Price`.")

# -------- 🎯 Goal-Linked Investing (NEW) --------
st.subheader("🎯 Goal-Linked Investing (Beta)")
with st.form("add_goal_form", clear_on_submit=True):
    g1, g2 = st.columns([2,1])
    with g1:
        goal_name = st.text_input("Goal Name", placeholder="Buy iPhone / MBA Fees / Bike / Emergency Fund")
    with g2:
        goal_emoji = st.text_input("Emoji (optional)", value="🎯", max_chars=2)

    goal_target = st.number_input("Target Amount (₹)", min_value=0.0, step=1000.0, value=50000.0, format="%.2f")
    goal_date = st.date_input("Target Date", value=date.today().replace(year=date.today().year + 1))
    goal_notes = st.text_area("Notes (optional)", placeholder="Why this goal matters?")

    submitted = st.form_submit_button("➕ Add Goal")
    if submitted:
        add_goal(goal_name or "My Goal", goal_target, goal_date.isoformat(), notes=goal_notes, emoji=goal_emoji or "🎯")
        st.success("Goal added!")

goals = load_goals()
if not goals:
    st.info("No goals yet. Add your first goal above.")
else:
    st.write("#### Your Goals")
    # Use current portfolio total value if available, else 0
    current_portfolio_value = total_value if df is not None else 0.0

    for g in goals:
        evald = evaluate_goal(current_portfolio_value, g)
        pct = evald["progress_pct"]
        left = evald["remaining_amount"]
        days_left = evald["days_left"]
        need_pm = evald["need_per_month"]
        on_track = evald["on_track"]

        box = st.container(border=True)
        with box:
            h1, h2 = st.columns([3,1])
            with h1:
                st.markdown(f"**{g.get('emoji','🎯')} {g['name']}** — Target: ₹{g['target_amount']:,.0f} by **{g['target_date']}**")
                st.progress(min(1.0, pct/100.0), text=f"{pct:.1f}% complete")
                if on_track:
                    st.success(evald["reason"])
                else:
                    st.warning(evald["reason"])
                st.caption(f"Remaining: ₹{left:,.0f} • Days left: {days_left} • Needed ≈ ₹{need_pm:,.0f}/month")
                if g.get("notes"):
                    st.caption(f"📝 {g['notes']}")
            with h2:
                if st.button("Delete", key=f"del_{g['id']}"):
                    remove_goal(g["id"])
                    st.toast("Goal removed.", icon="🗑️")
                    st.rerun()

# -------- Historical Candles --------
if df is not None:
    st.subheader("🔧 HISTORICAL PRICE COMPARISON")
    symbols = df["Symbol"].tolist()
    selected_symbol = st.selectbox("Pick a stock to view historical candlestick chart:", symbols)
    hist_data = get_stock_data(selected_symbol, period=period, interval=interval)
    if hist_data is not None and isinstance(hist_data.columns, pd.MultiIndex):
        hist_data.columns = hist_data.columns.get_level_values(-1)

    if hist_data is not None and not hist_data.empty and {"Open","High","Low","Close"}.issubset(hist_data.columns):
        fig2 = go.Figure(data=[go.Candlestick(
            x=hist_data.index,
            open=hist_data["Open"], high=hist_data["High"], low=hist_data["Low"], close=hist_data["Close"],
            increasing_line_color='green', decreasing_line_color='red'
        )])
        fig2.update_layout(title=f"{selected_symbol} Historical Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig2, use_container_width=True)
    elif hist_data is not None:
        st.error("❌ Historical data missing required columns.")
    else:
        st.warning("⚠️ No valid historical data found for this stock.")
