import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from fetch_data import get_stock_data, get_realtime_price
from alerts import check_price_alert, check_volume_alert
from portfolio_optimizer import fetch_data as fetch_portfolio_data, optimize_portfolio
from news_feed import get_news_feed
from institutional_flow import get_flow_dashboard, default_universe
st.set_page_config(page_title="StoxEye", layout="wide")

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
        json.dump({"watchlist": watchlist}, f)
def add_to_watchlist(symbol):
    watchlist = load_watchlist()
    if symbol not in watchlist:
        watchlist.append(symbol)
        save_watchlist(watchlist)
def remove_from_watchlist(symbol):
    watchlist = load_watchlist()
    if symbol in watchlist:
        watchlist.remove(symbol)
        save_watchlist(watchlist)
def safe_realtime_price(ticker):
    """Return float price or None. Avoids 'cannot convert series to float' errors."""
    try:
        p = get_realtime_price(ticker)
        if p is None:
            return None
        if isinstance(p, (pd.Series, pd.DataFrame, list, tuple)):
            s = pd.Series(p).dropna()
            if not s.empty:
                return float(pd.to_numeric(s, errors="coerce").dropna().iloc[-1])
            return None
        return float(p)
    except Exception:
        return None

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLCV single-level columns and numeric dtype."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(n, min_periods=n).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(n, min_periods=n).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index).fillna(method="bfill")

def _macd(close: pd.Series):
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    macd = fast - slow
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist

def make_trade_decision(df: pd.DataFrame, symbol: str):
    """
    Returns: decision ('BUY'/'HOLD'/'SELL'), reasons (list[str])
    Uses: SMA(20/50), RSI(14), MACD, Volume spike, Gap risk
    """
    df = ensure_ohlcv(df)
    if df is None or df.empty or not {"Open","High","Low","Close"}.issubset(df.columns):
        return "HOLD", [f"🟡 **Decision: HOLD**", "Insufficient data to compute indicators."]

    reasons = []
    score = 0.0

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    if len(close) >= 50:
        sma20 = _sma(close, 20)
        sma50 = _sma(close, 50)
        if sma20.iloc[-1] > sma50.iloc[-1]:
            score += 1.5
            reasons.append("📈 **Uptrend**: 20-SMA > 50-SMA (bullish).")
        elif sma20.iloc[-1] < sma50.iloc[-1]:
            score -= 1.5
            reasons.append("📉 **Downtrend**: 20-SMA < 50-SMA (bearish).")
    rsi = _rsi(close, 14)
    if not rsi.isna().all():
        r = float(rsi.iloc[-1])
        if r < 30:
            score += 1.0
            reasons.append(f"🟢 **RSI {r:.1f}**: Oversold → potential bounce.")
        elif r > 70:
            score -= 1.0
            reasons.append(f"🔴 **RSI {r:.1f}**: Overbought → risk of pullback.")
        else:
            reasons.append(f"⚖️ **RSI {r:.1f}**: Neutral.")
    macd, signal, hist = _macd(close)
    if not macd.isna().all():
        if (macd.iloc[-1] > signal.iloc[-1]) and (hist.iloc[-1] > 0):
            score += 1.0
            reasons.append("✅ **MACD**: Bullish crossover (MACD > Signal).")
        elif (macd.iloc[-1] < signal.iloc[-1]) and (hist.iloc[-1] < 0):
            score -= 1.0
            reasons.append("⚠️ **MACD**: Bearish crossover (MACD < Signal).")

    if not volume.isna().all() and volume.count() >= 20:
        vol_ratio = volume.iloc[-1] / (volume.rolling(20).mean().iloc[-1] + 1e-9)
        if vol_ratio >= 1.5:
            score += 0.5
            reasons.append(f"🔊 **Volume**: {vol_ratio:.1f}× above avg → strong participation.")
        elif vol_ratio <= 0.7:
            reasons.append(f"🤏 **Volume**: {vol_ratio:.1f}× below avg → weak confirmation.")

    if len(close) >= 2:
        gap_pct = (df["Open"].iloc[-1] - close.iloc[-2]) / (close.iloc[-2] + 1e-9) * 100
        if abs(gap_pct) > 2.0:
            reasons.append(f"⚡ **Gap**: {gap_pct:.2f}% at open → higher short-term volatility.")

    if score >= 1.5:
        decision = "BUY"
        reasons.insert(0, f"🟢 **Decision: BUY**  (score {score:+.1f})")
        reasons.append("📌 Why BUY? Trend & momentum supportive; consider scaling in with stop-loss.")
    elif score <= -1.5:
        decision = "SELL"
        reasons.insert(0, f"🔴 **Decision: SELL** (score {score:+.1f})")
        reasons.append("📌 Why SELL? Weak trend/momentum; consider reducing exposure/protecting gains.")
    else:
        decision = "HOLD"
        reasons.insert(0, f"🟡 **Decision: HOLD** (score {score:+.1f})")
        reasons.append("📌 Why HOLD? Mixed signals; wait for a clearer break or retest.")

    return decision, reasons

st.title("📈 StoxEye – Real-Time Stock Dashboard")
st.markdown("## 📊 Market Indices (Real-Time)")

c1, c2, c3 = st.columns(3)
with c1:
    nifty_price = safe_realtime_price("^NSEI")
    st.metric("🇮🇳 NIFTY 50", f"₹{nifty_price:.2f}" if nifty_price is not None else "N/A")
with c2:
    sensex_price = safe_realtime_price("^BSESN")
    st.metric("📈 SENSEX", f"₹{sensex_price:.2f}" if sensex_price is not None else "N/A")
with c3:
    banknifty_price = safe_realtime_price("^NSEBANK")
    st.metric("🏦 BANK NIFTY", f"₹{banknifty_price:.2f}" if banknifty_price is not None else "N/A")

st.markdown("Track. Analyze. Grow your investments like a pro.")
st.info("Enter a stock symbol and period to get started.")

symbol = st.text_input("Enter a stock symbol to get started:", "TCS.NS").strip().upper()
period = st.selectbox("Select period:", ["5d", "15d", "1mo", "3mo", "6mo", "1y"], index=1)
interval = st.selectbox("Select interval:", ["1h", "1d", "1wk"], index=1)

data = None
if symbol:
    raw = get_stock_data(symbol, period=period, interval=interval)
    data = ensure_ohlcv(raw)

    if data is not None and not data.empty:
        st.subheader(f"📊 Stock chart for {symbol}")
        if {"Open", "High", "Low", "Close"}.issubset(data.columns):
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
        else:
            st.error("Required columns missing for chart.")
        st.dataframe(data.tail())
        pa = check_price_alert(data)
        if pa: st.warning(pa)
        va = check_volume_alert(data)
        if va: st.info(va)
        try:
            decision, reasons = make_trade_decision(data.copy(), symbol)
            if decision == "BUY":
                st.success(" ".join(reasons[:1]))
            elif decision == "SELL":
                st.error(" ".join(reasons[:1]))
            else:
                st.warning(" ".join(reasons[:1]))
            with st.expander("See full analysis & reasons"):
                for r in reasons[1:]:
                    st.markdown(f"- {r}")
        except Exception as e:
            st.warning(f"Couldn’t compute the decision: {e}")
    else:
        st.error("No data found for this symbol.")

cwl1, cwl2 = st.columns(2)
with cwl1:
    if st.button("🔖 Add to Watchlist") and symbol:
        add_to_watchlist(symbol)
        st.success(f"{symbol} added to watchlist!")
with cwl2:
    if st.button("🗑️ Remove from Watchlist") and symbol:
        remove_from_watchlist(symbol)
        st.warning(f"{symbol} removed from watchlist.")

st.subheader("📰 Latest News Headlines")
news_items = get_news_feed(symbol) if symbol else []
if news_items:
    for article in news_items:
        st.markdown(f"🔹 [{article['headline']}]({article['url']})", unsafe_allow_html=True)
else:
    st.info("No recent news found.")

st.subheader("⚡ Institutional Power Tracker (Live)")
universe_choice = st.radio("Universe:", ["My Watchlist", "Default Largecaps", "Custom"], horizontal=True)

if universe_choice == "My Watchlist":
    universe = load_watchlist() or []
elif universe_choice == "Default Largecaps":
    universe = default_universe()
else:
    custom_input = st.text_input("Enter comma-separated symbols (e.g., TCS.NS, RELIANCE.NS, HDFCBANK.NS)")
    universe = [s.strip().upper() for s in custom_input.split(",") if s.strip()]

colA, colB = st.columns([1, 1])
with colA:
    top_n = st.slider("How many top signals to show?", 5, 50, 15, step=5)
with colB:
    refresh = st.button("🔄 Refresh Signals")

if refresh:
    if not universe:
        st.warning("Pick at least one symbol (watchlist is empty or custom not provided).")
    else:
        try:
            flow_df = get_flow_dashboard(universe)
            st.dataframe(flow_df.head(top_n), use_container_width=True)

            cbuys, csells = st.columns(2)
            with cbuys:
                st.markdown("### 🟢 Today’s Strong Buys")
                buys = flow_df[flow_df["Flow"].isin(["Strong Buy", "Buy"])].head(top_n)[
                    ["Symbol", "Price", "DayChange%", "VolXAvg", "Flow", "Score"]
                ]
                if not buys.empty:
                    st.dataframe(buys, use_container_width=True)
                else:
                    st.caption("No buy signals right now.")
            with csells:
                st.markdown("### 🔴 Today’s Strong Sells")
                sells = flow_df[flow_df["Flow"].isin(["Strong Sell", "Sell"])].head(top_n)[
                    ["Symbol", "Price", "DayChange%", "VolXAvg", "Flow", "Score"]
                ]
                if not sells.empty:
                    st.dataframe(sells, use_container_width=True)
                else:
                    st.caption("No sell signals right now.")

            strong = flow_df[flow_df["Flow"].isin(["Strong Buy", "Strong Sell"])].head(3)
            for _, r in strong.iterrows():
                direction = "📈" if "Buy" in r["Flow"] else "📉"
                st.toast(f"{direction} {r['Symbol']} | {r['Flow']} | Δ {r['DayChange%']}% | Vol× {r['VolXAvg']}", icon="⚡")
        except Exception as e:
            st.error(f"Failed to load Institutional Power Tracker: {e}")

st.subheader("📌 Your Watchlist")
watchlist = load_watchlist()
search_term = st.text_input("🔍 Search Watchlist:")
filtered_watchlist = [s for s in watchlist if search_term.upper() in s.upper()]
if filtered_watchlist:
    for stock in filtered_watchlist:
        st.markdown(f"- {stock}")
else:
    st.info("No matching stocks found.")

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
    portfolio_data = fetch_portfolio_data(symbols)

    if not portfolio_data.empty:
        result = optimize_portfolio(portfolio_data)
        st.success("✅ Optimal portfolio calculated!")

        st.write("### 🧻 Optimal Weights:")
        for sym, w in zip(symbols, result["weights"]):
            st.write(f"- **{sym}**: `{w * 100:.2f}%`")

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(result["weights"], labels=symbols, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.subheader("📊 Portfolio Allocation Pie Chart")
        st.pyplot(fig)

        st.write(f"📈 Expected Return: `{result['expected_return']:.2%}`")
        st.write(f"📉 Expected Risk: `{result['expected_risk']:.2%}`")
    else:
        st.error("❌ No data found for the given symbols.")

    st.subheader("📊 STOCK COMPARISON GRAPH")
    compare_symbols = st.multiselect("Select stocks to compare price trends:", symbols, default=symbols[:2])
    if compare_symbols:
        price_data = {}
        for sym in compare_symbols:
            comp = ensure_ohlcv(get_stock_data(sym, period="6mo", interval="1d"))
            if comp is not None and not comp.empty and "Adj Close" in comp.columns:
                price_data[sym] = comp["Adj Close"].astype(float)
        if price_data:
            comparison_df = pd.concat(price_data.values(), axis=1)
            comparison_df.columns = list(price_data.keys())
            fig = go.Figure()
            for sym in comparison_df.columns:
                fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[sym], mode="lines", name=sym))
            fig.update_layout(title="📈 Price Comparison Over Time",
                              xaxis_title="Date", yaxis_title="Adjusted Close Price")
            st.plotly_chart(fig, use_container_width=True)

st.subheader("💰 REAL-TIME PORTFOLIO VALUE")
if df is not None and all(col in df.columns for col in ["Symbol", "Quantity", "Buy Price"]):
    df["Live Price"] = 0.0
    df["Current Value"] = 0.0
    df["Investment"] = 0.0
    df["P&L"] = 0.0

    for idx, row in df.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        qty = float(row["Quantity"])
        buy_price = float(row["Buy Price"])
        live = ensure_ohlcv(get_stock_data(sym, period="1d", interval="1h"))
        if live is not None and not live.empty and "Close" in live.columns:
            latest_price = float(pd.to_numeric(live["Close"], errors="coerce").dropna().iloc[-1])
            df.at[idx, "Live Price"] = latest_price
            df.at[idx, "Current Value"] = qty * latest_price
            df.at[idx, "Investment"] = qty * buy_price
            df.at[idx, "P&L"] = df.at[idx, "Current Value"] - df.at[idx, "Investment"]

    total_investment = float(df["Investment"].sum())
    total_value = float(df["Current Value"].sum())
    total_profit = total_value - total_investment
    df["Return %"] = np.where(df["Investment"] != 0, (df["P&L"] / df["Investment"]) * 100, 0.0)

    st.dataframe(df[["Symbol", "Quantity", "Buy Price", "Live Price",
                     "Investment", "Current Value", "P&L", "Return %"]])

    st.success(f"📊 Total Investment: ₹{total_investment:,.2f}")
    st.info(f"💼 Current Portfolio Value: ₹{total_value:,.2f}")
    st.markdown(f"🔺 Profit / Loss: `{total_profit:+,.2f}` ₹")

    st.subheader("🧠 Smart Portfolio Insights")
    if not df.empty and (df["Return %"].notna().any()):
        best_stock = df.loc[df["Return %"].idxmax()]
        worst_stock = df.loc[df["Return %"].idxmin()]
        total_return_pct = (total_profit / total_investment) * 100 if total_investment else 0
        st.markdown(f"🔝 **Best Performer**: `{best_stock['Symbol']}` with `{best_stock['Return %']:.2f}%` return.")
        st.markdown(f"🔻 **Worst Performer**: `{worst_stock['Symbol']}` with `{worst_stock['Return %']:.2f}%` return.")
        st.markdown(f"📈 **Total Portfolio Return**: `{total_return_pct:.2f}%`")

        if total_return_pct > 10:
            st.balloons()
            st.toast("🎉 Massive Gains! You're smashing it, Venu!", icon="🚀")
        elif total_return_pct > 0:
            st.toast("📈 Gains noted. Stay consistent!", icon="✅")
        elif total_return_pct < 0:
            st.toast("📉 Portfolio in loss. Rebalance needed!", icon="⚠️")
        else:
            st.toast("📊 Break-even. Analyze deeper!", icon="ℹ️")

if df is not None and "Symbol" in df.columns:
    st.subheader("🔧 HISTORICAL PRICE COMPARISON")
    symbols = df["Symbol"].astype(str).tolist()
    selected_symbol = st.selectbox("Pick a stock to view historical candlestick chart:", symbols)
    hist_data = ensure_ohlcv(get_stock_data(selected_symbol, period=period, interval=interval))
    if hist_data is not None and not hist_data.empty:
        if {"Open", "High", "Low", "Close"}.issubset(hist_data.columns):
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
            st.error("❌ Historical data missing required columns.")
    else:
        st.warning("⚠️ No valid historical data found for this stock.")

    if st.button("Recalculate Portfolio"):
        st.toast("✅ Portfolio rebalanced successfully!", icon="⚙️")

st.info(
    "⚠️ **Caution / Disclaimer**: This app provides *analysis only* based on public market data "
    "and simple technical indicators. It is **not** investment advice. Markets involve risk and you "
    "**are solely responsible** for your decisions. Consider consulting a SEBI-registered advisor.",
    icon="⚠️",
)
