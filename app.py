# app.py
# StoxEye — Streamlit Dashboard (stable build)
# ------------------------------------------------------------
# Works with your existing helper modules. Safe fallbacks are included
# so the app doesn't crash if an optional module is missing.

import os
import json
from datetime import date

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------- Core modules ----------
from fetch_data import get_stock_data, get_realtime_price
from alerts import check_price_alert, check_volume_alert
from portfolio_optimizer import fetch_data as fetch_portfolio_data, optimize_portfolio
from news_feed import get_news_feed

# ---------- Optional modules (with safe fallbacks) ----------
try:
    from institutional_flow import get_flow_dashboard, default_universe
except Exception:
    def get_flow_dashboard(_): return pd.DataFrame()
    def default_universe(): return []

try:
    from advisor import analyze_stock
except Exception:
    def analyze_stock(data, symbol, alerts, vol):
        return "HOLD", "Advisor module missing; HOLD fallback."

try:
    from goals import load_goals, add_goal, remove_goal, evaluate_goal
except Exception:
    _GOAL_FILE = "data/goals.json"
    os.makedirs("data", exist_ok=True)

    def _goals_read():
        try:
            with open(_GOAL_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _goals_write(items):
        with open(_GOAL_FILE, "w") as f:
            json.dump(items, f, indent=2)

    def load_goals():
        return _goals_read()

    def add_goal(name, target_amount, target_date, notes="", emoji="🎯"):
        items = _goals_read()
        items.append({
            "id": f"g_{len(items)+1}",
            "name": name,
            "target_amount": float(target_amount),
            "target_date": str(target_date),
            "notes": notes,
            "emoji": emoji
        })
        _goals_write(items)

    def remove_goal(gid):
        items = [g for g in _goals_read() if g.get("id") != gid]
        _goals_write(items)

    def evaluate_goal(current_value, g):
        target = float(g.get("target_amount", 0.0))
        progress = 0.0 if target <= 0 else min(100.0, (float(current_value) / target) * 100.0)
        return {
            "progress_pct": progress,
            "remaining_amount": max(0.0, target - float(current_value)),
            "days_left": 0,
            "need_per_month": 0.0,
            "on_track": progress >= 50.0,
            "reason": "Goal status (fallback)."
        }

try:
    from risk_utils import (
        get_sector_map,
        sector_exposure,
        position_weights,
        estimate_volatility,
        build_risk_summary,
    )
except Exception:
    def get_sector_map(symbols): return {s: "Unknown" for s in symbols}

    def sector_exposure(df, sec_map):
        if df is None or df.empty:
            return pd.DataFrame(columns=["Sector", "Value"])
        tmp = df.copy()
        tmp["Sector"] = tmp["Symbol"].map(sec_map).fillna("Unknown")
        if "Current Value" not in tmp.columns:
            tmp["Current Value"] = tmp.get("Quantity", 0) * tmp.get("Live Price", 0)
        return (
            tmp.groupby("Sector", as_index=False)["Current Value"]
               .sum()
               .rename(columns={"Current Value": "Value"})
        )

    def position_weights(df):
        if df is None or df.empty:
            return pd.Series(dtype=float)
        vals = df.get("Current Value", pd.Series([0]*len(df)))
        total = float(vals.sum()) or 1.0
        out = (vals / total * 100.0).round(2)
        out.index = df["Symbol"]
        return out

    def estimate_volatility(symbol, lookback="6mo", interval="1d"): return 0.0
    def build_risk_summary(w, sec_df, vol_map): return [("info", "Risk summary (fallback).")]

# -------------------- App Config + Theme --------------------
st.set_page_config(page_title="StoxEye", page_icon="📈", layout="wide")
st.markdown(
    """
<style>
:root { --card-bg:#10151c; --card-border:#1f2a37; --accent:#16a34a; --warn:#f59e0b; --err:#ef4444; }
.block-container { padding-top: 0.75rem; }
.stMetric { background: var(--card-bg); border:1px solid var(--card-border); border-radius:12px; padding:.6rem; }
.stTabs [data-baseweb="tab"] { background:#0b0f15; border:1px solid #1c2430; border-radius:10px; padding:10px 14px; }
.stTabs [aria-selected="true"] { border-color:var(--accent); color:#fff !important; }
.stDataFrame { border:1px solid var(--card-border); border-radius:12px; overflow:hidden; }
hr { border-color:#1c2430; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Robust Watchlist Helpers --------------------
WATCHLIST_FILE = "data/watchlist.json"
os.makedirs("data", exist_ok=True)

def _wl_read():
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def _ensure_watchlist_file():
    data = _wl_read()
    if data is None:
        data = {"watchlist": []}
    elif isinstance(data, list):
        data = {"watchlist": data}
    elif isinstance(data, dict):
        data.setdefault("watchlist", [])
    else:
        data = {"watchlist": []}

    wl = data.get("watchlist", [])
    wl = [str(s).upper().strip() for s in wl if str(s).strip()]
    data["watchlist"] = sorted(set(wl))

    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f, indent=2)

_ensure_watchlist_file()

def load_watchlist():
    data = _wl_read()
    if isinstance(data, dict):
        wl = data.get("watchlist", [])
    elif isinstance(data, list):
        wl = data
    else:
        wl = []
    return [str(s).upper().strip() for s in wl if str(s).strip()]

def save_watchlist(wl):
    wl = [str(s).upper().strip() for s in wl if str(s).strip()]
    wl = sorted(set(wl))
    with open(WATCHLIST_FILE, "w") as f:
        json.dump({"watchlist": wl}, f, indent=2)

def add_to_watchlist(symbol):
    if not symbol:
        return
    wl = load_watchlist()
    s = symbol.upper().strip()
    if s and s not in wl:
        wl.append(s)
        save_watchlist(wl)

def remove_from_watchlist(symbol):
    if not symbol:
        return
    wl = load_watchlist()
    s = symbol.upper().strip()
    if s in wl:
        wl.remove(s)
        save_watchlist(wl)

# -------------------- Session portfolio holder --------------------
if "portfolio_df" not in st.session_state:
    st.session_state["portfolio_df"] = None

def set_portfolio(df: pd.DataFrame | None):
    if df is None or df.empty:
        st.session_state["portfolio_df"] = None
        return
    req = {"Symbol", "Quantity", "Buy Price"}
    if not req.issubset(df.columns):
        st.session_state["portfolio_df"] = None
        return
    out = df.copy()
    out["Symbol"] = out["Symbol"].astype(str).str.upper()
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce").fillna(0.0)
    out["Buy Price"] = pd.to_numeric(out["Buy Price"], errors="coerce").fillna(0.0)
    st.session_state["portfolio_df"] = out

def get_portfolio() -> pd.DataFrame | None:
    pf = st.session_state.get("portfolio_df", None)
    return pf.copy() if pf is not None else None

# -------------------- Header / Indices --------------------
st.title("📈 StoxEye — Markets Dashboard")
c1, c2, c3 = st.columns(3)
with c1:
    n = get_realtime_price("^NSEI")
    st.metric("🇮🇳 NIFTY 50", f"₹{n:.2f}" if n else "N/A")
with c2:
    s_ = get_realtime_price("^BSESN")
    st.metric("📈 SENSEX", f"₹{s_:.2f}" if s_ else "N/A")
with c3:
    b = get_realtime_price("^NSEBANK")
    st.metric("🏦 BANK NIFTY", f"₹{b:.2f}" if b else "N/A")
st.caption("⚠ Educational analysis only. This is NOT financial advice.")

# -------------------- Tabs --------------------
tabs = st.tabs(["📊 Markets", "📜 Watchlist", "💼 Portfolio", "🛡 Risk", "🎯 Goals", "⚡ Signals", "📰 News"])

# =========================================================
# 📊 Markets
# =========================================================
with tabs[0]:
    st.subheader("Market View")
    left, right = st.columns([2, 1])

    with left:
        symbol = st.text_input("Symbol", "TCS.NS", key="mkt_sym").upper()
        period = st.selectbox("Period", ["5d", "15d", "1mo", "3mo", "6mo", "1y"], index=2, key="mkt_prd")
        interval = st.selectbox("Interval", ["1h", "1d", "1wk"], index=1, key="mkt_int")

        data = None
        if symbol:
            data = get_stock_data(symbol, period=period, interval=interval)
            if data is not None and isinstance(data.columns, pd.MultiIndex):
                # Try to flatten; if symbol slice fails, keep last level
                try:
                    data = data[symbol]
                except Exception:
                    data.columns = data.columns.get_level_values(-1)

            if data is not None and not data.empty and {"Open", "High", "Low", "Close"}.issubset(data.columns):
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index,
                    open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
                    increasing_line_color="#16a34a", decreasing_line_color="#ef4444",
                )])
                fig.update_layout(title=f"{symbol} — Candlesticks",
                                  xaxis_title="Date", yaxis_title="Price (₹)",
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(data.tail(), use_container_width=True)
            else:
                st.info("No chartable data found for this symbol/period.")

    with right:
        st.markdown("#### Alerts & Advisor")
        if symbol and data is not None and not data.empty:
            p_alert = check_price_alert(data)
            v_alert = check_volume_alert(data)
            if p_alert: st.warning(p_alert)
            if v_alert: st.info(v_alert)

            vol = estimate_volatility(symbol, lookback="6mo", interval="1d")
            sig, reason = analyze_stock(data, symbol, {"price": p_alert, "volume": v_alert}, vol)
            if sig == "BUY":
                st.success(f"🟢 BUY — {reason}")
            elif sig == "SELL":
                st.error(f"🔴 SELL — {reason}")
            else:
                st.info(f"⚪ HOLD — {reason}")
        else:
            st.caption("Load a symbol to view alerts.")

        st.markdown("---")
        cA, cB = st.columns(2)
        with cA:
            if st.button("🔖 Add to Watchlist", key="add_wl_btn"):
                add_to_watchlist(symbol)
                st.toast(f"{symbol} added.", icon="✅")
        with cB:
            if st.button("🗑 Remove from Watchlist", key="rm_wl_btn"):
                remove_from_watchlist(symbol)
                st.toast(f"{symbol} removed.", icon="🗑")

# =========================================================
# 📜 Watchlist
# =========================================================
with tabs[1]:
    st.subheader("Your Watchlist")
    wl = load_watchlist()
    if not wl:
        st.info("Watchlist is empty. Add symbols from the Markets tab.")
    else:
        q = st.text_input("Search", "", key="wl_search")
        filtered = [s for s in wl if q.upper() in s.upper()] if q else wl
        st.write(", ".join(filtered) if filtered else "No matches.")

# =========================================================
# 💼 Portfolio
# =========================================================
with tabs[2]:
    st.subheader("Portfolio")
    st.caption("Upload a CSV with columns: Symbol, Quantity, Buy Price")

    up = st.file_uploader("Upload portfolio CSV", type=["csv"], key="port_up")
    df = None
    if up:
        try:
            df = pd.read_csv(up)
            set_portfolio(df)
            st.caption("Portfolio uploaded.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        if get_portfolio() is None:
            try:
                df = pd.read_csv("data/sample_stocks.csv")
                set_portfolio(df)
                st.caption("Using sample_stocks.csv from /data.")
            except FileNotFoundError:
                st.warning("No portfolio loaded yet.")
                df = None

    port = get_portfolio()
    if port is not None:
        port = port.copy()
        port["Live Price"] = 0.0
        port["Current Value"] = 0.0
        port["Investment"] = port["Quantity"] * port["Buy Price"]

        for i, r in port.iterrows():
            sym = r["Symbol"]
            live = get_stock_data(sym, period="1d", interval="1h")
            if live is not None and not live.empty and "Close" in live.columns:
                price = float(live["Close"].iloc[-1])
                port.at[i, "Live Price"] = price
                port.at[i, "Current Value"] = price * float(r["Quantity"])

        total_inv = float(port["Investment"].sum())
        total_val = float(port["Current Value"].sum())
        pnl = total_val - total_inv

        st.dataframe(
            port[["Symbol","Quantity","Buy Price","Live Price","Investment","Current Value"]],
            use_container_width=True
        )
        st.success(f"Investment: ₹{total_inv:,.2f}")
        st.info(f"Current Value: ₹{total_val:,.2f}")
        st.markdown(f"P&L: {pnl:+,.2f} ₹")

        # Optimizer
        st.markdown("### Optimizer")
        syms = port["Symbol"].tolist()
        pdata = fetch_portfolio_data(syms)
        if not pdata.empty:
            res = optimize_portfolio(pdata)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(res["weights"], labels=syms, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
            st.caption(f"Expected Return: {res['expected_return']:.2%} • Risk: {res['expected_risk']:.2%}")
        else:
            st.caption("Optimizer data not available.")

        # Comparison chart
        st.markdown("### Compare Prices")
        default_pick = syms[: min(3, len(syms))]
        pick = st.multiselect("Select", syms, default=default_pick, key="cmp_pick")
        if pick:
            price_map = {}
            for sname in pick:
                d_ = get_stock_data(sname, period="6mo", interval="1d")
                if d_ is not None and not d_.empty:
                    # Try Adj Close first; fall back to Close
                    series = None
                    if "Adj Close" in d_.columns:
                        series = d_["Adj Close"].rename(sname)
                    elif "Close" in d_.columns:
                        series = d_["Close"].rename(sname)
                    if series is not None:
                        price_map[sname] = series
            if price_map:
                comp = pd.concat(price_map.values(), axis=1)
                figc = go.Figure()
                for c in comp.columns:
                    figc.add_trace(go.Scatter(x=comp.index, y=comp[c], mode="lines", name=c))
                figc.update_layout(title="Price (6M)", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(figc, use_container_width=True)

        # Store priced portfolio for Risk/Goals
        st.session_state["portfolio_df"] = port
    else:
        st.info("Upload a portfolio to see valuation, optimizer and comparisons.")

# =========================================================
# 🛡 Risk
# =========================================================
with tabs[3]:
    st.subheader("🛡 Risk Dashboard (Beta)")

    base = get_portfolio()
    if base is None or base.empty or not {"Symbol","Quantity","Buy Price"}.issubset(base.columns):
        st.info("Upload a portfolio (Symbol, Quantity, Buy Price) in the Portfolio tab to see risk metrics.")
        st.stop()

    # Ensure Live Price + Current Value exist
    if "Live Price" not in base.columns or base["Live Price"].isna().all():
        base["Live Price"] = 0.0
        for i, r in base.iterrows():
            sym = r["Symbol"]
            live = get_stock_data(sym, period="1d", interval="1h")
            if live is not None and not live.empty and "Close" in live.columns:
                base.at[i, "Live Price"] = float(live["Close"].iloc[-1])

    port_risk = base.copy()
    port_risk["Quantity"] = pd.to_numeric(port_risk["Quantity"], errors="coerce").fillna(0.0)
    port_risk["Live Price"] = pd.to_numeric(port_risk["Live Price"], errors="coerce").fillna(0.0)
    port_risk["Current Value"] = port_risk["Quantity"] * port_risk["Live Price"]

    if port_risk["Current Value"].sum() <= 0:
        st.info("Could not fetch live prices right now. Try again later.")
        st.stop()

    symbols_in_port = port_risk["Symbol"].astype(str).str.upper().tolist()

    with st.spinner("Fetching sector classification…"):
        sec_map = get_sector_map(symbols_in_port)
        sec_df = sector_exposure(port_risk, sec_map)

    w = position_weights(port_risk)

    with st.spinner("Estimating volatility…"):
        vol_map = {s: estimate_volatility(s, lookback="6mo", interval="1d") for s in symbols_in_port}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("*Sector Allocation*")
        if not sec_df.empty:
            fig = go.Figure(data=[go.Pie(labels=sec_df["Sector"], values=sec_df["Value"], hole=0.35)])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No sector data available.")
    with c2:
        st.markdown("*Position Weights*")
        if not w.empty:
            fig2 = go.Figure(data=[go.Bar(x=w.index.tolist(), y=w.values.tolist())])
            fig2.update_layout(yaxis_title="Weight (%)", xaxis_title="Symbol")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("No positions to display.")

    tbl = port_risk[["Symbol", "Quantity", "Live Price", "Current Value"]].copy()
    tbl["Weight %"] = tbl["Symbol"].map(w.to_dict())
    tbl["Sector"] = tbl["Symbol"].map(sec_map).fillna("Unknown")
    tbl["Volatility % (Ann)"] = tbl["Symbol"].map(vol_map)
    st.markdown("*Position Details*")
    st.dataframe(tbl.sort_values("Weight %", ascending=False), use_container_width=True)

    msgs = build_risk_summary(w, sec_df, vol_map)
    for level, msg in msgs:
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)

    st.caption("⚠ Educational analysis only — not investment advice. Markets carry risk.")

# =========================================================
# 🎯 Goals
# =========================================================
with tabs[4]:
    st.subheader("Goal-Linked Investing (Beta)")
    port_g = get_portfolio()
    current_val = float(port_g["Current Value"].sum()) if (port_g is not None and "Current Value" in port_g.columns) else 0.0

    with st.form("goal_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            gname = st.text_input("Goal Name", "Buy iPhone / MBA Fees / Bike / Emergency Fund")
        with c2:
            gemoji = st.text_input("Emoji (optional)", "🎯", max_chars=2)

        gamt = st.number_input("Target Amount (₹)", min_value=0.0, step=1000.0, value=50000.0, format="%.2f")
        gdate = st.date_input("Target Date", value=date.today().replace(year=date.today().year + 1))
        gnotes = st.text_area("Notes (optional)", "")

        if st.form_submit_button("➕ Add Goal"):
            add_goal(gname or "My Goal", gamt, gdate.isoformat(), notes=gnotes, emoji=gemoji or "🎯")
            st.success("Goal added!")

    goals = load_goals()
    if not goals:
        st.info("No goals yet. Add one above.")
    else:
        st.markdown("#### Your Goals")
        for g in goals:
            ev = evaluate_goal(current_val, g)
            pct = float(ev.get("progress_pct", 0))
            left = float(ev.get("remaining_amount", 0))
            days_left = ev.get("days_left", 0)
            per_m = float(ev.get("need_per_month", 0))
            on_track = bool(ev.get("on_track", False))
            reason = ev.get("reason", "")

            box = st.container()
            with box:
                h1, h2 = st.columns([3, 1])
                with h1:
                    st.markdown(f"{g.get('emoji','🎯')} {g['name']} — Target: ₹{g['target_amount']:,.0f} by {g['target_date']}")
                    st.progress(min(1.0, pct/100.0), text=f"{pct:.1f}% complete")
                    (st.success if on_track else st.warning)(reason)
                    st.caption(f"Remaining: ₹{left:,.0f} • Days left: {days_left} • Need ≈ ₹{per_m:,.0f}/month")
                    if g.get("notes"): st.caption(f"📝 {g['notes']}")
                with h2:
                    if st.button("Delete", key=f"del_{g['id']}"):
                        remove_goal(g["id"])
                        st.toast("Goal removed.", icon="🗑")
                        st.rerun()

# =========================================================
# ⚡ Signals
# =========================================================
with tabs[5]:
    st.subheader("Institutional Flow Signals")

    choice = st.radio("Universe", ["My Watchlist", "Default Largecaps", "Custom"],
                      horizontal=True, key="sig_universe")
    if choice == "My Watchlist":
        universe = load_watchlist() or []
    elif choice == "Default Largecaps":
        universe = default_universe()
    else:
        raw = st.text_input("Comma-separated symbols", "", key="sig_custom")
        universe = [x.strip().upper() for x in raw.split(",") if x.strip()]

    colx, coly = st.columns([1, 1])
    with colx:
        top_n = st.slider("How many to show?", 5, 50, 15, step=5, key="sig_topn")
    with coly:
        refresh = st.button("🔄 Refresh", key="sig_refresh")

    if not refresh:
        st.info("Pick a universe and press Refresh to load signals.")
    else:
        if not universe:
            st.warning("Pick at least one symbol.")
        else:
            fd = get_flow_dashboard(universe)
            if fd is None or fd.empty:
                st.info("No signals right now.")
            else:
                st.dataframe(fd.head(top_n), use_container_width=True)

                wanted_cols = ["Symbol", "Price", "DayChange%", "VolXAvg", "Flow", "Score"]
                cols = [c for c in wanted_cols if c in fd.columns]

                lft, rgt = st.columns(2)
                with lft:
                    st.markdown("### 🟢 Today’s Strong Buys")
                    buys = fd[fd["Flow"].isin(["Strong Buy", "Buy"])].head(top_n)
                    if buys.empty:
                        st.caption("No buy signals right now.")
                    else:
                        st.dataframe(buys[cols] if cols else buys, use_container_width=True)

                with rgt:
                    st.markdown("### 🔴 Today’s Strong Sells")
                    sells = fd[fd["Flow"].isin(["Strong Sell", "Sell"])].head(top_n)
                    if sells.empty:
                        st.caption("No sell signals right now.")
                    else:
                        st.dataframe(sells[cols] if cols else sells, use_container_width=True)

                strong = fd[fd["Flow"].isin(["Strong Buy", "Strong Sell"])].head(3)
                for _, r in strong.iterrows():
                    icon = "📈" if "Buy" in str(r.get("Flow", "")) else "📉"
                    chg = r.get("DayChange%")
                    volx = r.get("VolXAvg")
                    chg_txt = f"{chg:.2f}" if isinstance(chg, (int, float)) else "—"
                    vol_txt = f"{volx:.2f}" if isinstance(volx, (int, float)) else "—"
                    st.toast(f"{icon} {r.get('Symbol','?')} | {r.get('Flow','?')} | Δ {chg_txt}% | Vol× {vol_txt}", icon="⚡")

# =========================================================
# 📰 News
# =========================================================
with tabs[6]:
    st.subheader("News")
    news_sym = st.text_input("Symbol for News", "TCS.NS", key="news_sym").upper()
    news_items = get_news_feed(news_sym) if news_sym else []
    if news_items:
        for a in news_items:
            st.markdown(f"🔹 [{a['headline']}]({a['url']})")
    else:
        st.info("No recent headlines.")

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
    portfolio_data = fetch_portfolio_data(symbols)
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
# --- after you've created df (either from upload or sample) ---
required_cols = {"Symbol", "Quantity", "Buy Price"}

if df is not None and not df.empty:
    # Optional: validate columns early
    if required_cols.issubset(df.columns):
        st.session_state["portfolio_df"] = df
    else:
        st.warning("Portfolio CSV must include columns: Symbol, Quantity, and Buy Price.")
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
# -------------------- 🛡️ RISK DASHBOARD (Beta) --------------------
st.subheader("🛡️ Risk Dashboard (Beta)")

required_cols = {"Symbol", "Quantity", "Buy Price"}

# Prefer the portfolio you already built; fall back to df
base = (
    st.session_state.get("portfolio_df").copy()
    if "portfolio_df" in st.session_state
    else (df.copy() if df is not None else None)
)

# Guard: need a portfolio with the required columns
if base is None or base.empty or not required_cols.issubset(base.columns):
    st.info("Upload a portfolio CSV above (with Symbol, Quantity, Buy Price) to see risk metrics.")
    st.stop()

# Ensure we have a live price and current value for risk calcs
if "Live Price" not in base.columns or base["Live Price"].isna().all():
    # Try to fill live prices now (best-effort)
    base["Live Price"] = 0.0
    for i, r in base.iterrows():
        sym = str(r["Symbol"]).upper()
        live = get_stock_data(sym, period="1d", interval="1h")
        if live is not None and not live.empty:
            base.at[i, "Live Price"] = float(live["Close"].iloc[-1])

# Build 'port' used by the rest of the dashboard
port = base.copy()
port["Quantity"] = port["Quantity"].astype(float)
port["Live Price"] = port["Live Price"].astype(float)
port["Current Value"] = port["Quantity"] * port["Live Price"]

# Nothing to show if still zero (no live prices could be fetched)
if port["Current Value"].sum() <= 0:
    st.info("Could not fetch live prices right now. Try again later.")
    st.stop()

symbols_in_port = (
    port["Symbol"].astype(str).str.upper().tolist()
    if "Symbol" in port.columns else []
)

# Sector map + exposure
with st.spinner("Fetching sector classification…"):
    sec_map = get_sector_map(symbols_in_port)
    sec_df = sector_exposure(port, sec_map)

# Position weights (for concentration checks)
w = position_weights(port)

# Volatility (cached; computed per symbol)
with st.spinner("Estimating volatility…"):
    vol_map = {s: estimate_volatility(s, lookback="6mo", interval="1d") for s in symbols_in_port}

# ===== Visuals =====
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Sector Allocation**")
    if not sec_df.empty:
        fig = go.Figure(data=[
            go.Pie(labels=sec_df["Sector"], values=sec_df["Value"],
                   hole=0.35, textinfo="label+percent")
        ])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No sector data available.")

with c2:
    st.markdown("**Position Weights**")
    if not w.empty:
        fig2 = go.Figure(data=[
            go.Bar(x=w.index.tolist(), y=w.values.tolist(),
                   hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
        ])
        fig2.update_layout(
            yaxis_title="Weight (%)",
            xaxis_title="Symbol",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("No positions to display.")

# Detailed table
tbl = port[["Symbol", "Quantity", "Live Price", "Current Value"]].copy()
tbl["Weight %"] = tbl["Symbol"].map(w.to_dict())
tbl["Sector"] = tbl["Symbol"].map(sec_map).fillna("Unknown")
tbl["Volatility % (Ann)"] = tbl["Symbol"].map(vol_map)

st.markdown("**Position Details**")
st.dataframe(
    tbl.sort_values("Weight %", ascending=False)
       .rename(columns={"Current Value": "Current Value (₹)"})
       .style.format({
           "Live Price": "₹{:.2f}",
           "Current Value (₹)": "₹{:.2f}",
           "Weight %": "{:.2f}%",
           "Volatility % (Ann)": "{:.1f}%"
       }),
    use_container_width=True
)

# Risk summary
msgs = build_risk_summary(w, sec_df, vol_map)
for level, msg in msgs:
    if level == "error":
        st.error(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.info(msg)

st.caption("⚠️ Educational analysis only — not investment advice. Markets carry risk.")
# -------------------------------------------------------------------


# Risk summary
msgs = build_risk_summary(w, sec_df, vol_map)
for level, msg in msgs:
    if level == "error":
        st.error(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.info(msg)

st.caption("⚠️ Educational analysis only — not investment advice. Markets carry risk.")
# -------------------------------------------------------------------


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
    hist_data = get_stock_data(selected_symbol, period=pd.Period, interval=pd.Interval)
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
