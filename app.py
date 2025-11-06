import os
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st

from train_15m import train_and_forecast, LOOK_BACK
from data_sources import fetch_ohlcv

# ------------------
# Streamlit Config
# ------------------
st.set_page_config(page_title="Crypto Forecast Bot • 15m (Fib + Liquidity)", layout="centered")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------
# Access Gate (use Streamlit Secrets)
# ------------------
PASSWORD = st.secrets.get("ACCESS_PASSWORD", None)
st.title("🔮 15m Crypto Forecast (Fib Confluence + Liquidity)")

st.markdown(
    "> ⚠️ **Educational only** — not financial advice. Models are experimental and noisy for long horizons."
)

if PASSWORD:
    pwd = st.text_input("Enter Access Password", type="password")
    if pwd != PASSWORD:
        st.warning("Access denied. Set ACCESS_PASSWORD in .streamlit/secrets.toml")
        st.stop()
    st.success("✅ Access granted.")
else:
    st.info("No ACCESS_PASSWORD set — running in open mode.")

# ------------------
# Sidebar Controls
# ------------------
st.sidebar.header("⚙️ Settings")
exchange = st.sidebar.radio("Exchange", ["bybit", "okx"], index=0, help="Data source for 15m OHLCV")

symbol_help = (
    "Bybit perp examples: BTCUSDT, ETHUSDT.\n"
    "OKX swap examples: BTC-USDT-SWAP, ETH-USDT-SWAP."
)
if exchange == "bybit":
    default_symbol = "BTCUSDT"
else:
    default_symbol = "BTC-USDT-SWAP"

symbol = st.sidebar.text_input("Symbol", value=default_symbol, help=symbol_help)

days = st.sidebar.slider("Horizon (days)", min_value=1, max_value=7, value=2, help="Number of *days* to forecast ahead at 15-minute resolution.")
steps = int(days * 24 * 4)  # 15m bars per day

retrain = st.sidebar.checkbox("Force retrain today", value=False)

# ------------------
# Helper: cache key + path
# ------------------
def forecast_path(symbol: str, exchange: str) -> str:
    os.makedirs("intraday_forecasts", exist_ok=True)
    safe_sym = symbol.replace("/", "-")
    return os.path.join("intraday_forecasts", f"{safe_sym}_{exchange}_15m.csv")

# ------------------
# Load cached forecast if valid
# ------------------
@st.cache_data(show_spinner=False)
def load_cached(path: str):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])  # columns: timestamp, pred_close
        return df
    except Exception:
        return None

# ------------------
# Train / Load Logic
# ------------------
path = forecast_path(symbol, exchange)
needs_train = True

cached = load_cached(path)
if cached is not None and len(cached) >= steps:
    # valid if file is from today (UTC) and has enough rows
    mtime = dt.datetime.utcfromtimestamp(os.path.getmtime(path))
    today_utc = dt.datetime.utcnow().date()
    if mtime.date() == today_utc and cached["timestamp"].iloc[0].tzinfo is None:
        # allow naive timestamps (assume UTC) from training script
        needs_train = not (cached.shape[0] >= steps)
    elif mtime.date() == today_utc:
        needs_train = not (cached.shape[0] >= steps)

if retrain:
    needs_train = True

colA, colB = st.columns([1,1])
with colA:
    if st.button("🛠 Train / Update Forecast"):
        needs_train = True

# Train if needed
if needs_train:
    with st.spinner(f"Training {symbol} on {exchange} (15m, Fib+Liquidity features)…"):
        try:
            out, _ = train_and_forecast(exchange=exchange, symbol=symbol, horizon_steps=steps)
            out.to_csv(path, index=False)
            st.success(f"✅ Forecast saved → {path}")
            cached = out
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()
else:
    st.caption("Using cached forecast from today.")

# ------------------
# Fetch recent actuals for overlay
# ------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_actuals(exchange: str, symbol: str, start: dt.datetime, end: dt.datetime):
    df = fetch_ohlcv(exchange, symbol, start, end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    return df

if cached is None or cached.empty:
    st.error("No forecast available.")
    st.stop()

forecast_df = cached.copy()
forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"], utc=True)

start_actuals = forecast_df["timestamp"].iloc[0] - pd.Timedelta(days=2)
end_actuals = forecast_df["timestamp"].iloc[-1]

actuals = fetch_actuals(exchange, symbol, start_actuals, end_actuals)

# ------------------
# Display Panels
# ------------------
st.subheader(f"📊 {symbol} • {exchange.upper()} • 15m")

# Last trained info
mtime_local = dt.datetime.fromtimestamp(os.path.getmtime(path))
st.caption(f"🕒 Last trained: {mtime_local.strftime('%Y-%m-%d %H:%M:%S')} (local)")

# Current price (from last fetched actual)
if actuals is not None and not actuals.empty:
    last_price = float(actuals["close"].iloc[-1])
    st.metric(label=f"💰 Current {symbol} Price", value=f"${last_price:,.2f}")

# Table preview + download
preview_rows = min(steps, 200)
st.dataframe(forecast_df.head(preview_rows))

csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Forecast CSV", csv_bytes, file_name=f"{symbol}_{exchange}_15m_forecast.csv", mime="text/csv")

# ------------------
# Chart
# ------------------
try:
    import altair as alt
    # Prepare long-form data for overlay
    pred = forecast_df.rename(columns={"timestamp": "time", "pred_close": "price"}).copy()
    pred["series"] = "Forecast"

    charts = []
    layers = []

    if actuals is not None and not actuals.empty:
        act = actuals[["start", "close"]].rename(columns={"start": "time", "close": "price"}).copy()
        act["time"] = pd.to_datetime(act["time"], utc=True)
        # Cut to reasonable range (last 7 days + forecast horizon)
        min_time = pred["time"].min() - pd.Timedelta(days=7)
        act = act[act["time"] >= min_time]
        act["series"] = "Actual"
        combined = pd.concat([act, pred], ignore_index=True)
    else:
        combined = pred

    line = alt.Chart(combined).mark_line().encode(
        x=alt.X("time:T", title="Time (UTC)"),
        y=alt.Y("price:Q", title="Price"),
        color=alt.Color("series:N", scale=alt.Scale(scheme="category10")),
        tooltip=["series","time:T","price:Q"]
    ).properties(height=380)

    st.altair_chart(line, use_container_width=True)
except Exception:
    # Fallback to Streamlit line_chart (forecast only)
    fplot = forecast_df.set_index("timestamp")["pred_close"]
    st.line_chart(fplot)

# ------------------
# Notes / Tips
# ------------------
st.markdown("---")
st.markdown(
    """
    **Notes**
⚠️ Disclaimer
This application is provided for educational and informational purposes only.
The forecasts and data presented are experimental outputs from machine learning models trained on historical market information and do not constitute financial advice.
Cryptocurrency markets are volatile and unpredictable; past performance or modeled predictions are not indicative of future results.
Always conduct your own research and consult with a licensed financial professional before making any trading or investment decisions.
By using this tool, you acknowledge that you assume full responsibility for any financial outcomes resulting from your actions.
    """
)
