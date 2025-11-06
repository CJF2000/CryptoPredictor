import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Feature engineering (VWAP/RSI/Fib/Liquidity)
from features import build_feature_frame
# Public US-accessible data sources
from data_sources import (
    fetch_okx_klines,
    fetch_coinbase_klines,
    fetch_bitfinex_klines,
)

# ==============================
# Streamlit Config + Access Gate
# ==============================
st.set_page_config(page_title="Crypto Forecast", layout="centered")
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

st.title("🔮 Crypto Forecast")

PASSWORD = st.secrets.get("ACCESS_PASSWORD", None)
if PASSWORD:
    pwd = st.text_input("Enter Access Password", type="password")
    if pwd != PASSWORD:
        st.warning("Access denied. Set ACCESS_PASSWORD in .streamlit/secrets.toml")
        st.stop()
    st.success("✅ Access granted.")
else:
    st.info("No ACCESS_PASSWORD set — running in open mode.")

st.markdown(
    "> ⚠️ **Educational only** — not financial advice. Forecasts are experimental and can be noisy, especially at longer horizons."
)

# ==============================
# UI: Coin + Forecast Horizon (days)
# ==============================
col1, col2 = st.columns([2, 1])
with col1:
    coin = st.text_input("🪙 Coin (base symbol)", value="BTC", help="Examples: BTC, ETH, SOL, XRP")
with col2:
    days = st.slider("📆 Days to Forecast", min_value=1, max_value=7, value=2)

# Always 15m bars
TIMEFRAME = "15m"
BARS_PER_DAY = 96
STEPS = days * BARS_PER_DAY   # 15m bars per day
LOOK_BACK = 64                # sequence length for 15m regime

# ==============================
# Symbol builders (15m only)
# ==============================

def build_symbols(base: str):
    base = base.upper().strip()
    return {
        "okx_swap": f"{base}-USDT-SWAP",  # perp
        "okx_spot": f"{base}-USDT",
        "coinbase": f"{base}-USD",
        "bitfinex": f"t{base}USD",
    }

# 15m params for each API
OKX_BAR = "15m"
COINBASE_GRANULARITY = 900   # seconds
BITFINEX_TF = "15m"
STEP_MS = 15 * 60 * 1000

# ==============================
# Auto-select a working data source (15m only)
# ==============================

def try_fetch_sample_15m(base_symbol: str, lookback_days: int = 14):
    """Try OKX perp → Coinbase spot → Bitfinex spot → OKX spot. Return (exchange, symbol, df)."""
    syms = build_symbols(base_symbol)
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days)

    try_order = [
        ("okx", syms["okx_swap"]),
        ("coinbase", syms["coinbase"]),
        ("bitfinex", syms["bitfinex"]),
        ("okx", syms["okx_spot"]),
    ]

    for exc, sym in try_order:
        try:
            if exc == "okx":
                df = fetch_okx_klines(sym, start, end, bar=OKX_BAR, limit=100)
            elif exc == "coinbase":
                df = fetch_coinbase_klines(sym, start, end, granularity=COINBASE_GRANULARITY)
            else:
                df = fetch_bitfinex_klines(sym, start, end, timeframe=BITFINEX_TF, limit=10_000)
            if df is not None and not df.empty and len(df) >= max(LOOK_BACK + 50, 200):
                return exc, sym, df
        except Exception:
            continue
    raise RuntimeError("No public source returned enough 15m data for the chosen coin.")

# ==============================
# Model helpers (15m)
# ==============================
FEATURES_KEEP = [
    "close","high","low","volume",
    "vwap","rsi14","fib_conf","fib_nearest_bps",
    "dist_swing_high_bps","dist_swing_low_bps","dist_equal_highs_bps","dist_equal_lows_bps"
]

EPOCHS = 50
BATCH_SIZE = 256
VAL_SPLIT = 0.1
SEED = 42
np.random.seed(SEED)


def windowize(X, y, look_back=LOOK_BACK):
    Xs, ys = [], []
    for i in range(look_back, len(X)):
        Xs.append(X[i - look_back : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_model(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        LSTM(96, return_sequences=True),
        Dropout(0.15),
        LSTM(64),
        Dense(1),  # predict close only
    ])
    m.compile(optimizer="adam", loss="mae")
    return m


def train_and_forecast_from_df(df_raw: pd.DataFrame, horizon_steps: int):
    df = df_raw.rename(columns=str.lower).set_index("start")
    feats = build_feature_frame(df)
    feats = feats[FEATURES_KEEP].copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feats)
    tgt_scaler = MinMaxScaler()
    y_scaled = tgt_scaler.fit_transform(feats[["close"]])

    X_seq, y_seq = windowize(X_scaled, y_scaled, look_back=LOOK_BACK)
    n = len(X_seq)
    v = max(1, int(n * VAL_SPLIT))
    Xtr, ytr, Xv, yv = X_seq[:-v], y_seq[:-v], X_seq[-v:], y_seq[-v:]

    model = build_model((X_seq.shape[1], X_seq.shape[2]))
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[es])

    # iterative forecast on last window
    last_window = X_scaled[-LOOK_BACK:].copy()
    preds_scaled = []
    window = last_window
    close_idx = FEATURES_KEEP.index("close")

    for _ in range(horizon_steps):
        x = window.reshape(1, *window.shape)
        y_hat = model.predict(x, verbose=0)[0][0]  # scaled close
        preds_scaled.append([y_hat])
        next_row = window[-1].copy()
        next_row[close_idx] = y_hat
        window = np.vstack([window[1:], next_row])

    preds = tgt_scaler.inverse_transform(np.array(preds_scaled)).ravel()
    return feats.index, preds

# ==============================
# Cache helpers
# ==============================

def cache_path(base_symbol: str, days: int):
    os.makedirs("intraday_forecasts", exist_ok=True)
    tag = f"{base_symbol.upper()}_15m_{days}d"
    return os.path.join("intraday_forecasts", f"{tag}.csv")

@st.cache_data(show_spinner=False)
def load_cached(path: str):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            return df
        except Exception:
            return None
    return None

# ==============================
# Train / Load Flow (always 15m)
# ==============================
path = cache_path(coin, days)
force_retrain = st.checkbox("Force retrain today", value=False)

use_cache = False
cached = load_cached(path)
if cached is not None and len(cached) >= STEPS:
    mtime = dt.datetime.utcfromtimestamp(os.path.getmtime(path)).date()
    if mtime == dt.datetime.utcnow().date() and not force_retrain:
        use_cache = True

if st.button("🛠 Train / Update Forecast"):
    force_retrain = True
    use_cache = False

if not use_cache:
    with st.spinner(f"Resolving data source and training {coin} on 15m…"):
        try:
            exc, sym, sample = try_fetch_sample_15m(coin, lookback_days=30)
            # fetch full 3-year 15m history
            end = dt.datetime.utcnow()
            start = end - dt.timedelta(days=365*3)

            if exc == "okx":
                df_full = fetch_okx_klines(sym, start, end, bar=OKX_BAR, limit=100)
            elif exc == "coinbase":
                df_full = fetch_coinbase_klines(sym, start, end, granularity=COINBASE_GRANULARITY)
            else:
                df_full = fetch_bitfinex_klines(sym, start, end, timeframe=BITFINEX_TF, limit=10_000)

            if df_full is None or df_full.empty:
                raise RuntimeError("Selected source returned no data.")

            idx_hist, preds = train_and_forecast_from_df(df_full, horizon_steps=STEPS)

            # build future timestamps at 15m increments
            last_t = idx_hist[-1] + pd.Timedelta(minutes=15)
            future_idx = pd.date_range(last_t, periods=STEPS, freq="15min", tz="UTC")
            out = pd.DataFrame({"timestamp": future_idx, "pred_close": preds})
            out.to_csv(path, index=False)
            cached = out
            st.success(f"✅ Trained on {exc.upper()} ({sym}) • saved → {path}")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()
else:
    st.caption("Using cached forecast from today.")

# ==============================
# Fetch actuals (15m) for overlay
# ==============================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_actuals_15m_auto(base_symbol: str, start: dt.datetime, end: dt.datetime):
    try:
        exc, sym, _ = try_fetch_sample_15m(base_symbol, lookback_days=7)
        if exc == "okx":
            df = fetch_okx_klines(sym, start, end, bar=OKX_BAR, limit=100)
        elif exc == "coinbase":
            df = fetch_coinbase_klines(sym, start, end, granularity=COINBASE_GRANULARITY)
        else:
            df = fetch_bitfinex_klines(sym, start, end, timeframe=BITFINEX_TF, limit=10_000)
        return df.rename(columns=str.lower)
    except Exception:
        # final fallback: OKX spot 15m
        try:
            df = fetch_okx_klines(f"{base_symbol.upper()}-USDT", start, end, bar=OKX_BAR, limit=100)
            return df.rename(columns=str.lower)
        except Exception:
            return pd.DataFrame()

# ==============================
# Display
# ==============================
if cached is None or cached.empty:
    st.error("No forecast available.")
    st.stop()

forecast_df = cached.copy()
forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"], utc=True)

mtime_local = dt.datetime.fromtimestamp(os.path.getmtime(path))
st.caption(f"🕒 Last trained: {mtime_local.strftime('%Y-%m-%d %H:%M:%S')} (local)")

# Overlay with recent actuals
start_actuals = forecast_df["timestamp"].iloc[0] - pd.Timedelta(days=7)
end_actuals = forecast_df["timestamp"].iloc[-1]
actuals = fetch_actuals_15m_auto(coin, start_actuals, end_actuals)

# Metric
if actuals is not None and not actuals.empty:
    last_price = float(actuals["close"].iloc[-1])
    st.metric(label=f"💰 Current {coin} Price", value=f"${last_price:,.2f}")

# Table + download
preview_rows = min(STEPS, 200)
st.dataframe(forecast_df.head(preview_rows))
st.download_button(
    "📥 Download Forecast CSV",
    forecast_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{coin.upper()}_15m_{days}d_forecast.csv",
    mime="text/csv",
)

# Chart
try:
    import altair as alt
    pred = forecast_df.rename(columns={"timestamp": "time", "pred_close": "price"}).copy()
    pred["series"] = "Forecast"

    if actuals is not None and not actuals.empty:
        act = actuals[["start", "close"]].rename(columns={"start": "time", "close": "price"}).copy()
        act["time"] = pd.to_datetime(act["time"], utc=True)
        act = act[act["time"] >= pred["time"].min() - pd.Timedelta(days=7)]
        act["series"] = "Actual"
        combined = pd.concat([act, pred], ignore_index=True)
    else:
        combined = pred

    line = alt.Chart(combined).mark_line().encode(
        x=alt.X("time:T", title="Time (UTC)"),
        y=alt.Y("price:Q", title="Price"),
        color=alt.Color("series:N"),
        tooltip=["series", "time:T", "price:Q"],
    ).properties(height=380)

    st.altair_chart(line, use_container_width=True)
except Exception:
    st.line_chart(forecast_df.set_index("timestamp")["pred_close"])

st.markdown("---")
st.markdown(
    """
### ⚠️ Disclaimer
This application is provided for educational and informational purposes only. The forecasts are experimental outputs from machine learning models trained on historical data and do not constitute financial advice. Markets are volatile; past performance or model predictions are not indicative of future results. Always do your own research and consult a licensed professional. By using this tool, you accept full responsibility for any financial outcomes.
"""
)
