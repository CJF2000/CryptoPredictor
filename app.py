import os
import time
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

# =====================================================
# Streamlit Config + Theme
# =====================================================
st.set_page_config(page_title="🔮 Crypto Forecast Crystal Ball", layout="centered")

# --- CSS for background, doors, glow effect ---
st.markdown("""
<style>
body {
    background: radial-gradient(circle at center, #0b0014 0%, #000 100%);
    color: white;
}
h1, h2, h3, h4, h5, h6 {
    text-align: center;
    color: #e0d7ff;
    text-shadow: 0 0 15px #7a47ff;
}
.door-container {
    position: relative;
    width: 100%;
    height: 400px;
    overflow: hidden;
    background: black;
}
.left-door, .right-door {
    position: absolute;
    top: 0;
    width: 50%;
    height: 100%;
    background: linear-gradient(145deg, #3c0080, #1a0033);
    transition: all 2s ease;
    z-index: 10;
}
.left-door { left: 0; border-right: 2px solid #b999ff; }
.right-door { right: 0; border-left: 2px solid #b999ff; }
.door-open .left-door { transform: translateX(-100%); }
.door-open .right-door { transform: translateX(100%); }
.crystal-container {
    text-align: center;
    margin-top: 40px;
}
.glow {
    animation: pulse 2s infinite alternate;
}
@keyframes pulse {
  from { filter: drop-shadow(0 0 10px #8f7fff); }
  to { filter: drop-shadow(0 0 30px #b999ff); }
}
</style>
""", unsafe_allow_html=True)

CRYSTAL_BALL_GIF = "https://media.giphy.com/media/du3J3cXyzhj75IOgvA/giphy.gif"

# =====================================================
# Password Gate (Door Animation)
# =====================================================
if "unlocked" not in st.session_state:
    st.session_state.unlocked = False

st.title("🔮 Welcome to the Crystal Ball Forecast")

if not st.session_state.unlocked:
    password = st.text_input("Enter Access Password", type="password")
    if password == st.secrets.get("ACCESS_PASSWORD", "Crypto_Forecast777"):
        st.session_state.unlocked = True
        with st.spinner("✨ The doors are opening..."):
            time.sleep(2)
        st.markdown(
            "<div class='door-container door-open'><div class='left-door'></div><div class='right-door'></div></div>",
            unsafe_allow_html=True,
        )
        time.sleep(2)
        st.experimental_rerun()
    else:
        st.warning("Access Denied.")
        st.stop()

# =====================================================
# Core Settings (always 15m candles)
# =====================================================
TIMEFRAME = "15m"
BARS_PER_DAY = 96
LOOK_BACK = 64

OKX_BAR = "15m"
COINBASE_GRANULARITY = 900
BITFINEX_TF = "15m"

FEATURES_KEEP = [
    "close","high","low","volume",
    "vwap","rsi14","fib_conf","fib_nearest_bps",
    "dist_swing_high_bps","dist_swing_low_bps","dist_equal_highs_bps","dist_equal_lows_bps"
]

EPOCHS = 50
BATCH_SIZE = 256
VAL_SPLIT = 0.1
np.random.seed(42)

# =====================================================
# Utility Functions
# =====================================================
def build_symbols(base: str):
    base = base.upper().strip()
    return {
        "okx_swap": f"{base}-USDT-SWAP",
        "okx_spot": f"{base}-USDT",
        "coinbase": f"{base}-USD",
        "bitfinex": f"t{base}USD",
    }

def try_fetch_sample_15m(base_symbol: str, lookback_days: int = 14):
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
            if df is not None and not df.empty and len(df) >= LOOK_BACK + 50:
                return exc, sym, df
        except Exception:
            continue
    raise RuntimeError("No source returned enough 15m data for the chosen coin.")

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
        Dense(1),
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
    X_seq, y_seq = windowize(X_scaled, y_scaled)
    v = max(1, int(len(X_seq) * VAL_SPLIT))
    Xtr, ytr, Xv, yv = X_seq[:-v], y_seq[:-v], X_seq[-v:], y_seq[-v:]
    model = build_model((X_seq.shape[1], X_seq.shape[2]))
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[es])
    last_window = X_scaled[-LOOK_BACK:].copy()
    preds_scaled = []
    window = last_window
    close_idx = FEATURES_KEEP.index("close")
    for _ in range(horizon_steps):
        x = window.reshape(1, *window.shape)
        y_hat = model.predict(x, verbose=0)[0][0]
        preds_scaled.append([y_hat])
        next_row = window[-1].copy()
        next_row[close_idx] = y_hat
        window = np.vstack([window[1:], next_row])
    preds = tgt_scaler.inverse_transform(np.array(preds_scaled)).ravel()
    return feats.index, preds

def cache_path(base_symbol: str, days: int):
    os.makedirs("intraday_forecasts", exist_ok=True)
    return os.path.join("intraday_forecasts", f"{base_symbol.upper()}_15m_{days}d.csv")

@st.cache_data(show_spinner=False)
def load_cached(path: str):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=["timestamp"])
        except Exception:
            return None
    return None

# =====================================================
# Crystal Ball Interface
# =====================================================
st.markdown(
    f"<div class='crystal-container'><img src='{CRYSTAL_BALL_GIF}' width='300' class='glow'></div>",
    unsafe_allow_html=True,
)
st.markdown("<h2>Peer into the crystal ball...</h2>", unsafe_allow_html=True)

coin = st.selectbox("🪙 Choose Your Coin", ["BTC", "ETH", "XRP", "SOL", "SUI"])
days = st.slider("📆 Days to Forecast", 1, 7, 2)
STEPS = days * BARS_PER_DAY

if st.button("🔮 Reveal the Future"):
    with st.spinner(f"Consulting the spirits for {coin}..."):
        try:
            exc, sym, sample = try_fetch_sample_15m(coin, lookback_days=30)
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
            last_t = idx_hist[-1] + pd.Timedelta(minutes=15)
            future_idx = pd.date_range(last_t, periods=STEPS, freq="15min", tz="UTC")
            forecast_df = pd.DataFrame({"timestamp": future_idx, "pred_close": preds})
            st.success("✨ The vision is clear!")
            st.balloons()
            st.markdown(f"<h3 style='text-align:center;'>📈 {coin}-USD Forecast for Next {days} Days</h3>", unsafe_allow_html=True)
            st.line_chart(forecast_df.set_index("timestamp")["pred_close"])
        except Exception as e:
            st.error(f"Training failed: {e}")

st.markdown("---")
st.markdown("""
### ⚠️ Disclaimer
This app is for **educational and informational purposes only**. The forecasts are **experimental** outputs of machine learning models trained on historical data and **do not constitute financial advice**. Markets are volatile; past performance or predictions are **not guarantees of future results**. Use responsibly.
""")
