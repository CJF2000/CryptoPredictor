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
# Streamlit Config
# =====================================================
st.set_page_config(page_title="🔮 Crypto Forecast", layout="centered")
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      .stDeployButton {display:none;}
      body {background-color: #0b0014; color: #e0d7ff;}
      h1, h2, h3, h4, h5, h6 {text-align: center;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔮 Crypto Forecast")

# =====================================================
# Password Gate
# =====================================================
if "unlocked" not in st.session_state:
    st.session_state.unlocked = False

if not st.session_state.unlocked:
    password = st.text_input("Enter Access Password", type="password")
    if password == st.secrets.get("ACCESS_PASSWORD", "Crypto_Forecast777"):
        st.session_state.unlocked = True
        st.success("✅ Access granted.")
        st.rerun()
    elif password:
        st.warning("Access denied. Incorrect password.")
        st.stop()
    else:
        st.info("Please enter your access password to continue.")
        st.stop()

# =====================================================
# Config / Constants (Always 15m)
# =====================================================
TIMEFRAME = "15m"
BARS_PER_DAY = 96
LOOK_BACK = 64
EPOCHS = 50
BATCH_SIZE = 256
VAL_SPLIT = 0.1
np.random.seed(42)

OKX_BAR = "15m"
COINBASE_GRANULARITY = 900
BITFINEX_TF = "15m"

FEATURES_KEEP = [
    "close","high","low","volume",
    "vwap","rsi14","fib_conf","fib_nearest_bps",
    "dist_swing_high_bps","dist_swing_low_bps","dist_equal_highs_bps","dist_equal_lows_bps"
]

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
    """Auto-select a source with enough 15m data: OKX perp → Coinbase spot → Bitfinex spot → OKX spot."""
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
    """Train LSTM on engineered features and produce iterative close forecasts."""
    df = df_raw.rename(columns=str.lower).set_index("start")
    feats = build_feature_frame(df)  # from features.py (VWAP, RSI, Fib confluence, Liquidity)
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

    # Iterative forecast
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

def ensure_forecast_dir():
    os.makedirs("intraday_forecasts", exist_ok=True)

def forecast_path(coin: str) -> str:
    ensure_forecast_dir()
    return os.path.join("intraday_forecasts", f"{coin.upper()}_15m_forecast.csv")

# Robust normalizer for actuals used by confidence metric
def normalize_actuals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["start", "close"])
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    if "start" not in out.columns:
        if "timestamp" in out.columns:
            out["start"] = out["timestamp"]
        elif "time" in out.columns:
            out["start"] = out["time"]
        elif out.index.name in ("start", "time", "timestamp"):
            out["start"] = out.index
        else:
            out["start"] = pd.NaT
    if "close" not in out.columns and "closing_price" in out.columns:
        out["close"] = out["closing_price"]
    out["start"] = pd.to_datetime(out["start"], utc=True, errors="coerce")
    out = out.dropna(subset=["start", "close"]).sort_values("start")
    return out[["start", "close"]]

# =====================================================
# Forecast UI (Train All + Per-Coin + Confidence)
# =====================================================
st.subheader("Forecast Dashboard")

# 🛠 Train All Coins Button (once per day cached)
if st.button("🛠 Train All Coins for Today"):
    with st.spinner("Training all coins (BTC, ETH, XRP, SOL, SUI)... this may take several minutes."):
        try:
            for base in ["BTC", "ETH", "XRP", "SOL", "SUI"]:
                path_all = forecast_path(base)
                # Skip if already trained today
                if os.path.exists(path_all):
                    mtime = dt.datetime.fromtimestamp(os.path.getmtime(path_all))
                    if mtime.date() == dt.datetime.utcnow().date():
                        st.info(f"ℹ️ {base}: already trained today — skipping.")
                        continue

                exc, sym, _ = try_fetch_sample_15m(base, lookback_days=30)
                end = dt.datetime.utcnow()
                start = end - dt.timedelta(days=365 * 3)

                if exc == "okx":
                    df_full = fetch_okx_klines(sym, start, end, bar=OKX_BAR, limit=100)
                elif exc == "coinbase":
                    df_full = fetch_coinbase_klines(sym, start, end, granularity=COINBASE_GRANULARITY)
                else:
                    df_full = fetch_bitfinex_klines(sym, start, end, timeframe=BITFINEX_TF, limit=10_000)

                if df_full is None or df_full.empty:
                    st.warning(f"⚠️ {base}: No data returned from {exc.upper()}")
                    continue

                horizon_steps = 7 * BARS_PER_DAY  # default 7D for bulk run
                idx_hist, preds = train_and_forecast_from_df(df_full, horizon_steps=horizon_steps)
                last_t = idx_hist[-1] + pd.Timedelta(minutes=15)
                future_idx = pd.date_range(last_t, periods=horizon_steps, freq="15min", tz="UTC")
                forecast_df_all = pd.DataFrame({"timestamp": future_idx, "pred_close": preds})
                forecast_df_all.to_csv(path_all, index=False)
                st.success(f"✅ {base} trained and cached using {exc.upper()} ({sym})")
            st.success("🎯 All forecasts updated for today!")
        except Exception as e:
            st.error(f"Training failed: {e}")

st.write("---")
st.subheader("Generate Individual Forecast")

coin = st.selectbox("🪙 Choose Coin", ["BTC", "ETH", "XRP", "SOL", "SUI"])
days = st.slider("📆 Days to Forecast", 1, 7, 2)
STEPS = days * BARS_PER_DAY
path = forecast_path(coin)

# Determine if we can use cache (trained today)
use_cache = False
mtime = None
if os.path.exists(path):
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
    if mtime.date() == dt.datetime.utcnow().date():
        use_cache = True

if st.button("🔮 Generate Forecast"):
    with st.spinner(f"{'Loading' if use_cache else 'Training'} model for {coin}..."):
        try:
            if use_cache:
                forecast_df = pd.read_csv(path, parse_dates=["timestamp"])
                st.success(f"✅ Loaded cached forecast for {coin} (trained {mtime.strftime('%Y-%m-%d')})")
            else:
                exc, sym, _ = try_fetch_sample_15m(coin, lookback_days=30)
                end = dt.datetime.utcnow()
                start = end - dt.timedelta(days=365 * 3)

                if exc == "okx":
                    df_full = fetch_okx_klines(sym, start, end, bar=OKX_BAR, limit=100)
                elif exc == "coinbase":
                    df_full = fetch_coinbase_klines(sym, start, end, granularity=COINBASE_GRANULARITY)
                else:
                    df_full = fetch_bitfinex_klines(sym, start, end, timeframe=BITFINEX_TF, limit=10_000)

                if df_full is None or df_full.empty:
                    raise RuntimeError("No data returned from source.")

                idx_hist, preds = train_and_forecast_from_df(df_full, horizon_steps=STEPS)
                last_t = idx_hist[-1] + pd.Timedelta(minutes=15)
                future_idx = pd.date_range(last_t, periods=STEPS, freq="15min", tz="UTC")
                forecast_df = pd.DataFrame({"timestamp": future_idx, "pred_close": preds})
                ensure_forecast_dir()
                forecast_df.to_csv(path, index=False)
                st.success(f"✅ Forecast complete using {exc.upper()} ({sym}) and cached for today")

            # --- Confidence metric (robust) ---
            try:
                end_check = dt.datetime.utcnow()
                start_check = end_check - dt.timedelta(days=3)
                exc_a, sym_a, _ = try_fetch_sample_15m(coin, lookback_days=7)

                if exc_a == "okx":
                    df_actual = fetch_okx_klines(sym_a, start_check, end_check, bar=OKX_BAR, limit=500)
                elif exc_a == "coinbase":
                    df_actual = fetch_coinbase_klines(sym_a, start_check, end_check, granularity=COINBASE_GRANULARITY)
                else:
                    df_actual = fetch_bitfinex_klines(sym_a, start_check, end_check, timeframe=BITFINEX_TF, limit=10_000)

                df_actual = normalize_actuals(df_actual).rename(columns={"close": "actual"})

                if df_actual.empty:
                    confidence = None
                else:
                    f_sorted = forecast_df.sort_values("timestamp").copy()
                    a_sorted = df_actual.sort_values("start").copy()

                    merged = pd.merge_asof(
                        f_sorted, a_sorted,
                        left_on="timestamp", right_on="start",
                        direction="nearest",
                        tolerance=pd.Timedelta(minutes=15)
                    )

                    if "actual" not in merged or merged["actual"].isna().all():
                        confidence = None
                    else:
                        merged = merged.dropna(subset=["actual", "pred_close"])
                        merged["diff_pct"] = (merged["pred_close"] - merged["actual"]).abs() / merged["actual"] * 100.0
                        confidence = round((merged["diff_pct"] <= 1.0).mean() * 100.0, 2)
            except Exception:
                confidence = None

            # --- Display results ---
            st.metric(
                label="🤖 Model Confidence (past 3 days, ±1%)",
                value=f"{confidence:.1f}%" if confidence is not None else "N/A",
            )
            st.line_chart(forecast_df.set_index("timestamp")["pred_close"])
            st.download_button(
                "📥 Download Forecast CSV",
                forecast_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{coin}_forecast.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Forecast failed: {e}")

st.markdown("---")
st.markdown("""
### ⚠️ Disclaimer
This application is for **educational and informational purposes only**. The forecasts are **experimental outputs** from machine learning models trained on historical data and **do not constitute financial advice**. Markets are volatile; past performance or model predictions are **not indicative of future results**. Use at your own discretion.
""")
