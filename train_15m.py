# train_15m.py
import os
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from data_sources import fetch_ohlcv
from features import build_feature_frame

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

LOOK_BACK = 64              # longer memory for 15m regime
FORECAST_STEPS = 7*24*4     # 7 days * 24h * 4 bars/h = 672 steps (you can reduce to e.g., 96–192)
EPOCHS = 60
BATCH_SIZE = 256
VAL_SPLIT = 0.1

TARGETS = ["close"]         # predict close; you can expand to [close, high, low]
FEATURES_KEEP = [
    "close","high","low","volume",
    "vwap","rsi14","fib_conf","fib_nearest_bps",
    "dist_swing_high_bps","dist_swing_low_bps","dist_equal_highs_bps","dist_equal_lows_bps"
]

def windowize(X, y, look_back=LOOK_BACK):
    Xs, ys = [], []
    for i in range(look_back, len(X)):
        Xs.append(X[i-look_back:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_model(input_shape):
    m = Sequential([
        Input(shape=input_shape),        # (look_back, n_features)
        LSTM(96, return_sequences=True),
        Dropout(0.15),
        LSTM(64),
        Dense(len(TARGETS))
    ])
    m.compile(optimizer="adam", loss="mae")
    return m

def prepare(exchange, symbol):
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=365*3)
    df = fetch_ohlcv(exchange, symbol, start, end)
    if df.empty or len(df) < 5000:
        raise RuntimeError("Not enough data fetched.")
    df = df.rename(columns=str.lower)
    df = df.set_index("start")
    feats = build_feature_frame(df)
    feats = feats[FEATURES_KEEP]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feats)
    y_scaled = scaler.fit_transform(feats[TARGETS])  # re-fit to target scale? better use separate scaler:
    # Better:
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(feats[TARGETS])

    X_seq, y_seq = windowize(X_scaled, y_scaled)
    n = len(X_seq)
    v = max(1, int(n * VAL_SPLIT))
    return feats.index, X_seq[:-v], y_seq[:-v], X_seq[-v:], y_seq[-v:], scaler, target_scaler, feats, df

def iterative_forecast(model, last_window, steps, n_features, target_scaler):
    preds = []
    window = last_window.copy()  # shape (look_back, n_features)
    for _ in range(steps):
        x = window.reshape(1, *window.shape)
        y_hat = model.predict(x, verbose=0)[0]  # scaled
        preds.append(y_hat)
        # naive: append predicted *target* only; keep other features persistent
        next_row = window[-1].copy()
        # Put predicted close back into the close feature position (index 0 in FEATURES_KEEP)
        close_idx = FEATURES_KEEP.index("close")
        next_row[close_idx] = y_hat[0]
        window = np.vstack([window[1:], next_row])
    preds = np.array(preds)
    inv = target_scaler.inverse_transform(preds)
    return inv[:, 0]  # close

def train_and_forecast(exchange: str, symbol: str, horizon_steps=FORECAST_STEPS, epochs=EPOCHS):
    idx, Xtr, ytr, Xv, yv, scaler, tgt_scaler, feats, raw = prepare(exchange, symbol)
    model = build_model((Xtr.shape[1], Xtr.shape[2]))
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=epochs, batch_size=BATCH_SIZE, verbose=0, callbacks=[es])

    last_window = scaler.transform(feats.iloc[-LOOK_BACK:])
    n_features = last_window.shape[1]
    preds = iterative_forecast(model, last_window, horizon_steps, n_features, tgt_scaler)

    start_time = feats.index[-1] + pd.Timedelta(minutes=15)
    dates = pd.date_range(start_time, periods=horizon_steps, freq="15min", tz="UTC")
    out = pd.DataFrame({"timestamp": dates, "pred_close": preds})
    return out, model

if __name__ == "__main__":
    # Example: Bybit BTCUSDT perpetual
    out, _ = train_and_forecast(exchange="bybit", symbol="BTCUSDT", horizon_steps=96, epochs=40)
    os.makedirs("intraday_forecasts", exist_ok=True)
    out.to_csv("intraday_forecasts/BTCUSDT_bybit_15m.csv", index=False)
    print("Saved: intraday_forecasts/BTCUSDT_bybit_15m.csv")
