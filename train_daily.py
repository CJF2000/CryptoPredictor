# train_daily.py

import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

def calculate_vwap(df):
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def prepare_data(df, look_back=30):
    df['VWAP'] = calculate_vwap(df)
    df = df.dropna()
    features = ['Close', 'High', 'Low', 'VWAP']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, :3])
    return np.array(X), np.array(y), scaler, df

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future(model, recent_input, scaler, steps=7):
    future_input = recent_input.copy()
    future_preds_scaled = []
    for _ in range(steps):
        input_seq = future_input.reshape(1, 30, 4)
        pred = model.predict(input_seq, verbose=0)[0]
        future_preds_scaled.append(pred)
        dummy_vwap = future_input[-1, 3]
        next_input = np.append(pred, dummy_vwap).reshape(1, 4)
        future_input = np.vstack([future_input[1:], next_input])
    future_preds_scaled = np.array(future_preds_scaled)
    padded_preds = np.hstack([future_preds_scaled, np.zeros((steps, 1))])
    return scaler.inverse_transform(padded_preds)[:, :3]

def train_and_save_forecast(coin, forecast_days=7, epochs=150):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    df = yf.download(coin, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    
    if df.shape[0] < 100:
        print(f"Not enough data for {coin}")
        return

    X, y, scaler, df_full = prepare_data(df)
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    recent_scaled = scaler.transform(df_full[['Close', 'High', 'Low', 'VWAP']].iloc[-30:])
    preds = predict_future(model, recent_scaled, scaler, steps=forecast_days)

    start_date = pd.to_datetime("today").normalize()
    future_dates = [(start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(forecast_days)]

    df_pred = pd.DataFrame(preds, columns=['Close', 'High', 'Low'])
    df_pred.insert(0, 'Date', future_dates)

    output_dir = "daily_forecasts"
    os.makedirs(output_dir, exist_ok=True)
    df_pred.to_csv(f"{output_dir}/{coin}_forecast.csv", index=False)
    print(f"{coin} forecast saved.")

# Run daily for all coins
if __name__ == "__main__":
    coins = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD']
    for coin in coins:
        train_and_save_forecast(coin)
