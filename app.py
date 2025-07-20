import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
import datetime

st.set_page_config(page_title="Crypto Predictor", layout="centered")
st.title("🔮 7-Day Crypto Price Predictor (LSTM)")

# --- User Input ---
user_coin = st.text_input("Enter a crypto symbol (e.g., BTC-USD, ETH-USD):", "BTC-USD")

# --- Helper: VWAP Calculation ---
def calculate_vwap(df):
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

# --- Prepare LSTM Data ---
def prepare_data(df, look_back=30):
    df['VWAP'] = calculate_vwap(df)
    df = df.dropna()
    features = ['Close', 'High', 'Low', 'VWAP']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, :3])
    return np.array(X), np.array(y), scaler, df

# --- Build LSTM Model ---
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=64, return_sequences=True),
        LSTM(units=32),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Predict Future ---
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

# --- Run App Logic ---
if user_coin:
    with st.spinner("Fetching data and predicting with LSTM..."):
        try:
            end_date = datetime.datetime.now()
            start_date = datetime.datetime(2014, 1, 1)
            df = yf.download(user_coin, start=start_date, end=end_date).dropna()

            if df.shape[0] < 100:
                st.error("⚠️ Not enough historical data to predict.")
            else:
                X, y, scaler, df_full = prepare_data(df)
                model = build_model((X.shape[1], X.shape[2]))
                model.fit(X, y, epochs=20, batch_size=32, verbose=0)

                recent_scaled = scaler.transform(df_full[['Close', 'High', 'Low', 'VWAP']].iloc[-30:])
                preds = predict_future(model, recent_scaled, scaler, steps=7)

                last_date = df.index[-1]
                future_dates = [(last_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
                df_pred = pd.DataFrame(preds, columns=['Predicted Close (USD)', 'Predicted High (USD)', 'Predicted Low (USD)'])
                df_pred.insert(0, 'Date', future_dates)

                st.subheader(f"📈 Prediction for {user_coin}")
                st.dataframe(df_pred, use_container_width=True)

                # Download CSV
                csv = df_pred.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, f"{user_coin}_7_day_forecast.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")
