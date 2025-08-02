import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import os
import joblib
from train_daily import prepare_data, build_model

from tensorflow.keras.models import load_model

st.set_page_config(page_title="Crypto Forecast Bot", layout="centered")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("📈 Crypto Forecast Bot")
st.write("This bot predicts Close, High, and Low prices 7 days into the future using technical indicators and LSTM.")

COINS = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD"]
look_back = 30
forecast_days = 7

@st.cache_data(show_spinner=False)
def load_data(coin):
    df = yf.download(coin, period="5y", interval="1d", progress=False)
    return df

def train_and_save(coin):
    df = load_data(coin)
    try:
        X, y, scaler, df_full = prepare_data(df, look_back)
    except ValueError as e:
        st.error(f"{coin} - {e}")
        return None, None, None
    model = build_model(X.shape[1:])
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    model.save(f"models/{coin}.h5")
    joblib.dump(scaler, f"models/{coin}_scaler.gz")
    return model, scaler, df_full

def load_model_and_scaler(coin):
    model = load_model(f"models/{coin}.h5")
    scaler = joblib.load(f"models/{coin}_scaler.gz")
    return model, scaler

def forecast(model, scaler, df):
    try:
        _, _, _, df_processed = prepare_data(df, look_back)
    except ValueError as e:
        st.error(f"Forecast error: {e}")
        return []

    if len(df_processed) < look_back:
        st.error("Not enough data to generate a forecast.")
        return []

    latest_window = df_processed[-look_back:]

    features = ['Close', 'High', 'Low', 'VWAP', 'Return', 'Momentum', 'MACD_Hist', 
                'RSI_Delta', 'StochRSI', 'Williams_%R', 'BB%', 'ATR']

    X_input = scaler.transform(latest_window[features])
    forecasts = []

    for _ in range(forecast_days):
        input_seq = np.expand_dims(X_input[-look_back:], axis=0)
        pred = model.predict(input_seq, verbose=0)[0]
        forecasts.append(pred)
        next_input = np.append(X_input[-look_back + 1:], [pred.tolist() + [0]*(len(X_input[0]) - 3)], axis=0)
        X_input = next_input

    predicted_scaled = np.array(forecasts)
    padded_input = np.zeros((forecast_days, len(X_input[0])))
    padded_input[:, :3] = predicted_scaled
    predicted_prices = scaler.inverse_transform(padded_input)[:, :3]

    return predicted_prices

# UI
selected_coin = st.selectbox("Choose a coin", COINS)
if not os.path.exists("models"):
    os.makedirs("models")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔁 Run the Bot"):
        with st.spinner(f"Fetching data and training model for {selected_coin}..."):
            model, scaler, df_full = train_and_save(selected_coin)
            if model is not None:
                forecasted = forecast(model, scaler, df_full)
                if len(forecasted) > 0:
                    st.subheader("📊 7-Day Forecast")
                    forecast_df = pd.DataFrame(forecasted, columns=["Close", "High", "Low"])
                    forecast_df.index = pd.date_range(start=datetime.date.today() + datetime.timedelta(days=1), periods=7)
                    st.line_chart(forecast_df)

                    csv = forecast_df.to_csv().encode('utf-8')
                    st.download_button("Download CSV", data=csv, file_name=f"{selected_coin}_forecast.csv", mime='text/csv')

with col2:
    if st.button("📦 Train All Coins"):
        for coin in COINS:
            with st.spinner(f"Training model for {coin}..."):
                train_and_save(coin)
        st.success("✅ All models trained and saved.")

