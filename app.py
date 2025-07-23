import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

# --- UI Setup ---
st.set_page_config(page_title="Crypto Forecast", layout="wide")
st.title("Crypto Forecast Bot")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Options ---
st.sidebar.header("Options")
coin = st.sidebar.selectbox("Select Coin", ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD"])
forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=15, value=7)
look_back = st.sidebar.slider("Look Back Days", min_value=10, max_value=90, value=30)

# --- Prepare Data Function ---
def prepare_data(df, look_back=30, forecast_days=7):
    df.index = pd.to_datetime(df.index)

    macd = MACD(close=df['Close'])
    df['MACD'] = pd.Series(macd.macd().values.ravel(), index=df.index)

    rsi = RSIIndicator(close=df['Close'])
    df['RSI'] = pd.Series(rsi.rsi().values.ravel(), index=df.index)

    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
    df['ATR'] = pd.Series(atr.average_true_range().values.ravel(), index=df.index)

    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['DayOfWeek'] = df.index.dayofweek

    df = df.dropna()

    features = ['Close', 'High', 'Low', 'Volume', 'VWAP', 'MACD', 'RSI', 'ATR', 'DayOfWeek']
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df[features])

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(df[['Close', 'High', 'Low']])

    X, y = [], []
    for i in range(look_back, len(df) - forecast_days + 1):
        X.append(X_scaled[i - look_back:i])
        y_seq = y_scaled[i: i + forecast_days]
        y.append(y_seq.flatten())

    return np.array(X), np.array(y), scaler_X, scaler_y, df

# --- Build LSTM Model ---
def build_model(input_shape, output_units):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Forecasting Function ---
def make_forecast(model, recent_data):
    prediction = model.predict(np.expand_dims(recent_data, axis=0))
    return prediction[0]

# --- Run Forecast ---
if st.button("Run Forecast"):
    with st.spinner(f"Fetching data and training model for {coin}..."):
        df = yf.download(coin, start="2014-01-01", end=datetime.datetime.now())

        if df.shape[0] < 100:
            st.error("⚠️ Not enough data to evaluate.")
        else:
            X, y, scaler_X, scaler_y, df_full = prepare_data(df, look_back, forecast_days)
            X_recent = scaler_X.transform(df_full.iloc[-look_back:][['Close', 'High', 'Low', 'Volume', 'VWAP', 'MACD', 'RSI', 'ATR', 'DayOfWeek']])
            model = build_model((X.shape[1], X.shape[2]), y.shape[1])
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            prediction_scaled = make_forecast(model, X_recent)

            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(forecast_days, 3))

            future_dates = pd.date_range(df_full.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame(prediction, columns=['Predicted Close', 'Predicted High', 'Predicted Low'], index=future_dates)

            st.subheader(f"{coin} - {forecast_days}-Day Forecast")
            st.line_chart(forecast_df[['Predicted Close']])
            st.dataframe(forecast_df, use_container_width=True)
