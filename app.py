import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# Technical indicators
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(page_title="Crypto Forecast Bot", layout="centered")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Crypto Forecast Bot")
st.markdown("""
> ⚠️ **Disclaimer:** This tool is for educational/informational purposes only.  
> It is not financial advice—use it to help confirm trends, not to predict the future.
""")

# Password protection
password = st.text_input("Enter Access Password", type="password")
if password != "brickedalpha":
    st.warning("Access denied. DM @Forecast_Wizard for the password.")
    st.stop()
st.success("✅ Access granted.")

# Config options
look_back = 30
forecast_days = st.slider("Forecast Days", 1, 15, 7)
coin = st.selectbox("Choose a coin", ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD'])

# -----------------------------
# Data Preparation (final fixed)
# -----------------------------
def prepare_data(df, look_back=30, forecast_days=7):
    df.index = pd.to_datetime(df.index)

    # --- Indicators (using raw internal arrays to avoid shape errors) ---
    macd = MACD(close=df['Close'])
    df['MACD'] = macd._macd.flatten()

    rsi = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi._rsi.flatten()

    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
    df['ATR'] = atr._atr.flatten()

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

# -----------------------------
# Model Builder
# -----------------------------
def build_model(input_shape, output_units):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(output_units)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# -----------------------------
# Forecast Prediction
# -----------------------------
def make_forecast(model, X_input, scaler_y, forecast_days):
    pred_vector = model.predict(np.expand_dims(X_input, axis=0), verbose=0)[0]
    pred_scaled = pred_vector.reshape(forecast_days, 3)
    predictions = scaler_y.inverse_transform(pred_scaled)
    df_pred = pd.DataFrame(predictions, columns=['Close', 'High', 'Low'])
    df_pred['High'] = df_pred[['Close', 'High']].max(axis=1)
    df_pred['Low'] = df_pred[['Close', 'Low']].min(axis=1)
    df_pred['Low'] = df_pred['Low'].clip(lower=0)
    return df_pred

# -----------------------------
# Run Forecast
# -----------------------------
if st.button("Run Forecast"):
    with st.spinner(f"Fetching data and training model for {coin}..."):
        df = yf.download(coin, start="2014-01-01", end=datetime.datetime.now())
        if df.shape[0] < 100:
            st.error("⚠️ Not enough data to evaluate.")
        else:
            X, y, scaler_X, scaler_y, df_full = prepare_data(df, look_back, forecast_days)
            X_recent = scaler_X.transform(df_full.iloc[-look_back:][['Close', 'High', 'Low', 'Volume', 'VWAP', 'MACD', 'RSI', 'ATR', 'DayOfWeek']])
            num_features = X.shape[2]
            output_units = forecast_days * 3
            model = build_model((look_back, num_features), output_units)
            model.fit(X, y, epochs=20, batch_size=32, verbose=0)
            df_pred = make_forecast(model, X_recent, scaler_y, forecast_days)

            start_date = pd.to_datetime("today").normalize()
            future_dates = [(start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(forecast_days)]
            df_pred.insert(0, 'Date', future_dates)

            st.success("Forecast complete!")
            st.dataframe(df_pred)
            st.line_chart(df_pred.set_index("Date"))

            csv = df_pred.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, f"{coin}_forecast.csv", "text/csv")

# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("---")
st.markdown("### Terms of Use & Disclaimer")
st.markdown("""
This site is for **educational and informational purposes only**. The predictions generated are based on historical data and machine learning models and **should not be interpreted as financial advice or investment guidance**. We do not guarantee accuracy, reliability, or performance of any forecasts.

You are solely responsible for any financial decisions you make. We are not registered financial advisors and do not offer personalized trading strategies.

Any contributions or donations made are considered **voluntary support for continued development** and do not constitute a purchase of financial products or services.

**Use at your own risk.**
""")

