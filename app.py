import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# 🔒 Hide Streamlit UI elements: hamburger, footer, header, and floating 'Manage App'
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    .st-emotion-cache-19rxjzo {display: none;} /* Extra floating controls (Manage App) */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Crypto Forecast Bot", layout="centered")

# ----------------------------- INTRO -----------------------------
st.title("🔮 Crypto Forecast Bot")
st.markdown("""
Welcome to the 7-day **Crypto Price Predictor**.

📈 Powered by LSTM neural networks  
🔒 Access is **password-protected** — DM [@YourTelegram](https://t.me/YourTelegram) to get in.  
💸 Suggested donation: **$10/month**

""")

# ----------------------------- PASSWORD WALL -----------------------------
password = st.text_input("Enter Access Password", type="password")

if password != "brickedalpha":  # <- Change this monthly
    st.warning("Access denied. DM @YourTelegram to get your password.")
    st.stop()

st.success("✅ Access granted. Welcome!")

# ----------------------------- LSTM HELPERS -----------------------------
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

# ----------------------------- APP UI -----------------------------
coin = st.selectbox("🪙 Choose a coin", ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD'])
forecast_days = st.slider("📆 Forecast Days", 1, 15, 7)

if st.button("🚀 Run Forecast"):
    with st.spinner(f"Fetching and training {coin}..."):
        df = yf.download(coin, start="2014-01-01", end=datetime.datetime.now())
        if df.shape[0] < 100:
            st.error("⚠️ Not enough data to evaluate.")
        else:
            X, y, scaler, df_full = prepare_data(df)
            model = build_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            recent_scaled = scaler.transform(df_full[['Close', 'High', 'Low', 'VWAP']].iloc[-30:])
            preds = predict_future(model, recent_scaled, scaler, steps=forecast_days)

            last_date = df.index[-1]
            future_dates = [(last_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, forecast_days + 1)]
            df_pred = pd.DataFrame(preds, columns=['Close', 'High', 'Low'])
            df_pred.insert(0, 'Date', future_dates)

            st.success("📊 Forecast complete!")
            st.dataframe(df_pred)
            st.line_chart(df_pred.set_index("Date"))

            csv = df_pred.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, f"{coin}_forecast.csv", "text/csv")

