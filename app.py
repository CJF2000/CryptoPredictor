# app.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import os
from train_daily import train_and_save_forecast  # manual retrain access

st.set_page_config(page_title="Crypto Forecast Bot", layout="centered")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    .st-emotion-cache-19rxjzo {display: none;}
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Crypto Forecast Bot")
st.markdown("""
> ⚠️ **Disclaimer:** This tool is for educational and informational purposes only.
> It is not financial advice and should not be used to make investment decisions.
""")

st.markdown("""
Welcome to the 7-day **Crypto Price Predictor**.

📈 Powered by AI (LSTM neural networks)  
🔒 Access is password-protected — DM [@Forecast_Wizard](https://t.me/Forecast_Wizard) to unlock.  
💸  **$10/month or 50 for lifetime access**
""")

password = st.text_input("Enter Access Password", type="password")
if password != "Crypto_Forecast777":
    st.warning("Access denied. DM @Forecast_Wizard on Telegram to get your password.")
    st.stop()
st.success("✅ Access granted.")

st.header("📊 Forecast Dashboard")

coin = st.selectbox("🪙 Choose a coin", ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD'])

# Show current price
try:
    current_data = yf.Ticker(coin).history(period="1d", interval="1m")
    if not current_data.empty:
        current_price = current_data["Close"].iloc[-1]
        st.metric(label=f"💰 Current {coin} Price", value=f"${current_price:,.2f}")
    else:
        st.warning("Live price unavailable.")
except Exception as e:
    st.warning(f"Couldn't fetch price: {e}")

forecast_days = st.slider("📆 Forecast Days", 1, 15, 7)

if st.button("🚀 Load Forecast"):
    path = f"daily_forecasts/{coin}_forecast.csv"
    if os.path.exists(path):
        df_pred = pd.read_csv(path)
        df_pred = df_pred.head(forecast_days)
        st.success("📈 Forecast loaded!")
        st.dataframe(df_pred)
        st.line_chart(df_pred.set_index("Date"))
        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, f"{coin}_forecast.csv", "text/csv")
    else:
        st.error("❌ Forecast not available. Try again later.")

if st.button("🔁 Manually Retrain This Coin"):
    with st.spinner(f"Training new model for {coin}..."):
        try:
            train_and_save_forecast(coin, forecast_days=forecast_days, epochs=150)
            df_pred = pd.read_csv(f"daily_forecasts/{coin}_forecast.csv").head(forecast_days)
            st.success("✅ Manual training complete!")
            st.dataframe(df_pred)
            st.line_chart(df_pred.set_index("Date"))
        except Exception as e:
            st.error(f"🚨 Manual training failed: {e}")

st.markdown("---")
st.markdown("### Terms of Use & Disclaimer")
st.markdown("""
This site is for **educational and informational purposes only**. The predictions generated are based on historical data and machine learning models and **should not be interpreted as financial advice or investment guidance**. Use at your own risk.
""")

