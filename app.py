import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import xgboost as xgb

st.set_page_config(page_title="Crypto Forecast Bot", layout="centered")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    .st-emotion-cache-19rxjzo {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Crypto Forecast Bot")
st.markdown("""
> ⚠️ **Disclaimer:** This tool is for educational and informational purposes only.  
> It is not financial advice and should not be used to make investment decisions.  
> Use it to help spot trends — not to predict the future.
""")

st.markdown("""
Welcome to the 7-day **Crypto Price Predictor**.

📈 Powered by AI (LSTM neural networks + XGBoost ensemble)  
🔐 Access is password-protected — DM [@Forest_Wizard](https://t.me/Forecast_Wizard) on Telegram or Discord to unlock.  
💸 Suggested donation: **$10/month or 50 for lifetime access**   
Cashapp: ForecastWizard  
Venmo: Forecast_Wizard   
Accept Other Forms of Payment Just DM Me For Access
""")

password = st.text_input("Enter Access Password", type="password")
if password != "brickedalpha":
    st.warning("Access denied. DM @Forest_Wizard to get your password.")
    st.stop()
st.success("✅ Access granted.")

# ------------------ Core Functions ------------------
def calculate_vwap(df):
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def prepare_data(df, look_back=30):
    df['VWAP'] = calculate_vwap(df)
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['MA20'] + 2 * df['STD20']
    df['Lower_BB'] = df['MA20'] - 2 * df['STD20']
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()

    features = ['Close', 'High', 'Low', 'VWAP', 'MACD', 'RSI',
                'Upper_BB', 'Lower_BB', 'ATR', 'OBV', 'Return']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    price_scaler = MinMaxScaler()
    price_scaler.fit(df[['Close', 'High', 'Low']])

    X_lstm, y_lstm = [], []
    for i in range(look_back, len(scaled)):
        X_lstm.append(scaled[i - look_back:i])
        y_lstm.append(scaled[i, :3])

    X_xgb = [scaled[i - look_back:i].flatten() for i in range(look_back, len(scaled))]
    y_xgb = df[['Close', 'High', 'Low']].values[look_back:]

    return np.array(X_lstm), np.array(y_lstm), np.array(X_xgb), np.array(y_xgb), scaler, price_scaler, df

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def predict_lstm(model, recent_input, scaler, steps=7):
    future_input = recent_input.copy()
    future_preds_scaled = []
    for _ in range(steps):
        input_seq = future_input.reshape(1, 30, future_input.shape[1])
        pred_scaled = model.predict(input_seq, verbose=0)[0]
        future_preds_scaled.append(pred_scaled)
        last_features = future_input[-1, 3:].copy()
        next_input = np.append(pred_scaled, last_features).reshape(1, -1)
        future_input = np.vstack([future_input[1:], next_input])

    padded = np.hstack([future_preds_scaled, np.zeros((steps, future_input.shape[1] - 3))])
    preds_unscaled = scaler.inverse_transform(padded)[:, :3]
    return preds_unscaled

# ------------------ UI ------------------
st.header("📊 Forecast Dashboard")

coin = st.selectbox("🪙 Choose a coin", ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD'])
try:
    current_data = yf.Ticker(coin).history(period="1d", interval="1m")
    if not current_data.empty:
        current_price = current_data["Close"].iloc[-1]
        st.metric(label=f"💰 Current {coin} Price", value=f"${current_price:,.2f}")
    else:
        st.warning("Live price unavailable.")
except Exception as e:
    st.warning(f"Couldn't fetch price: {e}")

forecast_days = st.slider("🗖️ Forecast Days", 1, 15, 7)

if st.button("🚀 Run Forecast"):
    with st.spinner(f"Fetching and training {coin}..."):
        df = yf.download(coin, start="2014-01-01", end=datetime.datetime.now())
        if df.shape[0] < 100:
            st.error("⚠️ Not enough data to evaluate.")
        else:
            X_lstm, y_lstm, X_xgb, y_xgb, scaler, price_scaler, df_full = prepare_data(df)

            lstm_model = build_lstm((X_lstm.shape[1], X_lstm.shape[2]))
            lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)
            recent_lstm = scaler.transform(df_full.iloc[-30:][['Close', 'High', 'Low', 'VWAP', 'MACD', 'RSI',
                                                              'Upper_BB', 'Lower_BB', 'ATR', 'OBV', 'Return']])
            preds_lstm = predict_lstm(lstm_model, recent_lstm, scaler, steps=forecast_days)

            preds_xgb = []
            for i in range(3):
                model = xgb.XGBRegressor()
                X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb[:, i], test_size=0.2, shuffle=False)
                model.fit(X_train, y_train)
                pred_scaled = [model.predict(X_xgb[-1].reshape(1, -1))[0]] * forecast_days
                preds_xgb.append(pred_scaled)

            preds_xgb = np.array(preds_xgb).T
            preds_xgb_unscaled = preds_xgb

            start_date = datetime.datetime.now().date()
            future_dates = [(start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(forecast_days)]

            df_pred = pd.DataFrame({
                'Date': future_dates,
                'Close_LSTM': preds_lstm[:, 0],
                'High_LSTM': preds_lstm[:, 1],
                'Low_LSTM': preds_lstm[:, 2],
                'Close_XGB': preds_xgb_unscaled[:, 0],
                'High_XGB': preds_xgb_unscaled[:, 1],
                'Low_XGB': preds_xgb_unscaled[:, 2],
                'Close': (preds_lstm[:, 0] + preds_xgb_unscaled[:, 0]) / 2,
                'High': (preds_lstm[:, 1] + preds_xgb_unscaled[:, 1]) / 2,
                'Low': (preds_lstm[:, 2] + preds_xgb_unscaled[:, 2]) / 2
            })

            st.success("📈 Forecast complete!")
            st.dataframe(df_pred[['Date', 'Close', 'High', 'Low']])
            st.line_chart(df_pred.set_index("Date")[['Close']])

            csv = df_pred[['Date', 'Close', 'High', 'Low']].to_csv(index=False).encode("utf-8")
            st.download_button("📅 Download CSV", csv, f"{coin}_forecast.csv", "text/csv")

st.markdown("---")
st.markdown("### Terms of Use & Disclaimer")
st.markdown("""
This site is for **educational and informational purposes only**. The predictions generated are based on historical data and machine learning models and **should not be interpreted as financial advice or investment guidance**. We do not guarantee accuracy, reliability, or performance of any forecasts.

You are solely responsible for any financial decisions you make. We are not registered financial advisors and do not offer personalized trading strategies.

Any contributions or donations made are considered **voluntary support for continued development** and do not constitute a purchase of financial products or services.

**Use at your own risk.**
""")
