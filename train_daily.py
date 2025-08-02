import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

def calculate_technical_indicators(df):
    st.write("Starting row count:", len(df))

    df['Return'] = df['Close'].pct_change()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    st.write("After Return & Momentum:", len(df.dropna()))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    st.write("After MACD:", len(df.dropna()))

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Delta'] = df['RSI'].diff()
    st.write("After RSI:", len(df.dropna()))

    rsi_min = df['RSI'].rolling(window=14).min()
    rsi_max = df['RSI'].rolling(window=14).max()
    df['StochRSI'] = (df['RSI'] - rsi_min) / (rsi_max - rsi_min + 1e-10)
    st.write("After StochRSI:", len(df.dropna()))

    high14 = df['High'].rolling(14).max()
    low14 = df['Low'].rolling(14).min()
    df['Williams_%R'] = -100 * (high14 - df['Close']) / (high14 - low14 + 1e-10)
    st.write("After Williams %R:", len(df.dropna()))

    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    df['BB%'] = (df['Close'] - lower_band) / (upper_band - lower_band + 1e-10)
    st.write("After Bollinger Bands:", len(df.dropna()))

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    st.write("After ATR:", len(df.dropna()))

    df = df.dropna()
    st.write("Final usable row count:", len(df))

    if len(df) < 50:
        raise ValueError("Insufficient data after calculating indicators.")
    return df


def calculate_vwap(df):
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def prepare_data(df, look_back=30):
    df['VWAP'] = calculate_vwap(df)
    df = calculate_technical_indicators(df)

    features = ['Close', 'High', 'Low', 'VWAP', 'Return', 'Momentum', 'MACD_Hist', 
                'RSI_Delta', 'StochRSI', 'Williams_%R', 'BB%', 'ATR']
    
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
    model.compile(optimizer='adam', loss='mse')
    return model


