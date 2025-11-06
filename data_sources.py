# data_sources.py
import time
import math
import requests
import pandas as pd
from typing import Literal, Optional

EXC = Literal["bybit", "okx"]

# --- Helpers
def _to_ms(ts):  # accepts pd.Timestamp/int/str
    if isinstance(ts, (int, float)):
        return int(ts)
    return int(pd.Timestamp(ts, tz="UTC").value // 10**6)

def _sleep_rl():
    time.sleep(0.2)  # mild back-off to respect public endpoints

# --- Bybit: /v5/market/kline (USDT perps via category=linear)
# Docs: GET /v5/market/kline (interval e.g. 15)  :contentReference[oaicite:0]{index=0}
def fetch_bybit_klines(symbol: str, start, end, interval="15", category="linear", limit=1000) -> pd.DataFrame:
    """Return a DataFrame with columns: start, open, high, low, close, volume."""
    url = "https://api.bybit.com/v5/market/kline"
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    out = []
    cursor = None
    while True:
        params = {
            "category": category,      # linear = USDT/USDC perps
            "symbol": symbol,
            "interval": interval,      # "15" for 15m
            "start": start_ms,
            "end": min(end_ms, start_ms + 1000*15*60*1000),  # clamp per page (limit * interval)
            "limit": limit
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {data}")
        rows = data["result"]["list"]
        out.extend(rows)
        cursor = data["result"].get("nextPageCursor")
        # advance window
        last_ts = int(rows[-1][0]) if rows else start_ms
        start_ms = last_ts + 15*60*1000
        if not rows or start_ms >= end_ms or not cursor:
            break
        _sleep_rl()

    if not out:
        return pd.DataFrame()
    # Bybit returns newest-first; normalize ascending and cast
    df = pd.DataFrame(out, columns=["start","open","high","low","close","volume","turnover"])
    df = df.sort_values("start")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], unit="ms", utc=True)
    return df[["start","open","high","low","close","volume"]].dropna()

# --- OKX: /api/v5/market/candles-history + /candles paging
# Public REST has rate limits; see OKX API v5 guide.  :contentReference[oaicite:1]{index=1}
def fetch_okx_klines(instId: str, start, end, bar="15m", limit=100):
    """
    Returns DataFrame with columns: start, open, high, low, close, volume.
    OKX history endpoints typically return up to ~100 per page for history; page newest->oldest.  :contentReference[oaicite:2]{index=2}
    """
    base = "https://www.okx.com"
    endpoint_hist = "/api/v5/market/history-candles"
    endpoint_live = "/api/v5/market/candles"
    start_ms, end_ms = _to_ms(start), _to_ms(end)

    out = []
    after = None  # OKX uses 'before'/'after' cursors; we’ll walk forward via time windows
    ts = end_ms
    while ts > start_ms:
        params = {"instId": instId, "bar": bar, "limit": limit, "before": ts}
        r = requests.get(base + endpoint_hist, params=params, timeout=30)
        if r.status_code == 404:
            r = requests.get(base + endpoint_live, params={"instId": instId, "bar": bar, "limit": limit}, timeout=30)
        r.raise_for_status()
        rows = r.json().get("data", [])
        if not rows:
            break
        out.extend(rows)
        # rows: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        oldest = int(rows[-1][0])
        ts = oldest - 1
        _sleep_rl()

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["start","open","high","low","close","volume","volCcy","volQuote","confirm"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], unit="ms", utc=True)
    df = df.sort_values("start")
    return df[["start","open","high","low","close","volume"]].dropna()

def fetch_ohlcv(exchange: EXC, symbol: str, start, end):
    if exchange == "bybit":
        return fetch_bybit_klines(symbol, start, end, interval="15", category="linear", limit=1000)
    elif exchange == "okx":
        return fetch_okx_klines(symbol, start, end, bar="15m", limit=100)
    else:
        raise ValueError("exchange must be 'bybit' or 'okx'")
