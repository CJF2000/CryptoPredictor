# data_sources.py
import time
import requests
import pandas as pd
from typing import Literal

EXC = Literal["okx", "bitfinex", "coinbase"]

def _to_ms(ts):
    if isinstance(ts, (int, float)):
        return int(ts)
    return int(pd.Timestamp(ts, tz="UTC").value // 10**6)

def _sleep_rl():
    time.sleep(0.2)

# ---------- OKX ----------
# history: /api/v5/market/history-candles (newest->oldest), fallback: /candles
def fetch_okx_klines(instId: str, start, end, bar="15m", limit=100) -> pd.DataFrame:
    base = "https://www.okx.com"
    endpoint_hist = "/api/v5/market/history-candles"
    endpoint_live = "/api/v5/market/candles"
    start_ms, end_ms = _to_ms(start), _to_ms(end)

    out = []
    ts = end_ms
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    while ts > start_ms:
        params = {"instId": instId, "bar": bar, "limit": limit, "before": ts}
        r = requests.get(base + endpoint_hist, params=params, headers=headers, timeout=30)
        if r.status_code == 404 or r.json().get("data") is None:
            r = requests.get(base + endpoint_live, params={"instId": instId, "bar": bar, "limit": limit}, headers=headers, timeout=30)
        r.raise_for_status()
        rows = r.json().get("data", [])
        if not rows:
            break
        out.extend(rows)
        oldest = int(rows[-1][0])  # ms
        ts = oldest - 1
        _sleep_rl()

    if not out:
        return pd.DataFrame()

    # rows: [ts, o, h, l, c, vol, volCcy, volQuote, confirm]
    df = pd.DataFrame(out, columns=["start","open","high","low","close","volume","volCcy","volQuote","confirm"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], unit="ms", utc=True)
    df = df.sort_values("start")
    return df[["start","open","high","low","close","volume"]].dropna()

# ---------- Bitfinex ----------
# /v2/candles/trade:15m:tBTCUSD/hist?start=...&end=...&limit=10000&sort=1
def fetch_bitfinex_klines(symbol: str, start, end, timeframe="15m", limit=10000) -> pd.DataFrame:
    url = f"https://api-pub.bitfinex.com/v2/candles/trade:{timeframe}:{symbol}/hist"
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    out = []
    cur_start = start_ms
    while cur_start < end_ms:
        params = {"start": cur_start, "end": end_ms, "limit": limit, "sort": 1}  # oldest->newest
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        rows = r.json() or []
        if not rows:
            break
        out.extend(rows)
        last_ts = rows[-1][0]
        cur_start = last_ts + 900_000  # 15m
        _sleep_rl()

    if not out:
        return pd.DataFrame()

    # rows: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
    df = pd.DataFrame(out, columns=["start","open","close","high","low","volume"])
    df = df[["start","open","high","low","close","volume"]]
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], unit="ms", utc=True)
    df = df.sort_values("start")
    return df.dropna()

# ---------- Coinbase ----------
# /products/{product_id}/candles?granularity=900&start=...&end=...
# Returns arrays: [time, low, high, open, close, volume] newest->oldest
def fetch_coinbase_klines(product_id: str, start, end, granularity=900) -> pd.DataFrame:
    base = "https://api.exchange.coinbase.com"
    endpoint = f"/products/{product_id}/candles"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    start_iso = pd.Timestamp(start, tz="UTC").isoformat()
    end_iso = pd.Timestamp(end, tz="UTC").isoformat()

    seconds_per_page = 300 * granularity  # ~300 bars per request
    out = []
    cursor = pd.Timestamp(start_iso)
    t1 = pd.Timestamp(end_iso)

    while cursor < t1:
        window_end = min(cursor + pd.Timedelta(seconds=seconds_per_page), t1)
        params = {"granularity": granularity, "start": cursor.isoformat(), "end": window_end.isoformat()}
        r = requests.get(base + endpoint, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        rows = r.json() or []
        if rows:
            out.extend(rows)
        cursor = window_end
        _sleep_rl()

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out, columns=["start_s","low","high","open","close","volume"])
    df["start"] = pd.to_datetime(df["start_s"], unit="s", utc=True)
    df = df[["start","open","high","low","close","volume"]]
    df = df.sort_values("start")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()
