# features.py
import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP that resets at each UTC calendar day (session).
    Typical price = (H+L+C)/3.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    grp = df["start"].dt.floor("D")
    cum_pv = (tp * df["volume"]).groupby(grp).cumsum()
    cum_v  = (df["volume"]).groupby(grp).cumsum().replace(0, np.nan)
    return cum_pv / cum_v

def swings(df: pd.DataFrame, lb: int = 10):
    """
    Simple swing high/low detection using rolling window extrema.
    Returns two Series of booleans for swing highs/lows.
    """
    high = df["high"]; low = df["low"]
    sh = (high.shift(1) == high.shift(1).rolling(lb, center=False).max()) & \
         (high.shift(1) > high.shift(2)) & (high.shift(1) > high)
    sl = (low.shift(1) == low.shift(1).rolling(lb, center=False).min()) & \
         (low.shift(1) < low.shift(2)) & (low.shift(1) < low)
    return sh.fillna(False), sl.fillna(False)

def fib_levels_from_swing(high_val, low_val):
    """Return common retracement/extension levels for a swing."""
    diff = high_val - low_val
    levels = [
        low_val + 0.236*diff, low_val + 0.382*diff, low_val + 0.5*diff,
        low_val + 0.618*diff, low_val + 0.786*diff,
        high_val - 0.236*diff, high_val - 0.382*diff, high_val - 0.618*diff
    ]
    return np.array(sorted(set(levels)))

def fib_confluence_score(df: pd.DataFrame, windows=(20, 50, 120), tol_bps=8):
    """
    For each bar, compute a confluence score: how many fib levels (from multiple swing windows)
    lie within 'tol_bps' (basis points) of current close.
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    scores = np.zeros(len(df))
    nearest_dist = np.full(len(df), np.nan)

    for i in range(max(windows)+2, len(df)):
        levels = []
        for w in windows:
            h = high[i-w:i].max()
            l = low[i-w:i].min()
            levels.extend(fib_levels_from_swing(h, l))
        levels = np.array(levels)
        price = close[i]
        if levels.size == 0:
            continue
        # distance in bps
        dists_bps = 1e4 * np.abs(levels - price) / price
        conf_hits = (dists_bps <= tol_bps).sum()
        scores[i] = conf_hits
        nearest_dist[i] = dists_bps.min()
    return pd.Series(scores, index=df.index, name="fib_conf"), pd.Series(nearest_dist, index=df.index, name="fib_nearest_bps")

def equal_level_pools(prices: pd.Series, tol_bps=5, min_hits=3, lookback=200):
    """
    Detect 'equal highs/lows' liquidity pools in the recent window: cluster prices within tolerance.
    Returns two arrays of pool levels (sell-side highs, buy-side lows).
    """
    arr = prices.to_numpy()[-lookback:]
    levels = []
    for p in arr:
        levels.append(p)
    levels = np.array(levels)
    # cluster by rounding to tolerance bucket
    buckets = np.round(levels / (levels * tol_bps / 1e4))
    # count occurrences
    counts = pd.Series(buckets).value_counts()
    buckets_keep = counts[counts >= min_hits].index
    kept = levels[np.isin(buckets, buckets_keep)]
    above = np.unique(np.round(kept[kept >= prices.iloc[-1]], 2))
    below = np.unique(np.round(kept[kept <= prices.iloc[-1]], 2))
    return above, below

def liquidity_features(df: pd.DataFrame, tol_bps=5, min_hits=3, lookback=300):
    """
    Proxy 'liquidity' features:
      - distance (bps) to nearest 'equal-highs' pool above (sell-side liquidity)
      - distance (bps) to nearest 'equal-lows' pool below (buy-side liquidity)
      - distance (bps) to recent swing high/low
    """
    close = df["close"]
    sh, sl = swings(df, lb=10)
    # recent swings
    last_sh = df["high"][sh].rolling(1).apply(lambda x: x[-1] if len(x) else np.nan).ffill()
    last_sl = df["low"][sl].rolling(1).apply(lambda x: x[-1] if len(x) else np.nan).ffill()

    d_sh_bps = 1e4 * (last_sh - close).abs() / close
    d_sl_bps = 1e4 * (close - last_sl).abs() / close

    # equal-level pools
    above, below = equal_level_pools(close, tol_bps=tol_bps, min_hits=min_hits, lookback=lookback)
    def nearest_bps(levels, price):
        if levels is None or len(levels) == 0:
            return np.nan
        return float((1e4 * np.min(np.abs(levels - price) / price)))

    d_sell_bps = []
    d_buy_bps = []
    for p in close:
        d_sell_bps.append(nearest_bps(above, p))
        d_buy_bps.append(nearest_bps(below, p))

    return (
        pd.Series(d_sh_bps, index=df.index, name="dist_swing_high_bps"),
        pd.Series(d_sl_bps, index=df.index, name="dist_swing_low_bps"),
        pd.Series(d_sell_bps, index=df.index, name="dist_equal_highs_bps"),
        pd.Series(d_buy_bps, index=df.index, name="dist_equal_lows_bps"),
    )

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["close"] = df["close"].astype(float)
    f["high"] = df["high"].astype(float)
    f["low"] = df["low"].astype(float)
    f["volume"] = df["volume"].astype(float)

    f["vwap"] = session_vwap(df)
    f["rsi14"] = rsi(f["close"], 14)
    fib_conf, fib_near = fib_confluence_score(df)
    f["fib_conf"] = fib_conf
    f["fib_nearest_bps"] = fib_near
    d_sh, d_sl, d_eqh, d_eql = liquidity_features(df)
    f["dist_swing_high_bps"] = d_sh
    f["dist_swing_low_bps"] = d_sl
    f["dist_equal_highs_bps"] = d_eqh
    f["dist_equal_lows_bps"] = d_eql

    # Fill/clean
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f
