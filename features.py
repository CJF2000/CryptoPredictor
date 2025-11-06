# features.py
import numpy as np
import pandas as pd

# Helper: RSI (classic Wilder's, 14 default)
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

# Helper: session/cumulative VWAP
def vwap_df(df: pd.DataFrame) -> pd.Series:
    # assume df has 'close' and 'volume'
    pv = (df['close'] * df['volume']).cumsum()
    vol = df['volume'].cumsum().replace(0, np.nan)
    return pv / vol

# Helper: recent swing high/low and fib levels
def swing_levels(high: pd.Series, low: pd.Series, lookback: int = 96*7):
    # last N bars swing
    hh = high.rolling(lookback, min_periods=lookback//4).max()
    ll = low.rolling(lookback, min_periods=lookback//4).min()
    sw_high = hh
    sw_low = ll
    # fibs between last swing
    levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    # compute per-row fib grid using current swing hi/lo
    rng = (sw_high - sw_low).replace(0, np.nan)
    fib_grid = {f"fib_{int(l*1000)}": (sw_high - rng * l) for l in levels}
    return sw_high, sw_low, fib_grid

# Helper: basis points distance
def bps_dist(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).abs() / b.replace(0, np.nan) * 10000

# Helper: equal highs/lows detection (simple)
def equal_level(series: pd.Series, window: int = 48, tol_bps: float = 5.0) -> pd.Series:
    # find recent price that repeats (approx) within window; return that ref price
    ref = series.round(2)  # coarse bucket
    eq = ref.rolling(window).apply(lambda x: pd.Series(x).mode().iloc[0] if len(pd.Series(x).mode()) else np.nan, raw=False)
    # validate by tolerance
    price = series
    valid = (bps_dist(price, eq) <= tol_bps)
    return eq.where(valid)

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df indexed by UTC datetime with columns:
      close, high, low, volume
    Returns a feature frame including:
      vwap, rsi14, fib_conf, fib_nearest_bps,
      dist_swing_high_bps, dist_swing_low_bps,
      dist_equal_highs_bps, dist_equal_lows_bps
    """
    x = df.copy()

    # Core indicators
    x["vwap"] = vwap_df(x)
    x["rsi14"] = rsi(x["close"], 14)

    # Swing highs/lows & fib grid
    sw_high, sw_low, fib_grid = swing_levels(x["high"], x["low"], lookback=96*7)  # ~7 days on 15m
    x["swing_high"] = sw_high
    x["swing_low"] = sw_low

    # Distance to swing extremes (bps)
    x["dist_swing_high_bps"] = bps_dist(x["close"], x["swing_high"])
    x["dist_swing_low_bps"]  = bps_dist(x["close"], x["swing_low"])

    # Fib confluence: count of fib levels within 10 bps of price
    fib_cols = []
    for k, lvl in fib_grid.items():
        col = f"{k}"
        x[col] = lvl
        fib_cols.append(col)

    # nearest fib (bps) and confluence score (#levels within threshold)
    diffs = [bps_dist(x["close"], x[c]) for c in fib_cols]
    diffs_df = pd.concat(diffs, axis=1)
    diffs_df.columns = fib_cols

    x["fib_nearest_bps"] = diffs_df.min(axis=1)
    x["fib_conf"] = (diffs_df.le(10.0)).sum(axis=1).astype(float)  # within 10 bps → counts as confluence

    # Liquidity: "equal highs/lows" rough proxies
    # Build reference levels from highs/lows separately (using close for distance)
    eq_high_ref = equal_level(x["high"], window=96, tol_bps=8.0)
    eq_low_ref  = equal_level(x["low"],  window=96, tol_bps=8.0)
    x["dist_equal_highs_bps"] = bps_dist(x["close"], eq_high_ref)
    x["dist_equal_lows_bps"]  = bps_dist(x["close"], eq_low_ref)

    # Keep only required columns + original OHLCV for safety
    keep = [
        "close","high","low","volume",
        "vwap","rsi14","fib_conf","fib_nearest_bps",
        "dist_swing_high_bps","dist_swing_low_bps","dist_equal_highs_bps","dist_equal_lows_bps",
    ]
    out = x[keep].replace([np.inf, -np.inf], np.nan).dropna()
    return out
