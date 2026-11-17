"""Features and labels. Every feature at candle t uses only candles up to and including t (no look-ahead)."""
import numpy as np
import pandas as pd

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]

PATTERN_DIR = {"bull_engulf": 1, "bear_engulf": -1, "hammer": 1, "shooting_star": -1,
               "morning_star": 1, "evening_star": -1, "three_soldiers": 1, "three_crows": -1}


def klines_to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df.index = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df.index.name = "time"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def rsi(c, p=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, p=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, adjust=False).mean()


def patterns(df, e20) -> pd.DataFrame:
    o, h, l, c = df.open, df.high, df.low, df.close
    body = (c - o).abs()
    bull, bear = c > o, c < o
    up, lo = h - np.maximum(o, c), np.minimum(o, c) - l
    avgb = body.shift(1).rolling(20, min_periods=5).mean()
    o1, c1, b1 = o.shift(1), c.shift(1), body.shift(1)
    o2, c2, b2 = o.shift(2), c.shift(2), body.shift(2)
    bull1, bear1 = bull.shift(1, fill_value=False), bear.shift(1, fill_value=False)
    bull2, bear2 = bull.shift(2, fill_value=False), bear.shift(2, fill_value=False)
    big = lambda b: b > 0.6 * avgb

    p = pd.DataFrame(index=df.index)
    p["bull_engulf"] = bear1 & bull & (o <= c1) & (c >= o1) & (body > b1)
    p["bear_engulf"] = bull1 & bear & (o >= c1) & (c <= o1) & (body > b1)
    p["hammer"] = (body > 0) & (lo >= 2 * body) & (up <= 0.5 * body) & (c < e20)
    p["shooting_star"] = (body > 0) & (up >= 2 * body) & (lo <= 0.5 * body) & (c > e20)
    p["morning_star"] = bear2 & (b2 > avgb) & (b1 < 0.3 * b2) & bull & (c > (o2 + c2) / 2)
    p["evening_star"] = bull2 & (b2 > avgb) & (b1 < 0.3 * b2) & bear & (c < (o2 + c2) / 2)
    p["three_soldiers"] = bull2 & bull1 & bull & (c1 > c2) & (c > c1) & big(b2) & big(b1) & big(body)
    p["three_crows"] = bear2 & bear1 & bear & (c1 < c2) & (c < c1) & big(b2) & big(b1) & big(body)
    return p.astype(int)


def build_features(df: pd.DataFrame):
    c = df.close
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    a = atr(df)
    f = patterns(df, e20)
    pats = list(PATTERN_DIR)
    for col in pats:
        f[col + "_3"] = f[col].rolling(3, min_periods=1).max()
    f["pattern_score_3"] = sum(f[col + "_3"] * d for col, d in PATTERN_DIR.items())

    f["ret_1"], f["ret_3"], f["ret_12"] = c.pct_change(1), c.pct_change(3), c.pct_change(12)
    f["dist_e20"], f["dist_e50"], f["dist_e200"] = (c - e20) / a, (c - e50) / a, (c - e200) / a
    f["e20_e50"] = (e20 - e50) / a
    f["e20_slope"] = e20.diff(5) / a
    f["rsi"] = rsi(c)
    f["atr_pct"] = a / c
    rng = (df.high - df.low).replace(0, np.nan)
    f["range_atr"] = rng / a
    f["close_pos"] = (c - df.low) / rng
    f["upper_wick"] = (df.high - np.maximum(df.open, c)) / a
    f["lower_wick"] = (np.minimum(df.open, c) - df.low) / a
    v = df.volume
    f["vol_z"] = (v - v.rolling(50).mean()) / v.rolling(50).std()
    f["hour"], f["dow"] = df.index.hour, df.index.dayofweek
    return f, a


def triple_barrier(df: pd.DataFrame, atr_s: pd.Series, horizon=5, k=1.0) -> pd.Series:
    """1 if price touches close + k*ATR before close - k*ATR within `horizon` candles, 0 if the
    lower barrier comes first. Timeouts and candles that touch both barriers stay NaN."""
    c, h, l, a = df.close.values, df.high.values, df.low.values, atr_s.values
    y = np.full(len(c), np.nan)
    for i in range(len(c) - horizon):
        if np.isnan(a[i]):
            continue
        tp, sl = c[i] + k * a[i], c[i] - k * a[i]
        for j in range(i + 1, i + horizon + 1):
            hit_tp, hit_sl = h[j] >= tp, l[j] <= sl
            if hit_tp and hit_sl:
                break
            if hit_tp:
                y[i] = 1
                break
            if hit_sl:
                y[i] = 0
                break
    return pd.Series(y, index=df.index, name="y")
