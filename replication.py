"""Confirmatory test of H1 and H2 exactly as written in PREREGISTRATION.md.

Computes the pre-registered quantities directly from the price data (not from pattern_stats CSVs):
  H1: up rate 5 candles after three black crows (any trend)    >  base up rate over all candles
  H2: up rate 5 candles after three white soldiers (downtrend) <  base up rate over downtrend candles
"Up" means the close 5 candles later is strictly above the signal close, for signals and base alike.
Pattern and trend definitions are imported unchanged from features.py.

Usage:
  python replication.py                         # confirmatory set (the decision is made here)
  python replication.py --group exploratory     # BTC, ETH, SOL, XRP, BNB, DOGE (exploratory only)
  python replication.py --group discovery       # ADA (discovery sample)
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from features import ema, patterns

GROUPS = {
    "confirmatory": ["AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT", "ATOMUSDT"],
    "exploratory": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"],
    "discovery": ["ADAUSDT"],
}
HYPOTHESES = {
    "H1": {"pattern": "three_crows", "trend": "all", "predict": "greater", "contrarian": 1},
    "H2": {"pattern": "three_soldiers", "trend": "down", "predict": "less", "contrarian": -1},
}
WARMUP, HORIZON, FEE = 200, 5, 0.001


def one_symbol(symbol, interval):
    df = pd.read_parquet(f"data/{symbol}_{interval}.parquet")
    c = df.close
    e20, e50 = ema(c, 20), ema(c, 50)
    P = patterns(df, e20)
    fwd = c.shift(-HORIZON) / c - 1
    valid = fwd.notna() & (np.arange(len(df)) >= WARMUP)
    down = valid & ~(e20 > e50)  # same rule as pattern_stats.py: "up" only when EMA20 > EMA50
    rows = []
    for h, spec in HYPOTHESES.items():
        regime = valid if spec["trend"] == "all" else down
        m = regime & (P[spec["pattern"]] == 1)
        n, ups = int(m.sum()), int((fwd[m] > 0).sum())
        base = float((fwd[regime] > 0).mean())
        rate = ups / n if n else np.nan
        right_dir = (rate > base) if spec["predict"] == "greater" else (rate < base)
        p = binomtest(ups, n, base, alternative=spec["predict"]).pvalue if n else np.nan
        move = float((fwd[m] * spec["contrarian"]).mean()) if n else np.nan
        rows.append({"hypothesis": h, "symbol": symbol, "n": n, "ups": ups, "up_rate": rate,
                     "base_up": base, "diff_pp": (rate - base) * 100, "predicted_dir": bool(right_dir),
                     "p_one_sided": p, "contrarian_move_%": move * 100,
                     "after_fee_%": (move - FEE) * 100})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS), default="confirmatory")
    ap.add_argument("--interval", default="1h")
    a = ap.parse_args()

    rows = []
    for s in GROUPS[a.group]:
        try:
            rows += one_symbol(s, a.interval)
        except FileNotFoundError:
            print(f"Missing data/{s}_{a.interval}.parquet, run data.py for it first.")
    if not rows:
        return
    out = pd.DataFrame(rows)

    print(f"Group: {a.group}  ({'decision sample' if a.group == 'confirmatory' else 'NOT confirmatory'})\n")
    for h, spec in HYPOTHESES.items():
        t = out[out.hypothesis == h]
        n, ups = int(t.n.sum()), int(t.ups.sum())
        base = float((t.n * t.base_up).sum() / n)
        p = binomtest(ups, n, base, alternative=spec["predict"]).pvalue
        k = int(t.predicted_dir.sum())
        show = t[["symbol", "n", "up_rate", "base_up", "diff_pp", "predicted_dir", "p_one_sided",
                  "contrarian_move_%", "after_fee_%"]].copy()
        for col in ("up_rate", "base_up"):
            show[col] = (show[col] * 100).round(1)
        show = show.round({"diff_pp": 1, "contrarian_move_%": 3, "after_fee_%": 3})
        show["p_one_sided"] = show["p_one_sided"].map(lambda x: f"{x:.4f}")
        print(f"{h}: {spec['pattern']}, trend {spec['trend']}, predicted up rate {spec['predict']} than base")
        print(show.to_string(index=False))
        print(f"Pooled: {ups}/{n} up = {ups / n * 100:.1f}% vs base {base * 100:.1f}%, one-sided p = {p:.6f}")
        print(f"Predicted direction in {k} of {len(t)} symbols")
        if a.group == "confirmatory":
            ok = k >= 4 and p < 0.01 and len(t) == 6
            print(f"Decision (>= 4 of 6 and pooled p < 0.01): {'REPLICATED' if ok else 'NOT REPLICATED'}")
        print()

    path = f"results/replication_{a.group}_{a.interval}.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
