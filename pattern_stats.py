"""How each pattern actually played out over the full downloaded history.

For every pattern, and separately inside uptrends and downtrends, this reports how often price moved
in the pattern's direction after N candles, compared with the base rate for that same trend regime
(so a pattern only gets credit for what it adds beyond the trend). P-values are Holm-corrected
because many patterns are tested at once.

Usage: python pattern_stats.py --symbol ADAUSDT --interval 1h --horizon 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import binomtest

from features import PATTERN_DIR, atr, ema, patterns, triple_barrier

WARMUP = 200


def double_patterns(df, k=5, tol=0.006, depth=0.015, min_gap=8):
    """Same rules as the web page. A pivot is confirmed k candles later, so the signal is placed there."""
    h, l, n = df.high.values, df.low.values, len(df)
    hmax = sliding_window_view(h, 2 * k + 1).max(axis=1)
    lmin = sliding_window_view(l, 2 * k + 1).min(axis=1)
    ph = [i for i in range(k, n - k) if h[i] == hmax[i - k]]
    pl = [i for i in range(k, n - k) if l[i] == lmin[i - k]]
    top, bot = np.zeros(n, int), np.zeros(n, int)
    for p, q in zip(ph, ph[1:]):
        if q - p >= min_gap and abs(h[p] - h[q]) / h[p] < tol and (min(h[p], h[q]) - l[p:q + 1].min()) / h[p] > depth:
            top[min(q + k, n - 1)] = 1
    for p, q in zip(pl, pl[1:]):
        if q - p >= min_gap and abs(l[p] - l[q]) / l[p] < tol and (h[p:q + 1].max() - max(l[p], l[q])) / l[p] > depth:
            bot[min(q + k, n - 1)] = 1
    return pd.Series(top, index=df.index), pd.Series(bot, index=df.index)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ADAUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--horizon", type=int, default=5, help="candles to look ahead")
    ap.add_argument("--k", type=float, default=1.0, help="barrier width in ATRs, for the barrier win rate")
    ap.add_argument("--fee", type=float, default=0.001, help="round-trip cost as a fraction (0.001 = 0.1%%)")
    ap.add_argument("--min-n", type=int, default=30, help="hide rows with fewer signals than this")
    a = ap.parse_args()
    key, H = f"{a.symbol.upper()}_{a.interval}", a.horizon

    df = pd.read_parquet(f"data/{key}.parquet")
    c = df.close
    e20, e50 = ema(c, 20), ema(c, 50)
    P = patterns(df, e20)
    P["double_top"], P["double_bottom"] = double_patterns(df)
    DIR = dict(PATTERN_DIR, double_top=-1, double_bottom=1)

    fwd = c.shift(-H) / c - 1
    tb = triple_barrier(df, atr(df), H, a.k)
    trend = pd.Series(np.where(e20 > e50, "up", "down"), index=df.index)
    valid = fwd.notna() & (np.arange(len(df)) >= WARMUP)

    rows = []
    for name, d in DIR.items():
        for regime in ("all", "up", "down"):
            in_regime = valid if regime == "all" else valid & (trend == regime)
            m = in_regime & (P[name] == 1)
            n = int(m.sum())
            if n == 0:
                continue
            base_up = float((fwd[in_regime] > 0).mean())
            base = base_up if d > 0 else 1 - base_up
            dr = fwd[m] * d
            wins = int((dr > 0).sum())
            tbm = m & tb.notna()
            bw = float(((tb[tbm] == 1) if d > 0 else (tb[tbm] == 0)).mean()) if tbm.any() else np.nan
            rows.append({"pattern": name, "dir": "bull" if d > 0 else "bear", "trend": regime, "n": n,
                         "win": wins / n, "base": base, "edge": wins / n - base,
                         "avg_move_%": dr.mean() * 100, "after_fees_%": (dr.mean() - a.fee) * 100,
                         "barrier_win": bw, "p": binomtest(wins, n, base).pvalue})

    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    out["sig"] = np.where(out["p_holm"] < 0.05, "*", "")

    Path("results").mkdir(exist_ok=True)
    path = f"results/pattern_stats_{key}_h{H}.csv"
    out.to_csv(path, index=False)

    span = f"{df.index[WARMUP].date()} to {df.index[-1].date()}"
    print(f"{key}, {span}, {int(valid.sum())} candles, look-ahead {H} candles, fee {a.fee * 100:.2f}% round trip\n")
    show = out[out["n"] >= a.min_n].copy()
    for col in ("win", "base", "edge", "barrier_win"):
        show[col] = (show[col] * 100).round(1)
    for col in ("avg_move_%", "after_fees_%"):
        show[col] = show[col].round(3)
    show["p_holm"] = show["p_holm"].map(lambda x: f"{x:.4f}")
    print(show.drop(columns="p").to_string(index=False))

    sig = out[(out["sig"] == "*") & (out["n"] >= a.min_n)]
    print(f"\n{len(sig)} of {len(out)} pattern and trend combinations differ from their base rate after Holm correction.")
    if len(sig):
        pos = sig[(sig["edge"] > 0) & (sig["after_fees_%"] > 0)]
        print(f"{len(pos)} of them beat the base rate AND stay positive after fees.")
    print("Note: signals on neighbouring candles overlap in time, so p-values are somewhat optimistic.")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
