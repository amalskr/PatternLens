"""Robustness checks for H1 and H2. Exploratory: these are NOT part of the pre-registered decision.

  A. Main result with a date-clustered bootstrap (signals on the same calendar day, across all coins,
     are resampled together, so cross-coin correlation and overlap are not counted as independent evidence)
  B. Overlapping signals removed (keep a signal only if the same signal did not fire in the previous 5 candles)
  C. Horizons 3, 5 and 10 candles
  D. Year by year stability
  E. Contrarian trade after maker (0.04%) and taker (0.10%) round-trip fees
  F. Pattern vs plain 3-candle return reversal: does the candle shape add anything?

Usage:
  python robustness.py                    # confirmatory symbols
  python robustness.py --group all        # all 13 symbols
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from features import ema, patterns
from replication import GROUPS

WARMUP, HORIZONS, B, SEED = 200, (3, 5, 10), 2000, 42
DIRECTION = {"H1": "greater", "H2": "less"}  # predicted up rate vs base
CONTRA = {"H1": 1, "H2": -1}  # contrarian trade: long after crows, short after soldiers


def dedup(mask, window=5):
    prev = mask.astype(int).rolling(window, min_periods=1).sum().shift(1).fillna(0) > 0
    return mask & ~prev


def collect(symbol, interval):
    df = pd.read_parquet(f"data/{symbol}_{interval}.parquet")
    c = df.close
    e20, e50 = ema(c, 20), ema(c, 50)
    P = patterns(df, e20)
    ok = pd.Series(np.arange(len(df)) >= WARMUP, index=df.index)
    down = ok & ~(e20 > e50)
    ret3 = c / c.shift(3) - 1

    h1 = ok & (P["three_crows"] == 1)
    h2 = down & (P["three_soldiers"] == 1)
    # Plain reversal rules with the same signal frequency as the pattern, no candle-shape conditions
    q1 = ret3[ok].quantile(h1.sum() / ok.sum())
    q2 = ret3[down].quantile(1 - h2.sum() / down.sum())
    r1 = ok & (ret3 <= q1)
    r2 = down & (ret3 >= q2)
    fams = {
        "H1": ("H1", ok, h1), "H2": ("H2", down, h2),
        "H1_dedup": ("H1", ok, dedup(h1)), "H2_dedup": ("H2", down, dedup(h2)),
        "H1_plain_reversal": ("H1", ok, r1), "H2_plain_reversal": ("H2", down, r2),
        "H1_reversal_not_pattern": ("H1", ok, r1 & ~h1), "H2_reversal_not_pattern": ("H2", down, r2 & ~h2),
        "H1_pattern_not_reversal": ("H1", ok, h1 & ~r1), "H2_pattern_not_reversal": ("H2", down, h2 & ~r2),
    }
    rows = []
    for h in HORIZONS:
        fwd = c.shift(-h) / c - 1
        valid = fwd.notna()
        for fam, (hyp, regime, sig) in fams.items():
            reg = regime & valid
            base = float((fwd[reg] > 0).mean())
            m = sig & valid
            rows.append(pd.DataFrame({
                "symbol": symbol, "family": fam, "hyp": hyp, "horizon": h, "time": df.index[m],
                "up": (fwd[m] > 0).astype(int).values, "base": base,
                "move": (fwd[m] * CONTRA[hyp]).values}))
    return pd.concat(rows, ignore_index=True)


def pooled(t, hyp):
    n, ups = len(t), int(t.up.sum())
    if n == 0:
        return {"n": 0}
    base = float(t.base.mean())  # mean over signals = signal-weighted base
    p = binomtest(ups, n, base, alternative=DIRECTION[hyp]).pvalue
    return {"n": n, "up_%": 100 * ups / n, "base_%": 100 * base, "diff_pp": 100 * (ups / n - base), "p": p}


def cluster_boot(t, hyp, rng):
    day = t.time.dt.floor("D")
    g = (t.up - t.base).groupby(day).agg(["sum", "count"])
    s, k = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), size=(B, len(g)))
    diffs = s[idx].sum(axis=1) / k[idx].sum(axis=1) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float((diffs <= 0).mean()) if DIRECTION[hyp] == "greater" else float((diffs >= 0).mean())
    return {"days": len(g), "ci_lo": lo, "ci_hi": hi, "boot_p": max(p, 1 / B)}


def fmt(d):
    if d.get("n", 0) == 0:
        return "n=0"
    out = f"n={d['n']:>6}  up {d['up_%']:5.1f}%  base {d['base_%']:5.1f}%  diff {d['diff_pp']:+5.1f} pp  p={d['p']:.2g}"
    if "ci_lo" in d:
        out += f"  | clustered 95% CI [{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}] pp over {d['days']} days, p={d['boot_p']:.3g}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS) + ["all"], default="confirmatory")
    ap.add_argument("--interval", default="1h")
    a = ap.parse_args()
    syms = sum(GROUPS.values(), []) if a.group == "all" else GROUPS[a.group]

    parts = []
    for s in syms:
        try:
            parts.append(collect(s, a.interval))
        except FileNotFoundError:
            print(f"Missing data/{s}_{a.interval}.parquet, skipped")
    S = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(SEED)
    summary = []

    def report(label, t, hyp, boot=False):
        d = pooled(t, hyp)
        if boot and d["n"]:
            d.update(cluster_boot(t, hyp, rng))
        summary.append({"section": label, "hyp": hyp, **d})
        print(f"  {hyp}  {fmt(d)}")

    print(f"ROBUSTNESS (exploratory, not part of the pre-registered decision)")
    print(f"Group: {a.group}, symbols: {', '.join(syms)}\n")
    h5 = S[S.horizon == 5]

    print("A. Main result, date-clustered bootstrap")
    for hyp in ("H1", "H2"):
        report("A_main", h5[h5.family == hyp], hyp, boot=True)

    print("\nB. Overlapping signals removed")
    for hyp in ("H1", "H2"):
        report("B_dedup", h5[h5.family == hyp + "_dedup"], hyp, boot=True)

    print("\nC. Horizons")
    for h in HORIZONS:
        print(f" {h} candles")
        for hyp in ("H1", "H2"):
            report(f"C_h{h}", S[(S.horizon == h) & (S.family == hyp)], hyp)

    print("\nD. By year (horizon 5)")
    for y in sorted(h5.time.dt.year.unique()):
        print(f" {y}")
        for hyp in ("H1", "H2"):
            t = h5[(h5.family == hyp) & (h5.time.dt.year == y)]
            report(f"D_{y}", t, hyp)

    print("\nE. Contrarian trade, mean move per signal after round-trip fees (horizon 5)")
    for hyp in ("H1", "H2"):
        t = h5[h5.family == hyp]
        gross = t.move.mean() * 100
        print(f"  {hyp}  gross {gross:+.3f}%  maker 0.04%: {gross - 0.04:+.3f}%  taker 0.10%: {gross - 0.10:+.3f}%  "
              f"(positive in {int((t.groupby('symbol').move.mean() * 100 - 0.10 > 0).sum())} of "
              f"{t.symbol.nunique()} symbols at taker fee)")
        summary.append({"section": "E_fees", "hyp": hyp, "gross_%": gross})

    print("\nF. Pattern vs plain 3-candle return reversal (same signal frequency, horizon 5)")
    for hyp in ("H1", "H2"):
        for fam, label in ((hyp, "pattern"), (hyp + "_plain_reversal", "plain reversal"),
                           (hyp + "_reversal_not_pattern", "reversal without pattern"),
                           (hyp + "_pattern_not_reversal", "pattern without big reversal")):
            d = pooled(h5[h5.family == fam], hyp)
            summary.append({"section": "F_" + fam, "hyp": hyp, **d})
            print(f"  {label:<30} {fmt(d)}")
        print()
    print("Reading F: if 'reversal without pattern' is about as strong as 'pattern', the candle shape adds\n"
          "nothing beyond the 3-candle return. If 'pattern without big reversal' still shows the effect,\n"
          "the shape carries information of its own.")

    path = f"results/robustness_{a.group}_{a.interval}.csv"
    pd.DataFrame(summary).to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
