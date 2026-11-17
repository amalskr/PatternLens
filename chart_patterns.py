"""Chart pattern dataset: the page's detection rules, ported to Python, run over full history.

For every pattern that is recognised, this records what was known when it was recognised (features)
and what happened afterwards (breakout side, target or stop). No feature uses a candle after the
pattern's confirmation candle.

IMPORTANT, the null control: a swing point is only confirmed after price has already moved 2 ATR away
from it. That move usually pushes price toward one side of the pattern before the pattern even exists,
so "which side did it break" looks predictable even on random data. Every file is therefore also run
on a shuffled copy (same candles, random order: no real patterns can exist). Any method's real edge is
its score on real data MINUS its score on the shuffled copy.

Usage:
  python chart_patterns.py                  # every data/<SYMBOL>_<interval>.parquet file
  python chart_patterns.py --interval 1h    # only one timeframe
Writes results/chart_patterns_dataset.csv (column null = 1 for shuffled copies).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import ema, rsi

WARMUP = 200


def atr_arr(h, l, c, p=14):
    out = np.empty(len(c))
    v = None
    for i in range(len(c)):
        tr = h[i] - l[i] if i == 0 else max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        v = tr if v is None else (v * (p - 1) + tr) / p
        out[i] = v
    return out


def zigzag(h, l, a, mult=2.0):
    piv, trend, hi, lo = [], 0, 0, 0
    for i in range(1, len(h)):
        th = mult * a[i]
        if trend != -1 and h[i] >= h[hi]:
            hi = i
        if trend != 1 and l[i] <= l[lo]:
            lo = i
        if trend != -1 and hi < i and h[hi] - l[i] >= th:
            piv.append({"i": hi, "p": h[hi], "t": "H", "conf": i})
            trend, lo = -1, hi + 1 + int(np.argmin(l[hi + 1:i + 1]))
        elif trend != 1 and lo < i and h[i] - l[lo] >= th:
            piv.append({"i": lo, "p": l[lo], "t": "L", "conf": i})
            trend, hi = 1, lo + 1 + int(np.argmax(h[lo + 1:i + 1]))
    return piv


def line(i1, p1, i2, p2):
    s = (p2 - p1) / (i2 - i1)
    return lambda i: p1 + s * (i - i1)


def detect(o, h, l, c):
    """Same rules as pattern-lens.html: two-line patterns, head and shoulders, double tops and bottoms,
    cup and handle. Returns a list of pattern dicts (not yet resolved)."""
    a = atr_arr(h, l, c)
    piv = zigzag(h, l, a)
    found = []

    def pole_of(w):
        s = w[0]["i"]
        return (w[0]["p"] - c[max(0, s - 12)]) / a[s]

    def tri(w):
        H = [x for x in w if x["t"] == "H"]
        L = [x for x in w if x["t"] == "L"]
        if len(H) < 2 or len(L) < 2:
            return None
        U = line(H[0]["i"], H[0]["p"], H[-1]["i"], H[-1]["p"])
        D = line(L[0]["i"], L[0]["p"], L[-1]["i"], L[-1]["p"])
        s, e = w[0]["i"], w[-1]["i"]
        ae = a[e]
        if any(abs(x["p"] - U(x["i"])) > 0.6 * ae for x in H) or any(abs(x["p"] - D(x["i"])) > 0.6 * ae for x in L):
            return None
        js = np.arange(s, e + 1)
        bad = np.sum((c[s:e + 1] > U(js) + 0.3 * ae) | (c[s:e + 1] < D(js) - 0.3 * ae))
        if bad > 0.1 * (e - s + 1):
            return None
        w0, w1 = U(s) - D(s), U(e) - D(e)
        if w0 < 1.5 * ae or w1 <= 0:
            return None
        pole = pole_of(w)
        has_pole = abs(pole) >= 3.5 and e - s <= 30 and w0 <= 0.6 * abs(pole) * ae
        su, sl, eps = (U(e) - U(s)) / ae, (D(e) - D(s)) / ae, 0.8
        flat = lambda x: abs(x) <= eps
        conv, par = w1 < 0.75 * w0, abs(w1 - w0) < 0.3 * w0
        name, d, height, bil = None, 0, w0, False
        if conv:
            if flat(su) and sl > eps:
                name, d, bil = "Ascending triangle", 1, True
            elif flat(sl) and su < -eps:
                name, d, bil = "Descending triangle", -1, True
            elif su < -eps and sl > eps:
                if has_pole:
                    name, d, height = ("Bullish pennant" if pole > 0 else "Bearish pennant"), int(np.sign(pole)), abs(pole) * ae
                else:
                    name, bil = "Symmetrical triangle", True
            elif su > eps and sl > eps:
                name, d = "Rising wedge", -1
            elif su < -eps and sl < -eps:
                name, d = "Falling wedge", 1
        elif par:
            if flat(su) and flat(sl):
                if abs(pole) >= 2.5:
                    name, d = ("Bullish rectangle" if pole > 0 else "Bearish rectangle"), int(np.sign(pole))
                else:
                    name, bil = "Consolidation channel", True
            elif has_pole and pole > 0 and su < eps and sl < eps:
                name, d, height = "Bull flag", 1, pole * ae
            elif has_pole and pole < 0 and su > -eps and sl > -eps:
                name, d, height = "Bear flag", -1, -pole * ae
        if not name:
            return None
        return dict(kind="lines", name=name, dir=d, bilateral=bil, s=s, e=e, U=U, D=D, height=height, a=ae,
                    conf=w[-1]["conf"], pole=pole)

    def hs(w):
        p0, p1, p2, p3, p4 = w
        inv = p0["t"] == "L"
        sg = -1 if inv else 1
        ae = a[p4["i"]]
        N = line(p1["i"], p1["p"], p3["i"], p3["p"])
        if sg * p2["p"] - max(sg * p0["p"], sg * p4["p"]) < 0.5 * ae:
            return None
        if abs(p0["p"] - p4["p"]) > 1.5 * ae:
            return None
        if abs((p3["p"] - p1["p"]) / (p3["i"] - p1["i"])) / ae > 0.1:
            return None
        if sg * (p0["p"] - N(p0["i"])) < 0.5 * ae or sg * (p4["p"] - N(p4["i"])) < 0.5 * ae:
            return None
        lv = p4["p"] + sg * 0.2 * ae
        return dict(kind="neck", name="Inverse head and shoulders" if inv else "Head and shoulders", dir=1 if inv else -1,
                    bilateral=False, s=p0["i"], e=p4["i"], N=N, height=sg * (p2["p"] - N(p2["i"])), a=ae,
                    conf=p4["conf"], inval=lv, stop_lv=lv, pole=pole_of(w))

    def dbl(w):
        p0, p1, p2 = w
        top = p0["t"] == "H"
        sg = 1 if top else -1
        ae = a[p2["i"]]
        if abs(p0["p"] - p2["p"]) > 0.6 * ae or sg * pole_of(w) < 2:
            return None
        lv = p1["p"]
        ext = max(p0["p"], p2["p"]) if top else min(p0["p"], p2["p"])
        st = ext + sg * 0.3 * ae
        return dict(kind="neck", name="Double top" if top else "Double bottom", dir=-sg, bilateral=False,
                    s=p0["i"], e=p2["i"], N=lambda i, lv=lv: lv, height=sg * (ext - lv), a=ae, conf=p2["conf"],
                    inval=st, stop_lv=st, pole=pole_of(w))

    for e in range(2, len(piv)):
        prev = found[-1] if found else None
        for n in (6, 5, 4, 3):
            s = e - n + 1
            if s < 0:
                continue
            w = piv[s:e + 1]
            hit = (n >= 4 and tri(w)) or (n == 5 and hs(w)) or (n == 3 and dbl(w))
            if not hit:
                continue
            hit["ps"], hit["pe"] = s, e
            if not prev or s >= prev["pe"]:
                found.append(hit)
                break
            grew = s >= prev["ps"] and hit["name"] == prev["name"]
            covers = s <= prev["ps"] and e - s > prev["pe"] - prev["ps"]
            if grew or covers:
                found[-1] = hit
                break

    # cup and handle (and inverse)
    cups = []
    for x in range(len(piv)):
        for y in range(x + 2, len(piv) - 1):
            A, B, Hd = piv[x], piv[y], piv[y + 1]
            ln = B["i"] - A["i"]
            if ln > 150:
                break
            if A["t"] != B["t"] or ln < 15:
                continue
            top = A["t"] == "H"
            sg, ae = (1 if top else -1), a[B["i"]]
            if abs(A["p"] - B["p"]) > ae:
                continue
            rim = max(A["p"], B["p"]) if top else min(A["p"], B["p"])
            seg_h, seg_l = h[A["i"] + 1:B["i"]], l[A["i"] + 1:B["i"]]
            if (top and np.any(seg_h > rim + 0.2 * ae)) or (not top and np.any(seg_l < rim - 0.2 * ae)):
                continue
            if top:
                ext_i = A["i"] + 1 + int(np.argmin(seg_l)); ext = l[ext_i]
            else:
                ext_i = A["i"] + 1 + int(np.argmax(seg_h)); ext = h[ext_i]
            depth = sg * (rim - ext)
            if depth < 3 * ae or ext_i < A["i"] + 0.2 * ln or ext_i > A["i"] + 0.8 * ln:
                continue
            near = np.sum(sg * (c[A["i"]:B["i"] + 1] - ext) <= 0.35 * depth)
            if near < 0.25 * (ln + 1):
                continue
            if sg * (B["p"] - Hd["p"]) > 0.5 * depth or Hd["i"] - B["i"] > 0.5 * ln:
                continue
            lv = Hd["p"] - sg * 0.2 * ae
            cups.append(dict(kind="neck", name="Cup and handle" if top else "Inverse cup and handle", dir=sg, bilateral=False,
                             s=A["i"], e=Hd["i"], N=line(A["i"], A["p"], B["i"], B["p"]), height=depth, a=ae,
                             conf=Hd["conf"], inval=lv, stop_lv=lv, pole=0.0))
    kept = []
    for cp in cups:
        if not any(cp["s"] <= k["e"] and k["s"] <= cp["e"] for k in kept):
            kept.append(cp)
    ov = lambda pt, k: max(0, min(pt["e"], k["e"]) - max(pt["s"], k["s"])) / max(1, pt["e"] - pt["s"])
    allp = [pt for pt in found if not any(ov(pt, k) > 0.5 for k in kept)] + kept
    return sorted(allp, key=lambda pt: pt["s"]), a


def resolve(pt, c, h, l):
    """Breakout side and trade result, scanning only candles after confirmation."""
    last, ln = len(c) - 1, pt["e"] - pt["s"]
    wait, sg, neck = max(10, ln), pt["dir"], pt["kind"] == "neck"
    up_b = (pt["N"] if sg > 0 else None) if neck else pt["U"]
    dn_b = (pt["N"] if sg < 0 else None) if neck else pt["D"]
    pt.update(brk=None, bdir=0, status="expired")
    for j in range(pt["conf"], min(last, pt["e"] + wait) + 1):
        if neck and (h[j] > pt["inval"] if sg < 0 else l[j] < pt["inval"]):
            pt["brk"], pt["bdir"] = j, -sg
            break
        if up_b and c[j] > up_b(j) + 0.1 * pt["a"]:
            pt["brk"], pt["bdir"] = j, 1
            break
        if dn_b and c[j] < dn_b(j) - 0.1 * pt["a"]:
            pt["brk"], pt["bdir"] = j, -1
            break
    if pt["brk"] is None:
        pt["status"] = "no breakout" if pt["e"] + wait < last else "forming"
        return
    d, b = pt["bdir"], pt["brk"]
    stop = pt["stop_lv"] if neck else (pt["D"](b) if d > 0 else pt["U"](b))
    entry = c[b]
    target = entry + d * pt["height"]
    pt.update(entry=entry, stop=stop, target=target, status="timeout")
    end = b + 3 * max(ln, 10)
    for j in range(b + 1, min(last, end) + 1):
        if (l[j] <= stop) if d > 0 else (h[j] >= stop):
            pt["status"] = "stop"; break
        if (h[j] >= target) if d > 0 else (l[j] <= target):
            pt["status"] = "target"; break
    if pt["status"] == "timeout" and end >= last:
        pt["status"] = "open"


def shuffled(df, seed):
    """Same candles in random order: keeps each candle's return, wicks and volume, destroys any pattern."""
    rng = np.random.default_rng(seed)
    o, h, l, c, v = (df[k].values for k in ("open", "high", "low", "close", "volume"))
    r = np.diff(np.log(c))
    orel, hrel = o[1:] / c[:-1], h[1:] / np.maximum(o[1:], c[1:])
    lrel = l[1:] / np.minimum(o[1:], c[1:])
    k = rng.permutation(len(r))
    nc = c[0] * np.exp(np.cumsum(r[k]))
    no = np.r_[c[0], nc[:-1]] * orel[k]
    return pd.DataFrame({"open": no, "high": np.maximum(no, nc) * hrel[k], "low": np.minimum(no, nc) * lrel[k],
                         "close": nc, "volume": v[1:][k]}, index=df.index[1:])


def build(df, symbol, interval, null):
    o, h, l, c, v = (df[k].values for k in ("open", "high", "low", "close", "volume"))
    cs = df["close"]
    e20, e50, e80, e200 = (ema(cs, n).values for n in (20, 50, 80, 200))
    rs = rsi(cs).values
    pats, a = detect(o, h, l, c)
    rows = []
    for pt in pats:
        j = pt["conf"]
        if j < WARMUP or j >= len(c):
            continue
        resolve(pt, c, h, l)
        aj = a[j]
        if pt["kind"] == "lines":
            U, D = pt["U"](j), pt["D"](j)
            pos = (c[j] - D) / (U - D) if U > D else 0.5
            width_ratio = (pt["U"](j) - pt["D"](j)) / max(1e-12, pt["U"](pt["s"]) - pt["D"](pt["s"]))
        else:
            pos = pt["dir"] * (c[j] - pt["N"](j)) / max(1e-12, pt["height"])  # >0 = on the textbook side
            width_ratio = np.nan
        tr = 1 if e20[j] > e50[j] else -1
        tb = pt["dir"]
        pos_vote = (1 if pos > 0.5 else -1) if pt["kind"] == "lines" else 0
        score = tb + tr + pos_vote
        vote_call = int(np.sign(score)) or tb or tr
        vol_ratio = v[max(0, j - 4):j + 1].mean() / max(1e-12, v[pt["s"]:pt["e"] + 1].mean())
        rows.append(dict(
            symbol=symbol, interval=interval, null=null, time=df.index[j], name=pt["name"], kind=pt["kind"],
            textbook=tb, bilateral=int(pt["bilateral"]),
            trend=tr, e20_slope=(e20[j] - e20[max(0, j - 5)]) / aj, htf_trend=1 if e80[j] > e200[j] else -1,
            dist_e200=(c[j] - e200[j]) / aj, pos=pos, width_ratio=width_ratio,
            length=pt["e"] - pt["s"], height_atr=pt["height"] / aj, pole_atr=pt.get("pole", 0.0),
            ret3_atr=(c[j] - c[max(0, j - 3)]) / aj, ret12_atr=(c[j] - c[max(0, j - 12)]) / aj,
            rsi=rs[j], vol_ratio=vol_ratio, atr_pct=aj / c[j], wait=j - pt["e"],
            vote_call=vote_call, status=pt["status"], bdir=pt["bdir"],
            broke_up=np.nan if pt["bdir"] == 0 else int(pt["bdir"] > 0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default=None, help="only this timeframe, e.g. 1h")
    a = ap.parse_args()
    files = sorted(Path("data").glob("*_*.parquet"))
    rows = []
    for f in files:
        sym, tf = f.stem.rsplit("_", 1)
        if a.interval and tf != a.interval:
            continue
        df = pd.read_parquet(f)
        r = build(df, sym, tf, 0)
        rn = build(shuffled(df, seed=abs(hash(f.stem)) % 2**32), sym, tf, 1)
        rows += r + rn
        print(f"{sym} {tf}: {len(r)} patterns, {len(rn)} in its shuffled copy")
    if not rows:
        raise SystemExit("No data files found in data/. Run data.py first.")
    ds = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    ds.to_csv("results/chart_patterns_dataset.csv", index=False)

    br = ds[ds.bdir != 0]
    print(f"\n{len(ds)} patterns ({(ds.null == 1).sum()} from shuffled copies). Saved results/chart_patterns_dataset.csv\n")

    def right(t, col):
        t = t[t.textbook != 0] if col == "textbook" else t
        return (t.bdir == t[col]).mean() * 100 if len(t) else np.nan

    def tgt(t):  # after the breakout: share of resolved trades that hit the target before the stop
        done = t[t.status.isin(["target", "stop"])]
        return (done.status == "target").mean() * 100 if len(done) else np.nan

    out = []
    for nm, g in br.groupby("name"):
        real, nul = g[g.null == 0], g[g.null == 1]
        out.append({"pattern": nm, "n_real": len(real), "n_null": len(nul), "textbook_dir": int(g.textbook.iloc[0]),
                    "textbook_real_%": right(real, "textbook"), "textbook_null_%": right(nul, "textbook"),
                    "vote_real_%": right(real, "vote_call"), "vote_null_%": right(nul, "vote_call"),
                    "target_real_%": tgt(real), "target_null_%": tgt(nul)})
    t = pd.DataFrame(out)
    t["textbook_edge_pp"] = t["textbook_real_%"] - t["textbook_null_%"]
    t["vote_edge_pp"] = t["vote_real_%"] - t["vote_null_%"]
    t["target_edge_pp"] = t["target_real_%"] - t["target_null_%"]
    print("Which side the breakout went, real data vs shuffled copy (edge = real minus shuffled):")
    print(t.sort_values("n_real", ascending=False).round(1).to_string(index=False))
    for lab, col in (("textbook", "textbook"), ("vote rule", "vote_call")):
        r, n = right(br[br.null == 0], col), right(br[br.null == 1], col)
        print(f"All patterns, {lab}: real {r:.1f}%, shuffled {n:.1f}%, edge {r - n:+.1f} pp")
    r, n = tgt(br[br.null == 0]), tgt(br[br.null == 1])
    print(f"All patterns, target before stop after breakout: real {r:.1f}%, shuffled {n:.1f}%, edge {r - n:+.1f} pp")


if __name__ == "__main__":
    main()
