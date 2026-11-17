"""Walk-forward training with an untouched hold-out period.

Usage:
  python train.py --symbol ADAUSDT --interval 1h            # develop: walk-forward results only
  python train.py --symbol ADAUSDT --interval 1h --final    # once, at the end: also score the hold-out
"""
import argparse

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from features import build_features, triple_barrier
from scipy.stats import binomtest

WARMUP = 200


def fit(train, feats, horizon):
    split = int(len(train) * 0.8)
    tr, cal = train.iloc[: split - horizon], train.iloc[split:]  # gap so labels do not overlap
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                               min_child_samples=100, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1)
    model.fit(tr[feats], tr.y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(cal[feats])[:, 1], cal.y)
    return model, iso


def predict(model, iso, X):
    return iso.predict(model.predict_proba(X)[:, 1])


def evaluate(p, y, up, dn):
    y = np.asarray(y)
    buy, sell = p >= up, p <= dn
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "base_up_rate": float(y.mean()),
        "buy_signals": int(buy.sum()),
        "buy_precision": float(y[buy].mean()) if buy.any() else float("nan"),
        "sell_signals": int(sell.sum()),
        "sell_precision": float(1 - y[sell].mean()) if sell.any() else float("nan"),
        "coverage": float((buy | sell).mean()),
    }


def show(name, m):
    print(f"{name:<10} n={m['n']:>6}  AUC={m['auc']:.3f}  base up={m['base_up_rate']:.3f}  "
          f"BUY {m['buy_signals']:>5} @ {m['buy_precision']:.3f}  "
          f"SELL {m['sell_signals']:>5} @ {m['sell_precision']:.3f}  coverage={m['coverage']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ADAUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--horizon", type=int, default=5, help="candles to look ahead")
    ap.add_argument("--k", type=float, default=1.0, help="barrier width in ATRs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdout-months", type=int, default=6)
    ap.add_argument("--up", type=float, default=0.60, help="BUY threshold on calibrated P(up)")
    ap.add_argument("--dn", type=float, default=0.40, help="SELL threshold on calibrated P(up)")
    ap.add_argument("--final", action="store_true", help="score the hold-out (do this once)")
    a = ap.parse_args()
    key = f"{a.symbol.upper()}_{a.interval}"

    df = pd.read_parquet(f"data/{key}.parquet")
    X, atr_s = build_features(df)
    y = triple_barrier(df, atr_s, a.horizon, a.k)
    feats = list(X.columns)
    data = X.join(y).iloc[WARMUP:].dropna(subset=["y"])

    cut = data.index.max() - pd.DateOffset(months=a.holdout_months)
    dev, hold = data[data.index < cut], data[data.index >= cut]
    print(f"{key}: {len(data)} labelled candles, dev until {cut.date()}, hold-out {len(hold)} candles")
    print(f"Label: +/-{a.k} ATR within {a.horizon} candles. BUY if P(up) >= {a.up}, SELL if <= {a.dn}\n")

    chunks = np.array_split(np.arange(len(dev)), a.folds + 1)
    results = []
    for f in range(1, a.folds + 1):
        train = dev.iloc[np.concatenate(chunks[:f])]
        test = dev.iloc[chunks[f][a.horizon:]]  # embargo
        model, iso = fit(train, feats, a.horizon)
        m = evaluate(predict(model, iso, test[feats]), test.y, a.up, a.dn)
        results.append(m)
        show(f"fold {f}", m)

    cv = {k: float(np.nanmean([r[k] for r in results])) for k in results[0]}
    show("average", cv)
    print("\nThe model is only useful if BUY precision beats 'base up' and SELL precision beats "
          "1 - 'base up', consistently across folds, after fees.")

    tot = sum(r["buy_signals"] for r in results)
    hits = sum(r["buy_signals"] * r["buy_precision"] for r in results if r["buy_signals"])
    base = cv["base_up_rate"]
    p = binomtest(round(hits), round(tot), base, alternative="greater").pvalue
    print(f"\nPooled BUY: {round(hits)}/{round(tot)} = {hits / tot:.3f} vs base {base:.3f}, p = {p:.4f}")

    model, iso = fit(dev, feats, a.horizon)
    imp = pd.Series(model.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\nTop features (gain):\n" + imp.head(10).round(0).to_string())

    hold_m = None
    if a.final:
        hold_m = evaluate(predict(model, iso, hold[feats]), hold.y, a.up, a.dn)
        print()
        show("HOLD-OUT", hold_m)
        print("Report this number. If you change anything after seeing it, it is no longer a clean hold-out.")

    joblib.dump({"model": model, "iso": iso, "features": feats, "horizon": a.horizon, "k": a.k,
                 "up": a.up, "dn": a.dn, "cv_summary": cv, "holdout_metrics": hold_m},
                f"models/{key}.joblib")
    print(f"\nSaved models/{key}.joblib")


if __name__ == "__main__":
    main()
