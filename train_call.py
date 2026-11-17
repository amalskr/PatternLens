"""Learned direction call for chart patterns, tested against simpler rules.

Question: once a chart pattern is recognised, which side will it break? Each method is scored on
patterns recognised AFTER the cutoff date, having learned only from patterns before it.

Methods compared:
  coin flip     50%
  majority      always the side that broke more often in training
  textbook      the pattern's textbook direction (trend for patterns with none)
  trend         EMA 20 above or below EMA 50
  vote rule     the page's current 3-vote rule
  logistic      learned model, chosen in advance as the product model (simple, well calibrated)
  lightgbm      learned model, for comparison only

Every method is also run on the shuffled copies (null = 1 in the dataset), where no real pattern can
exist. Because swing points are confirmed late, most methods look good even there. The number that
matters is the EDGE: real score minus shuffled score.

Usage:
  python chart_patterns.py                    # build the dataset first
  python train_call.py --cutoff 2024-07-01
"""
import argparse

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NUM = ["textbook", "bilateral", "trend", "e20_slope", "htf_trend", "dist_e200", "pos", "width_ratio", "length",
       "height_atr", "pole_atr", "ret3_atr", "ret12_atr", "rsi", "vol_ratio", "atr_pct", "wait"]


def features(ds, names, tfs):
    X = ds[NUM].copy()
    X["width_ratio"] = X["width_ratio"].fillna(1.0)
    X["rsi"] = X["rsi"].fillna(50.0)
    X["vol_ratio"] = X["vol_ratio"].clip(0, 10).fillna(1.0)
    for n in names:
        X["is_" + n] = (ds["name"] == n).astype(int)
    for tf in tfs:
        X["tf_" + tf] = (ds["interval"] == tf).astype(int)
    return X.astype(float)


def score(name, call, y, prob=None):
    right = int((call == y).sum())
    n = len(y)
    row = {"method": name, "n": n, "right_%": 100 * right / n, "p_vs_50%": binomtest(right, n, 0.5, alternative="greater").pvalue}
    if prob is not None:
        row["auc"] = roc_auc_score(y, prob)
        row["brier"] = brier_score_loss(y, prob)
        conf = (prob >= 0.6) | (prob <= 0.4)
        row["confident_share_%"] = 100 * conf.mean()
        row["confident_right_%"] = 100 * (call[conf] == y[conf]).mean() if conf.any() else np.nan
    return row


def evaluate(ds, cut, names, tfs, verbose):
    """Learn before cut, test after it, on one dataset (real or shuffled). Returns a table and the model."""
    y_all = (ds.bdir > 0).astype(int).values
    X = features(ds, names, tfs)
    tr, te = (ds.time < cut).values, (ds.time >= cut).values
    y_tr, y_te, t = y_all[tr], y_all[te], ds[te]
    to01 = lambda d: (np.asarray(d) > 0).astype(int)
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X[tr], y_tr)
    p_log = logit.predict_proba(X[te])[:, 1]
    gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(X[tr], y_tr)
    p_gbm = gbm.predict_proba(X[te])[:, 1]
    textbook = np.where(t.textbook != 0, t.textbook, t.trend)
    rows = [score("majority", np.full(len(y_te), int(y_tr.mean() >= 0.5)), y_te),
            score("textbook", to01(textbook), y_te),
            score("trend", to01(t.trend), y_te),
            score("vote rule", to01(t.vote_call), y_te),
            score("logistic", (p_log >= 0.5).astype(int), y_te, p_log),
            score("lightgbm", (p_gbm >= 0.5).astype(int), y_te, p_gbm)]
    extra = {"t": t, "y_te": y_te, "p_log": p_log, "logit": logit, "X": X, "y_all": y_all, "textbook": textbook}
    return pd.DataFrame(rows), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2024-07-01", help="learn before this date, test from it")
    a = ap.parse_args()
    all_ds = pd.read_csv("results/chart_patterns_dataset.csv", parse_dates=["time"])
    all_ds = all_ds[all_ds.bdir != 0].copy()  # only patterns that broke out have a side to predict
    names, tfs = sorted(all_ds["name"].unique()), sorted(all_ds["interval"].unique())
    cut = pd.Timestamp(a.cutoff, tz="UTC")
    real, null = all_ds[all_ds.null == 0].reset_index(drop=True), all_ds[all_ds.null == 1].reset_index(drop=True)
    print(f"Real: {len(real)} breakouts ({(real.time < cut).sum()} to learn from, {(real.time >= cut).sum()} to test). "
          f"Shuffled: {len(null)}. Timeframes: {', '.join(tfs)}.\n")
    r_res, r = evaluate(real, cut, names, tfs, True)
    n_res, _ = evaluate(null, cut, names, tfs, False)
    res = r_res[["method", "n", "right_%", "p_vs_50%"]].rename(columns={"right_%": "real_%", "n": "n_real"})
    res["shuffled_%"] = n_res["right_%"].values
    res["EDGE_pp"] = res["real_%"] - res["shuffled_%"]
    for col in ("auc", "brier", "confident_share_%", "confident_right_%"):
        res[col] = r_res.get(col)
    print("Which side did the breakout go? Test period only. EDGE = real minus shuffled.")
    print(res.round(3).to_string(index=False))
    print("\nOnly a positive EDGE is evidence that a method reads something real in the chart.")
    print("Signals in the same hours on several coins are correlated, so p-values are optimistic.\n")

    t, y_te, p_log = r["t"], r["y_te"], r["p_log"]
    by = []
    for nm, g in t.groupby("name"):
        idx = (t.name == nm).values
        yy = y_te[idx]
        if len(yy) < 30:
            continue
        by.append({"pattern": nm, "n": len(yy), "broke_up_%": 100 * yy.mean(),
                   "textbook_%": 100 * ((r["textbook"][idx] > 0).astype(int) == yy).mean() if g.textbook.iloc[0] else np.nan,
                   "vote_%": 100 * ((g.vote_call.values > 0).astype(int) == yy).mean(),
                   "logistic_%": 100 * ((p_log[idx] >= 0.5).astype(int) == yy).mean()})
    if by:
        print("By pattern, real data, test period (30 or more breakouts):")
        print(pd.DataFrame(by).sort_values("n", ascending=False).round(1).to_string(index=False))

    bins = pd.cut(p_log, [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    cal = pd.DataFrame({"bin": bins, "y": y_te}).groupby("bin", observed=True).agg(n=("y", "size"), broke_up_pct=("y", "mean"))
    cal["broke_up_pct"] = (cal["broke_up_pct"] * 100).round(1)
    print("\nLogistic calibration, real data (predicted chance of breaking up vs what happened):")
    print(cal.to_string())

    logit, X = r["logit"], r["X"]
    coef = pd.Series(logit[-1].coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    print("\nStrongest logistic weights (standardised; + pushes toward up):")
    print(coef.head(10).round(3).to_string())

    logit.fit(X, r["y_all"])  # product model: refit on all real breakouts
    joblib.dump({"model": logit, "names": names, "tfs": tfs, "num": NUM, "cutoff": a.cutoff,
                 "test": res.to_dict("records")}, "models/pattern_call.joblib")
    res.to_csv("results/pattern_call_test.csv", index=False)
    print("\nSaved models/pattern_call.joblib (refit on all real data) and results/pattern_call_test.csv")


if __name__ == "__main__":
    main()
