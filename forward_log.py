"""Forward log: an out-of-sample track record that cannot be fitted after the fact.

Each scan fetches the latest closed candles for a watchlist, runs the research detector, and:
  * records every pattern that is forming or running the FIRST time it is seen, with the lean and the
    trained model's P(up), before its outcome is known; that record is never changed afterwards;
  * fills in the outcome (breakout side, target or stop) once it happens.

Storage: an SQLite file locally, or Firestore when hosted (LOG_BACKEND=firestore).

Usage:
  python forward_log.py --loop 900      # scan every 15 minutes while this runs
  python forward_log.py --report        # print the track record so far
Watchlist: WATCHLIST=ADAUSDT,BTCUSDT  TIMEFRAMES=15m,1h,4h  (environment variables)
"""
import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from chart_patterns import build
from features import klines_to_df
from train_call import features

WATCH = os.environ.get("WATCHLIST", "ADAUSDT,BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT").split(",")
TFS = os.environ.get("TIMEFRAMES", "15m,1h,4h").split(",")
RESOLVED = {"target", "stop", "timeout", "no breakout"}
# From Study 2 (shuffled data): what chance alone gives, to compare the live numbers against
BASELINE = {"lean_side_random_%": 62.8, "target_random_%": 26.8, "target_research_%": 28.0}


class SqliteStore:
    def __init__(self, path="results/forward_log.sqlite"):
        Path(path).parent.mkdir(exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS log (key TEXT PRIMARY KEY, data TEXT)")

    def get(self, key):
        r = self.db.execute("SELECT data FROM log WHERE key = ?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, key, rec):
        self.db.execute("INSERT OR REPLACE INTO log VALUES (?, ?)", (key, json.dumps(rec)))
        self.db.commit()

    def all(self):
        return [json.loads(r[0]) for r in self.db.execute("SELECT data FROM log")]


class FirestoreStore:
    def __init__(self):
        from google.cloud import firestore  # installed only for hosting
        self.col = firestore.Client().collection("forward_log")

    def get(self, key):
        d = self.col.document(key).get()
        return d.to_dict() if d.exists else None

    def put(self, key, rec):
        self.col.document(key).set(rec)

    def all(self):
        return [d.to_dict() for d in self.col.stream()]


def get_store():
    return FirestoreStore() if os.environ.get("LOG_BACKEND") == "firestore" else SqliteStore()


def fetch(symbol, tf, limit=1000):
    r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                     params={"symbol": symbol, "interval": tf, "limit": limit}, timeout=15)
    r.raise_for_status()
    return klines_to_df(r.json()[:-1])  # closed candles only


def num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def scan(store, call=None):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    added = updated = 0
    for sym in WATCH:
        for tf in TFS:
            try:
                df = fetch(sym, tf)
            except Exception as e:
                print(f"{sym} {tf}: fetch failed ({type(e).__name__})")
                continue
            rows = build(df, sym, tf, 0)
            if call and rows:
                probs = call["model"].predict_proba(features(pd.DataFrame(rows), call["names"], call["tfs"]))[:, 1]
            else:
                probs = [None] * len(rows)
            for r, p in zip(rows, probs):
                key = f"{sym}|{tf}|{r['name']}|{pd.Timestamp(r['start']).isoformat()}"
                rec = store.get(key)
                if rec is None:
                    if r["status"] not in ("forming", "open"):
                        continue  # only patterns seen BEFORE their outcome enter the log
                    rec = {"key": key, "symbol": sym, "timeframe": tf, "pattern": r["name"], "start": key.split("|")[3],
                           "first_seen": now, "first_status": r["status"], "price_seen": float(df["close"].iloc[-1]),
                           "textbook": int(r["textbook"]), "lean": int(r["vote_call"]), "p_up": num(p),
                           "up_entry": num(r.get("up_entry")), "dn_entry": num(r.get("dn_entry")),
                           "status": r["status"], "bdir": int(r["bdir"]), "resolved": False}
                    added += 1
                elif rec["resolved"] or rec["status"] == r["status"]:
                    continue
                else:
                    updated += 1
                rec.update(status=r["status"], bdir=int(r["bdir"]), entry=num(r.get("entry")),
                           stop=num(r.get("stop")), target=num(r.get("target")), last_checked=now)
                if r["status"] in RESOLVED:
                    rec.update(resolved=True, resolved_at=now)
                store.put(key, rec)
    return {"scanned_at": now, "added": added, "updated": updated}


def summary(store):
    recs = store.all()
    if not recs:
        return {"patterns": 0, "baseline": BASELINE}
    d = pd.DataFrame(recs)
    seen_forming = d[(d.first_status == "forming") & (d.bdir != 0)]  # lean was recorded before the breakout
    trades = d[d.status.isin(["target", "stop"])]
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    lean_right = int((seen_forming.lean == seen_forming.bdir).sum())
    hits = int((trades.status == "target").sum())
    return {
        "since": d.first_seen.min(), "patterns": len(d), "open": int((~d.resolved).sum()), "resolved": int(d.resolved.sum()),
        "lean_judged": len(seen_forming), "lean_right": lean_right, "lean_right_%": pct(lean_right, len(seen_forming)),
        "trades_resolved": len(trades), "targets_hit": hits, "target_%": pct(hits, len(trades)),
        "baseline": BASELINE,
        "recent": d[d.resolved].sort_values("resolved_at", ascending=False).head(8)[
            ["symbol", "timeframe", "pattern", "status", "bdir", "lean", "resolved_at"]].to_dict("records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between scans; 0 = scan once")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    store = get_store()
    if a.report:
        print(json.dumps(summary(store), indent=1, default=str))
        return
    call_path = Path("models/pattern_call.joblib")
    call = joblib.load(call_path) if call_path.exists() else None
    while True:
        print(scan(store, call))
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
