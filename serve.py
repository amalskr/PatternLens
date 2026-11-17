"""Local model server for the Pattern Lens page.

Usage: uvicorn serve:app --port 8000
"""
import csv
import math
from pathlib import Path

import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from features import build_features, klines_to_df

app = FastAPI(title="Pattern Lens model server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
def clean(d):
    """JSON cannot carry NaN, so send null instead."""
    if d is None:
        return None
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


MODELS = {p.stem: joblib.load(p) for p in Path("models").glob("*.joblib")}


@app.get("/predict")
def predict(symbol: str = "ADAUSDT", interval: str = "1h"):
    key = f"{symbol.upper()}_{interval}"
    path = Path("models") / f"{key}.joblib"
    if key not in MODELS and path.exists():  # trained after the server started: load it now
        MODELS[key] = joblib.load(path)
    if key not in MODELS:
        raise HTTPException(404, f"No model for {key}. Run data.py and train.py for it first.")
    b = MODELS[key]
    r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                     params={"symbol": symbol.upper(), "interval": interval, "limit": 1000}, timeout=10)
    if r.status_code != 200:
        raise HTTPException(502, f"Binance returned HTTP {r.status_code}")
    df = klines_to_df(r.json()[:-1])  # drop the candle that is still forming
    X, _ = build_features(df)
    p = float(b["iso"].predict(b["model"].predict_proba(X[b["features"]].iloc[[-1]])[:, 1])[0])
    signal = "BUY" if p >= b["up"] else "SELL" if p <= b["dn"] else "NO TRADE"
    return {"symbol": symbol.upper(), "interval": interval, "candle_time": df.index[-1].isoformat(),
            "prob_up": round(p, 3), "signal": signal, "horizon": b["horizon"], "k": b["k"],
            "up": b["up"], "dn": b["dn"], "cv": clean(b["cv_summary"]), "holdout": clean(b["holdout_metrics"])}


@app.get("/models")
def models():
    return sorted(p.stem for p in Path("models").glob("*.joblib"))


@app.get("/stats")
def stats(symbol: str = "ADAUSDT", interval: str = "1h", horizon: int = 5):
    """Full-history pattern results written by pattern_stats.py, for the page's bias panel."""
    path = Path("results") / f"pattern_stats_{symbol.upper()}_{interval}_h{horizon}.csv"
    if not path.exists():
        raise HTTPException(404, f"Run: python pattern_stats.py --symbol {symbol.upper()} --interval {interval} --horizon {horizon}")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return [{"pattern": r["pattern"], "trend": r["trend"], "n": int(r["n"]), "win": float(r["win"]),
             "base": float(r["base"]), "p_holm": float(r["p_holm"])} for r in rows]


# ---------- Trained pattern model and honest pattern statistics ----------
import json
import os

import numpy as np
import pandas as pd
from pydantic import BaseModel

from chart_patterns import build
from train_call import features

CALL_PATH = Path("models/pattern_call.joblib")
CALL = joblib.load(CALL_PATH) if CALL_PATH.exists() else None
_PSTATS = None


def pattern_stats():
    """Per pattern, over the full history: how often the target was hit before the stop after a breakout,
    on real data and on shuffled copies (from chart_patterns.py)."""
    global _PSTATS
    if _PSTATS is None:
        _PSTATS = {}
        p = Path("results/chart_patterns_dataset.csv")
        if p.exists():
            ds = pd.read_csv(p, usecols=["name", "null", "status"])
            done = ds[ds.status.isin(["target", "stop"])]
            for nm, g in done.groupby("name"):
                real, nul = g[g.null == 0], g[g.null == 1]
                if len(real) >= 30 and len(nul) >= 30:
                    _PSTATS[nm] = {"n": int(len(real)), "target_real": float((real.status == "target").mean()),
                                   "target_null": float((nul.status == "target").mean())}
    return _PSTATS


class Candles(BaseModel):
    symbol: str
    interval: str
    candles: list[list[float]]  # [time_ms, open, high, low, close, volume], closed candles only


@app.post("/patterns")
def patterns(req: Candles):
    """Runs the research detector on the page's candles: patterns forming now or running after a breakout,
    each with the trained model's P(up) and the honest statistics for that pattern type."""
    df = pd.DataFrame(req.candles, columns=["t", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    rows = [r for r in build(df[["open", "high", "low", "close", "volume"]], req.symbol.upper(), req.interval, 0)
            if r["status"] in ("forming", "open")]
    probs = [None] * len(rows)
    if rows and CALL:
        X = features(pd.DataFrame(rows), CALL["names"], CALL["tfs"])
        probs = CALL["model"].predict_proba(X)[:, 1].tolist()
    st = pattern_stats()
    test = {r["method"]: r for r in (CALL or {}).get("test", [])}
    lg = test.get("logistic")
    return {
        "patterns": [{"name": r["name"], "status": r["status"], "bdir": int(r["bdir"]), "p_up": p,
                      "stats": st.get(r["name"])} for r, p in zip(rows, probs)],
        "model": None if not lg else {"real": lg["real_%"], "shuffled": lg["shuffled_%"], "edge": lg["EDGE_pp"]},
        "stats": st,
    }


# ---------- Forward log ----------
from fastapi import Header

import forward_log

_STORE = None


def store():
    global _STORE
    if _STORE is None:
        _STORE = forward_log.get_store()
    return _STORE


@app.post("/log/scan")
def log_scan(x_scan_token: str | None = Header(default=None)):
    """Called on a schedule (Cloud Scheduler when hosted). If SCAN_TOKEN is set, the caller must send it."""
    if os.environ.get("SCAN_TOKEN") and x_scan_token != os.environ["SCAN_TOKEN"]:
        raise HTTPException(403, "Wrong or missing X-Scan-Token header")
    res = forward_log.scan(store(), CALL)
    res["alerts_sent"] = forward_log.alert(res)
    return res


_SUMMARY = {"at": 0.0, "value": None}


@app.get("/log/summary")
def log_summary():
    """Reading the summary reads every logged pattern, so it is rebuilt at most every 5 minutes
    (keeps Firestore reads, and cost, low when the page refreshes every minute)."""
    import time
    if _SUMMARY["value"] is None or time.time() - _SUMMARY["at"] > 300:
        _SUMMARY["value"] = json.loads(json.dumps(forward_log.summary(store()), default=str))
        _SUMMARY["at"] = time.time()
    return _SUMMARY["value"]


# ---------- Prediction from the page's own candles (no Binance call from the server) ----------
@app.post("/predict")
def predict_from_candles(req: Candles):
    key = f"{req.symbol.upper()}_{req.interval}"
    if key not in MODELS and (Path("models") / f"{key}.joblib").exists():
        MODELS[key] = joblib.load(Path("models") / f"{key}.joblib")
    if key not in MODELS:
        raise HTTPException(404, f"No model for {key}. Run data.py and train.py for it first.")
    b = MODELS[key]
    df = pd.DataFrame(req.candles, columns=["t", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    X, _ = build_features(df[["open", "high", "low", "close", "volume"]])
    p = float(b["iso"].predict(b["model"].predict_proba(X[b["features"]].iloc[[-1]])[:, 1])[0])
    signal = "BUY" if p >= b["up"] else "SELL" if p <= b["dn"] else "NO TRADE"
    return {"symbol": req.symbol.upper(), "interval": req.interval, "candle_time": df.index[-1].isoformat(),
            "prob_up": round(p, 3), "signal": signal, "horizon": b["horizon"], "k": b["k"],
            "up": b["up"], "dn": b["dn"], "cv": clean(b["cv_summary"]), "holdout": clean(b["holdout_metrics"])}


# ---------- Hosting: Firebase Hosting forwards /api/... to this server ----------
@app.middleware("http")
async def strip_api_prefix(request, call_next):
    path = request.scope["path"]
    if path == "/api" or path.startswith("/api/"):
        request.scope["path"] = path[4:] or "/"
    return await call_next(request)


# ---------- Self-hosting: serve the page and protect everything with one password ----------
import base64
import secrets

from fastapi.responses import FileResponse, Response


@app.get("/", include_in_schema=False)
def page():
    return FileResponse("pattern-lens.html")


@app.middleware("http")
async def password(request, call_next):
    """If PAGE_PASSWORD is set, the browser asks for it once (any user name); nothing else changes."""
    pw = os.environ.get("PAGE_PASSWORD")
    if pw and request.method != "OPTIONS":
        ok, h = False, request.headers.get("authorization", "")
        if h.startswith("Basic "):
            try:
                ok = secrets.compare_digest(base64.b64decode(h[6:]).decode().partition(":")[2], pw)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Pattern Lens"'})
    return await call_next(request)
