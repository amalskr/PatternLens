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


# ---------- AI analyst ----------
ANALYST_RULES = """You explain a crypto chart to a beginner, using ONLY the facts in the JSON you are given.
Rules:
- Do not forecast prices and do not tell the reader to buy or sell.
- For any chart pattern, say how often its textbook target was reached before the stop in past data,
  and compare it with the same number on random (shuffled) data. If the two are close, say plainly that
  the pattern showed no real edge.
- If the facts say a model or rule has no edge over random data, say so in plain words.
- Mention the levels that matter right now (breakout level, stop, target) as plain prices.
- At most 130 words, short sentences, no jargon (say "average move size" instead of ATR).
- Write in {lang}.
- End with one short line saying this is not financial advice."""
_CACHE = {}


class Explain(BaseModel):
    lang: str  # "si" or "en"
    facts: dict


@app.post("/explain")
def explain(req: Explain):
    key = json.dumps([req.lang, req.facts], sort_keys=True)
    if key in _CACHE:
        return _CACHE[key]
    lang = "Sinhala (සිංහල), with crypto and pattern names left in English" if req.lang == "si" else "English"
    system = ANALYST_RULES.format(lang=lang)
    user = "Facts:\n" + json.dumps(req.facts, indent=1, ensure_ascii=False)
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):  # paid, better Sinhala
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                "content-type": "application/json"}, json={
                "model": os.environ.get("ANALYST_MODEL", "claude-haiku-5-5"), "max_tokens": 600,
                "system": system, "messages": [{"role": "user", "content": user}]})
            r.raise_for_status()
            text, engine = "".join(b.get("text", "") for b in r.json()["content"]), "Claude API"
        else:  # free, runs on this computer
            model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
            r = requests.post("http://localhost:11434/api/chat", timeout=180, json={
                "model": model, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
            text, engine = r.json()["message"]["content"], f"Ollama {model} (local)"
    except Exception as e:
        raise HTTPException(503, "No AI engine reachable. Free option: install Ollama and run "
                                 "'ollama pull qwen2.5:7b'. Paid option: set ANTHROPIC_API_KEY before starting the server. "
                                 f"({type(e).__name__})")
    out = {"text": text.strip(), "engine": engine}
    _CACHE[key] = out
    return out
