"""Local model server for the Pattern Lens page.

Usage: uvicorn serve:app --port 8000
"""
import math
from pathlib import Path

import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from features import build_features, klines_to_df
from fastapi.responses import RedirectResponse

app = FastAPI(title="Pattern Lens model server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
def clean(d):
    """JSON cannot carry NaN, so send null instead."""
    if d is None:
        return None
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


MODELS = {p.stem: joblib.load(p) for p in Path("models").glob("*.joblib")}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.get("/predict")
def predict(symbol: str = "ADAUSDT", interval: str = "1h"):
    key = f"{symbol.upper()}_{interval}"
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
    return sorted(MODELS)
