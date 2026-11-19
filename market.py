"""Market context around a symbol: what Bitcoin is doing, how closely the symbol follows it, futures
positioning (funding, open interest), the Crypto Fear and Greed index, and recent headlines.
All sources are free and need no API key. Results are cached for 5 minutes.
"""
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np
import requests

from features import ema, klines_to_df

FAPI = "https://fapi.binance.com"
FEEDS = {"CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/", "Cointelegraph": "https://cointelegraph.com/rss"}
KEYWORDS = {"ADAUSDT": ["cardano", " ada "], "BTCUSDT": ["bitcoin", "btc"], "ETHUSDT": ["ethereum", "eth "],
            "SOLUSDT": ["solana"], "XRPUSDT": ["xrp", "ripple"], "DOGEUSDT": ["dogecoin", "doge"]}
_CACHE = {}


def _cached(key, fn, ttl=300):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        val = fn()
    except Exception as e:
        val = {"error": type(e).__name__}
    _CACHE[key] = (time.time(), val)
    return val


def _klines(symbol, tf, limit=300):
    r = requests.get(f"{FAPI}/fapi/v1/klines", params={"symbol": symbol, "interval": tf, "limit": limit}, timeout=10)
    r.raise_for_status()
    return klines_to_df(r.json()[:-1])  # closed candles


def btc_state():
    def build():
        out = {}
        for tf in ("15m", "1h"):
            df = _klines("BTCUSDT", tf)
            c = df["close"]
            e20, e50 = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1]
            out[tf] = {"trend": "up" if e20 > e50 else "down"}
        h = _klines("BTCUSDT", "1h", 30)["close"]
        out["price"] = float(h.iloc[-1])
        out["chg_1h_%"] = float((h.iloc[-1] / h.iloc[-2] - 1) * 100)
        out["chg_24h_%"] = float((h.iloc[-1] / h.iloc[-25] - 1) * 100)
        moves = (h.pct_change().abs() * 100).iloc[-25:-1]
        out["fast"] = bool(abs(out["chg_1h_%"]) > 2 * moves.mean())  # this hour moved more than twice a usual hour
        return out
    return _cached("btc", build)


def correlation(symbol):
    """How closely the symbol's 15m moves followed Bitcoin's over the last 2 days (0 = not at all, 1 = exactly)."""
    def build():
        a = _klines(symbol, "15m", 200)["close"].pct_change()
        b = _klines("BTCUSDT", "15m", 200)["close"].pct_change()
        return {"corr_2d": float(np.corrcoef(a.iloc[-192:], b.iloc[-192:])[0, 1])}
    return _cached(f"corr|{symbol}", build)


def positioning(symbol):
    def build():
        p = requests.get(f"{FAPI}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=10).json()
        oi = requests.get(f"{FAPI}/futures/data/openInterestHist",
                          params={"symbol": symbol, "period": "1h", "limit": 25}, timeout=10).json()
        out = {"funding_%": float(p["lastFundingRate"]) * 100}
        if isinstance(oi, list) and len(oi) >= 25:
            out["oi_chg_24h_%"] = (float(oi[-1]["sumOpenInterest"]) / float(oi[0]["sumOpenInterest"]) - 1) * 100
        return out
    return _cached(f"pos|{symbol}", build)


def fear_greed():
    def build():
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()["data"][0]
        return {"value": int(d["value"]), "label": d["value_classification"]}
    return _cached("fng", build, ttl=1800)


def headlines(symbol, n=6):
    """Latest headlines; ones that mention the symbol first, then Bitcoin and the market in general."""
    def build():
        items = []
        for src, url in FEEDS.items():
            try:
                root = ET.fromstring(requests.get(url, timeout=10, headers={"User-Agent": "PatternLens"}).content)
            except Exception:
                continue
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                try:
                    ts = parsedate_to_datetime(it.findtext("pubDate")).timestamp()
                except Exception:
                    ts = 0
                items.append({"title": title, "link": (it.findtext("link") or "").strip(), "source": src, "ts": ts})
        seen, uniq = set(), []
        for x in sorted(items, key=lambda x: -x["ts"]):  # newest first, one copy of each title
            if x["title"].lower() not in seen:
                seen.add(x["title"].lower()); uniq.append(x)
        items = uniq
        keys = KEYWORDS.get(symbol, [])
        own = [x for x in items if any(k in f" {x['title'].lower()} " for k in keys)]
        rest = [x for x in items if x not in own]
        return {"items": (own[:4] + rest)[:n], "matched": len(own)}
    return _cached(f"news|{symbol}", build, ttl=600)


def context(symbol):
    return {"symbol": symbol, "btc": btc_state(), "corr": correlation(symbol), "positioning": positioning(symbol),
            "fear_greed": fear_greed(), "news": headlines(symbol)}
