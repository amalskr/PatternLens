"""One combined call (BUY, SELL or WAIT) for the next hour, from everything the system knows, and an
honest record of how those calls turned out.

Votes (positive = up):
  EMA 15m         +1 / -1   EMA 20 above or below EMA 50
  Bitcoin 1h      +1 / -1   Bitcoin's 1h trend (ADA mostly follows Bitcoin)
  Breakout        +1 / -1   a running chart pattern breakout, unless price came back inside the pattern
  Candles       +0.5 / -0.5 candlestick patterns on the last 3 candles whose full-history effect was
                            significant, in the direction the data showed (not the textbook)
  BTC fast move +0.5 / -0.5 Bitcoin moving more than twice a usual hour, in that direction
  Funding       +0.5 / -0.5 extreme funding: very high favours a drop (long squeeze), negative a rise
Score >= 1.5: BUY. Score <= -1.5: SELL. Otherwise WAIT.

Every closed candle's call is stored with its price. Four candles later it is scored: a BUY is right if
the close is higher, a SELL if it is lower. Compare with how often price simply rises.
"""
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import ema, patterns, PATTERN_DIR

HORIZON = 4  # candles: one hour on 15m
DB = Path(os.environ.get("CALLS_DB", "results/calls.sqlite"))
LR = {"Bull engulfing": "bull_engulf", "Bear engulfing": "bear_engulf", "Hammer": "hammer",
      "Shooting star": "shooting_star", "Morning star": "morning_star", "Evening star": "evening_star",
      "Three soldiers": "three_soldiers", "Three crows": "three_crows"}


def _research(symbol, tf):
    p = Path(f"results/pattern_stats_{symbol}_{tf}_h5.csv")
    return pd.read_csv(p) if p.exists() else None


def compute(symbol, tf, df):
    """df: closed candles (open, high, low, close, volume). Returns the call with every vote explained."""
    import market
    from chart_patterns import build
    c = df["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    votes = []

    up = e20.iloc[-1] > e50.iloc[-1]
    votes.append({"k": "EMA 15m" if tf == "15m" else f"EMA {tf}", "v": 1 if up else -1,
                  "why": f"EMA 20 (green) {'above' if up else 'below'} EMA 50 (red)"})

    b = market.btc_state()
    if "1h" in b:
        t = b["1h"]["trend"]
        votes.append({"k": "Bitcoin 1h", "v": 1 if t == "up" else -1, "why": f"Bitcoin's 1h trend is {t}"})
        if b.get("fast"):
            s = 0.5 if b["chg_1h_%"] > 0 else -0.5
            votes.append({"k": "BTC fast move", "v": s, "why": f"Bitcoin moved {b['chg_1h_%']:+.1f}% this hour"})

    rows = [r for r in build(df, symbol, tf, 0) if r["status"] == "open"]
    if rows:
        r = rows[-1]
        if r.get("back_in"):
            votes.append({"k": "Breakout", "v": 0, "why": f"{r['name']} broke {'up' if r['bdir'] > 0 else 'down'} but came back inside"})
        else:
            votes.append({"k": "Breakout", "v": int(r["bdir"]), "why": f"{r['name']} broke {'up' if r['bdir'] > 0 else 'down'}"})

    st = _research(symbol, tf)
    if st is not None:
        p = patterns(df, e20).iloc[-3:]
        trend = "up" if up else "down"
        cv, names = 0.0, []
        for page_name, col in LR.items():
            if col in p and p[col].any():
                row = st[(st.pattern == col) & (st.trend == trend)]
                if len(row) and row.iloc[0].p_holm < 0.05:
                    edge = row.iloc[0].win - row.iloc[0].base
                    cv += 0.5 * np.sign(edge) * PATTERN_DIR[col]
                    names.append(page_name)
        if names:
            votes.append({"k": "Candles", "v": float(np.clip(cv, -0.5, 0.5)),
                          "why": f"{', '.join(names)}: significant in the full history, direction as the data showed"})

    pos = market.positioning(symbol)
    fr = pos.get("funding_%") if isinstance(pos, dict) else None
    if isinstance(fr, (int, float)) and (fr > 0.05 or fr < 0):
        votes.append({"k": "Funding", "v": -0.5 if fr > 0.05 else 0.5,
                      "why": f"funding {fr:.3f}%: {'many longs, squeeze risk down' if fr > 0.05 else 'many shorts, squeeze risk up'}"})

    score = float(sum(v["v"] for v in votes))
    call = "BUY" if score >= 1.5 else "SELL" if score <= -1.5 else "WAIT"
    return {"symbol": symbol, "tf": tf, "candle": df.index[-1].isoformat(), "price": float(c.iloc[-1]),
            "score": score, "call": call, "votes": votes, "horizon": HORIZON,
            "summary": plain_summary(symbol, tf, call, votes)}


def plain_summary(symbol, tf, call, votes):
    """Two or three plain sentences, no numbers: what Bitcoin does, what the symbol does, what follows."""
    coin = symbol.replace("USDT", "")
    v = {x["k"]: x for x in votes}
    parts = []
    btc = v.get("Bitcoin 1h")
    if btc:
        fast = " fast" if "BTC fast move" in v and (v["BTC fast move"]["v"] > 0) == (btc["v"] > 0) else ""
        parts.append(f"Bitcoin is {'rising' if btc['v'] > 0 else 'falling'}{fast}.")
    own = []
    ema_vote = next((x for x in votes if x["k"].startswith("EMA")), None)
    if ema_vote:
        own.append(f"{coin}'s short-term trend is {'up' if ema_vote['v'] > 0 else 'down'}")
    br = v.get("Breakout")
    if br:
        name = br["why"].split(" broke ")[0].lower()
        name = ("an " if name[0] in "aeiou" else "a ") + name
        if br["v"] == 0:
            own.append(f"its breakout from {name} is failing")
        else:
            own.append(f"it broke {'up' if br['v'] > 0 else 'down'} out of {name}")
    cd = v.get("Candles")
    if cd and cd["v"]:
        own.append(f"its last candles have often been followed by a {'bounce' if cd['v'] > 0 else 'drop'}")
    fd = v.get("Funding")
    if fd:
        own.append("many traders are betting on a rise, so a sudden drop is possible" if fd["v"] < 0
                   else "many traders are betting on a fall, so a sudden jump is possible")
    if own:
        text = own[0] if len(own) == 1 else ", ".join(own[:-1]) + " and " + own[-1]
        parts.append(text[0].upper() + text[1:] + ".")
    hours = {"15m": 0.25, "1h": 1, "4h": 4}.get(tf, 0.25) * HORIZON
    when = "the next hour" if hours == 1 else f"the next {hours:g} hours"
    if call == "BUY":
        parts.append(f"So {symbol} may go up in {when}.")
    elif call == "SELL":
        parts.append(f"So {symbol} may go down in {when}.")
    else:
        parts.append(f"The signals point different ways, so no clear direction for {symbol} in {when}.")
    return " ".join(parts)


def _db():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB, timeout=15)
    con.execute("""CREATE TABLE IF NOT EXISTS calls (symbol TEXT, tf TEXT, candle TEXT, call TEXT, score REAL,
                   price REAL, later REAL, outcome TEXT, PRIMARY KEY (symbol, tf, candle))""")
    return con


def track(symbol, tf, df):
    """Record this candle's call (once) and score calls that are now HORIZON candles old.
    Returns the new call if it differs from the previous candle's call."""
    con = _db()
    prev = con.execute("SELECT candle, call FROM calls WHERE symbol=? AND tf=? ORDER BY candle DESC LIMIT 1",
                       (symbol, tf)).fetchone()
    if prev and prev[0] == df.index[-1].isoformat():
        con.close()
        return None  # this candle's call is already recorded; the record is never changed
    res = compute(symbol, tf, df)
    con.execute("INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?,?,NULL,NULL)",
                (symbol, tf, res["candle"], res["call"], res["score"], res["price"]))
    closes = {t.isoformat(): float(v) for t, v in df["close"].items()}
    times = list(closes)
    for cand, call, price in con.execute("SELECT candle, call, price FROM calls WHERE symbol=? AND tf=? AND later IS NULL",
                                         (symbol, tf)).fetchall():
        if cand in closes and times.index(cand) + HORIZON < len(times):
            later = closes[times[times.index(cand) + HORIZON]]
            outcome = "up" if later > price else "down" if later < price else "flat"
            con.execute("UPDATE calls SET later=?, outcome=? WHERE symbol=? AND tf=? AND candle=?",
                        (later, outcome, symbol, tf, cand))
    con.commit()
    con.close()
    changed = prev is None or prev[1] != res["call"]
    return res if changed else None


def summary(symbol, tf):
    con = _db()
    d = pd.read_sql_query("SELECT * FROM calls WHERE symbol=? AND tf=? AND outcome IS NOT NULL", con, params=(symbol, tf))
    con.close()
    if d.empty:
        return {"scored": 0}
    pct = lambda a, b: round(100 * float(a) / b, 1) if b else None
    buy, sell = d[d.call == "BUY"], d[d.call == "SELL"]
    return {"since": d.candle.min(), "scored": len(d), "base_up_%": pct((d.outcome == "up").sum(), len(d)),
            "buy": len(buy), "buy_right_%": pct((buy.outcome == "up").sum(), len(buy)),
            "sell": len(sell), "sell_right_%": pct((sell.outcome == "down").sum(), len(sell)),
            "wait": int((d.call == "WAIT").sum())}
