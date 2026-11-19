"""Telegram alerts.

  * Model signal: a message when a trained model (models/<SYMBOL>_<interval>.joblib) changes to BUY or
    SELL on a closed candle. Note: these models scored 0.500 in testing, the same as a coin flip.
  * Pattern events (ALERT_PATTERNS=1): a chart pattern starts forming, breaks out, or reaches its
    target or stop. Sent by forward_log.py after each scan.

Setup:
  1. In Telegram, open @BotFather, send /newbot, copy the token it gives you.
  2. Send any message to your new bot, then run:  TELEGRAM_TOKEN=... python alerts.py --chat-id
  3. Test:  TELEGRAM_TOKEN=... TELEGRAM_CHAT_ID=... python alerts.py --test
  4. Run with the forward log:  TELEGRAM_TOKEN=... TELEGRAM_CHAT_ID=... python forward_log.py --loop 300
  5. Right now, whatever the signals are:  python alerts.py --now
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import requests

from features import atr, build_features

STATE = Path("results/alert_state.json")
# Telegram alerts only for these symbols (the forward log still records every symbol in WATCHLIST)
ALERT_SYMBOLS = [x for x in os.environ.get("ALERT_SYMBOLS", "").split(",") if x]


def wanted(symbol):
    return not ALERT_SYMBOLS or symbol in ALERT_SYMBOLS
TF_MIN = {"15m": 15, "1h": 60, "4h": 240}


def enabled():
    return bool(os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send(text):
    r = requests.post(f"https://api.telegram.org/bot{os.environ['TELEGRAM_TOKEN']}/sendMessage",
                      json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text, "disable_web_page_preview": True},
                      timeout=15)
    if not r.ok:
        print(f"Telegram error {r.status_code}: {r.text[:200]}")
    return r.ok


def _state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(st):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def _model_status(path, fetch):
    sym, tf = path.stem.rsplit("_", 1)
    b = joblib.load(path)
    df = fetch(sym, tf)
    X, _ = build_features(df)
    p = float(b["iso"].predict(b["model"].predict_proba(X[b["features"]].iloc[[-1]])[:, 1])[0])
    signal = "BUY" if p >= b["up"] else "SELL" if p <= b["dn"] else "NO TRADE"
    px, step = float(df["close"].iloc[-1]), b["k"] * float(atr(df).iloc[-1])
    hours = TF_MIN.get(tf, 60) * b["horizon"] / 60
    span = f"{hours:g} hours" if hours >= 2 else f"{int(hours * 60)} minutes"
    return dict(sym=sym, tf=tf, p=p, signal=signal, px=px, step=step, span=span, candle=df.index[-1].isoformat(),
                auc=(b.get("cv_summary") or {}).get("auc") or float("nan"))


def model_signals(fetch):
    """Check every trained model on its latest closed candle; alert when the signal changes to BUY or SELL."""
    if os.environ.get("ALERT_MODEL", "1") != "1":
        return 0
    st, sent = _state(), 0
    for path in sorted(Path("models").glob("*_*.joblib")):
        if path.stem == "pattern_call":
            continue
        try:
            m = _model_status(path, fetch)
        except Exception as e:
            print(f"alerts: {path.stem} failed ({type(e).__name__})")
            continue
        sym, tf, p, signal, candle = m["sym"], m["tf"], m["p"], m["signal"], m["candle"]
        px, step, span, auc = m["px"], m["step"], m["span"], m["auc"]
        key = f"model|{path.stem}"
        last = st.get(key, {})
        if last.get("candle") == candle:
            continue  # this candle was already checked
        st[key] = {"candle": candle, "signal": signal}
        if signal == "NO TRADE" or signal == last.get("signal"):
            continue  # alert only when a trade signal appears or flips
        msg = (f"✨ AI model: {signal} signal, {sym} {tf}\n"
               f"Price {px:.5g}, candle closed {pd.Timestamp(candle).tz_convert('Asia/Colombo'):%d %b %H:%M} (Sri Lanka)\n"
               + (f"Next {span}: {100 * p:.0f}% chance price reaches {px + step:.5g} before {px - step:.5g}.\n" if signal == "BUY" else
                  f"Next {span}: {100 * (1 - p):.0f}% chance price drops to {px - step:.5g} before {px + step:.5g}.\n")
               + 
               f"Model test score {auc:.3f} (0.500 = coin flip).\n{URL}")
        if send(msg):
            sent += 1
    _save(st)
    return sent


def _pattern_rates():
    p = Path("results/chart_patterns_dataset.csv")
    if not p.exists():
        return {}
    cols = pd.read_csv(p, nrows=0).columns
    ds = pd.read_csv(p, usecols=[c for c in ("name", "null", "status", "back_in") if c in cols])
    done = ds[ds.status.isin(["target", "stop"])]
    out = {}
    for nm, g in done.groupby("name"):
        r, n = g[g.null == 0], g[g.null == 1]
        if len(r) >= 30 and len(n) >= 30:
            out[nm] = ((r.status == "target").mean() * 100, (n.status == "target").mean() * 100)
    if "back_in" in ds:  # all patterns together: target rate once price came back inside
        bi = done[(done.null == 0) & (done.back_in == 1)]
        if len(bi) >= 30:
            out["_back_in"] = (bi.status == "target").mean() * 100
    return out


def _btc_line(bdir):
    """Is Bitcoin's 1h trend with or against this breakout? Empty if Bitcoin data is not available."""
    try:
        import market
        b = market.btc_state()
        t = b["1h"]["trend"]
        side = "with" if (t == "up") == (bdir > 0) else "against"
        return f"\nBTC 1h trend {t} ({b['chg_1h_%']:+.1f}% last hour): breakout is {side} Bitcoin."
    except Exception:
        return ""


def pattern_events(events):
    """events from forward_log.scan: new patterns, breakouts and results."""
    if not events or os.environ.get("ALERT_PATTERNS") != "1":
        return 0
    rates, sent = _pattern_rates(), 0
    f = lambda v: "n/a" if v is None else f"{v:.5g}"
    for ev in events:
        r = ev["rec"]
        if not wanted(r["symbol"]):
            continue
        head = f"{r['symbol']} {r['timeframe']} {r['pattern']}"
        hist = "" if ev["type"] == "failing" else f"\nHistory: targets hit {rates[r['pattern']][0]:.0f}% (random data {rates[r['pattern']][1]:.0f}%)." if r["pattern"] in rates else ""
        if ev["type"] == "new":
            if r["status"] != "forming":
                continue
            lv = f"breaks up above {f(r.get('up_entry'))}, down below {f(r.get('dn_entry'))}" if r.get("up_entry") else "watching the neckline"
            msg = f"New pattern forming: {head}\nIt {lv}.{hist}"
        elif ev["type"] == "breakout":
            msg = (f"Breakout: {head} broke {'up' if r['bdir'] > 0 else 'down'} at {f(r.get('entry'))}\n"
                   f"Target {f(r.get('target'))}, stop {f(r.get('stop'))}.{hist}{_btc_line(r['bdir'])}")
        elif ev["type"] == "failing":
            back = f"\nHistory: after coming back inside, targets hit only {rates['_back_in']:.0f}%." if "_back_in" in rates else ""
            msg = (f"BREAKOUT FAILING - {head}: price closed back inside the pattern "
                   f"(stop {f(r.get('stop'))}).{back}")
        else:
            res = {"target": "reached its target", "stop": "hit its stop", "timeout": "ran out of time",
                   "no breakout": "ended without a breakout"}.get(r["status"], r["status"])
            msg = f"Result: {head} {res}."
        if send(f"{msg}\n{URL}"):
            sent += 1
    return sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-id", action="store_true", help="print the chat id of whoever messaged the bot")
    ap.add_argument("--test", action="store_true", help="send a test message")
    ap.add_argument("--now", action="store_true", help="send the current model signals and open patterns now")
    ap.add_argument("--cross-now", action="store_true", help="send where EMA 20 and EMA 50 stand now on CROSS_TFS")
    a = ap.parse_args()
    if a.chat_id:
        r = requests.get(f"https://api.telegram.org/bot{os.environ['TELEGRAM_TOKEN']}/getUpdates", timeout=15).json()
        chats = {u["message"]["chat"]["id"]: u["message"]["chat"].get("first_name", "") for u in r.get("result", []) if "message" in u}
        print(chats or "No messages yet: send any message to your bot in Telegram first, then run this again.")
    elif a.test:
        print("sent" if send("Pattern Lens: Telegram alerts are working.") else "failed")
    elif a.now:
        import forward_log as fl
        lines = ["✨ AI model signals now (test score 0.500 = coin flip):"]
        for path in sorted(Path("models").glob("*_*.joblib")):
            if path.stem == "pattern_call":
                continue
            try:
                m = _model_status(path, fl.fetch)
                lines.append(f"{m['sym']} {m['tf']}: {m['signal']}, {100 * m['p']:.0f}% chance of "
                             f"{m['px'] + m['step']:.5g} before {m['px'] - m['step']:.5g} (next {m['span']})")
            except Exception as e:
                lines.append(f"{path.stem}: failed ({type(e).__name__})")
        recs = [r for r in fl.get_store().all() if not r.get("resolved")]
        lines.append(f"\nOpen in the forward log: {len(recs)}")
        for r in sorted(recs, key=lambda r: (r["symbol"], r["timeframe"]))[:15]:
            st = "forming" if r["status"] == "forming" else f"broke {'up' if r['bdir'] > 0 else 'down'}"
            lines.append(f"{r['symbol']} {r['timeframe']} {r['pattern']}: {st}")
        print("sent" if send("\n".join(lines) + f"\n{URL}") else "failed")
    elif a.cross_now:
        import forward_log as fl
        tfs = os.environ.get("CROSS_TFS", "15m").split(",")
        lines = ["EMA 20 (green) vs EMA 50 (red) right now:"]
        for sym in fl.WATCH:
            for tf in tfs:
                try:
                    df = fl.fetch(sym, tf)
                except Exception as e:
                    lines.append(f"{sym} {tf}: fetch failed ({type(e).__name__})")
                    continue
                c = df["close"].tolist()
                e20, e50 = _ema(c, A20), _ema(c, A50)
                px, atr_now = c[-1], float(atr(df).iloc[-1])
                dec = 2 if px >= 100 else 3 if px >= 10 else 4 if px >= 0.1 else 6
                need, up = cross_price(e20[-1], e50[-1]), e20[-1] > e50[-1]
                dist = abs(need - px)
                lines.append(f"{sym} {tf}: EMA 20 {'above' if up else 'below'} EMA 50. "
                             f"A close {'below' if up else 'above'} {need:.{dec}f} crosses ({'SELL' if up else 'BUY'}), "
                             f"{dist / atr_now:.1f} candle moves away.")
        print("sent" if send("\n".join(lines)) else "failed")



def call_alerts(changed):
    """ALERT_CALL=1: a message when the combined next-hour call changes to BUY or SELL."""
    if os.environ.get("ALERT_CALL", "0") != "1":
        return 0
    return sum(send(f"CALL: {r['call']} - {r['symbol']} {URL}") for r in changed if r["call"] != "WAIT")


# ---------- EMA 20 / EMA 50 crossover: early warning and confirmation ----------
A20, A50 = 2 / 21, 2 / 51  # EMA smoothing factors, the same EMAs the page draws
URL = os.environ.get("PUBLIC_URL", "http://tradepattern.duckdns.org/")
EARLY = os.environ.get("ALERT_CROSS_EARLY", "0") == "1"  # "signal soon" warnings before the cross
COOL, APART, LOOK = 6, 0.6, 12  # no signal within 6 candles of a cross; warning needs lines 0.6 ATR apart lately


def _ema(values, a):
    e = values[0]
    out = []
    for v in values:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def cross_price(e20, e50):
    """The close that would make EMA 20 equal EMA 50 on the next candle.
    From e' = a*P + (1 - a)*e for both EMAs, solved for P."""
    return ((1 - A50) * e50 - (1 - A20) * e20) / (A20 - A50)


def cross_signals(fetch):
    """For each watched symbol on CROSS_TFS (default 15m):
      * confirmed: EMA 20 crossed EMA 50 on the last closed candle -> BUY or SELL
      * early warning: the lines are closing in and the next close needed to cross is within one
        average candle move (ATR) of the price now. Sent once per approach."""
    if os.environ.get("ALERT_CROSS", "1") != "1":
        return 0
    import forward_log as fl
    st, sent = _state(), 0
    tfs = os.environ.get("CROSS_TFS", "15m").split(",")
    for sym in [x for x in fl.WATCH if wanted(x)]:
        for tf in tfs:
            try:
                df = fetch(sym, tf)
            except Exception as e:
                print(f"cross: {sym} {tf} fetch failed ({type(e).__name__})")
                continue
            c = df["close"].tolist()
            e20, e50 = _ema(c, A20), _ema(c, A50)
            gap = [a - b for a, b in zip(e20, e50)]
            n = len(c) - 1
            px, a = c[-1], float(atr(df).iloc[-1])
            key = f"cross|{sym}|{tf}"
            last = st.get(key, {})
            if last.get("candle") == df.index[-1].isoformat():
                continue  # this candle was already checked
            rec = {"candle": df.index[-1].isoformat(), "warned": last.get("warned")}
            crosses = [i for i in range(1, n) if gap[i] * gap[i - 1] < 0]  # crosses before this candle
            since = n - crosses[-1] if crosses else 10 ** 6
            label = f"{sym} {tf}" if len(tfs) > 1 else sym
            if gap[n] * gap[n - 1] < 0:  # crossed on the last closed candle
                side = "BUY" if gap[n] > 0 else "SELL"
                if since >= COOL:  # not a quick flip back in a sideways market
                    sent += send(f"{side} SIGNAL - {label} {URL}")
                rec["warned"] = None
            elif since >= COOL:
                need = cross_price(e20[n], e50[n])
                side = "BUY" if gap[n] < 0 else "SELL"
                closing = abs(gap[n]) < abs(gap[n - 3])  # lines getting closer over the last 3 candles
                dist = (need - px) if side == "BUY" else (px - need)  # how far price must still move
                apart = max(abs(g) for g in gap[n - LOOK:n])  # were the lines clearly apart lately?
                moving = (px > e20[n] and e20[n] > e20[n - 1]) if side == "BUY" else (px < e20[n] and e20[n] < e20[n - 1])
                if EARLY and closing and moving and apart >= APART * a and dist <= a and rec["warned"] != side:
                    sent += send(f"{side} SIGNAL SOON - {label}: {URL}")
                    rec["warned"] = side
                elif not closing or dist > 2 * a:
                    rec["warned"] = None  # lines moved apart again: allow a new warning later
            st[key] = rec
    _save(st)
    return sent


if __name__ == "__main__":
    main()
