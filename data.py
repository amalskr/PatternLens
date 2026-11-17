"""Download Binance USDT-M futures klines from Binance Data Vision (free) into one Parquet file.

Usage: python data.py --symbol ADAUSDT --interval 1h --start 2021-01
"""
import argparse
import io
import zipfile
from datetime import date

import pandas as pd
import requests

from features import klines_to_df

URL = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/{i}/{s}-{i}-{y}-{m:02d}.zip"


def month_range(start: str):
    y, m = map(int, start.split("-"))
    today = date.today()
    while (y, m) < (today.year, today.month):  # completed months only
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ADAUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2021-01", help="YYYY-MM")
    a = ap.parse_args()
    s, i = a.symbol.upper(), a.interval

    frames = []
    for y, m in month_range(a.start):
        r = requests.get(URL.format(s=s, i=i, y=y, m=m), timeout=60)
        if r.status_code == 404:
            print(f"{y}-{m:02d}: not available, skipped")
            continue
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
        raw = raw[raw[0].astype(str) != "open_time"]  # newer files include a header row
        frames.append(klines_to_df(raw.values.tolist()))
        print(f"{y}-{m:02d}: {len(frames[-1])} candles")

    if not frames:
        raise SystemExit("No data downloaded. Check the symbol, interval and start month.")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    out = f"data/{s}_{i}.parquet"
    df.to_parquet(out)
    print(f"Saved {len(df)} candles to {out} ({df.index[0]} to {df.index[-1]})")


if __name__ == "__main__":
    main()
