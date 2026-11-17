# Pattern Lens

Candlestick pattern scanner with a walk-forward tested model. Runs on your own computer, costs nothing.

## Setup (once)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, if LightGBM fails to import: `brew install libomp`.

## 1. Download history (free, Binance Data Vision)

```bash
python data.py --symbol ADAUSDT --interval 1h --start 2021-01
```

## 2. Train and test

```bash
python train.py --symbol ADAUSDT --interval 1h
```

Read the fold table. The model has an edge only if BUY precision beats "base up" and
SELL precision beats 1 minus "base up", in most folds, not just on average.
Try other `--horizon`, `--k`, `--up`, `--dn` values here.

When you are done experimenting, run once with `--final` to score the last 6 months,
which the model has never seen. That is the number to report in a paper.

## 3. Run the model server

```bash
uvicorn serve:app --port 8000
```

Open `pattern-lens.html` in your browser. The "Model signal" panel connects to the server
automatically. Train one model per symbol and timeframe you want to see.

Not financial advice. A model that beats a coin flip in a backtest can still lose money live.
