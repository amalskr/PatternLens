# Pre-registration: three-candle reversal replication

Registered: 17 November 2026. The confirmatory symbols listed below had not been downloaded or analysed
when this file was committed.

## Origin

Exploratory analysis of ADAUSDT perpetual futures, 1h candles, 9 January 2021 to 31 October 2026
(50,915 candles, horizon 5, Holm-corrected over 30 pattern and trend combinations) found that the only
significant results ran opposite to the textbook reading of the pattern:

* Three black crows (all trends): down rate 44.8% vs base 51.2%, n = 1228, p_holm = 0.0002
* Three black crows (downtrend): down rate 43.6% vs base 50.5%, n = 755, p_holm = 0.0044
* Three white soldiers (downtrend): up rate 42.6% vs base 49.5%, n = 516, p_holm = 0.0494

These were found after testing 30 combinations, so ADAUSDT is treated as the discovery sample only.
It is excluded from all confirmatory tests below.

## Samples

* **Discovery:** ADAUSDT.
* **Exploratory second sample:** BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT, DOGEUSDT. `pattern_stats.py`
  was run on these before this file was written, so they cannot serve as a confirmatory test. Their
  output files are committed unchanged in `results/` together with this file, and they will be reported
  as exploratory only.
* **Confirmatory:** AVAXUSDT, LINKUSDT, LTCUSDT, DOTUSDT, TRXUSDT, ATOMUSDT. Not downloaded or analysed
  at registration.

## Hypotheses

* **H1.** After a three black crows signal (any trend), the share of cases where the close 5 candles
  later is above the signal close is greater than the base up rate over all candles.
* **H2.** After a three white soldiers signal in a downtrend (EMA 20 below EMA 50), the share of cases
  where the close 5 candles later is above the signal close is lower than the base up rate over all
  downtrend candles.

## Data (fixed)

* Confirmatory symbols: AVAXUSDT, LINKUSDT, LTCUSDT, DOTUSDT, TRXUSDT, ATOMUSDT (USDT-M perpetual futures)
* Interval: 1h
* Period: January 2021 (or first available month) to the last completed month at download time
* Source: Binance Data Vision monthly klines, via `data.py` as committed with this file

## Method (fixed)

* Pattern definitions, trend definition, warm-up (200 candles) and horizon (5 candles) exactly as in
  `features.py` and `pattern_stats.py` as committed with this file. No changes to definitions or
  thresholds after this commit.
* Per symbol: one-sided exact binomial test of the signal win count against the regime base rate,
  in the direction stated by the hypothesis.
* Pooled: sum signal counts and directional wins across the six symbols; one-sided exact binomial test
  against the signal-weighted average of the per-symbol base rates.

## Decision rule (fixed)

A hypothesis is counted as **replicated** only if both hold:

1. The effect is in the predicted direction in at least 4 of the 6 symbols, and
2. The pooled one-sided p-value is below 0.01.

Otherwise it is reported as **not replicated**. Both outcomes will be reported in full.

## Secondary analyses (exploratory, labelled as such when reported)

* The exploratory second sample (BTC, ETH, SOL, XRP, BNB, DOGE)
* Horizons 3 and 10 on the six confirmatory symbols
* Average move after a 0.1% round-trip fee for a contrarian trade (long after three crows,
  short after three soldiers in downtrend)
* Removing overlapping signals (keep only the first signal in any 5-candle window)
* Other intervals (15m, 4h)

## Known limitations stated in advance

* Overlapping signals make per-signal p-values somewhat optimistic.
* Crypto returns are correlated across symbols, so the six replications are not fully independent.
* Fees and funding vary by account tier; 0.1% round trip is an assumption.
