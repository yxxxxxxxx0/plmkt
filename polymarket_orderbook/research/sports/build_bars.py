"""Build multi-resolution bars from the polymarket_sports tick data.

Source: ../../../polymarket_sports/orderbook/orderbook_*.parquet (read-only).
See polymarket_sports/reports/DATA_AUDIT.md for why only ~12% of that data is
usable -- the median token there is a dead longshot with a 1-cent daily range.

What this keeps
---------------
The LIVE subset only: per day, tokens with >= --min-updates updates, a daily mid
range >= --min-range, median spread <= --max-spread and a median mid inside
[0.05, 0.95]. Those are the in-play esports/match markets; everything else is a
season future that never moves.

Features
--------
This feed carries best_bid/best_ask but NO bid/ask SIZES and no depth levels,
so the MLB feature set (10-level depth, near-touch dollars, book slope) cannot
be rebuilt and is not faked. What it does carry, and the MLB data did not, is
the raw delta stream -- so genuine order-flow features are available here for
the first time:

    n_updates        deltas in the bar (update intensity)
    n_buy / n_sell   deltas per side
    flow_imb         (n_buy - n_sell) / (n_buy + n_sell)
    size_buy/sell    summed |change_size| per side
    size_imb         dollar-weighted version of the same
    near_frac        share of deltas landing within 2 ticks of the mid

Bars are strictly backward-looking: the value stamped on a bar is the last
observation at or before its close, and every flow feature aggregates only
deltas inside the bar. Bars with no observation are left missing; the series is
broken there rather than filled.

    python build_bars.py --resolutions 1 5 15 60 300
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPORTS = os.path.abspath(os.path.join(ROOT, "..", "polymarket_sports"))
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "sports")
COLS = ["timestamp_created_at", "token_id", "market_id", "best_bid", "best_ask",
        "mid_price", "spread", "change_price", "change_size", "change_side"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def live_tokens(d, min_updates, min_range, max_spread):
    g = d.groupby("token_id", sort=False)
    a = pd.DataFrame({"n": g.size(),
                      "rng": g.mid_price.max() - g.mid_price.min(),
                      "sp": g.spread.median(),
                      "mid": g.mid_price.median()})
    keep = a[(a.n >= min_updates) & (a.rng >= min_range)
             & (a.sp <= max_spread) & (a.mid >= 0.05) & (a.mid <= 0.95)]
    return set(keep.index)


def bars_for(d, res_s):
    """Aggregate the delta stream into bars of res_s seconds."""
    step = res_s * 1000
    d = d.copy()
    d["bar"] = d.timestamp_created_at // step
    d["is_buy"] = (d.change_side == "BUY").astype(np.int8)
    d["asz"] = d.change_size.abs()
    d["near"] = (np.abs(d.change_price - d.mid_price) <= 0.02).astype(np.int8)
    # precomputed so the aggregation stays vectorised; a per-group lambda here
    # made the build ~50s/day
    d["asz_buy"] = d.asz * d.is_buy

    g = d.groupby(["token_id", "bar"], sort=True, observed=True)
    B = g.agg(
        ts=("timestamp_created_at", "last"),
        mid=("mid_price", "last"),
        best_bid=("best_bid", "last"),
        best_ask=("best_ask", "last"),
        spread=("spread", "last"),
        n_updates=("is_buy", "size"),
        n_buy=("is_buy", "sum"),
        size_buy=("asz_buy", "sum"),
        size_all=("asz", "sum"),
        near_cnt=("near", "sum"),
        market_id=("market_id", "last"),
    ).reset_index()
    B["n_sell"] = B.n_updates - B.n_buy
    B["size_sell"] = B.size_all - B.size_buy
    tot = B.n_updates.replace(0, np.nan)
    B["flow_imb"] = (B.n_buy - B.n_sell) / tot
    sz = (B.size_buy + B.size_sell).replace(0, np.nan)
    B["size_imb"] = (B.size_buy - B.size_sell) / sz
    B["near_frac"] = B.near_cnt / tot
    return B.drop(columns=["near_cnt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=int, nargs="*", default=[1, 5, 15, 60, 300])
    ap.add_argument("--min-updates", type=int, default=3000)
    ap.add_argument("--min-range", type=float, default=0.10)
    ap.add_argument("--max-spread", type=float, default=0.05)
    ap.add_argument("--max-days", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    files = sorted(glob.glob(os.path.join(SPORTS, "orderbook", "orderbook_*.parquet")))
    if a.max_days:
        files = files[:a.max_days]
    out = {r: [] for r in a.resolutions}
    stats = []
    for fp in files:
        date = os.path.basename(fp)[10:20]
        d = pq.read_table(fp, columns=COLS).to_pandas()
        toks = live_tokens(d, a.min_updates, a.min_range, a.max_spread)
        d = d[d.token_id.isin(toks)]
        if not len(d):
            log("%s: no live tokens" % date)
            continue
        d = d.sort_values(["token_id", "timestamp_created_at"], kind="stable")
        row = dict(date=date, live_tokens=len(toks), live_rows=len(d))
        for r in a.resolutions:
            B = bars_for(d, r)
            B["date"] = date
            out[r].append(B)
            row["bars_%ds" % r] = len(B)
        stats.append(row)
        log("%s: %d live tokens, %s rows -> %s"
            % (date, len(toks), "{:,}".format(len(d)),
               ", ".join("%ds:%s" % (r, "{:,}".format(row["bars_%ds" % r]))
                         for r in a.resolutions)))
        del d

    S = pd.DataFrame(stats)
    S.to_csv(os.path.join(RES, "bar_build_stats.csv"), index=False)
    for r in a.resolutions:
        B = pd.concat(out[r], ignore_index=True)
        B["ts_dt"] = pd.to_datetime(B.ts, unit="ms")
        f = os.path.join(CACHE, "bars_%ds.parquet" % r)
        B.to_parquet(f, index=False)
        log("resolution %3ds -> {:,} bars, {:,} tokens  ->  %s"
            .format(len(B), B.token_id.nunique()) % (r, f))
    print()
    print(S.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
