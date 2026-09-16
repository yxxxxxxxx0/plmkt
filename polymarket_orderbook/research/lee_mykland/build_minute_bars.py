"""Step 2: build a 1-minute mid-price series from the event-level book feed.

Source is the recorder's `top_of_book_*.csv` (opened read-only, never written).
Those are event-level: one row per top-of-book change, median gap 27 ms.

Price definition
----------------
    mid = (best_bid + best_ask) / 2

computed here from the raw bid/ask columns. The feed also carries its own
`midpoint` column; inspect_data.py verified it equals (bid+ask)/2 in 100.0% of
rows, so the two agree -- this script recomputes it anyway rather than trusting
a derived column.

Sampling rule (strictly backward-looking)
-----------------------------------------
The value stamped on minute M is the mid of the LAST valid event whose exchange
timestamp falls at or before the END of minute M. No observation after the
minute close is ever used, so the series is causal by construction.

An event is valid only if best_bid and best_ask are both present, the book is
not crossed (bid <= ask) and the mid lies strictly inside (0, 1) -- these are
probability contracts.

Minutes in which the asset produced no valid event carry the previous quote
forward, because a resting quote genuinely remains the prevailing price. That
carry-forward is capped at `--max-stale-min` (default 5). Past the cap the bar
is left MISSING and the series is broken there, so no return is ever computed
across a long silence. Every bar records `staleness_s` and `n_events`, and the
counts of carried and dropped bars are reported rather than hidden.

Series identity is `asset_id`. That is the tradable token; slug/market_type/line
collide across the two legs of a spread, which is the defect documented in
FINDINGS_SERIES_KEY.md.

    python build_minute_bars.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LIVE = os.path.join(ROOT, "data", "live")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "lee_mykland")
COLS = ["ts_exchange_ms", "event_slug", "market_type", "line", "outcome",
        "asset_id", "best_bid", "best_ask"]
MIN_MS = 60_000


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def build_one(path, max_stale_min):
    d = pacsv.read_csv(
        path, read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            include_columns=COLS,
            # asset_id is a 77-digit integer. Left to type inference pyarrow
            # reads it as float64, which is lossy and could silently merge two
            # tokens into one price series -- the defect in FINDINGS_SERIES_KEY.md.
            column_types={"asset_id": pa.string()})).to_pandas()
    n_raw = len(d)

    bid = d.best_bid.to_numpy(np.float64)
    ask = d.best_ask.to_numpy(np.float64)
    mid = (bid + ask) / 2.0
    ok = (np.isfinite(bid) & np.isfinite(ask) & (bid <= ask)
          & (mid > 0.0) & (mid < 1.0))
    stats = dict(file=os.path.basename(path), raw_rows=int(n_raw),
                 dropped_missing=int((~(np.isfinite(bid) & np.isfinite(ask))).sum()),
                 dropped_crossed=int((np.isfinite(bid) & np.isfinite(ask)
                                      & (bid > ask)).sum()),
                 dropped_out_of_range=int((np.isfinite(bid) & np.isfinite(ask)
                                           & (bid <= ask)
                                           & ((mid <= 0) | (mid >= 1))).sum()))
    d = d[ok].copy()
    d["mid"] = mid[ok]
    stats["valid_events"] = int(len(d))

    d["minute"] = d.ts_exchange_ms.to_numpy(np.int64) // MIN_MS
    d = d.sort_values(["asset_id", "ts_exchange_ms"], kind="stable")

    # last valid event at or before the end of each minute
    g = d.groupby(["asset_id", "minute"], sort=False, observed=True)
    bars = g.agg(mid=("mid", "last"), ts_last=("ts_exchange_ms", "last"),
                 n_events=("mid", "size")).reset_index()
    meta = d.groupby("asset_id", sort=False, observed=True).agg(
        event_slug=("event_slug", "first"), market_type=("market_type", "first"),
        line=("line", "first"), outcome=("outcome", "first")).reset_index()

    # reindex each asset onto a dense minute grid, carry forward under a cap
    out = []
    n_carried = n_dropped = 0
    for a, gg in bars.groupby("asset_id", sort=False, observed=True):
        gg = gg.sort_values("minute")
        lo, hi = int(gg.minute.iloc[0]), int(gg.minute.iloc[-1])
        full = pd.DataFrame({"minute": np.arange(lo, hi + 1, dtype=np.int64)})
        full = full.merge(gg, on="minute", how="left")
        full["asset_id"] = a
        observed = full.mid.notna().to_numpy()
        # minutes since the last observed quote (0 on an observed minute)
        idx = np.arange(len(full))
        last_obs = np.maximum.accumulate(np.where(observed, idx, -1))
        age_min = idx - last_obs
        full["mid"] = full.mid.ffill()
        full["ts_last"] = full.ts_last.ffill()
        full["n_events"] = full.n_events.fillna(0).astype(np.int32)
        too_stale = (age_min > max_stale_min) | (last_obs < 0)
        n_carried += int((~observed & ~too_stale).sum())
        n_dropped += int(too_stale.sum())
        full.loc[too_stale, "mid"] = np.nan
        full["carry_min"] = age_min
        out.append(full)
    B = pd.concat(out, ignore_index=True)
    B = B.merge(meta, on="asset_id", how="left")
    B["session"] = os.path.basename(path).replace("top_of_book_", "").replace(".csv", "")
    # staleness of the quote actually used, in seconds
    B["staleness_s"] = ((B.minute + 1) * MIN_MS - B.ts_last) / 1000.0
    stats.update(minute_bars=int(len(B)),
                 bars_observed=int((B.n_events > 0).sum()),
                 bars_carried_forward=int(n_carried),
                 bars_dropped_too_stale=int(n_dropped),
                 assets=int(B.asset_id.nunique()))
    return B, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stale-min", type=int, default=5,
                    help="longest silence carried forward; beyond this the bar "
                         "is missing and the series is broken there")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(LIVE, "top_of_book_2026-*.csv")))
    paths = [p for p in paths if "smoke" not in p and "tradetest" not in p]
    frames, stats = [], []
    for p in paths:
        log("building %s" % os.path.basename(p))
        B, s = build_one(p, a.max_stale_min)
        frames.append(B)
        stats.append(s)
        log("  %s valid events -> %s minute bars (%s observed, %s carried, "
            "%s dropped stale)"
            % ("{:,}".format(s["valid_events"]), "{:,}".format(s["minute_bars"]),
               "{:,}".format(s["bars_observed"]),
               "{:,}".format(s["bars_carried_forward"]),
               "{:,}".format(s["bars_dropped_too_stale"])))
        del B
    B = pd.concat(frames, ignore_index=True)
    # moneyline markets have line = null; without fillna the whole concatenated
    # label becomes NaN and the asset loses its human-readable name.
    B["label"] = (B.event_slug.fillna("?").astype(str) + " | "
                  + B.market_type.fillna("?").astype(str) + " | "
                  + B.line.fillna("-").astype(str) + " | "
                  + B.outcome.fillna("?").astype(str))
    f = os.path.join(CACHE, "minute_bars.parquet")
    B.to_parquet(f, index=False)
    S = pd.DataFrame(stats)
    S.to_csv(os.path.join(RES, "minute_bar_build_stats.csv"), index=False)
    print()
    print(S.to_string(index=False))
    valid = B.mid.notna()
    per = B[valid].groupby("asset_id").size()
    print("\ntotal minute bars %s | non-missing %s | assets %d"
          % ("{:,}".format(len(B)), "{:,}".format(int(valid.sum())),
             B.asset_id.nunique()))
    print("valid minutes per asset: median %.0f  max %.0f" % (per.median(), per.max()))
    for k in (120, 300, 600, 601):
        print("  assets with >= %d valid minutes: %d" % (k, int((per >= k).sum())))
    print("\nwrote", f)


if __name__ == "__main__":
    sys.exit(main())
