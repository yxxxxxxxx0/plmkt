"""Step 1: what price data actually exists, and which of it suits a jump test.

Reads the recorder's event-level top-of-book CSVs (the raw source, opened
read-only) and reports the facts the brief asks for: schema, row counts,
timestamp column and resolution, date range, assets, missing values, duplicate
timestamps, crossed/locked books, and per-asset lifespan.

Lifespan is the fact that decides the experiment. Lee-Mykland with K=600
minute observations needs 10 hours of trailing history before it can emit its
first statistic. These are per-game prediction-market contracts that are
created, traded and resolved within a day, so whether any series is long
enough is an empirical question, not an assumption.

    python inspect_data.py
"""
from __future__ import annotations

import glob
import json
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
RES = os.path.join(ROOT, "results", "lee_mykland")

COLS = ["ts_exchange_ms", "ts_recv_ms", "event_slug", "market_type", "line",
        "outcome", "asset_id", "best_bid", "best_ask", "midpoint"]
HKT = 8 * 3600 * 1000


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def read(path):
    t = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            include_columns=COLS,
            column_types={"asset_id": pa.string()}))
    return t.to_pandas()


def main():
    os.makedirs(RES, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(LIVE, "top_of_book_2026-*.csv")))
    # the smoke/tradetest files are not slates; exclude them explicitly
    paths = [p for p in paths if "smoke" not in p and "tradetest" not in p]
    if not paths:
        raise SystemExit("no top_of_book_*.csv found in %s" % LIVE)

    report = {"files": [], "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    spans_all = []
    for p in paths:
        log("reading %s (%.1f GB)" % (os.path.basename(p),
                                      os.path.getsize(p) / 1e9))
        d = read(p)
        ts = d.ts_exchange_ms.to_numpy(np.int64)
        # series identity: an asset_id IS the tradable token. slug/type/line
        # alone collide across the two legs of a spread (FINDINGS_SERIES_KEY.md)
        key = d.asset_id.astype(str)
        bid, ask = d.best_bid.to_numpy(), d.best_ask.to_numpy()
        both = np.isfinite(bid) & np.isfinite(ask)

        g = pd.DataFrame({"a": key, "ts": ts}).groupby("a").ts
        span_min = ((g.max() - g.min()) / 60000.0)
        spans_all.append(span_min)

        dt = pd.Series(ts).groupby(key.to_numpy()).diff().dropna()
        dt = dt[dt >= 0]

        r = dict(
            file=os.path.basename(p),
            size_gb=round(os.path.getsize(p) / 1e9, 2),
            rows=int(len(d)),
            assets=int(d.asset_id.nunique()),
            markets=int(d.event_slug.nunique()),
            market_types=sorted(d.market_type.dropna().unique().tolist()),
            outcomes=sorted(d.outcome.dropna().unique().tolist()),
            ts_min_hkt=str(pd.to_datetime(ts.min() + HKT, unit="ms")),
            ts_max_hkt=str(pd.to_datetime(ts.max() + HKT, unit="ms")),
            span_hours=round((ts.max() - ts.min()) / 3.6e6, 2),
            # timestamp resolution: gap between consecutive updates of the SAME asset
            dt_ms_p25=float(dt.quantile(.25)), dt_ms_median=float(dt.median()),
            dt_ms_p75=float(dt.quantile(.75)), dt_ms_p99=float(dt.quantile(.99)),
            frac_updates_under_1s=float((dt < 1000).mean()),
            missing_bid=int((~np.isfinite(bid)).sum()),
            missing_ask=int((~np.isfinite(ask)).sum()),
            missing_either=int((~both).sum()),
            frac_missing_either=float((~both).mean()),
            zero_or_neg_mid=int((both & (((bid + ask) / 2) <= 0)).sum()),
            crossed_bid_gt_ask=int((both & (bid > ask)).sum()),
            locked_bid_eq_ask=int((both & (bid == ask)).sum()),
            dup_ts_within_asset=int(
                pd.DataFrame({"a": key, "ts": ts}).duplicated().sum()),
            backwards_ts_within_asset=int((dt < 0).sum()) if len(dt) else 0,
            asset_span_min_median=float(span_min.median()),
            asset_span_min_max=float(span_min.max()),
            assets_span_over_600min=int((span_min > 600).sum()),
            assets_span_over_300min=int((span_min > 300).sum()),
        )
        # does the file's own midpoint column equal (bid+ask)/2 ?
        m = d.midpoint.to_numpy()
        ok = both & np.isfinite(m)
        r["midpoint_equals_bid_ask_mean"] = float(
            np.mean(np.abs(m[ok] - (bid[ok] + ask[ok]) / 2) < 1e-9))
        report["files"].append(r)
        log("  rows %s | assets %d | span %.1fh | median update gap %.0f ms"
            % ("{:,}".format(len(d)), r["assets"], r["span_hours"],
               r["dt_ms_median"]))
        del d

    S = pd.concat(spans_all)
    report["totals"] = dict(
        rows=int(sum(f["rows"] for f in report["files"])),
        assets=int(len(S)),
        assets_span_over_600min=int((S > 600).sum()),
        assets_span_over_300min=int((S > 300).sum()),
        span_min_median=float(S.median()),
    )
    f = os.path.join(RES, "data_inspection.json")
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== per file ===")
    print(pd.DataFrame(report["files"])[
        ["file", "rows", "assets", "markets", "span_hours", "dt_ms_median",
         "frac_missing_either", "crossed_bid_gt_ask", "dup_ts_within_asset",
         "asset_span_min_median", "assets_span_over_600min",
         "midpoint_equals_bid_ask_mean"]
    ].to_string(index=False))
    print("\n=== totals ===")
    print(json.dumps(report["totals"], indent=2))
    print("\nwrote", f)


if __name__ == "__main__":
    sys.exit(main())
