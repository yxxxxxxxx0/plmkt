"""Phase 1: what data actually exist, and are they fit for this question?

Writes a data-quality report covering both tracks:

  raw    the recorder's event stream, one JSON record per websocket book
         update. This is the source of truth and is opened READ ONLY.
  grid   the 200ms as-of resampling built by ../../jump_data.py, already
         audited by ../../audit_datasets.py.

Checks reported per file: record types, markets and tokens, time range,
inter-update interval distribution, depth levels actually present, crossed or
locked books, duplicated timestamps within a token, stale stretches, and the
share of updates that arrive faster than the grid can represent.

That last number is the one that decides whether a sub-second horizon can be
studied at all. Claiming to evaluate H=1s on a 200ms grid whose median book is
seconds old would be dishonest, so it is measured rather than assumed.

    python inspect_data.py                 # sample every recording
    python inspect_data.py --max-lines 2000000 --sessions 2026-09-13
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from jump_data import KEEP_TYPES, open_recording, session_tag  # noqa: E402

RES = os.path.join(ROOT, "results", "jump_prediction")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_raw():
    """Every recording, from both directories the recorder has written to."""
    out = []
    for d in (os.path.join(ROOT, "data", "live"),
              os.path.join(ROOT, "..", "data", "live")):
        for p in sorted(glob.glob(os.path.join(d, "books_*.jsonl*"))):
            if "smoke" in p or "tradetest" in p or p.endswith(".health.json"):
                continue
            out.append(os.path.abspath(p))
    # de-duplicate by session tag, preferring the larger file
    best = {}
    for p in out:
        t = session_tag(p)
        if t not in best or os.path.getsize(p) > os.path.getsize(best[t]):
            best[t] = p
    return dict(sorted(best.items()))


def inspect_raw(path, max_lines, grid_ms=200):
    """Stream one recording and accumulate quality statistics.

    Deliberately single-pass and allocation-light: the largest file is 47M
    records and this has to run beside other work on a 16 GB machine.
    """
    import orjson

    rec_types = collections.Counter()
    mt_counter = collections.Counter()
    assets, slugs = set(), set()
    key_assets = collections.defaultdict(set)     # the merge check, on raw
    last_ts, dup_ts, back_ts = {}, 0, 0
    gaps = collections.defaultdict(list)
    n_book = n_trade = 0
    crossed = locked = 0
    empty_side = 0
    depth_hist = collections.Counter()            # levels present per side
    ts_min, ts_max = None, None
    lag = []                                      # recv - ts, feed latency
    n = 0

    with open_recording(path) as fh:
        for line in fh:
            n += 1
            if max_lines and n > max_lines:
                break
            try:
                r = orjson.loads(line)
            except Exception:
                rec_types["unparseable"] += 1
                continue
            et = r.get("et") or ("book" if "bids" in r else "other")
            rec_types[et] += 1
            if et == "trade":
                n_trade += 1
                continue
            if "bids" not in r:
                continue

            mt = r.get("market_type")
            mt_counter[mt] += 1
            if mt not in KEEP_TYPES:
                continue
            aid, slug, ts = r.get("asset_id"), r.get("slug"), r.get("ts")
            if aid is None or ts is None:
                continue
            n_book += 1
            assets.add(aid)
            slugs.add(slug)
            key_assets[(slug, mt, r.get("line"))].add(aid)
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)
            if r.get("recv") and len(lag) < 200_000:
                lag.append(r["recv"] - ts)

            bids, asks = r.get("bids") or [], r.get("asks") or []
            depth_hist[("bid", min(len(bids), 10))] += 1
            depth_hist[("ask", min(len(asks), 10))] += 1
            if not bids or not asks:
                empty_side += 1
            else:
                if asks[0][0] < bids[0][0]:
                    crossed += 1
                elif asks[0][0] == bids[0][0]:
                    locked += 1

            p = last_ts.get(aid)
            if p is not None:
                if ts == p:
                    dup_ts += 1
                elif ts < p:
                    back_ts += 1
                elif len(gaps[aid]) < 400_000:
                    gaps[aid].append(ts - p)
            last_ts[aid] = ts

    g = np.concatenate([np.asarray(v, np.int64) for v in gaps.values()
                        if len(v) > 50]) if gaps else np.array([0])
    multi = {k: v for k, v in key_assets.items() if len(v) > 1}
    tot_levels = sum(depth_hist.values()) or 1
    full10 = sum(v for (s, l), v in depth_hist.items() if l >= 10) / tot_levels

    return dict(
        file=os.path.relpath(path, ROOT), size_bytes=os.path.getsize(path),
        lines_scanned=n - 1, truncated=bool(max_lines and n > max_lines),
        record_types=dict(rec_types), market_types=dict(mt_counter),
        book_records=n_book, trade_prints=n_trade,
        n_tokens=len(assets), n_markets=len(slugs),
        ts_min=ts_min, ts_max=ts_max,
        span_hours=(ts_max - ts_min) / 3.6e6 if ts_min else 0.0,
        obs_per_token=n_book / max(len(assets), 1),
        interval_ms={f"p{q}": float(np.percentile(g, q))
                     for q in (1, 5, 25, 50, 75, 90, 99)},
        interval_mean_ms=float(g.mean()),
        frac_faster_than_grid=float((g < grid_ms).mean()),
        frac_slower_than_1s=float((g > 1000).mean()),
        frac_stale_over_60s=float((g > 60_000).mean()),
        crossed_books=crossed, locked_books=locked, empty_side=empty_side,
        duplicate_ts=dup_ts, backwards_ts=back_ts,
        frac_full_10_levels=float(full10),
        feed_lag_ms_p50=float(np.percentile(lag, 50)) if lag else None,
        feed_lag_ms_p99=float(np.percentile(lag, 99)) if lag else None,
        nonunique_legacy_keys=len(multi),
        nonunique_key_types=dict(collections.Counter(k[1] for k in multi)),
    )


def inspect_grid(grid_dir):
    """Row counts and coverage of the already-built 200ms datasets."""
    import pyarrow.parquet as pq
    out = []
    for p in sorted(glob.glob(os.path.join(grid_dir, "feat_books_*.parquet"))):
        if "_trimmed" in p:
            continue
        name = os.path.basename(p)[len("feat_"):-len(".parquet")]
        f = pq.ParquetFile(p)
        n = f.metadata.num_rows
        tp = os.path.join(grid_dir, f"lob_{name}.npy")
        trim = os.path.join(grid_dir, f"feat_{name}_trimmed.parquet")
        n_trim = (pq.ParquetFile(trim).metadata.num_rows
                  if os.path.exists(trim) else None)
        # series identity from the trimmed file if present (smaller read)
        src = trim if os.path.exists(trim) else p
        ser = pq.read_table(src, columns=["series"]).column("series")
        ser = ser.combine_chunks().dictionary_encode().to_pandas()
        uniq = ser.cat.categories if hasattr(ser, "cat") else ser.unique()
        uniq = [str(x) for x in uniq]
        out.append(dict(
            session=name, rows=n, rows_in_game=n_trim,
            tensor_bytes=os.path.getsize(tp) if os.path.exists(tp) else None,
            n_series=len(uniq),
            n_markets=len({u.split("|")[0] for u in uniq}),
            market_types=dict(collections.Counter(u.split("|")[1] for u in uniq)),
            hours_of_grid=n * 0.2 / 3600.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=3_000_000,
                    help="records scanned per recording; 0 scans all (slow)")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--skip-raw", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    report = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                  max_lines_per_file=a.max_lines, raw=[], grid=[])

    raw = find_raw()
    if a.sessions:
        raw = {k: v for k, v in raw.items()
               if any(s in k for s in a.sessions)}
    log(f"{len(raw)} recording(s) found")
    if not a.skip_raw:
        for tag, p in raw.items():
            log(f"scanning {tag} ({os.path.getsize(p)/1e6:.0f} MB xz)...")
            report["raw"].append(inspect_raw(p, a.max_lines))
            r = report["raw"][-1]
            log(f"  {r['book_records']:,} books, {r['n_tokens']} tokens, "
                f"{r['n_markets']} markets, median gap "
                f"{r['interval_ms']['p50']:.0f}ms, "
                f"{r['frac_faster_than_grid']:.1%} faster than the grid")

    log("inspecting the built 200ms datasets...")
    report["grid"] = inspect_grid(os.path.join(ROOT, "data", "jump"))

    jp = os.path.join(RES, "data_quality_report.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    write_markdown(report, os.path.join(RES, "data_quality_report.md"))
    log(f"wrote {jp} and data_quality_report.md")


def write_markdown(rep, path):
    L = ["# Data quality report -- LOB jump prediction",
         "", f"Generated {rep['generated']}.",
         f"Raw recordings sampled at up to "
         f"{rep['max_lines_per_file']:,} records each.", ""]

    if rep["raw"]:
        L += ["## Raw event stream (source of truth, opened read-only)", "",
              "| session | books | trades | tokens | markets | span h | "
              "obs/token |",
              "|---|---|---|---|---|---|---|"]
        for r in rep["raw"]:
            tag = os.path.basename(r["file"])
            L.append(f"| {tag} | {r['book_records']:,} | {r['trade_prints']:,} "
                     f"| {r['n_tokens']} | {r['n_markets']} | "
                     f"{r['span_hours']:.1f} | {r['obs_per_token']:,.0f} |")
        L += ["", "### Timestamp resolution", "",
              "Interval between consecutive updates **of the same token**.", "",
              "| session | p1 | p25 | p50 | p75 | p90 | p99 | mean | "
              "< 200ms | > 60s |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rep["raw"]:
            i = r["interval_ms"]
            L.append(f"| {os.path.basename(r['file'])} | {i['p1']:.0f} | "
                     f"{i['p25']:.0f} | **{i['p50']:.0f}** | {i['p75']:.0f} | "
                     f"{i['p90']:.0f} | {i['p99']:,.0f} | "
                     f"{r['interval_mean_ms']:,.0f} | "
                     f"**{r['frac_faster_than_grid']:.1%}** | "
                     f"{r['frac_stale_over_60s']:.2%} |")
        L += ["", "All values in milliseconds. The **< 200ms** column is the "
              "share of updates that arrive faster than one grid slot: those "
              "events are collapsed by the 200ms resampling and are invisible "
              "to any model trained on it.", "",
              "### Integrity", "",
              "| session | crossed | locked | empty side | dup ts | "
              "backwards ts | full 10 levels | feed lag p50 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rep["raw"]:
            L.append(f"| {os.path.basename(r['file'])} | {r['crossed_books']:,} "
                     f"| {r['locked_books']:,} | {r['empty_side']:,} | "
                     f"{r['duplicate_ts']:,} | {r['backwards_ts']:,} | "
                     f"{r['frac_full_10_levels']:.1%} | "
                     f"{r['feed_lag_ms_p50']:.0f}ms |")
        bad = sum(r["nonunique_legacy_keys"] for r in rep["raw"])
        L += ["", f"Legacy-key collision check: **{bad}** "
              "(slug, market_type, line) keys map to more than one token "
              "across the sampled records -- the defect documented in "
              "`FINDINGS_SERIES_KEY.md`. The current pipeline keys on "
              "`asset_id` as well, so these no longer merge; the count is "
              "reported to show the collision is real in the raw feed.", ""]

    if rep["grid"]:
        L += ["## Built 200ms grid datasets", "",
              "Backward-only as-of resampling: each slot carries the most "
              "recent book at or before it. Silences beyond 60s are left as "
              "real gaps rather than filled.", "",
              "| session | grid rows | in-game rows | series | markets | "
              "grid hours |", "|---|---|---|---|---|---|"]
        for g in rep["grid"]:
            ig = f"{g['rows_in_game']:,}" if g["rows_in_game"] else "-"
            L.append(f"| {g['session']} | {g['rows']:,} | {ig} | "
                     f"{g['n_series']} | {g['n_markets']} | "
                     f"{g['hours_of_grid']:,.0f} |")
        tot = sum(g["rows"] for g in rep["grid"])
        tin = sum(g["rows_in_game"] or 0 for g in rep["grid"])
        L += ["", f"Total {tot:,} grid rows, {tin:,} in-game.", ""]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
