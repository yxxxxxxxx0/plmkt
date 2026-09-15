"""Per-match data quality: which games are trustworthy and which are not.

verify_recording.py judges a whole recording. That is too coarse to act on,
because a slate is 15 independent websockets and one dead socket does not
show up in a slate-wide average -- it just quietly poisons whichever games it
covered. Every model result so far pooled all matches together, so a match
with broken coverage contributed to the numbers exactly as if it were sound.

What is measured, per match slug:

  coverage      wall-clock span actually covered by updates, versus the span
                between the first and last record. A match recorded for 3 of
                its 9 innings looks complete unless you check.
  worst gap     the longest silence on any of its assets. With the recorder
                fix (conn/heartbeat records) a gap can be attributed; on older
                recordings it cannot, and that is reported honestly rather
                than guessed.
  re-seeds      on reconnect only the current book is recovered, so every
                price_change during the outage is gone. More than one seed per
                asset means known-missing data.
  lag spread    p90-p50 of (recv - ts), skew-invariant. A match whose socket
                was starved shows a wide spread even when the slate looks fine.
  trade prints  whether fills can be told from cancels for this match.
  ts regress    timestamps going backwards within an asset.

The verdict is deliberately conservative: a match is GOOD only if nothing is
known to be missing. Anything with a re-seed is at best SUSPECT, because the
loss is certain even when its size is not.

    python match_quality.py data/live/books_2026-09-10.jsonl.xz
    python match_quality.py data/live/*.jsonl --json data/jump/match_quality.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import orjson


def opener(path):
    if path.endswith(".xz"):
        import lzma
        return lzma.open(path, "rb")
    return open(path, "rb")


def scan(path, lag_sample=50):
    per = defaultdict(lambda: dict(
        n=0, first=None, last=None, assets=set(), seeds=defaultdict(int),
        trades=0, regress=0, lags=[], gaps=defaultdict(int),
        prev_ts=dict(), conn=0, outage_s=0.0, heartbeats=0, books=0))
    n = 0
    with opener(path) as fh:
        for raw in fh:
            n += 1
            try:
                r = orjson.loads(raw)
            except Exception:
                continue
            slug = r.get("slug")
            if not slug:
                continue
            d = per[slug]
            et = r.get("et")
            ts = r.get("ts")

            # liveness records added by the recorder fix
            if et == "heartbeat":
                d["heartbeats"] += 1
                continue
            if et == "conn":
                d["conn"] += 1
                d["outage_s"] += float(r.get("outage_s") or 0.0)
                continue

            d["n"] += 1
            if ts is None:
                continue
            d["first"] = ts if d["first"] is None else min(d["first"], ts)
            d["last"] = ts if d["last"] is None else max(d["last"], ts)
            if et == "trade":
                d["trades"] += 1
            if et == "seed":
                d["seeds"][r.get("asset_id")] += 1
            if et == "book":
                d["books"] += 1
            aid = r.get("asset_id")
            if aid:
                d["assets"].add(aid)
                p = d["prev_ts"].get(aid)
                if p is not None:
                    if ts < p:
                        d["regress"] += 1
                    else:
                        g = ts - p
                        if g > d["gaps"][aid]:
                            d["gaps"][aid] = g
                d["prev_ts"][aid] = ts
            recv = r.get("recv")
            if recv is not None and d["n"] % lag_sample == 0:
                d["lags"].append(recv - ts)
    return per, n


def judge(m):
    """GOOD / SUSPECT / BROKEN, with the reason."""
    reasons = []
    if m["reseeds"] > 0:
        reasons.append(f"{m['reseeds']} asset(s) re-seeded: price_change "
                       f"messages during those outages are unrecoverable")
    if m["outage_s"] > 60:
        reasons.append(f"{m['outage_s']:.0f}s of recorded outage")
    if m["coverage"] < 0.90:
        reasons.append(f"only {m['coverage']:.0%} of the span covered")
    if m["lag_spread_ms"] > 2000:
        reasons.append(f"lag dispersion {m['lag_spread_ms']:,.0f}ms")
    if m["worst_gap_s"] > 600 and not m["attributable"]:
        reasons.append(f"{m['worst_gap_s']:,.0f}s gap that cannot be "
                       f"attributed (no liveness records in this recording)")
    if m["n"] < 1000:
        reasons.append(f"only {m['n']:,} records")

    if m["reseeds"] > 0 or m["coverage"] < 0.75 or m["n"] < 1000:
        return "BROKEN", reasons
    if reasons:
        return "SUSPECT", reasons
    return "GOOD", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    files = []
    for p in a.paths:
        files.extend(sorted(glob.glob(p)) or [p])

    out = []
    for f in files:
        if not os.path.exists(f):
            print(f"missing: {f}")
            continue
        print(f"\n{'='*112}\n{os.path.basename(f)}\n{'='*112}")
        per, nlines = scan(f)
        has_liveness = any(v["heartbeats"] or v["conn"] for v in per.values())
        print(f"  {nlines:,} records, {len(per)} match(es), "
              f"liveness records: {'yes' if has_liveness else 'NO (pre-fix '
              'recording -- gaps cannot be attributed)'}")
        rows = []
        for slug, d in sorted(per.items()):
            if d["first"] is None:
                continue
            span = (d["last"] - d["first"]) / 1000.0
            gaps = np.array(list(d["gaps"].values()), dtype=float) / 1000.0
            worst = float(gaps.max()) if len(gaps) else 0.0
            # coverage: span minus the summed worst-gap per asset is a poor
            # estimate, so use the median asset's covered fraction instead
            covered = (1.0 - (np.median(gaps) / span)) if span > 0 and len(gaps) else 1.0
            lags = np.array(d["lags"]) if d["lags"] else np.array([0])
            reseeds = sum(1 for v in d["seeds"].values() if v > 1)
            m = dict(
                slug=slug, n=d["n"], assets=len(d["assets"]),
                span_h=round(span / 3600, 2),
                coverage=round(float(max(0.0, min(1.0, covered))), 4),
                worst_gap_s=round(worst, 1),
                reseeds=reseeds, trades=d["trades"],
                outage_s=round(d["outage_s"], 1),
                heartbeats=d["heartbeats"],
                attributable=bool(d["heartbeats"] or d["conn"]),
                lag_spread_ms=float(np.percentile(lags, 90) - np.percentile(lags, 50)),
                regress=d["regress"])
            m["verdict"], m["reasons"] = judge(m)
            rows.append(m)
            out.append(dict(file=os.path.basename(f), **m))

        R = sorted(rows, key=lambda r: (r["verdict"] != "BROKEN",
                                        r["verdict"] != "SUSPECT", r["slug"]))
        print(f"\n  {'match':<30} {'verdict':>8} {'recs':>10} {'assets':>7} "
              f"{'span_h':>7} {'cover':>7} {'worstgap':>9} {'reseed':>7} "
              f"{'trades':>7} {'lagspr':>8}")
        for m in R:
            print(f"  {m['slug']:<30} {m['verdict']:>8} {m['n']:>10,} "
                  f"{m['assets']:>7} {m['span_h']:>7.2f} {m['coverage']:>7.1%} "
                  f"{m['worst_gap_s']:>9,.0f} {m['reseeds']:>7} "
                  f"{m['trades']:>7,} {m['lag_spread_ms']:>8,.0f}")
        for m in R:
            if m["reasons"]:
                print(f"\n  {m['slug']} [{m['verdict']}]")
                for r in m["reasons"]:
                    print(f"    - {r}")
        c = {}
        for m in R:
            c[m["verdict"]] = c.get(m["verdict"], 0) + 1
        print(f"\n  summary: " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
