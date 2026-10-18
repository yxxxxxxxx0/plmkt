"""Per-match data quality: which games are trustworthy and which are not.

verify_recording.py judges a whole recording. That is too coarse to act on,
because a slate is 15 independent websockets and one dead socket does not
show up in a slate-wide average -- it just quietly poisons whichever games it
covered. Every model result so far pooled all matches together, so a match
with broken coverage contributed to the numbers exactly as if it were sound.

What is measured, per match slug:

  game cover    share of the GAME WINDOW's minutes that carry at least one
                record, plus how late the recording started and how early it
                stopped. This is the number that decides the verdict, because
                it is the only one measured against something outside the
                recording itself.
  stream cover  wall-clock span actually covered by updates, versus the span
                between the first and last record. Continuity WITHIN what was
                captured; it says nothing about what was never captured.
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
loss is certain even when its size is not. A match with no game window is
SUSPECT rather than GOOD, because its coverage cannot be established at all.

DEFECT FIXED 2026-09-22 -- why `game cover` had to be added
-----------------------------------------------------------
`stream cover` was the only coverage number here, and it is computed as
`1 - median(per-asset largest gap) / span`, where `span` runs from a match's
FIRST record to its LAST. Both ends move with the data, so a recording that
never started cannot lower it.

On books_2026-09-21, mlb-tor-bal was graded **GOOD at 99.9%** while 90% of the
game was never recorded. The slate's first run connected at 22:35Z and then
streamed nothing until a second run took over at 01:33Z -- twenty minutes
before the final out. Only 2 of the match's 42 assets were ever seen before
01:33Z, so the MEDIAN asset's first record was 01:33Z and its largest internal
gap was 16 seconds. Measured from there, the recording looks continuous,
because it is: it is continuous over the sliver of the game it caught.

Ground truth: 21 of that game's 198 minutes carry any record at all. That is
what `game cover` reports, and the same blind spot in match_filter.py's
truncation rule (which checks only the tail) was fixed in the same pass.

    python match_quality.py data/live/books_2026-09-10.jsonl.xz
    python match_quality.py data/live/*.jsonl --json data/jump/match_quality.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import orjson

BASE = os.path.dirname(os.path.abspath(__file__))
WIN = os.path.join(BASE, "data", "game_windows.json")


def load_windows(path=WIN):
    """slug -> (start_ms, end_ms), the same file match_filter.py trims on.

    Parsed with datetime rather than pandas so this stays a dependency-light
    scanner; game_windows.json only ever holds ISO-8601 UTC strings.
    """
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        if not (isinstance(v, dict) and v.get("start_utc") and v.get("end_utc")):
            continue
        try:
            a = dt.datetime.fromisoformat(v["start_utc"].replace("Z", "+00:00"))
            b = dt.datetime.fromisoformat(v["end_utc"].replace("Z", "+00:00"))
        except ValueError:
            continue
        out[k] = (int(a.timestamp() * 1000), int(b.timestamp() * 1000))
    return out


def opener(path):
    if path.endswith(".xz"):
        import lzma
        return lzma.open(path, "rb")
    return open(path, "rb")


def scan(path, lag_sample=50):
    per = defaultdict(lambda: dict(
        n=0, first=None, last=None, assets=set(), seeds=defaultdict(int),
        trades=0, regress=0, lags=[], gaps=defaultdict(int),
        prev_ts=dict(), conn=0, outage_s=0.0, heartbeats=0, books=0,
        minutes=set()))
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
            # one bucket per wall-clock minute: a few hundred ints per match,
            # and the only record of WHEN the recording was actually live
            d["minutes"].add(ts // 60000)
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


def window_coverage(minutes, window):
    """How much of the GAME WINDOW carries data, measured minute by minute.

    Returns (game_cover, late_start_s, early_stop_s) or (None, None, None)
    when the match has no window and coverage therefore cannot be judged.

    `minutes` is the set of ts // 60000 buckets seen for the match. Counting
    occupied buckets inside the window is deliberately crude: it cannot tell a
    minute with one record from a minute with ten thousand, and it is not
    supposed to. It answers the one question the stream metrics cannot --
    was the recorder live at this point in the game at all.
    """
    if not window or not minutes:
        return None, None, None
    st, en = window
    a, b = st // 60000, en // 60000
    total = b - a + 1
    if total <= 0:
        return None, None, None
    inside = sorted(m for m in minutes if a <= m <= b)
    if not inside:
        return 0.0, (en - st) / 1000.0, (en - st) / 1000.0
    return (len(inside) / total,
            max(0.0, (inside[0] * 60000 - st) / 1000.0),
            max(0.0, (en - (inside[-1] + 1) * 60000) / 1000.0))


def pct(x):
    """A share as a 7-wide column, or `n/a` when the match has no window."""
    return "    n/a" if x is None else f"{x:>7.1%}"


def judge(m):
    """GOOD / SUSPECT / BROKEN, with the reason."""
    reasons = []
    if m["reseeds"] > 0:
        reasons.append(f"{m['reseeds']} asset(s) re-seeded: price_change "
                       f"messages during those outages are unrecoverable")
    if m["outage_s"] > 60:
        reasons.append(f"{m['outage_s']:.0f}s of recorded outage")

    # game-window coverage first: it is the only check that can see data that
    # was never captured, rather than gaps inside data that was.
    g = m.get("game_cover")
    if g is None:
        reasons.append("no game window: coverage cannot be established")
    else:
        if g < 0.90:
            reasons.append(f"only {g:.0%} of the GAME covered "
                           f"({m['game_minutes_seen']:,} of "
                           f"{m['game_minutes']:,} minutes carry any record)")
        if m["late_start_s"] > 120:
            reasons.append(f"started {m['late_start_s'] / 60:.0f} min after "
                           f"first pitch")
        if m["early_stop_s"] > 300:
            reasons.append(f"stopped {m['early_stop_s'] / 60:.0f} min before "
                           f"the final out")

    if m["coverage"] < 0.90:
        reasons.append(f"only {m['coverage']:.0%} of the captured span covered")
    if m["lag_spread_ms"] > 2000:
        reasons.append(f"lag dispersion {m['lag_spread_ms']:,.0f}ms")
    if m["worst_gap_s"] > 600 and not m["attributable"]:
        reasons.append(f"{m['worst_gap_s']:,.0f}s gap that cannot be "
                       f"attributed (no liveness records in this recording)")
    if m["n"] < 1000:
        reasons.append(f"only {m['n']:,} records")

    if (m["reseeds"] > 0 or m["coverage"] < 0.75 or m["n"] < 1000
            or (g is not None and g < 0.75)):
        return "BROKEN", reasons
    if reasons:
        return "SUSPECT", reasons
    return "GOOD", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--windows", default=WIN,
                    help="game_windows.json; without it no match can be "
                         "judged GOOD, because coverage is unestablishable")
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
        windows = load_windows(a.windows)
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
            w = windows.get(slug)
            gcov, late, early = window_coverage(d["minutes"], w)
            gmin = ((w[1] // 60000) - (w[0] // 60000) + 1) if w else 0
            seen = (sum(1 for x in d["minutes"]
                        if w[0] // 60000 <= x <= w[1] // 60000) if w else 0)
            m = dict(
                slug=slug, n=d["n"], assets=len(d["assets"]),
                span_h=round(span / 3600, 2),
                game_cover=(None if gcov is None else round(float(gcov), 4)),
                game_minutes=gmin, game_minutes_seen=seen,
                late_start_s=(None if late is None else round(late, 1)),
                early_stop_s=(None if early is None else round(early, 1)),
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
              f"{'span_h':>7} {'game':>7} {'stream':>7} {'worstgap':>9} {'reseed':>7} "
              f"{'trades':>7} {'lagspr':>8}")
        for m in R:
            print(f"  {m['slug']:<30} {m['verdict']:>8} {m['n']:>10,} "
                  f"{m['assets']:>7} {m['span_h']:>7.2f} {pct(m['game_cover'])} "
                  f"{m['coverage']:>7.1%} "
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
