"""Is this recording fit to draw conclusions from? Run it before trusting one.

The 2026-08-26 recording captured about 6% of the feed and was unrepairable,
and that was discovered long afterwards by noticing something odd downstream.
Nothing about the file announces it: a 4 GB JSONL of well-formed records looks
exactly like a good recording, and the viewer renders it happily. The failure
modes are all invisible unless specifically measured, so this measures them.

What it checks and why each one matters
--------------------------------------
delivery lag (recv - ts)
    Local receive time minus the EXCHANGE's own event time. Read this
    carefully, because it is easy to over-read: `live_recorder.py` sets
    ts from `msg["timestamp"]` (line 367), so local slowness does NOT move
    event times. High lag means the process ran behind, not that the clock on
    the data is wrong -- event ordering and timing survive it.

    What it does mean is loss of headroom. The original disaster was
    `flush()` + `os.fsync()` per row on the event loop: the busiest second
    needed ~25s of disk time to write 1s of data, the loop froze past the 20s
    keepalive, and sockets were killed -- and on reconnect only the current
    book is recovered, so every `price_change` during the outage is gone.
    Lag is the leading indicator of that; re-seeds are the actual damage. So
    sustained lag is a warning about margin, and re-seeds and ts-fallback are
    the failures.

ts falling back to local time
    If the exchange omits `timestamp`, the recorder substitutes receive time
    (lines 367-371). Those rows have genuinely corrupted event times, and they
    are detectable because ts == recv exactly. This is the one path by which
    local lag can actually poison event timing.

re-seeds per asset
    On reconnect only the current book is recovered -- every `price_change`
    during the outage is gone for good. Order is recoverable, timing is not.
    One seed per asset is normal (the initial snapshot). More means outages,
    and the count is how many.

coverage gaps
    A market that stops updating for minutes either went quiet or its socket
    died, and those look identical in the data. Reported per asset so a single
    dead socket cannot hide inside a slate-wide average.

trade prints
    `last_trade_price` used to be dropped, which left every size decrease at
    the touch ambiguous between a fill and a cancel -- and cancels carry no
    adverse selection, so their presence biases maker estimates optimistically.
    Roughly 45% of one-tick touch moves turned out to be reprices. A recording
    without prints cannot answer any maker question honestly.

timestamp regressions and millisecond collisions
    The exchange clock is millisecond-resolution and up to 78 snapshots can
    share one millisecond; the fast rebuild path assumes ts never goes
    backwards within an asset and falls back to a slow exact loop when it
    does. Counting them here says in advance which path a rebuild will take.

Usage:
    python verify_recording.py data/live/books_hkt0909.jsonl
    python verify_recording.py ../data/live/books_2026-08-30.jsonl --max-lines 5000000
    python verify_recording.py data/live/*.jsonl --json report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import orjson

# A recording that trips any of these is not safe to draw conclusions from
# without saying so explicitly in the write-up.
LAG_SUSTAINED_FAIL_MS = 2_000
LAG_SPREAD_WARN_MS = 500
GAP_WARN_S = 60
RESEED_WARN = 2


def human(n):
    for u, d in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= d:
            return f"{n/d:.2f} {u}"
    return f"{n} B"


def verify(path, max_lines=None, lag_sample=200):
    t0 = time.time()
    et = Counter()
    mtypes = Counter()
    n = bad = 0
    lags = []
    # Lag bucketed by hour into the run. This is the check that separates a
    # startup burst -- hundreds of assets seeding at once, harmless -- from
    # sustained event-loop stalling, which makes every timestamp in the file
    # unreliable. Both look identical in a pooled percentile.
    lag_by_hour = defaultdict(list)
    anchor = None
    ts_fallback = 0
    per = defaultdict(lambda: dict(n=0, seeds=0, first=None, last=None,
                                   max_gap=0, regress=0, prev=None,
                                   trades=0, ms_collide=0))
    first_ts = last_ts = None

    # xz archives are read transparently: compress_raw.py removes the original
    # after proving a round-trip, so a compressed slate is the only copy.
    if path.endswith(".xz"):
        import lzma
        opener = lambda p: lzma.open(p, "rb")
    else:
        opener = lambda p: open(p, "rb")
    with opener(path) as fh:
        for raw in fh:
            n += 1
            if max_lines and n > max_lines:
                break
            try:
                r = orjson.loads(raw)
            except Exception:
                bad += 1
                continue
            e = r.get("et") or "?"
            et[e] += 1
            if r.get("market_type"):
                mtypes[r["market_type"]] += 1

            ts, recv = r.get("ts"), r.get("recv")
            if ts is None:
                continue
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
            if anchor is None:
                anchor = ts
            # the one way local lag can corrupt event time: the exchange
            # omitted `timestamp` and receive time was substituted
            if recv is not None and ts == recv:
                ts_fallback += 1
            if recv is not None and n % lag_sample == 0:
                lags.append(recv - ts)
                lag_by_hour[max(0, (ts - anchor) // 3_600_000)].append(recv - ts)

            aid = r.get("asset_id") or r.get("slug") or "?"
            d = per[aid]
            d["n"] += 1
            if e == "seed":
                d["seeds"] += 1
            if e == "trade":
                d["trades"] += 1
            if d["first"] is None:
                d["first"] = ts
            d["last"] = ts
            p = d["prev"]
            if p is not None:
                if ts < p:
                    d["regress"] += 1
                elif ts == p:
                    d["ms_collide"] += 1
                else:
                    d["max_gap"] = max(d["max_gap"], ts - p)
            d["prev"] = ts

    lags = np.array(lags, dtype=np.int64) if lags else np.array([0])
    dur_s = (last_ts - first_ts) / 1000 if first_ts and last_ts else 0
    assets = list(per.items())
    gaps = np.array([d["max_gap"] for _, d in assets]) / 1000 if assets else np.array([0])
    seeds = np.array([d["seeds"] for _, d in assets]) if assets else np.array([0])

    rep = dict(
        path=path, size_bytes=os.path.getsize(path), lines=n, unparseable=bad,
        truncated=bool(max_lines and n > max_lines),
        duration_s=dur_s,
        first_ts=first_ts, last_ts=last_ts,
        event_types=dict(et), market_types=dict(mtypes),
        n_assets=len(per),
        trade_prints=int(et.get("trade", 0)),
        lag_ms=dict(p50=int(np.percentile(lags, 50)),
                    p90=int(np.percentile(lags, 90)),
                    p99=int(np.percentile(lags, 99)),
                    max=int(lags.max())),
        asset_gap_s=dict(p50=float(np.percentile(gaps, 50)),
                         p90=float(np.percentile(gaps, 90)),
                         max=float(gaps.max())),
        reseeds=dict(max=int(seeds.max()),
                     n_assets_over_1=int((seeds > 1).sum())),
        ts_regressions=int(sum(d["regress"] for _, d in assets)),
        ms_collisions=int(sum(d["ms_collide"] for _, d in assets)),
        ts_fallback=ts_fallback,
        ts_fallback_frac=ts_fallback / max(n, 1),
        lag_by_hour=[dict(hour=int(h), n=len(v),
                          p50=int(np.percentile(v, 50)),
                          p90=int(np.percentile(v, 90)),
                          p99=int(np.percentile(v, 99)))
                     for h, v in sorted(lag_by_hour.items()) if v],
        parse_seconds=round(time.time() - t0, 1),
    )
    # `recv - ts` is local receive time minus exchange time, so it contains
    # clock skew as well as delivery lag -- one file here sits at p50 = -209ms,
    # meaning the local clock trails the exchange. Skew shifts every quantile
    # equally; a stalling event loop widens the upper tail. So the dispersion
    # p90 - p50 is the skew-invariant measure, and it is what the verdict uses.
    rep["lag_spread_ms"] = rep["lag_ms"]["p90"] - rep["lag_ms"]["p50"]
    late = [h for h in rep["lag_by_hour"] if h["hour"] >= 1]
    rep["lag_sustained_spread_ms"] = (
        int(np.median([h["p90"] - h["p50"] for h in late])) if late else None)

    # ---- verdict ---------------------------------------------------------- #
    fails, warns = [], []

    # Event time comes from the exchange, so lag costs headroom, not accuracy.
    # It is a FAIL only when it actually damaged something -- which shows up as
    # re-seeds (lost messages) or ts falling back to local time.
    if rep["ts_fallback_frac"] > 0.01:
        fails.append(
            f"{rep['ts_fallback']:,} rows ({rep['ts_fallback_frac']:.1%}) have "
            f"ts == recv: the exchange omitted `timestamp` and local receive "
            f"time was substituted, so those event times are corrupted by "
            f"whatever the process lag was at that moment")
    sus = rep["lag_sustained_spread_ms"]
    if sus is not None and sus > LAG_SUSTAINED_FAIL_MS:
        warns.append(
            f"delivery lag dispersion {sus:,}ms (p90-p50) sustained past hour "
            f"1: the recorder ran with little headroom for the whole session. "
            f"Event times still come from the exchange so timing is intact, "
            f"and re-seeds are the check on whether anything was actually "
            f"lost -- but this is close to the 20s keepalive that killed the "
            f"sockets in the 2026-08-26 run")
    elif rep["lag_spread_ms"] > LAG_SUSTAINED_FAIL_MS:
        warns.append(
            f"pooled lag dispersion {rep['lag_spread_ms']:,}ms but only "
            f"{sus:,}ms after hour 1 -- a startup burst (many assets seeding "
            f"at once) rather than sustained pressure")
    elif rep["lag_spread_ms"] > LAG_SPREAD_WARN_MS:
        warns.append(f"lag dispersion {rep['lag_spread_ms']:,}ms is elevated")
    if rep["trade_prints"] == 0:
        warns.append("no trade prints: fine for book-dynamics work, but every "
                     "size decrease at the touch is ambiguous between a fill "
                     "and a cancel, and cancels carry no adverse selection -- "
                     "so maker P&L from this file is biased optimistically")
    if rep["reseeds"]["max"] >= RESEED_WARN:
        warns.append(f"{rep['reseeds']['n_assets_over_1']} asset(s) re-seeded "
                     f"(max {rep['reseeds']['max']}x): price_change messages "
                     f"were lost during those outages and cannot be recovered")
    if rep["asset_gap_s"]["max"] > GAP_WARN_S:
        warns.append(f"largest per-asset gap {rep['asset_gap_s']['max']:,.0f}s "
                     f"-- a quiet market and a dead socket look identical here")
    if rep["ts_regressions"]:
        warns.append(f"{rep['ts_regressions']:,} timestamp regressions: the "
                     f"rebuild will fall back to the slow exact path for those "
                     f"assets (correct, just slower)")
    if rep["unparseable"]:
        warns.append(f"{rep['unparseable']:,} unparseable line(s) -- usually a "
                     f"truncated final line from a killed recorder")
    rep["fails"], rep["warns"] = fails, warns
    rep["verdict"] = "FAIL" if fails else ("WARN" if warns else "OK")
    # A recording can be sound for one question and unusable for another, so
    # say which rather than collapsing it to one verdict.
    rep["usable_for"] = dict(
        book_dynamics=not fails,
        maker_pnl=not fails and rep["trade_prints"] > 0,
        latency_analysis=not fails and (sus is None or sus <= LAG_SPREAD_WARN_MS),
    )
    return rep


def show(rep):
    print(f"\n{'='*78}\n{rep['path']}\n{'='*78}")
    print(f"  {human(rep['size_bytes'])}, {rep['lines']:,} lines"
          + ("  [TRUNCATED SCAN]" if rep["truncated"] else "")
          + f", parsed in {rep['parse_seconds']}s")
    if rep["duration_s"]:
        print(f"  span {rep['duration_s']/3600:.2f} h, {rep['n_assets']} assets, "
              f"{rep['lines']/max(rep['duration_s'],1):,.0f} records/s")
    print(f"  events: " + "  ".join(f"{k}={v:,}" for k, v in
                                    sorted(rep["event_types"].items(),
                                           key=lambda kv: -kv[1])))
    if rep["market_types"]:
        print(f"  market types: " + "  ".join(
            f"{k}={v:,}" for k, v in sorted(rep["market_types"].items(),
                                            key=lambda kv: -kv[1])[:8]))
    L = rep["lag_ms"]
    print(f"  delivery lag ms: p50={L['p50']:,} p90={L['p90']:,} "
          f"p99={L['p99']:,} max={L['max']:,}")
    print(f"    dispersion (p90-p50, skew-invariant): "
          f"{rep['lag_spread_ms']:,}ms pooled, "
          f"{rep['lag_sustained_spread_ms']:,}ms sustained after hour 1"
          if rep["lag_sustained_spread_ms"] is not None else
          f"    dispersion: {rep['lag_spread_ms']:,}ms (single hour)")
    if len(rep["lag_by_hour"]) > 1:
        print("    by hour into run:  " + "  ".join(
            f"h{h['hour']}:{h['p90']-h['p50']:,}" for h in rep["lag_by_hour"]))
    G = rep["asset_gap_s"]
    print(f"  per-asset max gap s: p50={G['p50']:.1f} p90={G['p90']:.1f} "
          f"max={G['max']:.1f}")
    print(f"  re-seeds: max {rep['reseeds']['max']}, "
          f"{rep['reseeds']['n_assets_over_1']} asset(s) > 1")
    print(f"  trade prints: {rep['trade_prints']:,}")
    print(f"  ts regressions {rep['ts_regressions']:,}, "
          f"ms collisions {rep['ms_collisions']:,}, "
          f"ts==recv fallback {rep['ts_fallback']:,} "
          f"({rep['ts_fallback_frac']:.2%})")
    print(f"\n  VERDICT: {rep['verdict']}   usable for: " + ", ".join(
        f"{k}={'yes' if v else 'NO'}" for k, v in rep["usable_for"].items()))
    for f in rep["fails"]:
        print(f"    [FAIL] {f}")
    for w in rep["warns"]:
        print(f"    [warn] {w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max-lines", type=int, default=None,
                    help="scan only the first N lines (quick check)")
    ap.add_argument("--json", default=None, help="write reports to this path")
    a = ap.parse_args()

    files = []
    for p in a.paths:
        files.extend(sorted(glob.glob(p)) or [p])
    reps = []
    for f in files:
        if not os.path.exists(f):
            print(f"missing: {f}")
            continue
        rep = verify(f, max_lines=a.max_lines)
        show(rep)
        reps.append(rep)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(reps, fh, indent=2)
        print(f"\nwrote {a.json}")
    worst = "OK"
    for r in reps:
        if r["verdict"] == "FAIL":
            worst = "FAIL"
        elif r["verdict"] == "WARN" and worst == "OK":
            worst = "WARN"
    print(f"\n{len(reps)} file(s) checked; worst verdict {worst}")
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
