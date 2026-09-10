"""Collect several consecutive slates, verifying each one before moving on.

`run_slate.py` already does one slate properly -- refresh the slate shortly
before first pitch, record until the MLB API reports every game Final, fetch
the real game windows afterwards, compress with an md5 round-trip. This adds
the two things the jump study actually needs:

  more than one day   Every result in reports/FINDINGS.md rests on a single
                      day of a single sport, which is the binding limitation
                      on all of them. Three days is not a lot but it is enough
                      to see whether a finding survives a different day.

  a verdict per day   A recording that captured 6% of the feed looks exactly
                      like a good one. verify_recording.py is run immediately
                      after each slate and its verdict is written into a
                      manifest, so a bad day is known at collection time
                      rather than discovered downstream weeks later.

Timing matters and is the reason this waits rather than firing everything at
once: Polymarket lists only the moneyline for a future game and adds the other
~15 markets close to game time, so planning a slate early silently records an
eighth of the book.

Usage:
    python collect_days.py --hkt-dates 2026-09-10 2026-09-11 --dry-run
    python collect_days.py --days 3
    python collect_days.py --verify-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(BASE, "data", "live")
MANIFEST = os.path.join(LIVE, "sessions.json")
PY = sys.executable
HKT = dt.timezone(dt.timedelta(hours=8))
SLUG_DATE = re.compile(r"-(\d{4}-\d{2}-\d{2})$")


def log(m):
    print(f"[{dt.datetime.now(HKT):%Y-%m-%d %H:%M:%S} HKT] {m}", flush=True)


def read_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as fh:
            return json.load(fh)
    return []


def write_manifest(rows):
    os.makedirs(LIVE, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


def first_pitch_hkt(hkt_date: str):
    """Earliest game start for an HKT date, from the MLB schedule.

    Polymarket slugs carry the US date, so an HKT date spans two US dates and
    both have to be scanned -- the same trap plan_slate.py documents.
    """
    sys.path.insert(0, BASE)
    import plan_slate as ps

    d = dt.date.fromisoformat(hkt_date)
    starts = []
    for us in (d - dt.timedelta(days=1), d):
        try:
            for g in ps.schedule(str(us)):
                t = dt.datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
                h = t.astimezone(HKT)
                if h.date().isoformat() == hkt_date:
                    starts.append(h)
        except Exception as e:
            log(f"  schedule fetch failed for {us}: {type(e).__name__}: {e}")
    return min(starts) if starts else None


def session_tag():
    """The tag run_slate.py will have used, derived the same way it derives it."""
    sys.path.insert(0, BASE)
    import importlib
    import matches as m
    importlib.reload(m)
    slugs = [x["slug"] for x in m.matches]
    dates = sorted({g.group(1) for s in slugs if (g := SLUG_DATE.search(s))})
    return (dates[0] if len(dates) == 1 else None), len(slugs)


def verify(path):
    """Run the health check and return its parsed verdict."""
    rep_path = os.path.join(LIVE, os.path.basename(path) + ".health.json")
    rc = subprocess.run([PY, "-u", "verify_recording.py", path,
                         "--json", rep_path], cwd=BASE).returncode
    if os.path.exists(rep_path):
        with open(rep_path, encoding="utf-8") as fh:
            reps = json.load(fh)
        if reps:
            r = reps[0]
            return dict(verdict=r["verdict"], fails=r["fails"],
                        warns=r["warns"], lines=r["lines"],
                        trade_prints=r["trade_prints"],
                        lag_sustained_spread_ms=r.get("lag_sustained_spread_ms"),
                        health_report=rep_path)
    return dict(verdict="UNKNOWN", fails=[f"verify exited {rc}"], warns=[])


def collect_one(hkt_date, lead_minutes, duration_hours, dry_run, no_compress):
    fp = first_pitch_hkt(hkt_date)
    if fp is None:
        log(f"{hkt_date}: no scheduled games found; skipping")
        return None
    start_at = fp - dt.timedelta(minutes=lead_minutes)
    log(f"{hkt_date}: first pitch {fp:%H:%M} HKT, "
        f"start recording {start_at:%H:%M} HKT "
        f"({lead_minutes} min lead), run up to {duration_hours}h")
    if dry_run:
        return dict(hkt_date=hkt_date, first_pitch=fp.isoformat(),
                    start_at=start_at.isoformat(), planned=True)

    wait = (start_at - dt.datetime.now(HKT)).total_seconds()
    if wait > 0:
        log(f"  sleeping {wait/3600:.2f}h until {start_at:%Y-%m-%d %H:%M} HKT")
        time.sleep(wait)
    elif wait < -3600:
        log(f"  start time was {-wait/3600:.1f}h ago; recording a partial "
            f"slate is worse than skipping it")
        return dict(hkt_date=hkt_date, skipped="start time already passed")

    cmd = [PY, "-u", "run_slate.py", "--hkt-date", hkt_date,
           "--duration-hours", str(duration_hours)]
    if no_compress:
        cmd.append("--no-compress")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=BASE).returncode
    log(f"  run_slate exit {rc} after {(time.time()-t0)/3600:.2f}h")

    tag, n_games = session_tag()
    row = dict(hkt_date=hkt_date, tag=tag, n_games=n_games, run_slate_rc=rc,
               finished_at=dt.datetime.now(HKT).isoformat())
    if not tag:
        row["skipped"] = "could not derive a single session tag"
        return row
    for cand in (f"books_{tag}.jsonl", f"books_{tag}.jsonl.xz"):
        p = os.path.join(LIVE, cand)
        if os.path.exists(p):
            row["path"] = p
            row["size_bytes"] = os.path.getsize(p)
            break
    if "path" not in row:
        row["skipped"] = f"no books_{tag}.jsonl produced"
        return row
    if row["path"].endswith(".jsonl"):
        log(f"  verifying {os.path.basename(row['path'])}...")
        row.update(verify(row["path"]))
    else:
        row["verdict"] = "SKIPPED (compressed before verify)"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hkt-dates", nargs="*", default=None)
    ap.add_argument("--days", type=int, default=None,
                    help="collect this many consecutive HKT dates from tomorrow")
    ap.add_argument("--lead-minutes", type=float, default=45.0,
                    help="start recording this long before the first game")
    ap.add_argument("--duration-hours", type=float, default=14.0)
    ap.add_argument("--no-compress", action="store_true",
                    help="leave the raw jsonl in place (needed if you intend "
                         "to rebuild the jump dataset from it straight away)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-verify every recording already in data/live")
    a = ap.parse_args()

    rows = read_manifest()

    if a.verify_only:
        import glob
        for p in sorted(glob.glob(os.path.join(LIVE, "books_*.jsonl"))):
            log(f"verifying {os.path.basename(p)}")
            r = dict(path=p, size_bytes=os.path.getsize(p),
                     checked_at=dt.datetime.now(HKT).isoformat())
            r.update(verify(p))
            rows = [x for x in rows if x.get("path") != p] + [r]
            write_manifest(rows)
        log(f"manifest -> {MANIFEST}")
        return 0

    dates = a.hkt_dates or []
    if a.days:
        today = dt.datetime.now(HKT).date()
        dates += [str(today + dt.timedelta(days=i + 1)) for i in range(a.days)]
    if not dates:
        raise SystemExit("pass --hkt-dates, --days, or --verify-only")

    log(f"plan: {len(dates)} slate(s) -- {', '.join(dates)}")
    for d in dates:
        r = collect_one(d, a.lead_minutes, a.duration_hours, a.dry_run,
                        a.no_compress)
        if r and not a.dry_run:
            rows.append(r)
            write_manifest(rows)
            log(f"  {d}: verdict {r.get('verdict', r.get('skipped'))}")
    if not a.dry_run:
        log(f"manifest -> {MANIFEST}")
        ok = [r for r in rows if r.get("verdict") == "OK"]
        log(f"{len(ok)} of {len(rows)} recorded session(s) verified clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
