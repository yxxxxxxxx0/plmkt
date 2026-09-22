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

Unattended operation. run_slate_daily.ps1 starts this detached and exits
immediately, so this process outlives the Task Scheduler instance that
launched it. Three consequences are designed for here:

  own log file      Its stdout belongs to a parent that is already gone, so
                    printing to it can raise. Everything is written to
                    logs/collector.log and stdout is best-effort only. This is
                    what failed on 2026-09-22: the wrapper was killed at 13:30,
                    the orphaned collector slept on, and the first print() when
                    it woke at 05:50 hit a dead pipe and killed it silently --
                    at exactly the moment it was supposed to start recording.

  heartbeat         logs/collector_state.json carries pid, slate and a
                    timestamp refreshed every 30s. The watchdog uses it to tell
                    a working collector from a wedged one; "the pid still
                    exists" was the check that let the dead collector above
                    suppress the whole night's restarts.

  --auto            Picks the slate itself from the MLB schedule, instead of
                    the caller assuming "tomorrow". A tick at 06:00 has to
                    restart *today's* slate, not queue tomorrow's.

Usage:
    python collect_days.py --hkt-dates 2026-09-10 2026-09-11 --dry-run
    python collect_days.py --days 3
    python collect_days.py --auto            # what the scheduled task runs
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
import threading
import time

BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(BASE, "data", "live")
LOGS = os.path.join(BASE, "logs")
LOGFILE = os.path.join(LOGS, "collector.log")
STATEFILE = os.path.join(LOGS, "collector_state.json")
MANIFEST = os.path.join(LIVE, "sessions.json")
PY = sys.executable
HKT = dt.timezone(dt.timedelta(hours=8))
SLUG_DATE = re.compile(r"-(\d{4}-\d{2}-\d{2})$")


_logfh = None


def open_log():
    """Own the log file rather than inheriting a parent's stdout."""
    global _logfh
    os.makedirs(LOGS, exist_ok=True)
    _logfh = open(LOGFILE, "a", encoding="utf-8")


def log(m):
    line = f"[{dt.datetime.now(HKT):%Y-%m-%d %H:%M:%S} HKT] {m}"
    if _logfh is not None:
        _logfh.write(line + chr(10))
        _logfh.flush()
    try:
        print(line, flush=True)
    except OSError:
        pass  # detached: the pipe our parent owned is gone. Never fatal.


def keep_awake(on=True):
    """Ask Windows not to sleep while a slate is pending or recording.

    A missed slate is unrecoverable, so this is asserted for the whole run --
    including the hours spent waiting for first pitch, which is most of it.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
    except Exception as e:
        log(f"  keep_awake({on}) failed: {type(e).__name__}: {e}")


_state = dict(slate=None, phase="starting")


def set_phase(phase, slate=None):
    _state["phase"] = phase
    if slate is not None:
        _state["slate"] = slate


def heartbeat():
    """Publish liveness for run_slate_daily.ps1 to read.

    Written whole-file then renamed so a reader never sees a half-written file
    and concludes the collector is dead.
    """
    tmp = STATEFILE + ".tmp"
    try:
        os.makedirs(LOGS, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dict(pid=os.getpid(), slate=_state["slate"],
                           phase=_state["phase"],
                           heartbeat_at=dt.datetime.now(HKT).isoformat()), fh)
        os.replace(tmp, STATEFILE)
    except OSError:
        pass


def start_beating(interval=30.0):
    """Beat from a daemon thread, for the whole life of the process.

    Beating only at phase boundaries is not enough: verifying a 20 GB
    recording blocks the main thread for many minutes, and during it no
    live_recorder is running either -- so the watchdog would see a collector
    with a stale heartbeat and no recording, conclude it was wedged, and kill
    it in the middle of the verify. A thread that beats regardless of what the
    main thread is doing means the heartbeat measures the process, not the
    phase.
    """
    def beat():
        while True:
            heartbeat()
            keep_awake(True)
            time.sleep(interval)

    heartbeat()
    t = threading.Thread(target=beat, name="heartbeat", daemon=True)
    t.start()
    return t


def sleep_until(target):
    """Wait for a wall-clock instant, not for an interval.

    One long time.sleep() measures elapsed time, so a clock correction or a
    suspend/resume silently shifts the start of a recording that has to begin
    45 minutes before a specific first pitch. Re-reading the clock in short
    naps keeps the target fixed.
    """
    while True:
        left = (target - dt.datetime.now(HKT)).total_seconds()
        if left <= 0:
            return
        time.sleep(min(30.0, left))


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
        log(f"  waiting {wait/3600:.2f}h until {start_at:%Y-%m-%d %H:%M} HKT")
        set_phase(f"waiting for {start_at:%m-%d %H:%M} HKT first pitch", hkt_date)
        sleep_until(start_at)
        log(f"  woke at {dt.datetime.now(HKT):%Y-%m-%d %H:%M:%S} HKT")
    elif wait < -3600:
        log(f"  start time was {-wait/3600:.1f}h ago; recording a partial "
            f"slate is worse than skipping it")
        return dict(hkt_date=hkt_date, skipped="start time already passed")

    cmd = [PY, "-u", "run_slate.py", "--hkt-date", hkt_date,
           "--duration-hours", str(duration_hours)]
    if no_compress:
        cmd.append("--no-compress")
    set_phase("recording", hkt_date)
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
    # Verify whatever we ended up with, compressed or not. This used to run
    # only on a bare .jsonl, so a slate that compressed itself first was
    # recorded as "SKIPPED (compressed before verify)" and never looked at
    # again -- which is how books_2026-09-17 (9 games, all GOOD, 14,313 trade
    # prints) sat unverified and unused for five days. verify_recording.py
    # reads .xz directly; it is only slower, and the whole point of the step
    # is that an unverified recording is indistinguishable from a broken one.
    set_phase("verifying", hkt_date)
    log(f"  verifying {os.path.basename(row['path'])}"
        f"{' (compressed -- slower)' if not row['path'].endswith('.jsonl') else ''}...")
    row.update(verify(row["path"]))
    return row


def pick_auto_slate(lead_minutes, grace_minutes):
    """The one HKT slate a tick fired *now* should be recording.

    The rule the old wrapper used -- "the slate is always tomorrow" -- is only
    right between about 14:00 and midnight. A watchdog tick at 06:00 on
    2026-09-22, restarting after the collector died, queued 2026-09-23 and
    walked past the slate that was starting ten minutes earlier. So consider
    today first and fall through to tomorrow only once today's start time is
    past recovering.
    """
    now = dt.datetime.now(HKT)
    for off in (0, 1):
        d = str(now.date() + dt.timedelta(days=off))
        fp = first_pitch_hkt(d)
        if fp is None:
            log(f"auto: {d} has no scheduled games")
            continue
        start_at = fp - dt.timedelta(minutes=lead_minutes)
        late = (now - start_at).total_seconds() / 60.0
        if late <= grace_minutes:
            when = f"{-late:.0f} min from now" if late < 0 else f"{late:.0f} min ago"
            log(f"auto: recording {d} (first pitch {fp:%m-%d %H:%M} HKT, "
                f"start {start_at:%m-%d %H:%M} HKT, {when})")
            return d
        log(f"auto: {d} start {start_at:%m-%d %H:%M} HKT was {late:.0f} min "
            f"ago, past the {grace_minutes:.0f} min grace")
    return None


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
    ap.add_argument("--auto", action="store_true",
                    help="work out which slate is due now from the MLB "
                         "schedule; what the scheduled task runs")
    ap.add_argument("--grace-minutes", type=float, default=60.0,
                    help="with --auto, still take today's slate if its start "
                         "was at most this long ago")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-verify every recording already in data/live")
    a = ap.parse_args()

    open_log()
    log(f"=== collect_days pid {os.getpid()} :: {' '.join(sys.argv[1:])} ===")
    start_beating()

    rows = read_manifest()

    if a.verify_only:
        import glob
        # Both extensions. Globbing only "*.jsonl" meant the catch-up path
        # could never re-verify an archive, so a session compressed before
        # its verify ran had no route back to being checked at all.
        paths = sorted(set(glob.glob(os.path.join(LIVE, "books_*.jsonl")))
                       | set(glob.glob(os.path.join(LIVE, "books_*.jsonl.xz"))))
        for p in paths:
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
    if a.auto:
        d = pick_auto_slate(a.lead_minutes, a.grace_minutes)
        if d is None:
            log("auto: nothing to record now; exiting so the next tick retries")
            return 0
        dates.append(d)
    if not dates:
        raise SystemExit("pass --hkt-dates, --days, --auto, or --verify-only")

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
    try:
        sys.exit(main())
    finally:
        keep_awake(False)
