"""
Unattended end-to-end runner for ONE slate. Intended for Task Scheduler.

Steps, in order, each logged to logs/slate_<hkt-date>.log:

  1. plan_slate.py --write matches.py   -- refresh the slate. This MUST happen
     shortly before first pitch, not days ahead: Polymarket lists only the
     moneyline for a future game and adds the other ~15 markets close to game
     time. Configuring early would silently record 1/8th of the book.
  2. live_recorder.py --session <US date> --stop-when-final --duration <guard>
     Recording normally ends when the MLB API reports every game Final (plus a
     short grace for the post-game tail). --duration is only the backstop for
     the case where that API is unreachable for hours -- without it a stuck
     recorder would collide with the next slate.
  3. wait, then fetch_game_windows.py --merge, so each slate's kickoff/final-out
     is captured while matches.py still points at that slate.

Deliberately does NOT rebuild. Recording is the only irreplaceable step;
reconstruction is a pure function of the recording and can run any time. Doing
it unattended would double disk use and put an 11 GB-peak job next to a live
recorder on a 16 GB machine.

Usage:
    python run_slate.py --hkt-date 2026-08-29 --duration-hours 14
"""
import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "logs")
PY = sys.executable
SLUG_DATE = re.compile(r"-(\d{4}-\d{2}-\d{2})$")


def log(fh, msg):
    line = f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}"
    fh.write(line + "\n")
    fh.flush()
    print(line, flush=True)


def run(fh, args, label, timeout=None):
    """Runs a step, streaming its output into the log as it happens.

    subprocess.run() would buffer everything until the process exits -- which
    for a 20-hour recording means no visibility at all while it matters. This
    runs unattended for days, so the log has to be live.
    """
    log(fh, f"--- {label}: {' '.join(args)}")
    deadline = time.monotonic() + timeout if timeout else None
    p = subprocess.Popen(args, cwd=BASE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1)
    try:
        for ln in p.stdout:
            fh.write("    " + ln.rstrip() + chr(10))
            fh.flush()
            if deadline and time.monotonic() > deadline:
                log(fh, f"--- {label}: TIMEOUT; terminating")
                p.terminate()
                break
        p.wait(timeout=120)
    except Exception as e:
        log(fh, f"--- {label}: supervisor error {type(e).__name__}: {e}")
        try:
            p.kill()
        except Exception:
            pass
    rc = p.returncode if p.returncode is not None else 1
    log(fh, f"--- {label}: exit {rc}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hkt-date", required=True)
    ap.add_argument("--duration-hours", type=float, required=True)
    ap.add_argument("--final-grace-minutes", type=float, default=20.0,
                    help="stop recording this long after the MLB API reports every "
                         "game Final; --duration-hours stays as the hard backstop")
    ap.add_argument("--no-compress", action="store_true",
                    help="skip compressing this slate's recording afterwards")
    ap.add_argument("--settle-minutes", type=float, default=25.0,
                    help="wait after recording before fetching game windows, so "
                         "the MLB API has marked the last game Final")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, f"slate_{args.hkt_date}.log"), "a",
              encoding="utf-8") as fh:
        log(fh, f"===== SLATE {args.hkt_date} HKT =====")

        rc = run(fh, [PY, "plan_slate.py", "--hkt-date", args.hkt_date,
                      "--write", "matches.py"], "plan_slate", timeout=1800)
        if rc != 0:
            log(fh, "plan_slate failed; aborting slate")
            return 1

        sys.path.insert(0, BASE)
        import importlib
        import matches as matches_module
        importlib.reload(matches_module)
        slugs = [m["slug"] for m in matches_module.matches]
        if not slugs:
            log(fh, "no games in matches.py; aborting slate")
            return 1
        dates = sorted({m.group(1) for s in slugs if (m := SLUG_DATE.search(s))})
        tag = dates[0] if len(dates) == 1 else args.hkt_date
        log(fh, f"{len(slugs)} games, US date(s) {dates}, session tag '{tag}'")

        pks = [m.get("gamePk") for m in matches_module.matches if m.get("gamePk")]
        log(fh, f"{len(pks)} gamePk(s) available for --stop-when-final")

        secs = int(args.duration_hours * 3600)
        cmd = [PY, "live_recorder.py", "--session", tag, "--duration", str(secs)]
        if pks:
            cmd += ["--stop-when-final", str(args.final_grace_minutes)]
        run(fh, cmd, "live_recorder", timeout=secs + 1800)

        books = os.path.join(BASE, "data", "live", f"books_{tag}.jsonl")
        if os.path.exists(books):
            log(fh, f"recorded {os.path.getsize(books)/1e9:.2f} GB -> {books}")
        else:
            log(fh, f"WARNING: {books} does not exist")

        log(fh, f"settling {args.settle_minutes} min before window fetch")
        time.sleep(args.settle_minutes * 60)

        # fetch_game_windows reads matches.py to know which slugs to resolve, and
        # this run may have been in flight for 20 hours -- long enough for
        # something else to have rewritten that file for another slate. Rewrite
        # it for THIS slate first so the windows can't be fetched for the wrong
        # games.
        run(fh, [PY, "plan_slate.py", "--hkt-date", args.hkt_date,
                 "--write", "matches.py"], "plan_slate (re-pin)", timeout=1800)
        run(fh, [PY, "fetch_game_windows.py", "--merge"], "fetch_game_windows",
            timeout=1800)

        # Compress this slate's recording. compress_raw verifies a full md5
        # round-trip before it removes the original, so a bad archive cannot
        # cost the recording. Only this slate's file is named -- never a glob,
        # so a run that is still writing elsewhere can never be swept up.
        if args.no_compress:
            log(fh, "skipping compression (--no-compress)")
        elif os.path.exists(books):
            run(fh, [PY, "-u", "compress_raw.py", books], f"compress {tag}",
                timeout=6 * 3600)
            xz = books + ".xz"
            if os.path.exists(xz):
                log(fh, f"archived {os.path.getsize(xz)/1e6:.0f} MB -> {xz}")
            else:
                log(fh, "WARNING: no .xz produced; original left in place")
        else:
            log(fh, f"no {books} to compress")

        log(fh, f"===== SLATE {args.hkt_date} DONE =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
