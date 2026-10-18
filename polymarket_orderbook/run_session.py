"""
Unattended runner for the Polymarket + Kalshi recorders. Linux replacement
for run_slate.bat / run_slate.py, meant to run under systemd on the server.

Daemon mode (the systemd service):

    python run_session.py --daemon [--s3 s3://bucket/prefix] [--leagues epl ucl ...]

  * Every --replan-min minutes, asks pairing.py which games listed on BOTH
    venues kick off in the next ~day. Re-planning keeps up with Polymarket
    adding markets late and kickoffs moving.
  * --lead-min before kickoff, the due games are launched together as one
    group (football and MLB separately): a detached child that runs
    live_recorder.py and kalshi_recorder.py side by side, each in its own
    process so a crash in one never takes the other down.
  * Every launched game is written to data/sessions/state.json. A daemon
    restart re-reads it, so nothing is launched twice; and the children run
    in their own session, so restarting the daemon does not kill a
    recording in progress (the unit file sets KillMode=process).

Group mode (what the daemon spawns; also handy by hand):

    python run_session.py --run-group data/sessions/<tag>.json

  Runs both recorders until the games end, then compresses every output with
  compress_raw.py (md5 round-trip verified). The .xz files then go to S3
  (--s3), or else to data/outbox/<tag>/, which deploy/pull_recordings.sh
  on the Mac downloads and deletes from the server once verified.

Stopping:
  football  Kalshi recorder stops 10 min after Kalshi closes every market;
            Polymarket recorder stops at last kickoff + 2 h 50 m.
  MLB       both stop 20 min after the MLB API reports every game Final.
  Both have a --duration hard backstop.

One-off planning check:
    python run_session.py --plan-only --hours 24
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time

import pairing
from compress_raw import xz_name

BASE = os.path.dirname(os.path.abspath(__file__))
SESS_DIR = os.path.join(BASE, "data", "sessions")
LOGS = os.path.join(BASE, "logs")
STATE = os.path.join(SESS_DIR, "state.json")
PY = sys.executable
UTC = dt.timezone.utc
HKT = pairing.HKT

FOOTBALL_PM_AFTER = dt.timedelta(hours=2, minutes=50)   # PM recorder stop after last kickoff
FOOTBALL_BACKSTOP = dt.timedelta(hours=4)
MLB_BACKSTOP = dt.timedelta(hours=7)
LOW_DISK_GB = 5.0   # warn below this; one busy football day is a few GB raw
LATE_LIMIT = {"football": dt.timedelta(minutes=90), "mlb": dt.timedelta(hours=3)}


def log(msg):
    print(f"[{dt.datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE)


def game_key(g):
    return g["kalshi_events"][0]


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

def launch_group(games, args, state):
    now = dt.datetime.now(UTC)
    first = min(dt.datetime.fromisoformat(g["kickoff"]) for g in games)
    last = max(dt.datetime.fromisoformat(g["kickoff"]) for g in games)
    sport = games[0]["sport"]
    tag = f"{first.astimezone(HKT):%Y-%m-%d_%H%M}_{sport}"
    n = 2
    while os.path.exists(os.path.join(SESS_DIR, f"{tag}.json")):
        tag = f"{first.astimezone(HKT):%Y-%m-%d_%H%M}_{sport}{n}"
        n += 1
    if sport == "football":
        pm_end = last + FOOTBALL_PM_AFTER
        backstop = last + FOOTBALL_BACKSTOP
    else:
        pm_end = backstop = last + MLB_BACKSTOP
    spec = {
        "session": tag, "sport": sport, "games": games,
        "pm_duration": max(600, int((pm_end - now).total_seconds())),
        "kalshi_duration": max(600, int((backstop - now).total_seconds())),
        "s3": args.s3, "s3_keep_local": args.s3_keep_local,
        "pollers": args.pollers, "max_rps": args.max_rps,
    }
    path = os.path.join(SESS_DIR, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    os.makedirs(LOGS, exist_ok=True)
    out = open(os.path.join(LOGS, f"session_{tag}.log"), "a", encoding="utf-8")
    p = subprocess.Popen([PY, "-u", os.path.abspath(__file__), "--run-group", path],
                         cwd=BASE, stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True)
    for g in games:
        state[game_key(g)] = {"session": tag, "launched": now.isoformat(), "pid": p.pid,
                              "title": g["title"], "kickoff": g["kickoff"]}
    save_state(state)
    log(f"launched {tag} (pid {p.pid}): {len(games)} game(s): "
        + "; ".join(g["title"] for g in games))


def daemon(args):
    os.makedirs(SESS_DIR, exist_ok=True)
    lead = dt.timedelta(minutes=args.lead_min)
    games, next_plan = [], dt.datetime.min.replace(tzinfo=UTC)
    log(f"daemon up: lead {args.lead_min} min, replan every {args.replan_min} min, "
        f"leagues {args.leagues or 'all'}, mlb {not args.no_mlb}, s3 {args.s3 or 'off'}")
    while True:
        now = dt.datetime.now(UTC)
        if now >= next_plan:
            try:
                games, report = pairing.plan(now - dt.timedelta(hours=3), now + dt.timedelta(hours=30),
                                             args.leagues, mlb=not args.no_mlb, verbose=False)
                for r in report:
                    log(f"[pairing] {r}")
                log(f"planned {len(games)} game(s) on both venues in the next 30 h")
            except Exception as e:
                log(f"planning failed ({type(e).__name__}: {e}); keeping previous plan")
            next_plan = now + dt.timedelta(minutes=args.replan_min)

        free = disk_free_gb()
        if free < LOW_DISK_GB:
            log(f"WARNING: only {free:.1f} GB free -- is pull_recordings.sh running on the Mac? "
                f"Unpulled sessions are in data/outbox/")

        state = load_state()
        due = {}
        for g in games:
            if game_key(g) in state:
                continue
            k = dt.datetime.fromisoformat(g["kickoff"])
            if k - lead <= now <= k + LATE_LIMIT[g["sport"]]:
                due.setdefault(g["sport"], []).append(g)
        for sport, gs in due.items():
            try:
                launch_group(gs, args, state)
            except Exception as e:
                log(f"launch failed for {sport} group ({type(e).__name__}: {e})")

        pending = [dt.datetime.fromisoformat(g["kickoff"]) - lead for g in games
                   if game_key(g) not in state]
        wake = min([next_plan] + [t for t in pending if t > now])
        time.sleep(max(5.0, min((wake - dt.datetime.now(UTC)).total_seconds(), 600)))


# ---------------------------------------------------------------------------
# One group
# ---------------------------------------------------------------------------

def run_group(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    tag, games = spec["session"], spec["games"]
    log(f"===== SESSION {tag}: {len(games)} game(s) =====")
    for g in games:
        log(f"  {g['kickoff']}  {g['title']}  pm={g['pm_slug']}  kalshi={' '.join(g['kalshi_events'])}")

    pm_cmd = [PY, "-u", "live_recorder.py", "--session", tag,
              "--event-slugs", *[g["pm_slug"] for g in games],
              "--market-types", "moneyline", "--gzip",
              "--duration", str(spec["pm_duration"])]
    k_cmd = [PY, "-u", "kalshi_recorder.py", "--session", tag,
             "--events", *[e for g in games for e in g["kalshi_events"]],
             "--duration", str(spec["kalshi_duration"]),
             "--pollers", str(spec["pollers"]), "--max-rps", str(spec["max_rps"])]
    pks = [g["game_pk"] for g in games if g.get("game_pk")]
    if spec["sport"] == "mlb" and pks:
        for cmd in (pm_cmd, k_cmd):
            cmd += ["--stop-when-final", "20", "--game-pks", *map(str, pks)]
    if spec["sport"] == "football":
        k_cmd += ["--stop-when-closed", "10"]

    procs = {}
    for name, cmd in (("polymarket", pm_cmd), ("kalshi", k_cmd)):
        out = open(os.path.join(LOGS, f"session_{tag}_{name}.log"), "a", encoding="utf-8")
        procs[name] = subprocess.Popen(cmd, cwd=BASE, stdout=out, stderr=subprocess.STDOUT)
        log(f"started {name} recorder (pid {procs[name].pid})")
    for name, p in procs.items():
        log(f"{name} recorder exited {p.wait()}")

    files = [os.path.join(BASE, "data", "live", f"books_{tag}.jsonl"),
             os.path.join(BASE, "data", "live", f"books_{tag}.jsonl.gz"),
             os.path.join(BASE, "data", "live", f"top_of_book_{tag}.csv")]
    kdir = os.path.join(BASE, "data", "live", "kalshi")
    files += [os.path.join(kdir, f"{k}_{tag}{ext}") for k, ext in
              (("kbooks", ".jsonl"), ("ktrades", ".jsonl"), ("kpolls", ".jsonl"), ("ktop", ".csv"))]
    files = [f for f in files if os.path.exists(f)]
    for f in files:
        log(f"recorded {os.path.getsize(f)/1e6:9.1f} MB  {f}")

    if files:
        rc = subprocess.call([PY, "-u", "compress_raw.py", *files], cwd=BASE)
        log(f"compress_raw exited {rc}")
    xzs = [xz_name(f) for f in files if os.path.exists(xz_name(f))]
    missing = [f for f in files if not os.path.exists(xz_name(f))]
    if missing:
        log(f"WARNING: not compressed, originals kept: {missing}")

    if spec.get("s3") and xzs:
        dest = spec["s3"].rstrip("/") + f"/{tag}/"
        for x in xzs + [path]:
            rc = subprocess.call(["aws", "s3", "cp", "--only-show-errors", x, dest])
            if rc == 0 and x != path and not spec.get("s3_keep_local"):
                os.remove(x)
            log(f"s3 upload {'ok' if rc == 0 else f'FAILED ({rc}), kept locally'}: "
                f"{os.path.basename(x)} -> {dest}")
    elif xzs:
        to_outbox(tag, xzs, path)
    log(f"===== SESSION {tag} DONE =====")


def to_outbox(tag, xzs, spec_path):
    """Hand a finished session to deploy/pull_recordings.sh, which runs on
    the Mac, copies data/outbox/ down and deletes each file from the server
    once it has arrived intact -- so the server only ever holds sessions that
    have not been pulled yet.

    Built under a .tmp- name and renamed into place, so the puller (which
    skips .tmp-*) can never take a half-moved session."""
    outbox = os.path.join(BASE, "data", "outbox")
    tmp = os.path.join(outbox, f".tmp-{tag}")
    os.makedirs(tmp, exist_ok=True)
    for x in xzs:
        sub = "kalshi" if os.path.basename(x).startswith("k") else "polymarket"
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        os.replace(x, os.path.join(tmp, sub, os.path.basename(x)))
    shutil.copy2(spec_path, os.path.join(tmp, "session.json"))
    for f in os.listdir(LOGS):
        if f.startswith(f"session_{tag}"):
            shutil.copy2(os.path.join(LOGS, f), os.path.join(tmp, f))
    os.replace(tmp, os.path.join(outbox, tag))
    log(f"moved {len(xzs)} file(s) to outbox/{tag} for pull_recordings.sh")


def disk_free_gb(path=BASE):
    return shutil.disk_usage(path).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--daemon", action="store_true")
    mode.add_argument("--run-group", metavar="SPEC_JSON")
    mode.add_argument("--plan-only", action="store_true")
    ap.add_argument("--hours", type=float, default=24, help="--plan-only window")
    ap.add_argument("--leagues", nargs="*", default=None,
                    help=f"football leagues (default all: {' '.join(pairing.FOOTBALL)})")
    ap.add_argument("--no-mlb", action="store_true")
    ap.add_argument("--lead-min", type=float, default=30, help="start recording this long before kickoff")
    ap.add_argument("--replan-min", type=float, default=20)
    ap.add_argument("--pollers", type=int, default=4)
    ap.add_argument("--max-rps", type=float, default=25)
    ap.add_argument("--s3", default=None, metavar="s3://bucket/prefix",
                    help="upload compressed sessions here (needs the aws cli and an instance role)")
    ap.add_argument("--s3-keep-local", action="store_true", help="keep the .xz after uploading")
    args = ap.parse_args()

    if args.run_group:
        run_group(args.run_group)
    elif args.plan_only:
        now = dt.datetime.now(UTC)
        games, _ = pairing.plan(now, now + dt.timedelta(hours=args.hours), args.leagues,
                                mlb=not args.no_mlb)
        for g in games:
            k = dt.datetime.fromisoformat(g["kickoff"]).astimezone(HKT)
            print(f"{k:%m-%d %H:%M} HKT  start {k - dt.timedelta(minutes=args.lead_min):%H:%M}  "
                  f"{g['league']:<4} {g['title']}")
        print(f"{len(games)} game(s)")
    else:
        daemon(args)


if __name__ == "__main__":
    main()
