"""
Fetch timestamped scoring plays for every recorded game (MLB Stats API).

Why this exists: it is the only way to tell an INFORMED price move from noise.
A run scores, every market on that game must reprice, and that jump has a
cause you can point at. Every other jump is either slower information or
churn. Without these timestamps the two are indistinguishable in the book.

Anchor choice matters and is not obvious
----------------------------------------
`about.startTime` is when the at-bat began and `about.endTime` is when the
play finished -- for a home run those can be 2+ minutes apart, because
endTime includes the trot around the bases. Neither is when the market
learned anything.

The moment the outcome becomes known is the pitch that was put in play, so
the anchor used here is the timestamp of the playEvent with
`details.isInPlay`. Its `startTime` is the pitch release; the ball lands a
second or two later. All three are recorded so an analysis can shift the
anchor and check its own sensitivity rather than trusting this choice.

    python fetch_plays.py --all
    python fetch_plays.py --slugs mlb-ari-sf-2026-08-27

Output: data/plays/<slug>.json
"""
import argparse
import json
import os
import sys
import time

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = os.path.join(BASE, "data", "game_windows.json")
OUT_DIR = os.path.join(BASE, "data", "plays")
API = "https://statsapi.mlb.com/api/v1/game/{}/playByPlay"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def in_play_ts(play):
    """Timestamp of the pitch put in play -- the instant the outcome is known.

    Falls back to the last event, then to about.startTime, because a scoring
    play need not involve a batted ball (a walk with the bases loaded, a wild
    pitch, a balk) and those have no in-play event at all.
    """
    events = play.get("playEvents") or []
    for e in events:
        if (e.get("details") or {}).get("isInPlay"):
            return e.get("startTime"), "in_play"
    if events:
        return events[-1].get("startTime"), "last_event"
    return (play.get("about") or {}).get("startTime"), "about_start"


def fetch_game(session, pk):
    r = session.get(API.format(pk), timeout=45)
    r.raise_for_status()
    return r.json().get("allPlays", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    with open(WINDOWS, "r", encoding="utf-8") as f:
        windows = json.load(f)
    slugs = args.slugs or (sorted(windows) if args.all else [])
    if not slugs:
        raise SystemExit("pass --slugs or --all")

    os.makedirs(args.out_dir, exist_ok=True)
    session = requests.Session()
    n_ok = n_sc = 0
    failed = []

    for slug in slugs:
        info = windows.get(slug)
        if not info or not info.get("gamePk"):
            failed.append(f"{slug}: no gamePk")
            continue
        out = os.path.join(args.out_dir, f"{slug}.json")
        if os.path.exists(out) and not args.force:
            continue
        try:
            plays = fetch_game(session, info["gamePk"])
        except Exception as e:
            failed.append(f"{slug}: {type(e).__name__} {e}")
            continue

        rows = []
        for p in plays:
            about = p.get("about") or {}
            res = p.get("result") or {}
            ts, src = in_play_ts(p)
            rows.append({
                "atBatIndex": about.get("atBatIndex"),
                "inning": about.get("inning"),
                "halfInning": about.get("halfInning"),
                "isScoringPlay": bool(about.get("isScoringPlay")),
                "event": res.get("event"),
                "description": res.get("description"),
                "awayScore": res.get("awayScore"),
                "homeScore": res.get("homeScore"),
                "rbi": res.get("rbi"),
                "anchor_utc": ts,
                "anchor_source": src,
                "about_start_utc": about.get("startTime"),
                "about_end_utc": about.get("endTime"),
            })
        sc = sum(1 for r in rows if r["isScoringPlay"])
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"slug": slug, "gamePk": info["gamePk"],
                       "n_plays": len(rows), "n_scoring": sc,
                       "plays": rows}, f, indent=1)
        n_ok += 1
        n_sc += sc
        log(f"  {slug}: {len(rows)} plays, {sc} scoring")
        time.sleep(0.2)

    log(f"done: {n_ok} game(s), {n_sc} scoring plays -> {args.out_dir}")
    if failed:
        log(f"FAILED ({len(failed)}): {failed[:5]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
