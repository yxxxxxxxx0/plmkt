"""
Writes data/game_windows.json: each tracked game's kickoff and the time of
its actual final out, from the MLB Stats API.

rebuild_multi.py uses these to trim each game's window. The distinction that
matters: the end time here is the end of the **last completed play**, not
market close -- Polymarket markets stay open for a while after the final out,
and trimming on market close pulls in a tail of post-game quoting.

Run this AFTER the games finish. Before then a game has no last play, and any
game still "In Progress" gets the last play so far, which is not its real end
-- rerun once it reads Final.

Teams are matched to Polymarket slugs by the away/home abbreviations already
in the slug, so a new slate needs no hand-written mapping.

Usage:
    python fetch_game_windows.py                 # dates inferred from matches.py
    python fetch_game_windows.py --date 2026-08-27
"""

import argparse
import json
import os
import re

import requests

import matches as matches_module

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "data", "game_windows.json")

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

_SLUG_RE = re.compile(r"^mlb-(?P<away>[a-z]+)-(?P<home>[a-z]+)-(?P<date>\d{4}-\d{2}-\d{2})$")


def slug_dates(slugs):
    out = set()
    for s in slugs:
        m = _SLUG_RE.match(s)
        if m:
            out.add(m.group("date"))
    return sorted(out)


def gamma_title(slug):
    try:
        rows = requests.get(GAMMA_EVENTS_URL, params={"slug": slug}, timeout=20).json()
    except Exception:
        return None
    return rows[0].get("title") if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", nargs="*", default=None,
                     help="US date(s) YYYY-MM-DD; default: inferred from matches.py slugs")
    ap.add_argument("--output", default=OUT_PATH)
    ap.add_argument("--merge", action="store_true",
                     help="merge into the existing file instead of replacing it")
    args = ap.parse_args()

    slugs = [m["slug"] for m in matches_module.matches]
    dates = args.date or slug_dates(slugs)
    if not dates:
        raise SystemExit("Could not infer any date from matches.py slugs; pass --date.")
    print(f"tracked slugs: {len(slugs)}   dates: {', '.join(dates)}")

    # Match Gamma's event title (which the viewer displays) to the MLB game,
    # via "Away vs. Home" -> ("Away", "Home").
    title_by_slug = {}
    for s in slugs:
        t = gamma_title(s)
        if t:
            title_by_slug[s] = t
    pair_to_slug = {}
    for s, t in title_by_slug.items():
        parts = [p.strip() for p in t.split(" vs. ")]
        if len(parts) == 2:
            pair_to_slug[(parts[0], parts[1])] = s

    result = {}
    if args.merge and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            result = json.load(f)

    matched = 0
    for date in dates:
        r = requests.get(SCHEDULE_URL, params={"sportId": 1, "date": date}, timeout=25)
        r.raise_for_status()
        games = []
        for d in r.json().get("dates", []):
            games.extend(d.get("games", []))

        for g in games:
            away = g["teams"]["away"]["team"]["name"]
            home = g["teams"]["home"]["team"]["name"]
            slug = pair_to_slug.get((away, home))
            if slug is None:
                continue
            pk = g["gamePk"]
            status = g["status"]["detailedState"]
            end = None
            try:
                fd = requests.get(FEED_URL.format(pk=pk), timeout=30).json()
                plays = fd["liveData"]["plays"]["allPlays"]
                status = fd["gameData"]["status"]["detailedState"]
                if plays:
                    end = plays[-1]["about"]["endTime"]
            except Exception as e:
                print(f"  [warn] {slug}: feed fetch failed ({e})")

            if end is None:
                print(f"  [skip] {slug}: no completed plays yet (status={status})")
                continue

            result[slug] = {
                "gamePk": pk,
                "start_utc": g["gameDate"],
                "end_utc": end,
                "status": status,
            }
            matched += 1
            warn = "" if status == "Final" else "   <-- NOT FINAL, rerun later"
            print(f"  {slug:<26} {g['gameDate']} -> {end}  {status}{warn}")

    unmatched = [s for s in slugs if s not in result]
    if unmatched:
        print(f"\nno window for {len(unmatched)} tracked slug(s): {', '.join(unmatched)}")
        print("  (games that have not started yet -- rerun after they finish)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.output} ({matched} game(s) resolved this run, {len(result)} total)")


if __name__ == "__main__":
    main()
