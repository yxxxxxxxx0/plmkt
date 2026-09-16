"""Read-only delay audit for current scheduled matchup games only."""
from __future__ import annotations

import csv
import datetime as dt
import re
import time
import argparse
from pathlib import Path

import requests

SPORT_CODES = {
    "Counter-Strike / CS2": "cs2", "NHL": "nhl", "NBA": "nba", "Formula 1": "f1",
    "Dota 2": "dota2", "Valorant": "val", "Esports (generic)": "cs2",
    "League of Legends": "lol", "Cricket": "cricipl", "Premier League": "epl",
    "MLS": "mls", "UFC": "ufc", "NFL": "nfl", "NCAA": "ncaab", "Ligue 1": "fl1",
    "Champions League": "ucl", "La Liga": "lal", "Serie A": "sea", "Bundesliga": "bun",
    "EPL": "epl", "Baseball": "mlb", "Football": "nfl", "Basketball": "nba",
    "Rugby": "ruprem", "WNBA": "wnba", "MLB": "mlb",
}
GAMMA_SPORTS = "https://gamma-api.polymarket.com/sports"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_MARKET = "https://clob.polymarket.com/markets/{}"
MATCH_TITLE = re.compile(r"^\s*[^|]+?\s+(?:vs\.?|versus)\s+[^|]+?\s*$", re.I)
PROP = re.compile(
    r"total|corners?|halftime|second half|first half|\b[12]H\b|map\s*\d|game\s*\d|round|kills?|"
    r"player|goals?|assists?|shots?|cards?|toss|batter|strikeout|champion|"
    r"qualif|pole|fastest|safety car|red flag|retire|draft|award|next fight|"
    r"standings?|playoffs?|finals?|season series|win total|clean sheets?",
    re.I,
)
MAIN_MARKET = re.compile(r"(^|\b)(who will win|match winner|moneyline|winner|win\??)(\b|$)", re.I)


def get(session, url, params=None):
    last = None
    for n in range(3):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(.5 * (n + 1))
    raise last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-open", action="store_true", help="include any open matchup, regardless of game date")
    args = parser.parse_args()
    root = Path("polymarket_sports")
    categories = [r["sports_group"] for r in csv.DictReader(
        (root / "reports" / "sports_coverage_by_classifier.csv").open(encoding="utf-8")
    )]
    session = requests.Session()
    sports = {x["sport"]: x for x in get(session, GAMMA_SPORTS)}
    rows = []
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for category in categories:
        code = SPORT_CODES[category]
        sport = sports.get(code)
        base = {"checked_utc": now, "dataset_category": category, "sport_code": code,
                "event_slug": "", "event_title": "", "question": "", "condition_id": "",
                "seconds_delay": "", "status": ""}
        if not sport:
            base["status"] = "no official sport code"
            rows.append(base); continue
        events = []
        for offset in range(0, 1000, 100):
            page = get(session, GAMMA_EVENTS, {"tag_id": sport["primaryTagId"],
                        "active": "true", "closed": "false", "limit": 100, "offset": offset})
            events.extend(page)
            if len(page) < 100:
                break
        games = []
        for e in events:
            if not (MATCH_TITLE.search(e.get("title", "")) and not PROP.search(e.get("title", ""))):
                continue
            if args.all_open:
                games.append(e)
                continue
            now_dt = dt.datetime.now(dt.timezone.utc)
            start = e.get("startTime") or e.get("eventDate") or ""
            try:
                parsed = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
                if parsed < now_dt - dt.timedelta(hours=4):
                    continue
            except ValueError:
                # If no parseable start is supplied, leave the event out of a
                # current-game audit rather than accidentally testing a stale event.
                continue
            games.append(e)
        if not games:
            base["status"] = "no current matchup event"
            rows.append(base); print(f"{category:<24} no current matchup event"); continue
        event = games[0]
        markets = [m for m in event.get("markets", []) if m.get("conditionId") and
                   m.get("active", True) and not m.get("closed", False) and
                   m.get("acceptingOrders", True) and not PROP.search(m.get("question", ""))]
        if not markets:
            base["status"] = "match event has no main non-prop market"
            rows.append(base); print(f"{category:<24} match found but no main market"); continue
        market = next((m for m in markets if m.get("question", "").strip() == event.get("title", "").strip()), None)
        if market is None:
            market = next((m for m in markets if MAIN_MARKET.search(m.get("question", ""))), markets[0])
        cid = market["conditionId"]
        try:
            details = get(session, CLOB_MARKET.format(cid))
            base.update(event_slug=event.get("slug", ""), event_title=event.get("title", ""),
                        question=market.get("question", ""), condition_id=cid,
                        seconds_delay=details.get("seconds_delay", ""), status="checked")
        except Exception as e:
            base.update(event_slug=event.get("slug", ""), event_title=event.get("title", ""),
                        question=market.get("question", ""), condition_id=cid, status=f"error: {e}")
        rows.append(base)
        print(f"{category:<24} {base['seconds_delay']:>2}s  {base['event_title'][:65]}")
    out = root / "reports" / "current_match_game_delay_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    checked = [r for r in rows if r["status"] == "checked"]
    print(f"\nChecked {len(checked)}/{len(rows)} categories; output={out}")


if __name__ == "__main__":
    main()
