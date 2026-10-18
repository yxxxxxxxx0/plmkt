"""
Finds games listed on BOTH Kalshi and Polymarket, and works out what to
record for each: the Kalshi moneyline event, the Polymarket event slug, the
real kickoff time and (for MLB) the gamePk. Only moneylines are recorded.

Matching, strongest evidence first:

  1. Team codes. Kalshi tickers end in the two team codes
     (KXEPLGAME-26OCT18BOUSUN -> BOUSUN); Polymarket slugs carry them too
     (epl-bou-sun-2026-10-18). Equal codes + kickoff within 6 h = match.
  2. Fuzzy team names + kickoff time, for leagues where the two venues
     abbreviate differently (Kalshi "Santander" vs Polymarket "Real Racing
     Club"). Ambiguous or weak matches are reported, never guessed.

Kickoff time comes from Polymarket's startTime, then the MLB API. Kalshi's
`occurrence_datetime` is ~3 h AFTER kickoff for football and can be stale for
rescheduled MLB games (ALCS G5 2026 kept a postponed game's time), so it is
only a fallback and is flagged when used.

Usage:
    python pairing.py --hkt-date 2026-10-19             # print the day's plan
    python pairing.py --hours 24 --json plan.json       # next 24 h, as JSON
    python pairing.py --hkt-date 2026-10-24 --leagues epl ucl
"""
import argparse
import datetime as dt
import difflib
import json
import re
import time
import unicodedata

import requests

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
HEADERS = {"User-Agent": "Mozilla/5.0 (orderbook-research)"}   # Gamma 403s python's default UA
HKT = dt.timezone(dt.timedelta(hours=8))
UTC = dt.timezone.utc

# Kalshi game series -> Polymarket sport code (from GET /sports). Verified
# 2026-10-18; Scottish Premiership had no overlapping games and is left out.
FOOTBALL = {
    "epl": "KXEPLGAME", "lal": "KXLALIGAGAME", "sea": "KXSERIEAGAME",
    "bun": "KXBUNDESLIGAGAME", "fl1": "KXLIGUE1GAME", "ucl": "KXUCLGAME",
    "uel": "KXUELGAME", "mls": "KXMLSGAME", "ere": "KXEREDIVISIEGAME",
    "por": "KXLIGAPORTUGALGAME", "elc": "KXEFLCHAMPIONSHIPGAME",
    "mex": "KXLIGAMXGAME", "bra": "KXBRASILEIROGAME",
}
MLB_GAME_SERIES = "KXMLBGAME"   # moneyline only; side markets are not recorded

FOOTBALL_KICKOFF_LAG = dt.timedelta(hours=3)   # occurrence_datetime - kickoff, observed

# KXEPLGAME-26OCT18BOUSUN / KXMLBGAME-26OCT172000TBCLE -> codes
_TICKER = re.compile(r"-\d{2}[A-Z]{3}\d{2}(?:\d{4})?([A-Z0-9]+)$")
_STOP = {"fc", "afc", "cf", "sc", "ac", "as", "ss", "us", "cd", "ud", "rc", "sd", "fk", "sk",
         "tc", "bv", "sv", "vfb", "vfl", "tsg", "rsc", "club", "de", "del", "la", "le", "the",
         "calcio", "futbol", "football", "sporting", "cp", "ec", "fbpa", "sfp", "ogc", "osc"}


def _get(url, **params):
    for wait in (2, 4, 8, 16):           # Kalshi 429s while the recorders poll
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 429:
            break
        time.sleep(wait)
    r.raise_for_status()
    return r.json()


def _ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", " ", s)
    toks = [t for t in re.findall(r"[a-z]+", s) if t not in _STOP and len(t) > 1]
    return " ".join(toks)


def name_sim(a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    if ta & {t for t in tb if len(t) >= 4}:
        return 0.9
    return difflib.SequenceMatcher(None, a, b).ratio()


def split_teams(title):
    title = re.sub(r"^Game \d+:\s*", "", title or "")
    parts = re.split(r"\s+vs\.?\s+", title, maxsplit=1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else None


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def kalshi_events(series):
    out, cursor = [], None
    while True:
        p = {"series_ticker": series, "status": "open", "limit": 200, "with_nested_markets": "true"}
        if cursor:
            p["cursor"] = cursor
        d = _get(f"{KALSHI}/events", **p)
        out += d.get("events", [])
        cursor = d.get("cursor")
        if not cursor or not d.get("events"):
            return out


def kalshi_codes(event_ticker):
    m = _TICKER.search(event_ticker)
    return m.group(1) if m else None


def kalshi_occurrence(ev):
    ts = [_ts(m.get("occurrence_datetime")) for m in ev.get("markets", []) if m.get("occurrence_datetime")]
    return min(ts) if ts else None


_SPORTS = {}


def pm_main_events(sport):
    """Polymarket's head-to-head event for each game. Side events
    ('... - Halftime Result', '... - More Markets') are skipped."""
    if not _SPORTS:
        for s in _get(f"{GAMMA}/sports"):
            _SPORTS[s["sport"]] = s
    s = _SPORTS.get(sport)
    if not s:
        return []
    # Gamma silently caps `limit` at 100, so page until an empty page rather
    # than until a short one.
    out, offset = [], 0
    for _ in range(100):
        rows = _get(f"{GAMMA}/events", series_id=s["series"], closed="false",
                    limit=100, offset=offset)
        if not rows:
            break
        for e in rows:
            if split_teams(e.get("title")) and " - " not in e["title"] and e.get("startTime"):
                out.append(e)
        offset += len(rows)
    return out


def pm_codes(slug):
    m = re.match(r"^[a-z0-9]+-([a-z0-9]+)-([a-z0-9]+)-\d{4}-\d{2}-\d{2}", slug or "")
    return (m.group(1) + m.group(2)).upper() if m else None


def mlb_games(start, end):
    d = _get(MLB_SCHEDULE, sportId=1, startDate=(start - dt.timedelta(days=1)).date().isoformat(),
             endDate=(end + dt.timedelta(days=1)).date().isoformat())
    return [g for day in d.get("dates", []) for g in day.get("games", [])]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_pm(kev, pms, approx_kickoff, used):
    """Best Polymarket event for a Kalshi event, or (None, reason)."""
    teams = split_teams(kev.get("title"))
    codes = kalshi_codes(kev["event_ticker"])
    window = dt.timedelta(hours=6)
    cands = []
    for e in pms:
        if e["slug"] in used:
            continue
        t = _ts(e["startTime"])
        if approx_kickoff and abs(t - approx_kickoff) > window:
            continue
        if codes and pm_codes(e["slug"]) == codes:
            cands.append((3.0, e))
            continue
        pt = split_teams(e["title"])
        if not (teams and pt):
            continue
        s = name_sim(teams[0], pt[0]) + name_sim(teams[1], pt[1])
        near = approx_kickoff and abs(t - approx_kickoff) <= dt.timedelta(minutes=20)
        if s >= 1.5 or (near and s >= 1.2):
            cands.append((s, e))
    if not cands:
        return None, "no Polymarket event"
    cands.sort(key=lambda x: -x[0])
    if len(cands) > 1 and cands[1][0] >= cands[0][0] - 0.05 and cands[0][0] < 3.0:
        return None, f"ambiguous: {cands[0][1]['slug']} / {cands[1][1]['slug']}"
    return cands[0][1], None


def match_mlb_game(teams, kickoff, games):
    best = None
    for g in games:
        away = g["teams"]["away"]["team"]["name"]
        home = g["teams"]["home"]["team"]["name"]
        s = name_sim(teams[0], away) + name_sim(teams[1], home)
        if s < 1.5:
            continue
        d = abs(_ts(g["gameDate"]) - kickoff)
        if d > dt.timedelta(hours=3):
            continue
        if best is None or d < best[0]:
            best = (d, g)
    return best[1] if best else None


def plan(start, end, leagues=None, mlb=True, verbose=True):
    """Games kicking off in [start, end) listed on both venues."""
    games, report = [], []
    leagues = leagues or list(FOOTBALL)

    for sport in leagues:
        series = FOOTBALL[sport]
        try:
            kev, pms = kalshi_events(series), pm_main_events(sport)
        except Exception as e:
            report.append(f"{sport}: fetch failed: {e}")
            continue
        used = set()
        for ev in kev:
            occ = kalshi_occurrence(ev)
            approx = occ - FOOTBALL_KICKOFF_LAG if occ else None
            if approx and not (start - dt.timedelta(hours=6) <= approx < end + dt.timedelta(hours=6)):
                continue
            pm, why = match_pm(ev, pms, approx, used)
            if pm is None:
                if approx and start <= approx < end:
                    report.append(f"{sport}: unpaired {ev['event_ticker']} ({ev.get('title')}): {why}")
                continue
            kickoff = _ts(pm["startTime"])
            if not (start <= kickoff < end):
                continue
            used.add(pm["slug"])
            games.append({
                "sport": "football", "league": sport, "title": pm["title"],
                "kickoff": kickoff.isoformat(), "kickoff_source": "polymarket",
                "kalshi_events": [ev["event_ticker"]], "pm_slug": pm["slug"],
                "game_pk": None,
                "kalshi_kickoff_lag_min": round((occ - kickoff).total_seconds() / 60) if occ else None,
            })

    if mlb:
        try:
            kev = kalshi_events(MLB_GAME_SERIES)
            pms = pm_main_events("mlb")
            sched = mlb_games(start, end)
        except Exception as e:
            report.append(f"mlb: fetch failed: {e}")
            kev = []
        used = set()
        for ev in kev:
            teams = split_teams(ev.get("title"))
            occ = kalshi_occurrence(ev)
            pm, why = match_pm(ev, pms, None, used)   # MLB occurrence can be stale: match on codes/names
            kickoff = _ts(pm["startTime"]) if pm else None
            src = "polymarket"
            g = match_mlb_game(teams, kickoff or occ, sched) if teams and (kickoff or occ) else None
            if g and not kickoff:
                kickoff, src = _ts(g["gameDate"]), "mlb_api"
            if pm is None:
                if kickoff and start <= kickoff < end:
                    report.append(f"mlb: unpaired {ev['event_ticker']} ({ev.get('title')}): {why}")
                continue
            if not (start <= kickoff < end):
                continue
            used.add(pm["slug"])
            games.append({
                "sport": "mlb", "league": "mlb", "title": pm["title"],
                "kickoff": kickoff.isoformat(), "kickoff_source": src,
                "kalshi_events": [ev["event_ticker"]], "pm_slug": pm["slug"],
                "game_pk": g["gamePk"] if g else None,
                "kalshi_kickoff_lag_min": round((occ - kickoff).total_seconds() / 60) if occ else None,
            })
            if not g:
                report.append(f"mlb: no gamePk for {pm['slug']}; it will stop on --duration only")

    games.sort(key=lambda g: g["kickoff"])
    if verbose:
        for r in report:
            print(f"[pairing] {r}")
    return games, report


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hkt-date", help="games kicking off on this HKT calendar day")
    g.add_argument("--hours", type=float, help="games kicking off in the next N hours")
    ap.add_argument("--leagues", nargs="*", default=None, help=f"subset of {list(FOOTBALL)}")
    ap.add_argument("--no-mlb", action="store_true")
    ap.add_argument("--json", help="write the plan here")
    a = ap.parse_args()
    if a.hkt_date:
        start = dt.datetime.fromisoformat(a.hkt_date).replace(tzinfo=HKT)
        end = start + dt.timedelta(days=1)
    else:
        start = dt.datetime.now(UTC)
        end = start + dt.timedelta(hours=a.hours or 24)
    games, _ = plan(start, end, a.leagues, mlb=not a.no_mlb)
    for x in games:
        k = _ts(x["kickoff"]).astimezone(HKT)
        print(f"{k:%m-%d %H:%M} HKT  {x['league']:<4} {x['title']:<55} "
              f"{x['pm_slug']:<34} {len(x['kalshi_events'])} kalshi ev"
              + (f"  gamePk {x['game_pk']}" if x["game_pk"] else "")
              + (f"  lag {x['kalshi_kickoff_lag_min']}m" if x["kalshi_kickoff_lag_min"] is not None else ""))
    print(f"{len(games)} games on both venues")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(games, f, indent=1)


if __name__ == "__main__":
    main()
