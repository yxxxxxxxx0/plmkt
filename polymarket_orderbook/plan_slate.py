"""
Finds the Polymarket event slugs for an upcoming MLB slate and prints a
ready-to-paste matches.py.

Polymarket slugs carry the **US** date, while you usually think in HKT
(UTC+8). A 19:05 ET game is 07:05 HKT the next morning, so "tomorrow's HKT
slate" is mostly today's US date. Pass --hkt-date to think in HKT and let this
work out the US dates, or --date to name US dates directly.

Every candidate slug is verified against the Gamma API before it is printed,
so a slug that does not exist (postponed game, different naming) is reported
rather than silently written into matches.py and failing at record time.

Usage:
    python plan_slate.py --hkt-date 2026-08-29 2026-08-30
    python plan_slate.py --date 2026-08-28
"""
import argparse
import datetime as dt

import requests

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
HKT = dt.timezone(dt.timedelta(hours=8))


# MLB's abbreviation is not always Polymarket's. Confirmed from real slugs:
# MLB "AZ" -> Polymarket "ari". Others are unverified guesses, so each
# candidate is tested against Gamma and the one that resolves wins.
ALIASES = {
    "az": ["ari", "az"],
    "ath": ["ath", "oak", "las"],
    "cws": ["cws", "chw"],
    "wsh": ["wsh", "was"],
    "sd": ["sd", "sdp"],
    "sf": ["sf", "sfg"],
    "tb": ["tb", "tbr"],
    "kc": ["kc", "kcr"],
}


def candidates(abbr):
    return ALIASES.get(abbr, [abbr])


_ABBR = {}


def team_abbrs():
    """id -> abbreviation. The schedule endpoint omits abbreviations unless
    hydrated, so fetch the team list once and join on team id."""
    if not _ABBR:
        r = requests.get(TEAMS_URL, params={"sportId": 1}, timeout=25)
        r.raise_for_status()
        for t in r.json().get("teams", []):
            if t.get("abbreviation"):
                _ABBR[t["id"]] = t["abbreviation"].lower()
    return _ABBR


def schedule(date):
    r = requests.get(SCHEDULE_URL, params={"sportId": 1, "date": date}, timeout=25)
    r.raise_for_status()
    out = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            abbr = team_abbrs()
            away = g["teams"]["away"]["team"]
            home = g["teams"]["home"]["team"]
            out.append({
                "gamePk": g["gamePk"],
                "away_abbr": abbr.get(away.get("id"), ""),
                "home_abbr": abbr.get(home.get("id"), ""),
                "away_name": away.get("name"),
                "home_name": home.get("name"),
                "gameDate": g.get("gameDate"),
                "status": g.get("status", {}).get("detailedState"),
                "us_date": d.get("date"),
            })
    return out


def gamma_event(slug):
    try:
        rows = requests.get(GAMMA_EVENTS_URL, params={"slug": slug}, timeout=20).json()
    except Exception:
        return None
    return rows[0] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", nargs="*", default=None, help="US date(s) YYYY-MM-DD")
    ap.add_argument("--write", metavar="PATH", default=None,
                    help="write the generated module to PATH (e.g. matches.py) "
                         "instead of only printing it")
    ap.add_argument("--hkt-date", nargs="*", default=None,
                    help="HKT date(s); US dates are derived as HKT-1 and HKT (both scanned)")
    args = ap.parse_args()

    us_dates = set(args.date or [])
    for h in (args.hkt_date or []):
        d = dt.date.fromisoformat(h)
        us_dates.add(str(d - dt.timedelta(days=1)))
        us_dates.add(str(d))
    if not us_dates:
        raise SystemExit("pass --date or --hkt-date")

    hkt_filter = set(args.hkt_date or [])
    rows = []
    for us in sorted(us_dates):
        for g in schedule(us):
            slug = f"mlb-{g['away_abbr']}-{g['home_abbr']}-{g['us_date']}"
            start = dt.datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
            hkt = start.astimezone(HKT)
            if hkt_filter and hkt.date().isoformat() not in hkt_filter:
                continue
            g["slug"] = slug
            g["hkt"] = hkt
            rows.append(g)

    rows.sort(key=lambda g: g["hkt"])
    print(f"{len(rows)} scheduled game(s) in window; verifying slugs against Gamma...\n")

    # A split doubleheader puts two MLB games on one date for one matchup, but
    # Polymarket opens a book for only one of them -- both candidate rows
    # therefore resolve to the SAME slug. Emitting both would subscribe twice
    # and, worse, could pair the slug with the wrong gamePk: --stop-when-final
    # would then watch a game that ends hours before the one being recorded and
    # cut the recording short. So the Gamma event's own gameStartTime decides
    # which scheduled game the slug actually is.
    seen = {}
    good, bad = [], []
    for g in rows:
        ev = None
        for a in candidates(g["away_abbr"]):
            for h in candidates(g["home_abbr"]):
                trial = f"mlb-{a}-{h}-{g['us_date']}"
                ev = gamma_event(trial)
                if ev:
                    g["slug"] = trial
                    break
            if ev:
                break
        if ev:
            gst = next((m.get("gameStartTime") for m in (ev.get("markets") or [])
                        if m.get("gameStartTime")), None)
            if gst:
                want = dt.datetime.fromisoformat(gst.replace("Z", "+00:00").replace("+00", "+00:00"))
                mine = dt.datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
                if abs((want - mine).total_seconds()) > 900:
                    print(f"  [SKIP] {g['slug']} pk={g['gamePk']} starts {mine:%H:%M}Z but the "
                          f"Polymarket book is for {want:%H:%M}Z (doubleheader other game)")
                    continue
            if g["slug"] in seen:
                print(f"  [DUP ] {g['slug']} already matched to pk={seen[g['slug']]}; skipping pk={g['gamePk']}")
                continue
            seen[g["slug"]] = g["gamePk"]
            g["title"] = ev.get("title") or f"{g['away_name']} vs. {g['home_name']}"
            g["n_markets"] = len(ev.get("markets") or [])
            good.append(g)
        else:
            bad.append(g)
        flag = "OK " if ev else "MISS"
        print(f"  [{flag}] {g['hkt']:%Y-%m-%d %H:%M} HKT  {g['slug']:<34}"
              f" {g['away_name']} @ {g['home_name']}"
              + (f"  ({g['n_markets']} markets)" if ev else f"  [{g['status']}]"))

    if bad:
        print(f"\n{len(bad)} slug(s) not found on Gamma -- postponed, not listed yet, "
              f"or named differently. Re-run closer to game time.")

    print("\n" + "=" * 72)
    print("matches.py  (paste over the `matches = [...]` block)")
    print("=" * 72)
    hkt_days = sorted({g["hkt"].date().isoformat() for g in good})
    print(f'"""\nThe MLB games tracked by live_recorder.py.\n')
    print(f'Current slate: {len(good)} games, HKT date(s) {", ".join(hkt_days)}.')
    print('Polymarket slugs carry the US date.\n"""\n')
    lines = ['"""', "The MLB games tracked by live_recorder.py.", "",
             f"Current slate: {len(good)} games, HKT date(s) {', '.join(hkt_days)}.",
             "Polymarket slugs carry the US date.",
             "", "Generated by plan_slate.py -- do not hand-edit; regenerate instead.",
             '"""', "", "matches = ["]
    for g in good:
        lines += ["    {",
                  f'        "time_hkt": "{g["hkt"]:%H:%M}",',
                  f'        "match": "{g["title"]}",',
                  f'        "slug": "{g["slug"]}",',
                  # gamePk lets live_recorder --stop-when-final ask the MLB API
                  # whether the games have actually ended, instead of guessing
                  # with a fixed timer.
                  f'        "gamePk": {g["gamePk"]},',
                  "    },"]
    lines.append("]")
    body = chr(10).join(lines) + chr(10)

    print(body)
    if args.write:
        if not good:
            raise SystemExit("refusing to write: zero verified slugs "
                             "(markets may not be listed yet -- retry closer to first pitch)")
        with open(args.write, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"[plan_slate] wrote {len(good)} games -> {args.write}")


if __name__ == "__main__":
    main()
