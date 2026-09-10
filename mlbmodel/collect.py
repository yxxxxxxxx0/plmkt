"""Pull MLB game, team and pitcher data from the free MLB Stats API.

statsapi.mlb.com needs no key and no auth. Every response is cached to
data/raw so re-runs are offline and the API is hit once per resource.

Three tables come out of this module:
  games.parquet          one row per regular-season game (result + probable SPs)
  team_logs.parquet      one row per team-game, hitting and pitching box lines
  pitcher_logs.parquet   one row per pitcher-game for anyone who started a game
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .config import API, RAW, PROC, SEASONS

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-research/1.0"})


def _get(path: str, params: dict, cache_key: str, ttl_days: float | None = None) -> dict:
    """GET with an on-disk JSON cache. ttl_days=None means cache forever."""
    fp = RAW / f"{cache_key}.json"
    if fp.exists():
        if ttl_days is None or (time.time() - fp.stat().st_mtime) < ttl_days * 86400:
            return json.loads(fp.read_text(encoding="utf-8"))
    for attempt in range(5):
        try:
            r = SESSION.get(API + path, params=params, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    fp.write_text(json.dumps(data), encoding="utf-8")
    return data


def _ip_to_outs(ip) -> float:
    """MLB reports innings pitched as '6.1' meaning 6 innings + 1 out."""
    if ip is None:
        return 0.0
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole) * 3 + int(frac or 0)
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- #
# 1. Schedule / results / probable starters
# --------------------------------------------------------------------------- #
def fetch_games(seasons=SEASONS) -> pd.DataFrame:
    rows = []
    for yr in seasons:
        # Recent seasons may still be in progress -> short TTL so results refresh.
        ttl = 0.5 if yr >= 2026 else None
        j = _get(
            "/schedule",
            dict(
                sportId=1,
                startDate=f"{yr}-01-01",
                endDate=f"{yr}-12-31",
                gameType="R",
                hydrate="probablePitcher,linescore,team,venue",
            ),
            f"schedule_{yr}",
            ttl_days=ttl,
        )
        for date in j.get("dates", []):
            for g in date.get("games", []):
                if g.get("status", {}).get("codedGameState") not in ("F", "O"):
                    continue  # not a completed game
                h, a = g["teams"]["home"], g["teams"]["away"]
                if h.get("score") is None or a.get("score") is None:
                    continue
                ls = g.get("linescore", {})
                rows.append(
                    dict(
                        game_pk=g["gamePk"],
                        date=g["officialDate"],
                        season=int(g["season"]),
                        game_num=g.get("gameNumber", 1),
                        day_night=g.get("dayNight", "day"),
                        venue_id=g.get("venue", {}).get("id"),
                        home_id=h["team"]["id"],
                        away_id=a["team"]["id"],
                        home_abbr=h["team"].get("abbreviation"),
                        away_abbr=a["team"].get("abbreviation"),
                        home_score=h["score"],
                        away_score=a["score"],
                        innings=ls.get("scheduledInnings", 9),
                        home_sp=(h.get("probablePitcher") or {}).get("id"),
                        away_sp=(a.get("probablePitcher") or {}).get("id"),
                    )
                )
        print(f"  schedule {yr}: {len(rows)} cumulative games")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)
    df["home_win"] = (df.home_score > df.away_score).astype(int)
    df.to_parquet(PROC / "games.parquet", index=False)
    return df


# --------------------------------------------------------------------------- #
# 2. Team-game box lines (hitting + pitching)
# --------------------------------------------------------------------------- #
HIT_COLS = ["runs", "hits", "homeRuns", "baseOnBalls", "strikeOuts", "atBats",
            "plateAppearances", "totalBases", "doubles", "triples", "hitByPitch",
            "stolenBases", "leftOnBase", "sacFlies"]
PIT_COLS = ["runs", "earnedRuns", "hits", "homeRuns", "baseOnBalls", "strikeOuts",
            "battersFaced", "outs", "numberOfPitches", "hitBatsmen"]


def fetch_team_logs(seasons=SEASONS, team_ids=None) -> pd.DataFrame:
    if team_ids is None:
        j = _get("/teams", dict(sportId=1, season=2024), "teams_2024")
        team_ids = sorted({t["id"] for t in j["teams"]})
        # A few franchises changed ids/names over the window; union across seasons.
        for yr in seasons:
            jj = _get("/teams", dict(sportId=1, season=yr), f"teams_{yr}")
            team_ids = sorted(set(team_ids) | {t["id"] for t in jj["teams"]})

    rows = {}
    for yr in seasons:
        ttl = 0.5 if yr >= 2026 else None
        for tid in team_ids:
            for group, cols in (("hitting", HIT_COLS), ("pitching", PIT_COLS)):
                j = _get(
                    f"/teams/{tid}/stats",
                    dict(stats="gameLog", group=group, season=yr, sportId=1, gameType="R"),
                    f"teamlog_{group}_{tid}_{yr}",
                    ttl_days=ttl,
                )
                for st in j.get("stats", []):
                    for sp in st.get("splits", []):
                        pk = sp.get("game", {}).get("gamePk")
                        if pk is None:
                            continue
                        key = (pk, tid)
                        rec = rows.setdefault(
                            key,
                            dict(game_pk=pk, team_id=tid, season=yr,
                                 date=sp.get("date"), is_home=sp.get("isHome"),
                                 opp_id=sp.get("opponent", {}).get("id")),
                        )
                        s = sp["stat"]
                        pre = "off_" if group == "hitting" else "def_"
                        for c in cols:
                            rec[pre + c] = s.get(c, 0)
                        if group == "pitching":
                            rec["def_outs"] = s.get("outs", 0) or _ip_to_outs(s.get("inningsPitched"))
        print(f"  team logs {yr}: {len(rows)} team-games")

    df = pd.DataFrame(list(rows.values()))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk", "team_id"]).reset_index(drop=True)
    df.to_parquet(PROC / "team_logs.parquet", index=False)
    return df


# --------------------------------------------------------------------------- #
# 3. Starting-pitcher game logs
# --------------------------------------------------------------------------- #
SP_KEEP = ["gamesStarted", "outs", "runs", "earnedRuns", "hits", "homeRuns",
           "baseOnBalls", "strikeOuts", "battersFaced", "numberOfPitches",
           "hitBatsmen", "intentionalWalks"]


def fetch_pitcher_logs(games: pd.DataFrame, seasons=SEASONS, batch=40) -> pd.DataFrame:
    """Game logs for every pitcher who was ever listed as a probable starter."""
    rows = []
    for yr in seasons:
        gy = games[games.season == yr]
        pids = sorted({int(p) for p in
                       pd.concat([gy.home_sp, gy.away_sp]).dropna().unique()})
        if not pids:
            continue
        ttl = 0.5 if yr >= 2026 else None
        for i in range(0, len(pids), batch):
            chunk = pids[i:i + batch]
            j = _get(
                "/people",
                dict(personIds=",".join(map(str, chunk)),
                     hydrate=f"stats(group=[pitching],type=[gameLog],season={yr},gameType=[R])"),
                f"splog_{yr}_{i//batch}",
                ttl_days=ttl,
            )
            for p in j.get("people", []):
                for st in p.get("stats", []):
                    for sp in st.get("splits", []):
                        if sp.get("gameType") not in (None, "R"):
                            continue
                        s = sp["stat"]
                        rec = dict(
                            pitcher_id=p["id"], season=yr,
                            game_pk=sp.get("game", {}).get("gamePk"),
                            date=sp.get("date"), is_home=sp.get("isHome"),
                            team_id=sp.get("team", {}).get("id"),
                            opp_id=sp.get("opponent", {}).get("id"),
                        )
                        for c in SP_KEEP:
                            rec[c] = s.get(c, 0)
                        rec["outs"] = s.get("outs", 0) or _ip_to_outs(s.get("inningsPitched"))
                        rows.append(rec)
        print(f"  pitcher logs {yr}: {len(pids)} pitchers, {len(rows)} cumulative rows")

    df = pd.DataFrame(rows).dropna(subset=["game_pk"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["pitcher_id", "game_pk"])
    df = df.sort_values(["pitcher_id", "date"]).reset_index(drop=True)
    df.to_parquet(PROC / "pitcher_logs.parquet", index=False)
    return df


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{name}.parquet")


def main():
    print("Fetching schedules...")
    games = fetch_games()
    print(f"games: {len(games)}  {games.date.min().date()} -> {games.date.max().date()}")
    print("Fetching team game logs...")
    tl = fetch_team_logs()
    print(f"team logs: {len(tl)}")
    print("Fetching pitcher game logs...")
    pl = fetch_pitcher_logs(games)
    print(f"pitcher logs: {len(pl)}")


if __name__ == "__main__":
    main()
