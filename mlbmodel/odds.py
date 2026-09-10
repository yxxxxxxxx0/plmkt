"""Free historical MLB moneylines from Action Network's public scoreboard API.

`api.actionnetwork.com/web/v1/scoreboard/mlb?date=YYYYMMDD` is an open,
key-less JSON endpoint that returns, for every game on a date, the last
odds snapshot from ~20 books plus public bet/handle percentages. The
snapshot timestamp sits at or just after first pitch, so it is effectively
the closing line -- which is what a model should be measured against.

One request per calendar date keeps this comfortably "mid-frequency":
a full season backfill is ~190 calls, and a daily update is one call.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from .config import RAW, PROC

AN = "https://api.actionnetwork.com/web/v1/scoreboard/mlb"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "application/json",
})

# Books ranked by how sharp / how widely available they are. The first one
# present for a game wins; book 15 is Action Network's consensus line.
BOOK_PREFERENCE = [15, 30, 68, 69, 75, 79, 123, 76, 71, 972, 264, 65]

# Action Network franchise names -> MLB Stats API names where they differ.
NAME_ALIASES = {
    "Athletics": "Athletics",
    "Oakland Athletics": "Athletics",
    "Sacramento Athletics": "Athletics",
    "Cleveland Indians": "Cleveland Guardians",
}


def american_to_prob(ml):
    # np.where evaluates both branches, so clamp away from the poles first.
    ml = np.asarray(ml, dtype=float)
    neg = np.where(ml < 0, ml, -100.0)
    pos = np.where(ml >= 0, ml, 100.0)
    return np.where(ml < 0, -neg / (-neg + 100.0), 100.0 / (pos + 100.0))


def american_to_decimal(ml):
    ml = np.asarray(ml, dtype=float)
    neg = np.where(ml < 0, ml, -100.0)
    pos = np.where(ml >= 0, ml, 100.0)
    return np.where(ml < 0, 1.0 + 100.0 / -neg, 1.0 + pos / 100.0)


def _fetch_date(datestr: str, refresh_days: float | None = None) -> dict:
    fp = RAW / f"an_{datestr}.json"
    if fp.exists() and (refresh_days is None or
                        (time.time() - fp.stat().st_mtime) < refresh_days * 86400):
        return json.loads(fp.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            r = SESSION.get(AN, params={"date": datestr}, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt == 3:
                return {"games": []}
            time.sleep(2 * (attempt + 1))
    fp.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_odds(dates) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(dates):
        ds = pd.Timestamp(d).strftime("%Y%m%d")
        j = _fetch_date(ds)
        for g in j.get("games", []):
            teams = {t["id"]: t for t in g.get("teams", [])}
            h = teams.get(g.get("home_team_id"), {})
            a = teams.get(g.get("away_team_id"), {})
            odds = [o for o in (g.get("odds") or []) if o.get("type") == "game"
                    and o.get("ml_home") is not None]
            if not odds or not h or not a:
                continue
            by_book = {}
            for o in odds:
                by_book.setdefault(o["book_id"], o)
            pick = next((by_book[b] for b in BOOK_PREFERENCE if b in by_book), None)
            if pick is None:
                pick = list(by_book.values())[0]
            # public betting split, taken from whichever snapshot carries it
            pub = next((o for o in odds if o.get("ml_home_public") is not None), {})
            rows.append(dict(
                date=pd.Timestamp(g["start_time"]).tz_convert("UTC").normalize().tz_localize(None),
                an_date=pd.Timestamp(ds),
                home_name=NAME_ALIASES.get(h.get("full_name"), h.get("full_name")),
                away_name=NAME_ALIASES.get(a.get("full_name"), a.get("full_name")),
                ml_home=pick["ml_home"], ml_away=pick["ml_away"],
                total=pick.get("total"), book_id=pick["book_id"],
                n_books=len(by_book),
                pub_home=pub.get("ml_home_public"),
                pub_home_money=pub.get("ml_home_money"),
                num_bets=g.get("num_bets"),
                # kept so the "is this really the close?" check in
                # reports/ can be reproduced -- see snapshot_lag() below
                snap_ts=pick.get("inserted"),
                start_ts=g.get("start_time"),
            ))
        if (i + 1) % 100 == 0:
            print(f"  odds {ds}: {len(rows)} rows")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_home_raw"] = american_to_prob(df.ml_home)
    df["p_away_raw"] = american_to_prob(df.ml_away)
    ov = df.p_home_raw + df.p_away_raw
    df["mkt_vig"] = ov - 1.0
    df["mkt_p_home"] = df.p_home_raw / ov          # proportional de-vig
    df["dec_home"] = american_to_decimal(df.ml_home)
    df["dec_away"] = american_to_decimal(df.ml_away)
    df.to_parquet(PROC / "odds.parquet", index=False)
    return df


def snapshot_lag(odds: pd.DataFrame) -> pd.Series:
    """Minutes between first pitch and the odds snapshot we treat as the close.

    Action Network stamps the final pre-game line a few minutes either side of
    first pitch. Anything materially positive would mean live in-game prices
    are contaminating the "closing" line, so this is worth checking before
    reporting closing-line value.
    """
    o = odds.dropna(subset=["snap_ts", "start_ts"])
    lag = (pd.to_datetime(o.snap_ts, format="mixed", utc=True)
           - pd.to_datetime(o.start_ts, format="mixed", utc=True)).dt.total_seconds() / 60
    return lag.describe(percentiles=[.05, .25, .5, .75, .95])


def attach_odds(games: pd.DataFrame, odds: pd.DataFrame,
                team_names: dict[int, str]) -> pd.DataFrame:
    """Join odds onto games by (date, home team, away team), tolerating the
    UTC/local date boundary by also trying the calendar date used by AN."""
    g = games.copy()
    g["home_name"] = g.home_id.map(team_names)
    g["away_name"] = g.away_id.map(team_names)
    o = odds.drop_duplicates(["date", "home_name", "away_name"])
    o = o.copy()
    o["snap_lag_min"] = (
        pd.to_datetime(o.snap_ts, format="mixed", utc=True)
        - pd.to_datetime(o.start_ts, format="mixed", utc=True)).dt.total_seconds() / 60
    keep = ["mkt_p_home", "ml_home", "ml_away", "dec_home", "dec_away",
            "mkt_vig", "pub_home", "pub_home_money", "total", "n_books",
            "snap_lag_min"]

    merged = g.merge(o[["date", "home_name", "away_name"] + keep],
                     on=["date", "home_name", "away_name"], how="left")
    miss = merged.mkt_p_home.isna()
    if miss.any():
        o2 = (o.drop(columns=["date"])
               .drop_duplicates(["an_date", "home_name", "away_name"])
               .rename(columns={"an_date": "date"}))
        fill = merged.loc[miss, ["date", "home_name", "away_name"]].merge(
            o2[["date", "home_name", "away_name"] + keep],
            on=["date", "home_name", "away_name"], how="left")
        for c in keep:
            merged.loc[miss, c] = fill[c].values
    return merged


def main(from_season: int = 2022):
    games = pd.read_parquet(PROC / "games.parquet")
    games = games[games.season >= from_season]
    dates = pd.date_range(games.date.min(), games.date.max(), freq="D")
    # only days that actually had games -> ~1/3 fewer calls
    have = set(games.date.dt.normalize())
    dates = [d for d in dates if d in have or (d - pd.Timedelta(days=1)) in have]
    print(f"fetching odds for {len(dates)} dates")
    df = fetch_odds(dates)
    print(f"odds rows: {len(df)}  {df.date.min()} -> {df.date.max()}")


if __name__ == "__main__":
    main()
