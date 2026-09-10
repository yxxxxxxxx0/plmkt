"""Score an upcoming slate and compare the model against Polymarket.

The trick that keeps this consistent with the backtest: an unplayed game is
appended to the history as a *placeholder row* with no statistics. Because
every feature is built with `shift(1)` before its window, the placeholder
picks up exactly the pre-game features it would have had in the backtest,
with no special-casing and therefore no chance of the two paths diverging.

    python -m mlbmodel.predict                 # today
    python -m mlbmodel.predict --date 2026-09-09
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import requests

from . import collect, features
from .advanced import OffsetGBM
from .config import API, OUT, TEST_SEASONS
from .models import ShrunkLogit

GAMMA = "https://gamma-api.polymarket.com"


# --------------------------------------------------------------------------- #
def upcoming(date: str) -> pd.DataFrame:
    r = requests.get(f"{API}/schedule", params=dict(
        sportId=1, startDate=date, endDate=date, gameType="R",
        hydrate="probablePitcher,team,venue"), timeout=60)
    r.raise_for_status()
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            h, a = g["teams"]["home"], g["teams"]["away"]
            rows.append(dict(
                game_pk=g["gamePk"], date=pd.Timestamp(g["officialDate"]),
                season=int(g["season"]), game_num=g.get("gameNumber", 1),
                day_night=g.get("dayNight", "night"),
                venue_id=g.get("venue", {}).get("id"),
                home_id=h["team"]["id"], away_id=a["team"]["id"],
                home_abbr=h["team"]["abbreviation"], away_abbr=a["team"]["abbreviation"],
                home_name=h["team"]["name"], away_name=a["team"]["name"],
                home_score=np.nan, away_score=np.nan, innings=9,
                home_sp=(h.get("probablePitcher") or {}).get("id"),
                away_sp=(a.get("probablePitcher") or {}).get("id"),
                home_sp_name=(h.get("probablePitcher") or {}).get("fullName"),
                away_sp_name=(a.get("probablePitcher") or {}).get("fullName"),
                home_win=np.nan,
                start_time=g.get("gameDate"),
            ))
    return pd.DataFrame(rows)


def _placeholders(slate: pd.DataFrame, team_logs: pd.DataFrame,
                  pitcher_logs: pd.DataFrame):
    """Empty stat rows so the rolling machinery treats the slate as 'next game'."""
    tl_rows = []
    for _, g in slate.iterrows():
        for side, opp in (("home", "away"), ("away", "home")):
            r = {c: 0.0 for c in team_logs.columns}
            r.update(game_pk=g.game_pk, team_id=g[f"{side}_id"], season=g.season,
                     date=g.date, is_home=(side == "home"), opp_id=g[f"{opp}_id"])
            tl_rows.append(r)
    tl = pd.concat([team_logs, pd.DataFrame(tl_rows)], ignore_index=True)

    pl_rows = []
    for _, g in slate.iterrows():
        for side in ("home", "away"):
            pid = g[f"{side}_sp"]
            if pd.isna(pid):
                continue
            r = {c: 0.0 for c in pitcher_logs.columns}
            r.update(pitcher_id=int(pid), season=g.season, game_pk=g.game_pk,
                     date=g.date, is_home=(side == "home"),
                     team_id=g[f"{side}_id"], gamesStarted=1)
            pl_rows.append(r)
    pl = pd.concat([pitcher_logs, pd.DataFrame(pl_rows)], ignore_index=True)
    return tl, pl


# --------------------------------------------------------------------------- #
def _poly_moneylines() -> dict[tuple[str, str, str], float]:
    """Every open MLB moneyline on Polymarket, keyed by (date, away, home).

    Polymarket's game events use the slug `mlb-{away}-{home}-{date}` and their
    moneyline market lists outcomes as [away team, home team] under the same
    full franchise names the MLB Stats API uses. Matching on that exact pair
    is reliable; fuzzy title matching is not.

    Only the `sports` tag ordered by volume surfaces game events -- the `mlb`
    tag returns season futures, and unordered paging never reaches them.
    """
    import json as _json

    out = {}
    for off in range(0, 1200, 100):
        try:
            r = requests.get(f"{GAMMA}/events", params={
                "tag_slug": "sports", "closed": "false", "limit": 100,
                "offset": off, "order": "volume24hr", "ascending": "false"},
                timeout=45)
            r.raise_for_status()
            events = r.json()
        except Exception as e:
            print(f"  (polymarket unavailable: {e})")
            break
        if not events:
            break
        for ev in events:
            slug = str(ev.get("slug", ""))
            if not slug.startswith("mlb-"):
                continue
            for m in ev.get("markets", []):
                if m.get("sportsMarketType") != "moneyline":
                    continue
                oc, pr = m.get("outcomes"), m.get("outcomePrices")
                if isinstance(oc, str):
                    oc, pr = _json.loads(oc), _json.loads(pr)
                if not oc or len(oc) != 2:
                    continue
                p_home = float(pr[1])
                # 0/1 means the game has already resolved -- not a live price
                if not (0.0 < p_home < 1.0):
                    continue
                date = str(m.get("gameStartTime", ""))[:10]
                out[(date, oc[0], oc[1])] = p_home
    return out


def polymarket_mlb(slate: pd.DataFrame) -> dict[int, float]:
    """Map {game_pk: Polymarket implied P(home win)} for the slate."""
    book = _poly_moneylines()
    if not book:
        return {}
    # Polymarket stamps game start in UTC, so a night game lands on the next
    # calendar day; try the game's own date and the day after.
    out = {}
    for _, g in slate.iterrows():
        for d in (g.date, g.date + pd.Timedelta(days=1)):
            k = (d.strftime("%Y-%m-%d"), g.away_name, g.home_name)
            if k in book:
                out[g.game_pk] = book[k]
                break
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--model", default="gbm_elo_offset",
                    choices=["gbm_elo_offset", "logit_l2", "elo_sp"])
    ap.add_argument("--edge", type=float, default=0.03)
    a = ap.parse_args()

    slate = upcoming(a.date)
    if slate.empty:
        print(f"no MLB games on {a.date}")
        return
    print(f"{len(slate)} games on {a.date}")

    games = collect.load("games")
    tl, pl = _placeholders(slate, collect.load("team_logs"), collect.load("pitcher_logs"))
    allg = pd.concat([games, slate.drop(columns=[c for c in slate.columns
                                                 if c not in games.columns])],
                     ignore_index=True)
    allg = allg.sort_values(["date", "game_pk"]).reset_index(drop=True)

    ds = features.build_dataset(allg, tl, pl, save=False)
    feat = features.feature_columns(ds)
    today = ds[ds.game_pk.isin(slate.game_pk)].copy()
    hist = ds[ds.home_win.notna() & ds.has_both_sp & (ds.season != 2020)]

    if a.model == "elo_sp":
        today["p_home"] = today.elo_sp_prob_home
    elif a.model == "logit_l2":
        m = ShrunkLogit(C=0.03, shrink=0.92, l1_ratio=0.0)
        m.fit(hist[feat].to_numpy(dtype=float), hist.home_win.to_numpy())
        today["p_home"] = m.predict_proba(today[feat].to_numpy(dtype=float))
    else:
        m = OffsetGBM().fit_frame(hist, feat, hist.home_win.to_numpy())
        today["p_home"] = m.predict_frame(today, feat)

    today = today.merge(slate[["game_pk", "home_abbr", "away_abbr", "home_sp_name",
                               "away_sp_name", "start_time"]].rename(
        columns={"home_abbr": "H", "away_abbr": "A"}), on="game_pk", how="left")

    pm = polymarket_mlb(slate)
    today["poly_p_home"] = today.game_pk.map(pm)
    print(f"matched {today.poly_p_home.notna().sum()}/{len(today)} "
          f"games to a live Polymarket moneyline")
    today["edge_home"] = today.p_home - today.poly_p_home
    # EV per $1 staked at the quoted price (Polymarket pays $1 per winning share)
    today["ev_home"] = today.p_home / today.poly_p_home - 1
    today["ev_away"] = (1 - today.p_home) / (1 - today.poly_p_home) - 1

    cols = ["A", "H", "away_sp_name", "home_sp_name", "elo_sp_prob_home",
            "p_home", "poly_p_home", "ev_home", "ev_away"]
    print(f"\n=== {a.date} -- model: {a.model} ===")
    print(today[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    picks = today[(today.ev_home > a.edge) | (today.ev_away > a.edge)]
    if len(picks):
        print(f"\n=== EV > {a.edge:.0%} vs Polymarket ===")
        print("  NOTE: the walk-forward backtest found no significant edge over "
              "the market\n  (see reports/FINDINGS.md). Treat these as model "
              "disagreements, not signals.")
        for _, r in picks.iterrows():
            side, ev = (("HOME " + str(r.H), r.ev_home) if r.ev_home > r.ev_away
                        else ("AWAY " + str(r.A), r.ev_away))
            print(f"  {r.A} @ {r.H}: {side}  model={r.p_home:.3f} "
                  f"poly={r.poly_p_home:.3f}  EV={ev:+.1%}")
    elif today.poly_p_home.notna().any():
        print(f"\nno game clears the {a.edge:.0%} EV threshold")

    today[cols + ["game_pk", "date"]].to_csv(OUT / f"slate_{a.date}.csv", index=False)
    print(f"\nwrote {OUT / f'slate_{a.date}.csv'}")


if __name__ == "__main__":
    main()
