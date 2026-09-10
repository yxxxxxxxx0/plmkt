"""Generic pipeline for any league Action Network covers (NBA, NHL, NFL, NCAAB).

MLB gets the deep treatment because the MLB Stats API hands over free
player-level data. For the other leagues there is no equally open box-score
feed, so this module works from what a single free endpoint gives per date:
final scores, the schedule, and the closing moneyline.

That is less information, but it is enough for the models that dominate
those sports anyway -- Elo plus schedule context. FiveThirtyEight's public
NBA Elo model used essentially these inputs.

Usage:  python -m mlbmodel.anysport nba --from-season 2022
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .backtest import evaluate, summarize
from .config import PROC, RAW, OUT
from .models import make_models
from .odds import SESSION, american_to_decimal, american_to_prob, BOOK_PREFERENCE

AN = "https://api.actionnetwork.com/web/v1/scoreboard/{league}"

# Sport-specific Elo settings. K and home-field are the two that matter;
# these are the conventional values for each league and are re-tuned below
# on pre-test seasons only.
SPORT_ELO = {
    "nba":   dict(K=20.0, HFA=85.0, revert=0.25, mov=3.0, season=(10, 6)),
    "nhl":   dict(K=6.0,  HFA=50.0, revert=0.30, mov=2.5, season=(10, 6)),
    "nfl":   dict(K=20.0, HFA=55.0, revert=0.33, mov=2.2, season=(9, 2)),
    "ncaab": dict(K=25.0, HFA=90.0, revert=0.40, mov=3.0, season=(11, 4)),
}


# --------------------------------------------------------------------------- #
def _parse_games(league: str, j: dict, date_fallback=None) -> list[dict]:
    """Pull completed regular-season games out of one scoreboard response."""
    rows = []
    for g in j.get("games", []):
        if g.get("status") != "complete" or g.get("type") != "reg":
            continue
        bs = g.get("boxscore") or {}
        hp, ap = bs.get("total_home_points"), bs.get("total_away_points")
        if hp is None or ap is None:
            continue
        teams = {t["id"]: t for t in g.get("teams", [])}
        h = teams.get(g["home_team_id"], {})
        a = teams.get(g["away_team_id"], {})
        odds = [o for o in (g.get("odds") or []) if o.get("type") == "game"
                and o.get("ml_home") is not None]
        by_book = {}
        for o in odds:
            by_book.setdefault(o["book_id"], o)
        pick = next((by_book[b] for b in BOOK_PREFERENCE if b in by_book),
                    (list(by_book.values())[0] if by_book else {}))
        date = (pd.Timestamp(g["start_time"]).tz_convert("UTC").tz_localize(None).normalize()
                if g.get("start_time") else date_fallback)
        rows.append(dict(
            game_pk=g["id"], date=date, season=g.get("season"),
            home_id=g["home_team_id"], away_id=g["away_team_id"],
            home_abbr=h.get("abbr"), away_abbr=a.get("abbr"),
            home_score=hp, away_score=ap,
            ml_home=pick.get("ml_home"), ml_away=pick.get("ml_away"),
            spread_home=pick.get("spread_home"), total=pick.get("total"),
            pub_home=pick.get("ml_home_public"),
        ))
    return rows


def _finalize(df: pd.DataFrame, league: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates("game_pk").sort_values(["date", "game_pk"]).reset_index(drop=True)
    df["home_win"] = (df.home_score > df.away_score).astype(int)
    ok = df.ml_home.notna() & df.ml_away.notna()
    df["mkt_p_home"] = np.nan
    ph = american_to_prob(df.loc[ok, "ml_home"])
    pa = american_to_prob(df.loc[ok, "ml_away"])
    df.loc[ok, "mkt_p_home"] = ph / (ph + pa)
    df["dec_home"] = np.nan
    df["dec_away"] = np.nan
    df.loc[ok, "dec_home"] = american_to_decimal(df.loc[ok, "ml_home"])
    df.loc[ok, "dec_away"] = american_to_decimal(df.loc[ok, "ml_away"])
    df.to_parquet(PROC / f"{league}_games.parquet", index=False)
    return df


def fetch_weekly(league: str, seasons, weeks=range(1, 19)) -> pd.DataFrame:
    """Week-indexed leagues (NFL). The scoreboard endpoint ignores `date` for
    the NFL and answers on (season, week, seasonType) instead -- roughly 18
    requests per season rather than 150 calendar days.
    """
    rows = []
    for yr in seasons:
        for wk in weeks:
            fp = RAW / f"an_{league}_s{yr}_w{wk}.json"
            if fp.exists():
                j = json.loads(fp.read_text(encoding="utf-8"))
            else:
                try:
                    r = SESSION.get(AN.format(league=league), params={
                        "season": yr, "week": wk, "seasonType": "reg"}, timeout=60)
                    r.raise_for_status()
                    j = r.json()
                except Exception:
                    time.sleep(2)
                    j = {"games": []}
                fp.write_text(json.dumps(j), encoding="utf-8")
            rows += [r for r in _parse_games(league, j) if r["season"] == yr]
        print(f"  {league} {yr}: {len(rows)} cumulative games")
    return _finalize(pd.DataFrame(rows), league)


def _in_season(league: str, d: pd.Timestamp) -> bool:
    """Skip the off-season entirely -- it is more than half the calendar for
    every one of these leagues and each skipped day is a saved request."""
    lo, hi = SPORT_ELO[league]["season"]      # (first month, last month), wrapping
    m = d.month
    return m >= lo or m <= hi


def fetch_league(league: str, start: str, end: str) -> pd.DataFrame:
    rows = []
    dates = [d for d in pd.date_range(start, end, freq="D") if _in_season(league, d)]
    print(f"  {league}: {len(dates)} in-season dates to fetch")
    for i, d in enumerate(dates):
        ds = d.strftime("%Y%m%d")
        fp = RAW / f"an_{league}_{ds}.json"
        if fp.exists():
            j = json.loads(fp.read_text(encoding="utf-8"))
        else:
            try:
                r = SESSION.get(AN.format(league=league), params={"date": ds}, timeout=60)
                r.raise_for_status()
                j = r.json()
            except Exception:
                time.sleep(2)
                j = {"games": []}
            fp.write_text(json.dumps(j), encoding="utf-8")

        rows += _parse_games(league, j, date_fallback=d.normalize())
        if (i + 1) % 200 == 0:
            print(f"  {league} {ds}: {len(rows)} games")

    return _finalize(pd.DataFrame(rows), league)


# --------------------------------------------------------------------------- #
def elo_and_form(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Elo plus the schedule/form features that carry most of the signal in
    the non-baseball leagues."""
    g = df.sort_values(["date", "game_pk"]).reset_index(drop=True).copy()
    K, HFA, REV, MOV = cfg["K"], cfg["HFA"], cfg["revert"], cfg["mov"]

    ratings, last_season = {}, {}
    eh = np.zeros(len(g)); ea = np.zeros(len(g)); pr = np.zeros(len(g))
    for i, (h, a, hs, as_, s) in enumerate(zip(
            g.home_id, g.away_id, g.home_score, g.away_score, g.season)):
        for t in (h, a):
            if t not in ratings:
                ratings[t] = 1500.0; last_season[t] = s
            elif last_season[t] != s:
                ratings[t] = 1500.0 + (1 - REV) * (ratings[t] - 1500.0)
                last_season[t] = s
        rh, ra = ratings[h], ratings[a]
        diff = rh - ra + HFA
        p = 1 / (1 + 10 ** (-diff / 400))
        eh[i], ea[i], pr[i] = rh, ra, p
        margin = abs(hs - as_)
        wdiff = diff if hs > as_ else -diff
        mult = np.log(margin + 1) * (MOV / (wdiff * 0.001 + MOV))
        shift = K * mult * ((1.0 if hs > as_ else 0.0) - p)
        ratings[h] = rh + shift; ratings[a] = ra - shift
    g["elo_home"], g["elo_away"], g["elo_prob_home"] = eh, ea, pr
    g["d_elo"] = g.elo_home - g.elo_away
    g["elo_logit"] = np.log(np.clip(pr, 1e-6, 1-1e-6) / (1 - np.clip(pr, 1e-6, 1-1e-6)))

    # long team-game frame for rolling form
    parts = []
    for side, opp in (("home", "away"), ("away", "home")):
        parts.append(pd.DataFrame(dict(
            game_pk=g.game_pk, date=g.date, season=g.season,
            team_id=g[f"{side}_id"], is_home=int(side == "home"),
            pf=g[f"{side}_score"], pa=g[f"{opp}_score"],
            win=(g[f"{side}_score"] > g[f"{opp}_score"]).astype(float))))
    t = pd.concat(parts).sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)
    t["one"] = 1.0
    t["margin"] = t.pf - t.pa

    key = "team_id"
    for w in (5, 15, 30):
        for c in ("pf", "pa", "margin", "win"):
            t[f"{c}_{w}"] = t.groupby(key, observed=True)[c].transform(
                lambda s: s.shift(1).rolling(w, min_periods=2).mean())
    t["wpct_std"] = t.groupby([key, "season"], observed=True)["win"].transform(
        lambda s: (s.shift(1).expanding().sum() + 5) / (s.shift(1).expanding().count() + 10))
    t["margin_std"] = t.groupby([key, "season"], observed=True)["margin"].transform(
        lambda s: s.shift(1).expanding().mean())
    prev = t.groupby(key, observed=True)["date"].shift(1)
    t["rest"] = (t.date - prev).dt.days.clip(0, 10).fillna(5)
    t["b2b"] = (t.rest <= 1).astype(int)
    t["g_last7"] = t.groupby(key, observed=True)["one"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()).fillna(0)
    t["game_no"] = t.groupby([key, "season"], observed=True).cumcount()

    fcols = [c for c in t.columns if c not in
             ("game_pk", "date", "season", "team_id", "is_home", "pf", "pa",
              "win", "one", "margin")]
    for side in ("home", "away"):
        tt = t[["game_pk", "team_id"] + fcols].rename(
            columns={c: f"{side}_{c}" for c in fcols})
        g = g.merge(tt.rename(columns={"team_id": f"{side}_id"}),
                    on=["game_pk", f"{side}_id"], how="left")
    for c in fcols:
        g[f"d_{c}"] = g[f"home_{c}"] - g[f"away_{c}"]

    g["month"] = g.date.dt.month
    return g


def features_for(g: pd.DataFrame) -> list[str]:
    return [c for c in g.columns if c.startswith("d_")] + ["elo_logit", "month"]


# --------------------------------------------------------------------------- #
def tune_elo(df: pd.DataFrame, cfg: dict, test_seasons) -> dict:
    """Grid-search K and home-field on pre-test seasons only."""
    ev = ~df.season.isin(test_seasons)
    y = df.home_win.values
    best, best_ll = dict(cfg), np.inf
    for K in [cfg["K"] * f for f in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]:
        for HFA in [cfg["HFA"] * f for f in (0.6, 0.8, 1.0, 1.2)]:
            c = dict(cfg, K=K, HFA=HFA)
            p = elo_and_form(df, c).sort_values(["date", "game_pk"]).elo_prob_home.values
            ll = log_loss(y[ev.values], np.clip(p[ev.values], 1e-6, 1-1e-6), labels=[0, 1])
            if ll < best_ll:
                best_ll, best = ll, c
    print(f"  tuned Elo: K={best['K']:.1f} HFA={best['HFA']:.0f} (logloss {best_ll:.4f})")
    return best


# Leagues the scoreboard endpoint indexes by (season, week) rather than date.
WEEKLY = {"nfl"}


def run_league(league: str, start: str, end: str, test_seasons: list[int],
               edge: float = 0.03):
    print(f"\n########## {league.upper()} ##########")
    fp = PROC / f"{league}_games.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
    elif league in WEEKLY:
        seasons = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
        df = fetch_weekly(league, seasons)
    else:
        df = fetch_league(league, start, end)
    if df.empty:
        print("no data"); return None, None
    df = df.dropna(subset=["season"])
    df["season"] = df.season.astype(int)
    print(f"{len(df)} games, seasons {sorted(df.season.unique())}, "
          f"odds on {df.mkt_p_home.notna().mean():.1%}")

    cfg = tune_elo(df, SPORT_ELO[league], test_seasons)
    g = elo_and_form(df, cfg)
    feat = features_for(g)

    from .backtest import walk_forward
    zoo = {k: v for k, v in make_models(len(feat)).items()
           if k in ("logit_l2", "lightgbm", "random_forest", "mlp")}
    res, preds = walk_forward(g, feat, test_seasons, min_train_seasons=2,
                              drop_odd_seasons=False, models=zoo, edge_thresh=edge)
    if res.empty:
        print("not enough history"); return None, None
    summ = summarize(res)
    cols = [c for c in ["model", "n", "winrate", "mkt_winrate", "logloss", "brier",
                        "auc", "bets", "bet_winrate", "flat_roi"] if c in summ.columns]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    res.to_csv(OUT / f"{league}_results_by_season.csv", index=False)
    summ.to_csv(OUT / f"{league}_summary.csv", index=False)
    return res, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("league", choices=list(SPORT_ELO))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default="2026-09-07")
    ap.add_argument("--test-seasons", type=int, nargs="*", default=None)
    ap.add_argument("--edge", type=float, default=0.03)
    a = ap.parse_args()
    starts = {"nba": "2018-10-01", "nhl": "2018-10-01",
              "nfl": "2016-09-01", "ncaab": "2021-11-01"}
    tests = {"nba": [2022, 2023, 2024, 2025], "nhl": [2022, 2023, 2024, 2025],
             "nfl": [2022, 2023, 2024, 2025], "ncaab": [2024, 2025]}
    run_league(a.league, a.start or starts[a.league], a.end,
               a.test_seasons or tests[a.league], a.edge)


if __name__ == "__main__":
    main()
