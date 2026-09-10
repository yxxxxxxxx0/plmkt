"""Feature engineering for MLB game outcome prediction.

Every feature in this module is *strictly pre-game*. The rule enforced
throughout is: a row's features may only use rows that are earlier in the
team's / pitcher's own history. That is done with `shift(1)` before any
rolling or expanding window, which is where most published baseball models
quietly leak (season-total stats that include the game being predicted).

Feature blocks
  A. Elo         team strength, plus a starting-pitcher-adjusted variant
  B. Team form   rolling rate stats built from summed numerators/denominators
  C. Pitcher     rolling game score / FIP / K-BB for the listed starter
  D. Bullpen     recent relief workload and effectiveness
  E. Context     rest, schedule density, park factor, day/night, series
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (ELO_START, ELO_K, ELO_HFA, ELO_REVERT, ELO_MOV_BASE,
                     SP_ELO_SCALE, SP_PRIOR_STARTS, SP_ROLL_STARTS, PROC)


# --------------------------------------------------------------------------- #
# A. Elo
# --------------------------------------------------------------------------- #
def elo_expected(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def build_elo(games: pd.DataFrame, sp_adj: pd.Series | None = None) -> pd.DataFrame:
    """Sequential Elo with margin-of-victory scaling and off-season regression.

    `sp_adj` (optional, indexed like `games`) is a per-game rating bump applied
    to each side for the listed starting pitcher. It shifts the win probability
    without being fed back into the team rating, exactly as FiveThirtyEight
    separates "Elo" from their pitcher-aware "rating".
    """
    ratings: dict[int, float] = {}
    last_season: dict[int, int] = {}
    out = np.zeros((len(games), 4))

    home_adj = sp_adj[0] if sp_adj is not None else np.zeros(len(games))
    away_adj = sp_adj[1] if sp_adj is not None else np.zeros(len(games))

    for i, (h, a, hs, as_, season) in enumerate(zip(
            games.home_id.values, games.away_id.values,
            games.home_score.values, games.away_score.values,
            games.season.values)):
        for t in (h, a):
            if t not in ratings:
                ratings[t] = ELO_START
                last_season[t] = season
            elif last_season[t] != season:
                # revert toward the mean between seasons (roster churn)
                ratings[t] = ELO_START + (1 - ELO_REVERT) * (ratings[t] - ELO_START)
                last_season[t] = season

        rh, ra = ratings[h], ratings[a]
        diff = rh - ra + ELO_HFA
        p_home = elo_expected(diff)
        # pitcher-aware probability (does not feed back into team Elo)
        p_home_sp = elo_expected(diff + home_adj[i] - away_adj[i])
        out[i] = (rh, ra, p_home, p_home_sp)

        # Unplayed games (used when scoring an upcoming slate) read the
        # ratings but must not update them.
        if not (np.isfinite(hs) and np.isfinite(as_)):
            continue

        margin = abs(hs - as_)
        winner_diff = diff if hs > as_ else -diff
        mov = np.log(margin + 1.0) * (ELO_MOV_BASE / (winner_diff * 0.001 + ELO_MOV_BASE))
        shift = ELO_K * mov * ((1.0 if hs > as_ else 0.0) - p_home)
        ratings[h] = rh + shift
        ratings[a] = ra - shift

    return pd.DataFrame(out, index=games.index,
                        columns=["elo_home", "elo_away", "elo_prob_home", "elo_sp_prob_home"])


# --------------------------------------------------------------------------- #
# Rolling helpers -- all shift(1) first so the current game is never included
# --------------------------------------------------------------------------- #
def _roll_sum(df, key, col, window):
    return (df.groupby(key, observed=True)[col]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum()))


def _expand_sum(df, key, col):
    return (df.groupby(key, observed=True)[col]
              .transform(lambda s: s.shift(1).expanding().sum()))


def _rate(df, key, num, den, window, name, league_rate, prior):
    """Rolling rate stat regressed toward the league mean by `prior` denominator.

    An empty window (a team's first ever game, or the first game of a season
    for the expanding windows) sums to NaN; filling with 0 lets the prior take
    over completely, which is the correct behaviour and keeps the column dense.
    """
    n = _roll_sum(df, key, num, window) if window else _expand_sum(df, key, num)
    d = _roll_sum(df, key, den, window) if window else _expand_sum(df, key, den)
    df[name] = (n.fillna(0.0) + prior * league_rate) / (d.fillna(0.0) + prior)
    return df


# --------------------------------------------------------------------------- #
# B + D. Team offence / defence / bullpen form
# --------------------------------------------------------------------------- #
def team_form(team_logs: pd.DataFrame, sp_outs: pd.DataFrame) -> pd.DataFrame:
    t = team_logs.sort_values(["team_id", "date", "game_pk"]).copy()
    t = t.fillna({c: 0 for c in t.columns if c.startswith(("off_", "def_"))})

    # ---- raw per-game building blocks ----
    t["off_pa"] = t["off_plateAppearances"].replace(0, np.nan).fillna(t["off_atBats"] + t["off_baseOnBalls"])
    t["off_ob"] = t["off_hits"] + t["off_baseOnBalls"] + t["off_hitByPitch"]
    t["off_obden"] = t["off_atBats"] + t["off_baseOnBalls"] + t["off_hitByPitch"] + t["off_sacFlies"]
    t["off_1b"] = t["off_hits"] - t["off_doubles"] - t["off_triples"] - t["off_homeRuns"]
    # wOBA-style linear weights (2020s scale) -- a far better offence signal than AVG
    t["off_woba_num"] = (0.690 * t["off_baseOnBalls"] + 0.720 * t["off_hitByPitch"] +
                         0.880 * t["off_1b"] + 1.250 * t["off_doubles"] +
                         1.590 * t["off_triples"] + 2.030 * t["off_homeRuns"])
    t["def_ip"] = t["def_outs"] / 3.0
    t["def_fip_num"] = (13 * t["def_homeRuns"] + 3 * (t["def_baseOnBalls"] + t["def_hitBatsmen"])
                        - 2 * t["def_strikeOuts"])
    t["gp"] = 1.0

    # relief workload = team outs minus the starter's outs
    t = t.merge(sp_outs, on=["game_pk", "team_id"], how="left")
    t["sp_outs"] = t["sp_outs"].fillna(15.0)
    t["pen_outs"] = (t["def_outs"] - t["sp_outs"]).clip(lower=0)

    LG = dict(woba=0.315, rpg=4.45, k=0.222, bb=0.085, hr=0.033, fip=3.15)
    key = "team_id"

    for w, tag in ((15, "15"), (40, "40"), (None, "std")):
        # NOTE: the "std" (expanding) windows are reset per season below.
        pass

    # --- rolling windows spanning season boundaries (recent form) ---
    for w in (15, 40):
        _rate(t, key, "off_woba_num", "off_pa", w, f"off_woba_{w}", LG["woba"], 150)
        _rate(t, key, "off_runs", "gp", w, f"off_rpg_{w}", LG["rpg"], 8)
        _rate(t, key, "off_strikeOuts", "off_pa", w, f"off_k_{w}", LG["k"], 150)
        _rate(t, key, "off_baseOnBalls", "off_pa", w, f"off_bb_{w}", LG["bb"], 150)
        _rate(t, key, "off_homeRuns", "off_pa", w, f"off_hr_{w}", LG["hr"], 150)
        _rate(t, key, "def_runs", "gp", w, f"def_rpg_{w}", LG["rpg"], 8)
        _rate(t, key, "def_fip_num", "def_ip", w, f"def_fipraw_{w}", LG["fip"] - 3.15, 40)
        _rate(t, key, "def_strikeOuts", "def_battersFaced", w, f"def_k_{w}", LG["k"], 150)
        _rate(t, key, "def_homeRuns", "def_battersFaced", w, f"def_hr_{w}", LG["hr"], 150)
        t[f"def_fip_{w}"] = t[f"def_fipraw_{w}"] + 3.15

    # --- season-to-date (expanding within season) ---
    skey = ["team_id", "season"]
    for num, den, nm, lg, pr in (("off_woba_num", "off_pa", "off_woba_std", LG["woba"], 300),
                                 ("off_runs", "gp", "off_rpg_std", LG["rpg"], 15),
                                 ("def_runs", "gp", "def_rpg_std", LG["rpg"], 15),
                                 ("def_fip_num", "def_ip", "def_fipraw_std", LG["fip"] - 3.15, 80)):
        _rate(t, skey, num, den, None, nm, lg, pr)
    t["def_fip_std"] = t["def_fipraw_std"] + 3.15

    # Pythagorean expectation (Pythagenpat exponent ~1.83 for modern MLB)
    # A 15-game prior at the league average keeps opening day sane.
    PR_G, PR_R = 15.0, 4.45
    rs = _expand_sum(t, skey, "off_runs").fillna(0.0) + PR_G * PR_R
    ra = _expand_sum(t, skey, "def_runs").fillna(0.0) + PR_G * PR_R
    gp = _expand_sum(t, skey, "gp").fillna(0.0) + PR_G
    t["pyth_std"] = rs ** 1.83 / (rs ** 1.83 + ra ** 1.83)
    t["rundiff_pg_std"] = (rs - ra) / gp

    # --- record / streak ---
    t["win"] = 0.0  # filled by caller-provided is_win
    if "is_win" in team_logs.columns:
        t["win"] = t["is_win"].astype(float)
    else:
        t["win"] = (t["off_runs"] > t["def_runs"]).astype(float)
    _rate(t, skey, "win", "gp", None, "wpct_std", 0.5, 20)
    _rate(t, key, "win", "gp", 10, "wpct_10", 0.5, 4)
    _rate(t, key, "win", "gp", 30, "wpct_30", 0.5, 8)

    # --- bullpen ---
    _rate(t, key, "pen_outs", "gp", 5, "pen_outs_5", 9.0, 1)
    _rate(t, key, "pen_outs", "gp", 15, "pen_outs_15", 9.0, 3)
    t["pen_er"] = t["def_earnedRuns"] * (t["pen_outs"] / t["def_outs"].clip(lower=1))
    _rate(t, key, "pen_er", "pen_outs", 40, "pen_erpo_40", 4.15 / 27.0, 120)
    t["pen_era_40"] = t["pen_erpo_40"] * 27.0

    # --- schedule context ---
    t["prev_date"] = t.groupby(key, observed=True)["date"].shift(1)
    t["rest_days"] = (t["date"] - t["prev_date"]).dt.days.clip(0, 7).fillna(5)
    t["g_last7"] = t.groupby(key, observed=True)["gp"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()).fillna(0)
    t["prev_opp"] = t.groupby(key, observed=True)["opp_id"].shift(1)
    t["series_game"] = (t["opp_id"] != t["prev_opp"]).astype(int)  # 1 = series opener
    t["prev_home"] = t.groupby(key, observed=True)["is_home"].shift(1)
    t["travelled"] = (t["is_home"] != t["prev_home"]).fillna(False).astype(int)
    t["game_no"] = t.groupby(skey, observed=True).cumcount()

    keep = ["game_pk", "team_id", "date", "season", "is_home"] + [
        c for c in t.columns if c.startswith(("off_woba_", "off_rpg_", "off_k_", "off_bb_",
                                              "off_hr_", "def_rpg_", "def_fip_", "def_k_",
                                              "def_hr_", "pyth_", "rundiff_", "wpct_",
                                              "pen_outs_", "pen_era_"))
        and not c.endswith("num")] + [
        "rest_days", "g_last7", "series_game", "travelled", "game_no"]
    return t[[c for c in keep if c in t.columns]].copy()


# --------------------------------------------------------------------------- #
# C. Starting pitcher form
# --------------------------------------------------------------------------- #
def pitcher_form(pl: pd.DataFrame) -> pd.DataFrame:
    # Form is measured over *starts only*. These pitchers' relief outings live
    # on a different scale (2 innings, no chance at a high game score) and
    # would bias the rolling averages downward.
    p = pl[pl.gamesStarted > 0].sort_values(["pitcher_id", "date", "game_pk"]).copy()
    p["ip"] = p["outs"] / 3.0
    # Bill James Game Score
    innings_completed = np.floor(p["outs"] / 3.0)
    p["gmsc"] = (50 + p["outs"] + 2 * np.clip(innings_completed - 4, 0, None)
                 + p["strikeOuts"] - 2 * p["hits"] - 4 * p["earnedRuns"]
                 - 2 * (p["runs"] - p["earnedRuns"]) - p["baseOnBalls"])
    p["fip_num"] = (13 * p["homeRuns"] + 3 * (p["baseOnBalls"] + p["hitBatsmen"])
                    - 2 * p["strikeOuts"])
    p["bf"] = p["battersFaced"].replace(0, np.nan).fillna(p["outs"] + p["hits"] + p["baseOnBalls"])
    p["starts"] = (p["gamesStarted"] > 0).astype(float)
    p["one"] = 1.0

    k = "pitcher_id"
    LG_GMSC, LG_FIP, LG_K, LG_BB, LG_HR, LG_IP = 50.0, 4.15, 0.222, 0.080, 0.032, 5.1

    def roll(col, w, name, lg, prior, den="one"):
        n = _roll_sum(p, k, col, w).fillna(0.0)
        d = _roll_sum(p, k, den, w).fillna(0.0)
        p[name] = (n + prior * lg) / (d + prior)

    for w in (5, SP_ROLL_STARTS):
        roll("gmsc", w, f"sp_gmsc_{w}", LG_GMSC, SP_PRIOR_STARTS)
        roll("ip", w, f"sp_ip_{w}", LG_IP, SP_PRIOR_STARTS)
        roll("earnedRuns", w, f"sp_er_{w}", LG_FIP / 9.0 * LG_IP, SP_PRIOR_STARTS)

    for w in (SP_ROLL_STARTS,):
        n_ip = _roll_sum(p, k, "ip", w).fillna(0.0)
        p[f"sp_fip_{w}"] = ((_roll_sum(p, k, "fip_num", w).fillna(0.0)
                             + SP_PRIOR_STARTS * LG_IP * (LG_FIP - 3.15))
                            / (n_ip + SP_PRIOR_STARTS * LG_IP)) + 3.15
        n_bf = _roll_sum(p, k, "bf", w).fillna(0.0)
        for col, nm, lg in (("strikeOuts", "sp_k", LG_K), ("baseOnBalls", "sp_bb", LG_BB),
                            ("homeRuns", "sp_hr", LG_HR)):
            p[f"{nm}_{w}"] = ((_roll_sum(p, k, col, w).fillna(0.0) + 100 * lg) / (n_bf + 100))
        p[f"sp_era_{w}"] = ((_roll_sum(p, k, "earnedRuns", w).fillna(0.0)
                             + SP_PRIOR_STARTS * LG_IP * LG_FIP / 9.0)
                            / (n_ip + SP_PRIOR_STARTS * LG_IP) * 9.0)

    p["sp_career_starts"] = p.groupby(k, observed=True)["starts"].transform(
        lambda s: s.shift(1).expanding().sum()).fillna(0)
    p["sp_rest"] = (p["date"] - p.groupby(k, observed=True)["date"].shift(1)).dt.days.clip(0, 30).fillna(5)
    p["sp_season_starts"] = p.groupby([k, "season"], observed=True)["starts"].transform(
        lambda s: s.shift(1).expanding().sum()).fillna(0)

    cols = ["game_pk", "pitcher_id", "team_id"] + [c for c in p.columns if c.startswith("sp_")]
    return p[cols].copy()


# --------------------------------------------------------------------------- #
# E. Park factor
# --------------------------------------------------------------------------- #
def park_factors(games: pd.DataFrame) -> pd.DataFrame:
    """Trailing 3-season run factor per venue, computed only from prior seasons."""
    g = games.copy()
    g["tot"] = g.home_score + g.away_score
    by = g.groupby(["venue_id", "season"])["tot"].agg(["sum", "count"]).reset_index()
    lg = g.groupby("season")["tot"].mean().rename("lg_rpg")
    by = by.merge(lg, on="season")
    by = by.sort_values(["venue_id", "season"])
    for c in ("sum", "count"):
        by[f"p{c}"] = by.groupby("venue_id")[c].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    by["plg"] = by.groupby("venue_id")["lg_rpg"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    # regressed toward 1.0 with a 400-game prior
    obs = by["psum"] / by["pcount"].replace(0, np.nan)
    by["park_factor"] = ((obs / by["plg"]) * by["pcount"] + 1.0 * 400) / (by["pcount"] + 400)
    by["park_factor"] = by["park_factor"].fillna(1.0)
    return by[["venue_id", "season", "park_factor"]]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
DIFF_COLS = [
    "off_woba_15", "off_woba_40", "off_woba_std", "off_rpg_15", "off_rpg_40", "off_rpg_std",
    "off_k_15", "off_bb_15", "off_hr_15", "off_hr_40",
    "def_rpg_15", "def_rpg_40", "def_rpg_std", "def_fip_15", "def_fip_40", "def_fip_std",
    "def_k_15", "def_hr_15", "pyth_std", "rundiff_pg_std",
    "wpct_10", "wpct_30", "wpct_std", "pen_outs_5", "pen_outs_15", "pen_era_40",
    "rest_days", "g_last7",
]
SP_DIFF_COLS = ["sp_gmsc_5", f"sp_gmsc_{SP_ROLL_STARTS}", f"sp_ip_{SP_ROLL_STARTS}",
                f"sp_fip_{SP_ROLL_STARTS}", f"sp_era_{SP_ROLL_STARTS}",
                f"sp_k_{SP_ROLL_STARTS}", f"sp_bb_{SP_ROLL_STARTS}", f"sp_hr_{SP_ROLL_STARTS}",
                "sp_career_starts", "sp_rest", "sp_season_starts"]


def build_dataset(games: pd.DataFrame, team_logs: pd.DataFrame,
                  pitcher_logs: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    g = games.sort_values(["date", "game_pk"]).reset_index(drop=True).copy()

    # --- starter form, mapped onto each game side ---
    sp_outs = (pitcher_logs.groupby(["game_pk", "team_id"], observed=True)["outs"]
               .max().rename("sp_outs").reset_index())
    spf = pitcher_form(pitcher_logs)
    tf = team_form(team_logs, sp_outs)

    for side, oth in (("home", "away"), ("away", "home")):
        t = tf.rename(columns={c: f"{side}_{c}" for c in tf.columns
                               if c not in ("game_pk", "team_id")})
        g = g.merge(t.rename(columns={"team_id": f"{side}_id"}),
                    on=["game_pk", f"{side}_id"], how="left")
        s = spf.drop(columns=["team_id"]).rename(
            columns={c: f"{side}_{c}" for c in spf.columns if c not in ("game_pk", "pitcher_id")})
        g = g.merge(s.rename(columns={"pitcher_id": f"{side}_sp"}),
                    on=["game_pk", f"{side}_sp"], how="left")

    # --- pitcher-adjusted Elo: convert game score above league into Elo points ---
    gm_h = g.get(f"home_sp_gmsc_{SP_ROLL_STARTS}", pd.Series(50.0, index=g.index)).fillna(50.0)
    gm_a = g.get(f"away_sp_gmsc_{SP_ROLL_STARTS}", pd.Series(50.0, index=g.index)).fillna(50.0)
    adj = (np.clip((gm_h - 50.0), -20, 20).values * SP_ELO_SCALE,
           np.clip((gm_a - 50.0), -20, 20).values * SP_ELO_SCALE)
    g = pd.concat([g, build_elo(g, sp_adj=adj)], axis=1)

    # --- park ---
    g = g.merge(park_factors(games), on=["venue_id", "season"], how="left")
    g["park_factor"] = g["park_factor"].fillna(1.0)

    # --- schedule-adjusted (Massey/ridge) offence and defence ---
    from .ratings import adjusted_ratings, attach_ratings
    g = attach_ratings(g, adjusted_ratings(team_logs))
    g["d_off_adj"] = g.home_off_adj - g.away_off_adj
    g["d_def_adj"] = g.home_def_adj - g.away_def_adj
    # net run expectancy edge: my bats vs your arms, both directions
    g["d_net_adj"] = ((g.home_off_adj - g.away_def_adj) -
                      (g.away_off_adj - g.home_def_adj))

    # --- differences (home minus away) ---
    for c in DIFF_COLS:
        h, a = f"home_{c}", f"away_{c}"
        if h in g.columns and a in g.columns:
            g[f"d_{c}"] = g[h] - g[a]
    for c in SP_DIFF_COLS:
        h, a = f"home_{c}", f"away_{c}"
        if h in g.columns and a in g.columns:
            g[f"d_{c}"] = g[h] - g[a]

    # --- cross terms: my offence vs your pitching ---
    g["d_off_vs_def"] = ((g.get("home_off_woba_40", 0) - g.get("away_def_fip_40", 0) / 100) -
                         (g.get("away_off_woba_40", 0) - g.get("home_def_fip_40", 0) / 100))
    g["d_elo"] = g.elo_home - g.elo_away
    g["elo_logit"] = np.log(g.elo_prob_home.clip(1e-4, 1 - 1e-4) /
                            (1 - g.elo_prob_home.clip(1e-4, 1 - 1e-4)))
    g["elo_sp_logit"] = np.log(g.elo_sp_prob_home.clip(1e-4, 1 - 1e-4) /
                               (1 - g.elo_sp_prob_home.clip(1e-4, 1 - 1e-4)))

    # --- context ---
    g["is_night"] = (g.day_night == "night").astype(int)
    g["month"] = g.date.dt.month
    g["doy"] = g.date.dt.dayofyear
    g["has_both_sp"] = g.home_sp.notna() & g.away_sp.notna()

    # `save=False` matters for predict.py: it appends unplayed games and would
    # otherwise clobber the odds-joined dataset the analysis modules read.
    if save:
        g.to_parquet(PROC / "dataset.parquet", index=False)
    return g


def feature_columns(df: pd.DataFrame, include_market: bool = False) -> list[str]:
    cols = [c for c in df.columns if c.startswith("d_")]
    cols += ["elo_logit", "elo_sp_logit", "park_factor", "is_night", "month",
             "home_game_no", "home_wpct_std", "away_wpct_std",
             "home_off_woba_40", "away_off_woba_40",
             "home_def_fip_40", "away_def_fip_40",
             "home_off_adj", "away_off_adj", "home_def_adj", "away_def_adj",
             "home_sp_gmsc_40", "away_sp_gmsc_40",
             "home_sp_fip_40", "away_sp_fip_40"]
    if include_market:
        cols += [c for c in df.columns if c.startswith("mkt_")]
    return [c for c in dict.fromkeys(cols) if c in df.columns]
