"""Opponent-adjusted team ratings via ridge regression (a Massey-style model).

Rolling averages of runs scored/allowed are confounded by schedule strength:
a team that has just played four series against bottom-five pitching staffs
looks better than it is. Solving

    runs_scored(i vs j, at venue v) ~ mu + off_i - def_j + hfa

by ridge on a trailing window removes that. The ridge penalty doubles as
regression toward the league mean, which is exactly what small samples need.

Ratings are refit every `refit_days` and then held constant until the next
refit, so a rating attached to a game is only ever built from games that
finished before it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


def adjusted_ratings(team_logs: pd.DataFrame, refit_days: int = 10,
                     lookback_days: int = 400, alpha: float = 60.0,
                     min_games: int = 400) -> pd.DataFrame:
    """Return one row per (date_from, team_id) with off/def ratings in runs/game."""
    t = team_logs[["date", "team_id", "opp_id", "is_home", "off_runs", "def_runs"]].dropna()
    t = t.sort_values("date").reset_index(drop=True)
    teams = np.sort(t.team_id.unique())
    idx = {tid: i for i, tid in enumerate(teams)}
    nt = len(teams)

    dates = pd.date_range(t.date.min(), t.date.max(), freq=f"{refit_days}D")
    rows = []
    for cut in dates:
        lo = cut - pd.Timedelta(days=lookback_days)
        w = t[(t.date < cut) & (t.date >= lo)]
        if len(w) < min_games:
            continue
        n = len(w)
        oi = w.team_id.map(idx).to_numpy()
        di = w.opp_id.map(idx).to_numpy()
        # design: [off_0..off_n-1 | def_0..def_n-1 | home]
        r = np.repeat(np.arange(n), 3)
        c = np.concatenate([np.stack([oi, nt + di, np.full(n, 2 * nt)], axis=1).ravel()])
        v = np.tile(np.array([1.0, -1.0, 1.0]), n)
        v[2::3] = w.is_home.to_numpy(dtype=float)
        X = sparse.csr_matrix((v, (r, c)), shape=(n, 2 * nt + 1))

        # recency weighting -- a half-life of ~120 days
        age = (cut - w.date).dt.days.to_numpy(dtype=float)
        sw = 0.5 ** (age / 120.0)

        m = Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg")
        m.fit(X, w.off_runs.to_numpy(dtype=float), sample_weight=sw)
        off = m.coef_[:nt]
        dfn = m.coef_[nt:2 * nt]
        for tid, i in idx.items():
            rows.append(dict(date_from=cut, team_id=tid,
                             off_adj=off[i], def_adj=dfn[i],
                             lg_rpg=float(m.intercept_)))
    return pd.DataFrame(rows)


def attach_ratings(df: pd.DataFrame, ratings: pd.DataFrame,
                   side_cols=("home_id", "away_id")) -> pd.DataFrame:
    """As-of join: each game gets the most recent rating strictly before it."""
    if ratings.empty:
        for s in ("home", "away"):
            df[f"{s}_off_adj"] = np.nan
            df[f"{s}_def_adj"] = np.nan
        return df
    r = ratings.sort_values("date_from")
    out = df.sort_values("date").copy()
    for side, col in zip(("home", "away"), side_cols):
        rr = r.rename(columns={"team_id": col, "off_adj": f"{side}_off_adj",
                               "def_adj": f"{side}_def_adj"})
        out = pd.merge_asof(
            out.sort_values("date"), rr.sort_values("date_from"),
            left_on="date", right_on="date_from", by=col,
            direction="backward", allow_exact_matches=False,
            suffixes=("", f"_{side}"))
        out = out.drop(columns=[c for c in ("date_from", "lg_rpg") if c in out.columns])
    return out.sort_values(["date", "game_pk"]).reset_index(drop=True)
