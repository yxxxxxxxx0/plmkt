"""Tune the Elo / pitcher-adjustment hyperparameters.

Fitted on seasons strictly before the test window, so the tuned values carry
no information from 2023-2026. Objective is log loss, not accuracy: accuracy
on ~2400 games has a standard error of ~1 percentage point and will happily
select noise.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config, features
from .config import OUT, PROC


def elo_probs(games, K, HFA, REVERT, MOV_BASE, sp_scale, gm_h, gm_a):
    ratings, last_season = {}, {}
    n = len(games)
    p_out = np.zeros(n)
    p_sp_out = np.zeros(n)
    hid = games.home_id.values; aid = games.away_id.values
    hs = games.home_score.values; as_ = games.away_score.values
    seas = games.season.values

    for i in range(n):
        h, a, season = hid[i], aid[i], seas[i]
        for t in (h, a):
            if t not in ratings:
                ratings[t] = 1500.0; last_season[t] = season
            elif last_season[t] != season:
                ratings[t] = 1500.0 + (1 - REVERT) * (ratings[t] - 1500.0)
                last_season[t] = season
        rh, ra = ratings[h], ratings[a]
        diff = rh - ra + HFA
        p = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        adj = (gm_h[i] - gm_a[i]) * sp_scale
        p_out[i] = p
        p_sp_out[i] = 1.0 / (1.0 + 10 ** (-(diff + adj) / 400.0))

        margin = abs(hs[i] - as_[i])
        wdiff = diff if hs[i] > as_[i] else -diff
        mov = np.log(margin + 1.0) * (MOV_BASE / (wdiff * 0.001 + MOV_BASE))
        shift = K * mov * ((1.0 if hs[i] > as_[i] else 0.0) - p)
        ratings[h] = rh + shift
        ratings[a] = ra - shift
    return p_out, p_sp_out


def main():
    df = pd.read_parquet(PROC / "dataset.parquet").sort_values(["date", "game_pk"])
    df = df.reset_index(drop=True)
    gm_h = np.clip(df.get("home_sp_gmsc_40", pd.Series(50.0, index=df.index)).fillna(50.0) - 50, -20, 20).values
    gm_a = np.clip(df.get("away_sp_gmsc_40", pd.Series(50.0, index=df.index)).fillna(50.0) - 50, -20, 20).values

    # evaluate on 2018-2022 only; 2015-2017 is Elo burn-in, 2023+ is held out
    ev = (df.season >= 2018) & (df.season <= 2022) & (~df.season.isin(config.ODD_SEASONS))
    y = df.home_win.values

    rows = []
    grid = itertools.product(
        [3.0, 4.0, 5.0, 6.0, 8.0],          # K
        [20.0, 24.0, 28.0, 32.0],           # HFA
        [0.20, 0.30, 0.40, 0.55],           # season revert
        [1.5, 2.2, 3.0],                    # MOV base
    )
    for K, HFA, REV, MOV in grid:
        p, _ = elo_probs(df, K, HFA, REV, MOV, 0.0, gm_h, gm_a)
        rows.append(dict(K=K, HFA=HFA, revert=REV, mov=MOV,
                         logloss=log_loss(y[ev], np.clip(p[ev], 1e-6, 1-1e-6), labels=[0, 1]),
                         acc=float(((p[ev] > .5).astype(int) == y[ev]).mean())))
    r = pd.DataFrame(rows).sort_values("logloss")
    print("=== Elo grid (top 10, tuned on 2018-2022) ===")
    print(r.head(10).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    best = r.iloc[0]

    rows2 = []
    for scale in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        _, ps = elo_probs(df, best.K, best.HFA, best.revert, best.mov, scale, gm_h, gm_a)
        rows2.append(dict(sp_scale=scale,
                          logloss=log_loss(y[ev], np.clip(ps[ev], 1e-6, 1-1e-6), labels=[0, 1]),
                          acc=float(((ps[ev] > .5).astype(int) == y[ev]).mean())))
    r2 = pd.DataFrame(rows2).sort_values("logloss")
    print("\n=== Starting-pitcher Elo scale (game-score point -> Elo points) ===")
    print(r2.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    out = dict(K=float(best.K), HFA=float(best.HFA), revert=float(best.revert),
               mov_base=float(best.mov), sp_scale=float(r2.iloc[0].sp_scale))
    (OUT / "elo_params.json").write_text(json.dumps(out, indent=2))
    print("\nbest:", out)


if __name__ == "__main__":
    main()
