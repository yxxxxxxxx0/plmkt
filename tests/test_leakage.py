"""Leakage and correctness checks.

The single biggest failure mode for game-prediction models is a feature that
quietly contains the result of the game being predicted -- season totals that
include today, a "last 15 games" window that is off by one, a rating updated
before it is read. Published MLB papers reporting 90%+ accuracy are almost
always this bug. These tests recompute a sample of features by brute force
and assert the pipeline agrees.

Run:  python -m pytest tests -q     (or)   python tests/test_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlbmodel import collect, features  # noqa: E402
from mlbmodel.config import PROC  # noqa: E402


def _load():
    return (collect.load("games"), collect.load("team_logs"), collect.load("pitcher_logs"))


def test_team_rolling_is_strictly_prior():
    """off_rpg_15 for a team's game N must equal the mean runs of games
    N-15..N-1 (regressed by the prior), never including game N."""
    games, tl, pl = _load()
    sp_outs = (pl.groupby(["game_pk", "team_id"])["outs"].max()
               .rename("sp_outs").reset_index())
    tf = features.team_form(tl, sp_outs)

    raw = tl.sort_values(["team_id", "date", "game_pk"])
    checked = 0
    rng = np.random.default_rng(0)
    for tid in rng.choice(raw.team_id.unique(), 6, replace=False):
        sub = raw[raw.team_id == tid].reset_index(drop=True)
        got = tf[tf.team_id == tid].set_index("game_pk")
        for n in rng.choice(np.arange(30, len(sub)), 12, replace=False):
            pk = sub.loc[n, "game_pk"]
            if pk not in got.index:
                continue
            window = sub.loc[max(0, n - 15):n - 1, "off_runs"]
            expect = (window.sum() + 8 * 4.45) / (len(window) + 8)   # prior in _rate
            assert abs(got.loc[pk, "off_rpg_15"] - expect) < 1e-6, (
                f"team {tid} game {n}: {got.loc[pk,'off_rpg_15']} != {expect}")
            checked += 1
    assert checked > 30
    print(f"  ok: {checked} rolling windows are strictly prior-only")


def test_pitcher_form_excludes_current_start():
    """A starter's rolling game score must not contain today's game score."""
    games, tl, pl = _load()
    spf = features.pitcher_form(pl)
    starts = pl[pl.gamesStarted > 0].sort_values(["pitcher_id", "date", "game_pk"])
    innings = np.floor(starts.outs / 3)
    starts = starts.assign(gmsc=(50 + starts.outs + 2 * np.clip(innings - 4, 0, None)
                                 + starts.strikeOuts - 2 * starts.hits
                                 - 4 * starts.earnedRuns
                                 - 2 * (starts.runs - starts.earnedRuns)
                                 - starts.baseOnBalls))
    checked = 0
    rng = np.random.default_rng(1)
    for pid in rng.choice(starts.pitcher_id.unique(), 8, replace=False):
        sub = starts[starts.pitcher_id == pid].reset_index(drop=True)
        if len(sub) < 8:
            continue
        got = spf[spf.pitcher_id == pid].set_index("game_pk")
        for n in range(3, min(len(sub), 12)):
            pk = sub.loc[n, "game_pk"]
            w = sub.loc[max(0, n - 5):n - 1, "gmsc"]
            expect = (w.sum() + 10 * 50.0) / (len(w) + 10)
            assert abs(got.loc[pk, "sp_gmsc_5"] - expect) < 1e-6
            checked += 1
    assert checked > 20
    print(f"  ok: {checked} pitcher windows exclude the current start")


def test_elo_is_pregame():
    """The Elo attached to a game must be the rating BEFORE it updates."""
    games, tl, pl = _load()
    g = games.sort_values(["date", "game_pk"]).reset_index(drop=True)
    e = features.build_elo(g)
    # A team's post-game rating implied by the update must equal the pre-game
    # rating on its next game.
    j = pd.concat([g, e], axis=1)
    tid = j.home_id.value_counts().index[0]
    rows = j[(j.home_id == tid) | (j.away_id == tid)].head(60)
    prev_post = None
    for _, r in rows.iterrows():
        pre = r.elo_home if r.home_id == tid else r.elo_away
        if prev_post is not None:
            assert abs(pre - prev_post) < 1e-6, "Elo not carried forward correctly"
        p = r.elo_prob_home
        margin = abs(r.home_score - r.away_score)
        diff = r.elo_home - r.elo_away + 24.0
        wdiff = diff if r.home_score > r.away_score else -diff
        mov = np.log(margin + 1) * (3.0 / (wdiff * 0.001 + 3.0))
        shift = 4.0 * mov * ((1.0 if r.home_score > r.away_score else 0.0) - p)
        prev_post = pre + (shift if r.home_id == tid else -shift)
    print("  ok: Elo ratings are pre-game and chain correctly")


def test_first_games_sit_at_priors():
    """Very early-career rows must be dominated by the league prior, not by
    a value borrowed from later in the season."""
    df = pd.read_parquet(PROC / "dataset.parquet")
    early = df[df.home_game_no == 0]
    assert len(early) > 50
    # game 1 of a season: the 15-game window is empty, so the prior dominates
    assert early.home_off_rpg_15.notna().all()
    spread = early.home_off_rpg_15.std()
    assert spread < 1.0, f"first-game rolling stat varies too much ({spread:.3f})"
    print(f"  ok: opening-day rolling stats sit near the prior (sd={spread:.3f})")


def test_shuffled_target_gives_no_signal():
    """Train on a shuffled label. Anything above ~52% means leakage."""
    from sklearn.impute import SimpleImputer
    import lightgbm as lgb

    df = pd.read_parquet(PROC / "dataset.parquet")
    df = df[df.has_both_sp & (df.season >= 2017)]
    feat = features.feature_columns(df)
    tr, te = df[df.season < 2024], df[df.season == 2024]
    rng = np.random.default_rng(7)
    ytr = rng.permutation(tr.home_win.to_numpy())

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(tr[feat].to_numpy(dtype=float))
    Xte = imp.transform(te[feat].to_numpy(dtype=float))
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                           verbose=-1, n_jobs=4, random_state=0).fit(Xtr, ytr)
    acc = float(((m.predict_proba(Xte)[:, 1] > 0.5) == te.home_win.to_numpy()).mean())
    assert acc < 0.53, f"shuffled-label model scored {acc:.3f} -- features leak"
    print(f"  ok: shuffled-label accuracy {acc:.4f} (chance level)")


def test_real_accuracy_is_plausible():
    """A real MLB model that scores much above the market (~58%) is a bug,
    not a discovery."""
    out = Path(__file__).resolve().parent.parent / "reports" / "summary.csv"
    if not out.exists():
        print("  skipped: run `python -m mlbmodel.run` first")
        return
    s = pd.read_csv(out)
    top = s.winrate.max()
    assert 0.52 < top < 0.62, f"top winrate {top:.4f} is outside the plausible band"
    print(f"  ok: best winrate {top:.4f} sits inside the published 55-62% band")


if __name__ == "__main__":
    for fn in [test_team_rolling_is_strictly_prior,
               test_pitcher_form_excludes_current_start,
               test_elo_is_pregame,
               test_first_games_sit_at_priors,
               test_shuffled_target_gives_no_signal,
               test_real_accuracy_is_plausible]:
        print(f"{fn.__name__}:")
        fn()
    print("\nall leakage checks passed")
