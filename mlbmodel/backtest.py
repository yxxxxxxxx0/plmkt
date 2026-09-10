"""Walk-forward backtest and betting simulation.

Protocol
--------
For each test season S the model is trained *only* on seasons < S, so no
information from S (or later) can reach the fit. Within S, predictions are
produced in one pass -- the features themselves are already causal, so an
in-season refit changes little; `refit_monthly=True` enables it anyway to
show the effect.

Reported per model / per season
  n, winrate (accuracy), log loss, Brier, AUC, calibration slope
  ROI at flat stake and at fractional Kelly against the closing moneyline
  CLV -- how often the model's pick beat the closing implied probability
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .advanced import FRAME_FIT_MODELS
from .baselines import EXTRA_BASELINES
from .config import ODD_SEASONS, OUT
from .models import FRAME_MODELS, make_models

ALL_FRAME_MODELS = {**FRAME_MODELS, **EXTRA_BASELINES}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def calibration_slope(p, y):
    """Regress y on the predicted logit. 1.0 = perfectly calibrated,
    <1 = over-confident, >1 = under-confident."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if len(np.unique(y)) < 2:
        return np.nan
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0][0])


def bet_sim(p_home, y, dec_home, dec_away, edge_thresh=0.03, kelly_frac=0.25,
            max_stake=0.05):
    """Flat-stake and fractional-Kelly simulation on the moneyline.

    A bet is placed on whichever side the model gives more edge than
    `edge_thresh` versus the book's own implied (vig-inclusive) price.
    """
    ok = np.isfinite(dec_home) & np.isfinite(dec_away) & np.isfinite(p_home)
    if ok.sum() == 0:
        return dict(bets=0)
    p_h, p_a = p_home[ok], 1 - p_home[ok]
    dh, da, yy = dec_home[ok], dec_away[ok], y[ok]

    ev_h = p_h * dh - 1.0
    ev_a = p_a * da - 1.0
    take_h = (ev_h > edge_thresh) & (ev_h >= ev_a)
    take_a = (ev_a > edge_thresh) & (ev_a > ev_h)
    bet = take_h | take_a

    if bet.sum() == 0:
        return dict(bets=0)

    dec = np.where(take_h, dh, da)[bet]
    p = np.where(take_h, p_h, p_a)[bet]
    won = np.where(take_h[bet], yy[bet] == 1, yy[bet] == 0)

    flat_pnl = np.where(won, dec - 1.0, -1.0)
    b = dec - 1.0
    k = np.clip((p * dec - 1.0) / b, 0, max_stake / kelly_frac) * kelly_frac
    kelly_pnl = np.where(won, k * b, -k)

    return dict(
        bets=int(bet.sum()),
        bet_rate=float(bet.mean()),
        bet_winrate=float(won.mean()),
        flat_roi=float(flat_pnl.mean()),
        flat_pnl=float(flat_pnl.sum()),
        # Stake-weighted return. NOT a compounded growth rate -- baseball
        # plays ~15 games at once, so bets cannot be compounded one at a time.
        # Use staking.simulate() for anything about bankroll growth.
        kelly_roi_on_turnover=float(kelly_pnl.sum() / max(k.sum(), 1e-9)),
        avg_dec=float(dec.mean()),
    )


def evaluate(p, y, frame=None, **betkw):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    # The market baseline has no prediction for the handful of games with no
    # posted line; score every model only where it actually produced one.
    ok = np.isfinite(p)
    if not ok.all():
        p, y = p[ok], y[ok]
        frame = frame.loc[ok] if frame is not None else None
    p = np.clip(p, 1e-6, 1 - 1e-6)
    row = dict(
        n=len(y),
        winrate=float(((p > 0.5).astype(int) == y).mean()),
        logloss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
        auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        cal_slope=calibration_slope(p, y),
        mean_p=float(p.mean()),
    )
    if frame is not None and "dec_home" in frame.columns:
        m = frame.mkt_p_home.to_numpy(dtype=float)
        sub = np.isfinite(m)
        if sub.sum() > 50:
            row["n_odds"] = int(sub.sum())
            row["winrate_odds_games"] = float(((p[sub] > .5).astype(int) == y[sub]).mean())
            row["mkt_winrate"] = float(((m[sub] > .5).astype(int) == y[sub]).mean())
            row["logloss_vs_mkt"] = float(log_loss(y[sub], p[sub], labels=[0, 1])
                                          - log_loss(y[sub], np.clip(m[sub], 1e-6, 1-1e-6), labels=[0, 1]))
            # CLV proxy: does the model disagree with the close in the right direction?
            row["corr_with_mkt"] = float(np.corrcoef(p[sub], m[sub])[0, 1])
        row.update({f"bet_{k}": v for k, v in bet_sim(
            p, y, frame.dec_home.to_numpy(dtype=float),
            frame.dec_away.to_numpy(dtype=float), **betkw).items()})
    return row


# --------------------------------------------------------------------------- #
# Walk-forward driver
# --------------------------------------------------------------------------- #
def walk_forward(df: pd.DataFrame, feat_cols: list[str], test_seasons: list[int],
                 min_train_seasons: int = 4, drop_odd_seasons: bool = True,
                 models: dict | None = None, edge_thresh: float = 0.03,
                 verbose: bool = True):
    d = df.sort_values(["date", "game_pk"]).reset_index(drop=True)
    if drop_odd_seasons:
        train_pool = d[~d.season.isin(ODD_SEASONS)]
    else:
        train_pool = d

    zoo = models if models is not None else make_models(len(feat_cols))
    preds, rows = [], []

    for season in test_seasons:
        tr = train_pool[train_pool.season < season]
        te = d[d.season == season]
        if tr.season.nunique() < min_train_seasons or len(te) == 0:
            continue
        Xtr, ytr = tr[feat_cols].to_numpy(dtype=float), tr.home_win.to_numpy()
        Xte, yte = te[feat_cols].to_numpy(dtype=float), te.home_win.to_numpy()

        # Baselines that read columns straight off the frame. Leagues without
        # starting pitchers (or without odds) simply skip the ones they cannot
        # compute rather than failing the whole run.
        season_preds = {}
        for name, factory in ALL_FRAME_MODELS.items():
            try:
                season_preds[name] = np.asarray(
                    factory().predict_proba_frame(te), dtype=float)
            except KeyError:
                continue

        for name, factory in zoo.items():
            m = factory()
            m.fit(Xtr, ytr)
            p = (m.predict_proba(Xte) if not hasattr(m, "predict_proba_frame")
                 else m.predict_proba_frame(te))
            if p.ndim > 1:
                p = p[:, 1]
            season_preds[name] = p

        # models that need the frame itself (offsets, run-margin targets)
        for name, factory in FRAME_FIT_MODELS.items():
            try:
                m = factory().fit_frame(tr, feat_cols, ytr)
                season_preds[name] = np.asarray(m.predict_frame(te, feat_cols), dtype=float)
            except KeyError:
                continue

        # simple ensemble of the fitted (non-baseline) models
        fitted = list(zoo) + list(FRAME_FIT_MODELS)
        stack = np.vstack([season_preds[n] for n in fitted if n in season_preds])
        season_preds["ensemble"] = stack.mean(axis=0)
        # ensemble blended with the strongest Elo variant, in logit space
        lg = lambda x: np.log(np.clip(x, 1e-6, 1-1e-6) / (1 - np.clip(x, 1e-6, 1-1e-6)))
        elo_key = "elo_sp" if "elo_sp" in season_preds else "elo"
        z = 0.6 * lg(season_preds["ensemble"]) + 0.4 * lg(season_preds[elo_key])
        season_preds["ensemble_elo"] = 1 / (1 + np.exp(-z))

        for name, p in season_preds.items():
            r = evaluate(p, yte, te, edge_thresh=edge_thresh)
            r.update(model=name, season=season, train_games=len(tr))
            rows.append(r)
            preds.append(pd.DataFrame(dict(
                game_pk=te.game_pk.values, date=te.date.values, season=season,
                model=name, p_home=p, home_win=yte,
                mkt_p_home=te.get("mkt_p_home", pd.Series(np.nan, index=te.index)).values,
                dec_home=te.get("dec_home", pd.Series(np.nan, index=te.index)).values,
                dec_away=te.get("dec_away", pd.Series(np.nan, index=te.index)).values)))
        if verbose:
            best = max((r for r in rows if r["season"] == season), key=lambda r: r["winrate"])
            print(f"  {season}: trained on {len(tr):>6} games, "
                  f"tested on {len(te):>5}  | best={best['model']} "
                  f"{best['winrate']:.4f}")

    res = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return res, pred_df


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    """Pooled (game-weighted) metrics across the test seasons."""
    def agg(g):
        w = g["n"]
        out = dict(
            seasons=len(g), n=int(w.sum()),
            winrate=float(np.average(g.winrate, weights=w)),
            logloss=float(np.average(g.logloss, weights=w)),
            brier=float(np.average(g.brier, weights=w)),
            auc=float(np.average(g.auc, weights=w)),
            cal_slope=float(np.average(g.cal_slope.fillna(1.0), weights=w)),
        )
        if "bet_bets" in g.columns and g.bet_bets.notna().any():
            bw = g.bet_bets.fillna(0)
            out["bets"] = int(bw.sum())
            # a model that never clears the edge threshold has no bet columns
            if bw.sum() > 0 and "bet_bet_winrate" in g.columns:
                out["bet_winrate"] = float(np.average(
                    g.bet_bet_winrate.fillna(0), weights=bw))
                out["flat_roi"] = float(g.bet_flat_pnl.fillna(0).sum() / bw.sum())
        if "mkt_winrate" in g.columns and g.mkt_winrate.notna().any():
            m = g.dropna(subset=["mkt_winrate"])
            out["mkt_winrate"] = float(np.average(m.mkt_winrate, weights=m.n_odds))
        return pd.Series(out)

    # Build the frame explicitly: `groupby.apply` returning a Series collapses
    # to a Series-of-Series under pandas 3, which is not what we want here.
    out = pd.DataFrame(
        [dict(model=name, **agg(g).to_dict())
         for name, g in res.groupby("model", observed=True)])
    return out.sort_values("logloss").reset_index(drop=True)
