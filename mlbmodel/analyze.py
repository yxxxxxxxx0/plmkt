"""Post-backtest analysis: calibration, market comparison, edge slicing.

Accuracy alone is a poor way to judge a betting model -- a model can be more
accurate than the market and still lose money, and can be less accurate and
still win. These diagnostics separate the two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import bet_sim
from .config import OUT, PROC


def calibration_table(p, y, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    d = pd.DataFrame(dict(p=p, y=y, bin=q))
    t = d.groupby("bin", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), actual=("y", "mean")).reset_index()
    t["gap"] = t.actual - t.pred
    t["bin"] = t["bin"].astype(str)
    return t


def market_comparison(pr: pd.DataFrame) -> pd.DataFrame:
    """For each model: accuracy, log loss and closing-line value vs the market."""
    from sklearn.metrics import log_loss
    d = pr[pr.mkt_p_home.notna() & pr.p_home.notna()].copy()
    rows = []
    for name, g in d.groupby("model", observed=True):
        y = g.home_win.to_numpy()
        p = np.clip(g.p_home.to_numpy(), 1e-6, 1 - 1e-6)
        m = np.clip(g.mkt_p_home.to_numpy(), 1e-6, 1 - 1e-6)
        # CLV proxy: when the model disagrees with the close, is it right?
        dis = np.abs(p - m) > 0.03
        side = (p > m)                              # model likes the home side
        won = np.where(side, y == 1, y == 0)
        rows.append(dict(
            model=name, n=len(g),
            acc=float(((p > .5).astype(int) == y).mean()),
            mkt_acc=float(((m > .5).astype(int) == y).mean()),
            logloss=float(log_loss(y, p, labels=[0, 1])),
            mkt_logloss=float(log_loss(y, m, labels=[0, 1])),
            corr_mkt=float(np.corrcoef(p, m)[0, 1]),
            mean_abs_dev=float(np.abs(p - m).mean()),
            disagree_rate=float(dis.mean()),
            disagree_hit=float(won[dis].mean()) if dis.sum() else np.nan,
        ))
    r = pd.DataFrame(rows)
    r["logloss_edge"] = r.mkt_logloss - r.logloss     # positive = beats the market
    return r.sort_values("logloss_edge", ascending=False)


def edge_curve(pr: pd.DataFrame, model: str, thresholds=None) -> pd.DataFrame:
    """ROI as a function of the required edge -- the shape matters more than
    any single number. A genuine edge decays smoothly; noise does not."""
    thresholds = thresholds or [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
    g = pr[(pr.model == model) & pr.dec_home.notna()]
    rows = []
    for th in thresholds:
        r = bet_sim(g.p_home.to_numpy(), g.home_win.to_numpy(),
                    g.dec_home.to_numpy(), g.dec_away.to_numpy(), edge_thresh=th)
        r["edge_thresh"] = th
        rows.append(r)
    return pd.DataFrame(rows)


def roi_significance(pr: pd.DataFrame, model: str, edge_thresh: float = 0.10,
                     n_boot: int = 5000, seed: int = 0) -> dict:
    """Is an apparently profitable ROI distinguishable from luck?

    Betting long-odds underdogs produces a wildly skewed P&L distribution: a
    handful of +400 winners can carry a strategy that is actually losing. A
    bootstrap over individual bets gives the honest interval, and the
    per-season breakdown shows whether the "edge" is one lucky year.
    """
    g = pr[(pr.model == model) & pr.dec_home.notna() & pr.p_home.notna()]
    rng = np.random.default_rng(seed)

    p = g.p_home.to_numpy(); y = g.home_win.to_numpy()
    dh = g.dec_home.to_numpy(); da = g.dec_away.to_numpy()
    ev_h, ev_a = p * dh - 1, (1 - p) * da - 1
    th = (ev_h > edge_thresh) & (ev_h >= ev_a)
    ta = (ev_a > edge_thresh) & (ev_a > ev_h)
    bet = th | ta
    if bet.sum() < 20:
        return dict(model=model, edge_thresh=edge_thresh, bets=int(bet.sum()))

    dec = np.where(th, dh, da)[bet]
    won = np.where(th[bet], y[bet] == 1, y[bet] == 0)
    pnl = np.where(won, dec - 1.0, -1.0)

    boot = np.array([pnl[rng.integers(0, len(pnl), len(pnl))].mean()
                     for _ in range(n_boot)])
    seasons = g.season.to_numpy()[bet]
    per_season = {int(s): float(pnl[seasons == s].mean()) for s in np.unique(seasons)}
    return dict(
        model=model, edge_thresh=edge_thresh, bets=int(bet.sum()),
        roi=float(pnl.mean()),
        roi_se=float(pnl.std(ddof=1) / np.sqrt(len(pnl))),
        ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
        p_roi_positive=float((boot > 0).mean()),
        seasons_profitable=f"{sum(v > 0 for v in per_season.values())}/{len(per_season)}",
        per_season={k: round(v, 4) for k, v in per_season.items()},
        avg_dec=float(dec.mean()), hit_rate=float(won.mean()),
    )


def naive_controls(pr: pd.DataFrame, model: str) -> pd.DataFrame:
    """The control every "I found an edge" result needs.

    A model that is roughly market-shaped but slightly flatter will flag an
    "edge" on whichever side is the longshot. If simply betting underdogs is
    profitable on its own -- the well-documented favourite-longshot bias --
    then the model contributed nothing and the ROI is not model skill.

    Each control below uses the closing line alone, no model at all.
    """
    g = pr[(pr.model == model) & pr.dec_home.notna() & pr.p_home.notna()].copy()
    y = g.home_win.to_numpy()
    dh, da = g.dec_home.to_numpy(), g.dec_away.to_numpy()
    m = g.mkt_p_home.to_numpy()

    def strat(name, take_home, mask=None):
        sel = np.ones(len(g), bool) if mask is None else mask
        if sel.sum() < 20:
            return None
        dec = np.where(take_home, dh, da)[sel]
        won = np.where(take_home[sel], y[sel] == 1, y[sel] == 0)
        pnl = np.where(won, dec - 1.0, -1.0)
        seasons = g.season.to_numpy()[sel]
        return dict(strategy=name, bets=int(sel.sum()), hit=float(won.mean()),
                    roi=float(pnl.mean()),
                    se=float(pnl.std(ddof=1) / np.sqrt(len(pnl))),
                    avg_dec=float(dec.mean()),
                    profit_seasons="/".join([
                        str(sum(pnl[seasons == s].mean() > 0 for s in np.unique(seasons))),
                        str(len(np.unique(seasons)))]))

    dog_is_home = m < 0.5
    rows = [
        strat("bet every underdog", dog_is_home),
        strat("bet every favourite", ~dog_is_home),
        strat("bet dogs priced > 2.40", dog_is_home,
              np.where(dog_is_home, dh, da) > 2.40),
        strat("bet dogs priced > 2.60", dog_is_home,
              np.where(dog_is_home, dh, da) > 2.60),
        strat("bet every home team", np.ones(len(g), bool)),
        strat("bet every away team", np.zeros(len(g), bool)),
    ]
    return pd.DataFrame([r for r in rows if r])


def model_vs_control(pr: pd.DataFrame, model: str, edge_thresh: float = 0.10,
                     seed: int = 0) -> dict:
    """Does the model's selection beat a price-matched random selection?

    Take the model's bets, then compare against betting the same number of
    games drawn from the same price band. If the two are indistinguishable,
    the ROI comes from the price band, not from the model.
    """
    g = pr[(pr.model == model) & pr.dec_home.notna() & pr.p_home.notna()]
    p, y = g.p_home.to_numpy(), g.home_win.to_numpy()
    dh, da = g.dec_home.to_numpy(), g.dec_away.to_numpy()
    ev_h, ev_a = p * dh - 1, (1 - p) * da - 1
    th = (ev_h > edge_thresh) & (ev_h >= ev_a)
    ta = (ev_a > edge_thresh) & (ev_a > ev_h)
    bet = th | ta
    if bet.sum() < 20:
        return {}

    dec_bet = np.where(th, dh, da)[bet]
    won = np.where(th[bet], y[bet] == 1, y[bet] == 0)
    model_roi = float(np.where(won, dec_bet - 1, -1).mean())

    # candidate pool: every side whose price falls in the model's price range
    lo, hi = dec_bet.min(), dec_bet.max()
    all_dec = np.concatenate([dh, da])
    all_won = np.concatenate([y == 1, y == 0])
    pool = (all_dec >= lo) & (all_dec <= hi)
    rng = np.random.default_rng(seed)
    pd_, pw = all_dec[pool], all_won[pool]
    sims = np.array([
        np.where(pw[i], pd_[i] - 1, -1).mean()
        for i in (rng.integers(0, pool.sum(), bet.sum()) for _ in range(3000))])
    return dict(
        model=model, edge_thresh=edge_thresh, bets=int(bet.sum()),
        model_roi=model_roi,
        price_band=f"{lo:.2f}-{hi:.2f}",
        control_roi=float(sims.mean()),
        control_ci=f"[{np.percentile(sims,2.5):+.3f}, {np.percentile(sims,97.5):+.3f}]",
        model_beats_control=float((model_roi > sims).mean()),
    )


def odds_quality(pr: pd.DataFrame, df: pd.DataFrame, model: str) -> tuple:
    """The control that decided this project.

    Action Network stamps each odds row with the time it was recorded. Only
    about half land within 15 minutes of first pitch; the rest are recorded
    hours later and are demonstrably worse prices -- the "market" baseline's
    own log loss degrades from 0.675 to 0.701 across those slices, which is
    the signature of a stale or mis-stamped line rather than a closing one.

    Any backtested profit that lives only in the stale slices is a data
    artifact. This function reports the market's own sharpness per slice and
    re-runs the edge curve on the clean subset, which is the only number
    worth quoting.
    """
    from sklearn.metrics import log_loss

    if "snap_lag_min" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    lag = df[["game_pk", "snap_lag_min"]].dropna()
    bins = [-1e9, 15, 60, 180, 1e9]
    labels = ["<=15 (clean)", "15-60", "60-180", ">180"]

    rows = []
    for name in ("market", model):
        m = pr[pr.model == name].merge(lag, on="game_pk", how="inner")
        m = m.dropna(subset=["p_home", "mkt_p_home"])
        m["slice"] = pd.cut(m.snap_lag_min, bins, labels=labels)
        for b, g in m.groupby("slice", observed=True):
            y = g.home_win.to_numpy()
            p = np.clip(g.p_home.to_numpy(), 1e-6, 1 - 1e-6)
            mk = np.clip(g.mkt_p_home.to_numpy(), 1e-6, 1 - 1e-6)
            rows.append(dict(model=name, slice=str(b), n=len(g),
                             logloss=log_loss(y, p, labels=[0, 1]),
                             acc=float(((p > .5) == y).mean()),
                             mkt_logloss=log_loss(y, mk, labels=[0, 1]),
                             mkt_acc=float(((mk > .5) == y).mean())))
    quality = pd.DataFrame(rows)

    clean_pks = set(lag[lag.snap_lag_min <= 15].game_pk)
    clean = pr[(pr.model == model) & pr.game_pk.isin(clean_pks)]
    curve = edge_curve(clean, model)
    curve.insert(0, "subset", "clean snapshots only")
    return quality, curve


def by_price_bucket(pr: pd.DataFrame, model: str) -> pd.DataFrame:
    """Where does the model beat the close -- on favourites or on dogs?"""
    g = pr[(pr.model == model) & pr.mkt_p_home.notna()].copy()
    g["bucket"] = pd.cut(g.mkt_p_home, [0, .40, .46, .50, .54, .60, 1.0],
                         labels=["big dog", "dog", "slight dog",
                                 "slight fav", "fav", "big fav"])
    rows = []
    for b, gg in g.groupby("bucket", observed=True):
        r = bet_sim(gg.p_home.to_numpy(), gg.home_win.to_numpy(),
                    gg.dec_home.to_numpy(), gg.dec_away.to_numpy(), edge_thresh=0.02)
        r.update(bucket=str(b), n=len(gg),
                 acc=float(((gg.p_home > .5).astype(int) == gg.home_win).mean()),
                 mkt_acc=float(((gg.mkt_p_home > .5).astype(int) == gg.home_win).mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def feature_importance(df, feat_cols, season_cut=2023, top=25):
    """Permutation-free importance: |standardised logistic coefficient| and
    LightGBM gain, both fit on pre-test seasons only."""
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tr = df[(df.season < season_cut) & (df.season != 2020)]
    X = SimpleImputer(strategy="median").fit_transform(tr[feat_cols].to_numpy(dtype=float))
    Xs = StandardScaler().fit_transform(X)
    y = tr.home_win.to_numpy()

    lr = LogisticRegression(C=0.03, max_iter=4000).fit(Xs, y)
    gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7,
                             max_depth=3, min_child_samples=250, reg_lambda=30,
                             verbose=-1, n_jobs=4, random_state=0).fit(X, y)
    imp = pd.DataFrame(dict(
        feature=feat_cols,
        logit_coef=lr.coef_[0],
        abs_coef=np.abs(lr.coef_[0]),
        gbm_gain=gbm.booster_.feature_importance("gain"),
    ))
    imp["gbm_gain"] /= imp.gbm_gain.sum()
    return imp.sort_values("abs_coef", ascending=False).head(top).reset_index(drop=True)


def market_residual_test(df: pd.DataFrame, feat_cols: list[str],
                         test_seasons=(2024, 2025, 2026)) -> pd.DataFrame:
    """The decisive question: do the free features add anything the closing
    line does not already contain?

    Fit a model with the market logit as a *fixed offset* -- it may only learn
    the residual. If the free data carries independent information, this beats
    the market on log loss. If it does not, the residual model lands on top of
    the market and the honest conclusion is that there is no edge here.
    """
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler

    d = df[df.mkt_p_home.notna() & df.home_win.notna()].copy()
    d["mkt_logit"] = np.log(np.clip(d.mkt_p_home, 1e-4, 1-1e-4) /
                            (1 - np.clip(d.mkt_p_home, 1e-4, 1-1e-4)))
    rows = []
    for s in test_seasons:
        tr, te = d[d.season < s], d[d.season == s]
        if len(tr) < 1500 or len(te) < 200:
            continue
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(tr[feat_cols].to_numpy(dtype=float))
        Xte = imp.transform(te[feat_cols].to_numpy(dtype=float))
        ytr, yte = tr.home_win.to_numpy(int), te.home_win.to_numpy(int)
        otr, ote = tr.mkt_logit.to_numpy(), te.mkt_logit.to_numpy()

        g = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.02, num_leaves=7,
                               max_depth=3, min_child_samples=300, reg_lambda=50,
                               verbose=-1, n_jobs=4, random_state=0)
        g.fit(Xtr, ytr, init_score=otr)
        p_gbm = 1 / (1 + np.exp(-(g.predict(Xte, raw_score=True) + ote)))

        sc = StandardScaler()
        lr = LogisticRegression(C=0.02, max_iter=4000)
        lr.fit(np.column_stack([sc.fit_transform(Xtr), otr]), ytr)
        p_lr = lr.predict_proba(np.column_stack([sc.transform(Xte), ote]))[:, 1]

        mkt = np.clip(te.mkt_p_home.to_numpy(), 1e-6, 1-1e-6)
        rows.append(dict(
            season=s, n=len(te),
            mkt_logloss=log_loss(yte, mkt, labels=[0, 1]),
            resid_gbm_logloss=log_loss(yte, np.clip(p_gbm, 1e-6, 1-1e-6), labels=[0, 1]),
            resid_logit_logloss=log_loss(yte, np.clip(p_lr, 1e-6, 1-1e-6), labels=[0, 1]),
            mkt_acc=float(((mkt > .5) == yte).mean()),
            resid_gbm_acc=float(((p_gbm > .5) == yte).mean()),
            mkt_coef=float(lr.coef_[0][-1]),
        ))
    r = pd.DataFrame(rows)
    if not r.empty:
        r["gbm_beats_mkt"] = r.mkt_logloss - r.resid_gbm_logloss
        r["logit_beats_mkt"] = r.mkt_logloss - r.resid_logit_logloss
    return r


def main():
    from . import features
    pr = pd.read_parquet(OUT / "predictions.parquet")
    df = pd.read_parquet(PROC / "dataset.parquet")
    df = df[df.has_both_sp]
    feat = features.feature_columns(df)
    fmt = lambda v: f"{v:,.4f}"

    print("=== FEATURE IMPORTANCE (fit on 2015-2022 only) ===")
    print(feature_importance(df, feat).to_string(index=False, float_format=fmt))

    if "mkt_p_home" in df.columns and df.mkt_p_home.notna().any():
        print("\n=== DO THE FREE FEATURES ADD ANYTHING THE CLOSE DOESN'T HAVE? ===")
        print("(model fit with the market logit as a fixed offset; positive "
              "'beats_mkt' = free data has independent information)")
        print(market_residual_test(df, feat).to_string(index=False, float_format=fmt))

    if pr.mkt_p_home.notna().any():
        print("\n=== VS CLOSING LINE ===")
        print(market_comparison(pr).to_string(index=False, float_format=fmt))

        # "market" trivially matches itself, so profile the best real model
        mc = market_comparison(pr)
        best = mc[mc.model != "market"].iloc[0].model
        print(f"\n=== EDGE CURVE: {best} ===")
        print(edge_curve(pr, best).to_string(index=False, float_format=fmt))
        print(f"\n=== BY PRICE BUCKET: {best} ===")
        print(by_price_bucket(pr, best).to_string(index=False, float_format=fmt))

        print(f"\n=== IS THE ROI REAL? bootstrap on {best} ===")
        for th in (0.03, 0.05, 0.10, 0.15):
            s = roi_significance(pr, best, th)
            if s.get("bets", 0) < 20:
                continue
            print(f"  edge>{th:.0%}: {s['bets']:>4} bets  "
                  f"ROI={s['roi']:+.3f} +/-{s['roi_se']:.3f}  "
                  f"95% CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]  "
                  f"P(ROI>0)={s['p_roi_positive']:.2f}  "
                  f"profitable seasons {s['seasons_profitable']}  "
                  f"avg price {s['avg_dec']:.2f}")
            print(f"            by season: {s['per_season']}")

        print("\n=== CONTROLS USING THE CLOSING LINE ALONE (no model) ===")
        print(naive_controls(pr, best).to_string(index=False, float_format=fmt))

        q, curve = odds_quality(pr, df, best)
        if not q.empty:
            print("\n=== ODDS QUALITY BY SNAPSHOT AGE ===")
            print("(the market's own log loss degrades in the stale slices -- "
                  "those are not closing lines)")
            print(q.to_string(index=False, float_format=fmt))
            print("\n=== EDGE CURVE ON CLEAN SNAPSHOTS ONLY (the honest number) ===")
            print(curve.to_string(index=False, float_format=fmt))

            clean_pks = set(df.loc[df.snap_lag_min <= 15, "game_pk"])
            clean_pr = pr[pr.game_pk.isin(clean_pks)]
            print("\n  bootstrap on that clean subset:")
            for th in (0.05, 0.10, 0.15):
                s = roi_significance(clean_pr, best, th)
                if s.get("bets", 0) < 20:
                    continue
                print(f"    edge>{th:.0%}: {s['bets']:>4} bets  ROI={s['roi']:+.3f}  "
                      f"95% CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]  "
                      f"P(ROI>0)={s['p_roi_positive']:.2f}  "
                      f"profitable seasons {s['seasons_profitable']}")
                print(f"              by season: {s['per_season']}")

        print(f"\n=== MODEL vs PRICE-MATCHED RANDOM SELECTION: {best} ===")
        for th in (0.05, 0.10, 0.15):
            c = model_vs_control(pr, best, th)
            if c:
                print(f"  edge>{th:.0%}: model ROI {c['model_roi']:+.3f} vs "
                      f"control {c['control_roi']:+.3f} {c['control_ci']} "
                      f"in price band {c['price_band']}  ->  "
                      f"model beats control in {c['model_beats_control']:.0%} of draws")

    print("\n=== CALIBRATION (best log-loss model) ===")
    from sklearn.metrics import log_loss
    valid = pr[pr.p_home.notna()]
    ll = valid.groupby("model", observed=True).apply(
        lambda g: log_loss(g.home_win, np.clip(g.p_home, 1e-6, 1-1e-6), labels=[0, 1]),
        include_groups=False)
    bm = ll.idxmin()
    g = valid[valid.model == bm]
    print(f"model = {bm}")
    print(calibration_table(g.p_home.to_numpy(), g.home_win.to_numpy())
          .to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
