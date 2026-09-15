"""5-fold grouped cross-validation over MATCHES.

The single 45/15/15 split gives one number with an unknown error bar, and
there is direct evidence that error bar is wide: on the previous test set 5 of
10 matches were profitable and the top 3 carried 97.6% of P&L. A result that
concentrated depends heavily on WHICH matches happened to land in test, so
"+0.3208 ticks/opp" could plausibly be +0.05 or +0.60 under a different
shuffle and the single split cannot tell you which.

This runs the whole thing K times instead:

    fold 1:  test = matches  1-15   train/val = the other 60
    fold 2:  test = matches 16-30   train/val = the other 60
    ...

Every match is tested exactly once, and the spread ACROSS folds is the error
bar the single split was missing. It also settles CNN vs GBM properly: both
see identical folds, so they can be compared per fold rather than on one draw.

"Grouped" means a match never splits across folds -- all of its rows travel
together. That is what keeps this honest; the original temporal split had 79
of 85 matches on both sides of the boundary and overfit accordingly
(+0.112 train-to-test AUC gap).

Early stopping needs a validation set, so each fold is NESTED: the 60
non-test matches split again into train and val. That means each fold trains
on ~48 matches rather than 60 -- slightly less data than the single split
used, which is the price of getting an error bar.

Data is loaded ONCE and the folds only re-index it; reloading per fold would
add ~10 minutes each on this machine.

    python cv_grouped.py --folds 5 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jump_model import frame_offsets
from jump_split import HAND, JD, TICK, load, subsample
from taker_signal import (label3, prepare, run_deep, sweep, taker_pnl,
                          touch_backing)


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logs", "cv_grouped.txt"), "a",
              encoding="utf-8") as fh:
        fh.write(line + "\n")


def fold_assignment(matches, k, seed=0):
    """Deterministically partition matches into k folds of near-equal size."""
    m = np.array(sorted(matches))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(m))
    folds = [set(m[order[i::k]]) for i in range(k)]
    return folds


def eval_edge(e, F, idx, tau):
    """Sign AUC and P&L per opportunity at a given threshold."""
    s = F.signed_ticks.to_numpy(np.float64)[idx]
    mv = s != 0
    auc = (roc_auc_score((s[mv] > 0).astype(int), e[mv])
           if mv.sum() > 200 and len(np.unique(s[mv] > 0)) > 1 else np.nan)
    pnl, side, traded = taker_pnl(e, F, idx, tau)
    hit = (((side > 0) == (s > 0))[traded & mv].mean()
           if (traded & mv).any() else np.nan)
    return dict(sign_auc=auc, pnl_per_opp=float(pnl.mean()),
                trade_frac=float(traded.mean()), hit=hit)


def pick_tau(e_val, F, va):
    R = sweep(e_val, F, va, np.round(np.arange(0.0, 0.96, 0.05), 3))
    return float(R.loc[R.pnl_per_opp.idxmax()].tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--label-ticks", type=int, default=4)
    ap.add_argument("--min-backing", type=float, default=200.0)
    ap.add_argument("--max-spread", type=float, default=2.0)
    ap.add_argument("--max-train", type=int, default=250_000)
    ap.add_argument("--max-val", type=int, default=120_000)
    ap.add_argument("--max-test", type=int, default=200_000)
    ap.add_argument("--max-rows-per-session", type=int, default=900_000)
    ap.add_argument("--val-frac", type=float, default=0.20,
                    help="fraction of the non-test matches used for the "
                         "early-stopping validation set")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pd.set_option("display.width", 240)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "logs"), exist_ok=True)

    log("=" * 78)
    log(f"GROUPED {a.folds}-FOLD CV  epochs<={a.epochs} patience={a.patience}")
    log("=" * 78)

    OFF, STR, _, lookback = frame_offsets(24, 24, 8)
    F, T, _tr, _va, _te = load(["moneyline", "total", "spread"],
                               lookback=lookback, log=log,
                               max_book_age_s=5.0, split="group",
                               use_trimmed=True,
                               max_rows_per_session=a.max_rows_per_session)
    F, h = prepare(F, 5.0, "sports", a.max_spread)

    # eligibility: same gates the single-split run uses, applied once
    elig = np.zeros(len(F), bool)
    elig[np.concatenate([_tr, _va, _te])] = True
    elig &= F.tight.to_numpy()
    log("computing touch backing...")
    bk = touch_backing(T)
    elig &= bk >= a.min_backing
    slug = F.series.astype(str).str.split("|").str[0].to_numpy()
    matches = sorted(set(slug[elig]))
    log(f"{elig.sum():,} eligible rows across {len(matches)} matches")

    folds = fold_assignment(matches, a.folds, seed=a.seed)
    y3 = label3(F.signed_ticks.to_numpy(np.float64), a.label_ticks)

    import lightgbm as lgb
    cols = [c for c in HAND if c in F.columns]
    rng = np.random.default_rng(a.seed)
    results = []

    for fi, test_ms in enumerate(folds):
        t0 = time.time()
        rest = [m for m in matches if m not in test_ms]
        nv = max(1, int(len(rest) * a.val_frac))
        perm = rng.permutation(len(rest))
        val_ms = set(np.array(rest)[perm[:nv]])
        train_ms = set(np.array(rest)[perm[nv:]])

        tr = np.where(elig & np.isin(slug, list(train_ms)))[0]
        va = np.where(elig & np.isin(slug, list(val_ms)))[0]
        te = np.where(elig & np.isin(slug, list(test_ms)))[0]
        tr = subsample(tr, a.max_train, seed=a.seed)
        va = subsample(va, a.max_val, seed=a.seed)
        te = subsample(te, a.max_test, seed=a.seed)

        log(f"\n--- fold {fi+1}/{a.folds}: "
            f"{len(train_ms)} train / {len(val_ms)} val / {len(test_ms)} test "
            f"matches | rows {len(tr):,}/{len(va):,}/{len(te):,} ---")

        # ---- CNN ---------------------------------------------------------
        out = []
        run_deep(T, F, tr, va, te, out, OFF, STR, a.label_ticks,
                 epochs=a.epochs, patience=a.patience, seed=a.seed,
                 tag=f"_cv{fi+1}")
        cnn_row = out[-1] if out else {}
        ce = np.load(os.path.join(JD, f"takeredge_cnn_direction_cv{fi+1}.npy"))
        # run_deep's report() already chose tau on val and scored test; recover
        # the same tau so the GBM is compared on equal terms
        tau = float(cnn_row.get("tau", 0.9))
        cnn = eval_edge(ce, F, te, tau)

        # ---- GBM on the identical fold ------------------------------------
        s_tr = F.signed_ticks.to_numpy(np.float64)[tr]
        fit = s_tr != 0
        g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=63, min_child_samples=200,
                               subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=5.0,
                               verbose=-1, n_jobs=8, random_state=a.seed)
        X = lambda i: np.nan_to_num(F.loc[i, cols].to_numpy(np.float32))
        g.fit(X(tr)[fit], (s_tr[fit] > 0).astype(int))
        gv = 2.0 * g.predict_proba(X(va))[:, 1] - 1.0
        gt = 2.0 * g.predict_proba(X(te))[:, 1] - 1.0
        gtau = pick_tau(gv, F, va)
        gbm = eval_edge(gt, F, te, gtau)

        # ---- momentum control ---------------------------------------------
        r = lambda x: 2.0 * pd.Series(x).rank(pct=True).to_numpy() - 1.0
        mv_ = -F.dmid_25.to_numpy(np.float64)
        mtau = pick_tau(r(mv_[va]), F, va)
        mom = eval_edge(r(mv_[te]), F, te, mtau)

        row = dict(fold=fi + 1, n_test_matches=len(test_ms), n_test=len(te),
                   cnn_auc=cnn["sign_auc"], cnn_pnl=cnn["pnl_per_opp"],
                   cnn_trade=cnn["trade_frac"], cnn_tau=tau,
                   gbm_auc=gbm["sign_auc"], gbm_pnl=gbm["pnl_per_opp"],
                   mom_auc=mom["sign_auc"], mom_pnl=mom["pnl_per_opp"],
                   minutes=round((time.time() - t0) / 60, 1))
        results.append(row)
        log(f"  fold {fi+1}: CNN auc {row['cnn_auc']:.4f} pnl "
            f"{row['cnn_pnl']:+.4f} | GBM auc {row['gbm_auc']:.4f} pnl "
            f"{row['gbm_pnl']:+.4f} | mom pnl {row['mom_pnl']:+.4f} "
            f"({row['minutes']} min)")
        pd.DataFrame(results).to_csv(os.path.join(JD, "cv_grouped.csv"),
                                     index=False)

    R = pd.DataFrame(results)
    log("\n" + "=" * 78)
    log("PER-FOLD RESULTS")
    log("=" * 78)
    log("\n" + R.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    log("\n" + "=" * 78)
    log("AGGREGATE  (mean +/- std across folds -- the error bar the single "
        "split could not give)")
    log("=" * 78)
    for name, ac, pc in (("CNN", "cnn_auc", "cnn_pnl"),
                         ("GBM", "gbm_auc", "gbm_pnl"),
                         ("momentum", "mom_auc", "mom_pnl")):
        log(f"  {name:>9}  AUC {R[ac].mean():.4f} +/- {R[ac].std():.4f}   "
            f"P&L {R[pc].mean():+.4f} +/- {R[pc].std():.4f} ticks/opp   "
            f"folds profitable {(R[pc] > 0).sum()}/{len(R)}")

    d = R.cnn_pnl - R.gbm_pnl
    log(f"\n  CNN minus GBM per fold: {', '.join(f'{x:+.4f}' for x in d)}")
    log(f"  mean {d.mean():+.4f} +/- {d.std():.4f}; "
        f"CNN wins {int((d > 0).sum())}/{len(d)} folds")
    if d.std() > 0:
        t = d.mean() / (d.std() / np.sqrt(len(d)))
        log(f"  paired t across folds: t = {t:.2f} "
            f"({'CNN better' if t > 2.5 else 'not distinguishable'})")
    log("\n  A model that only wins on some folds is a model whose single-"
        "split result was luck.")

    with open(os.path.join(JD, "cv_grouped_summary.json"), "w") as fh:
        json.dump(dict(config=vars(a), per_fold=results,
                       ts=time.strftime("%Y-%m-%dT%H:%M:%S")), fh, indent=2,
                  default=str)
    log(f"\nwrote data/jump/cv_grouped.csv and cv_grouped_summary.json")


if __name__ == "__main__":
    main()
