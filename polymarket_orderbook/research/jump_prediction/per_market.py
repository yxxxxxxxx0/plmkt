"""Is the result carried by every held-out market, or by a few of them?

Section 16 question 8. A pooled PR-AUC can be produced by a handful of unusual
games, so the headline number is only trustworthy if the lift survives market
by market. Each held-out market is scored against ITS OWN prevalence, because
markets differ a lot in base rate and a pooled floor would flatter markets that
simply jump more often.

The model is the handcrafted trajectory model (D), refit on train exactly as in
run_clean.py, on the corrected clean+durable label in the tight-book regime.

    python per_market.py --J 0.02 --H 30 --label both
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import splits as SP  # noqa: E402
from run_clean import families, MOVE_COLS  # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "jump_prediction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--label", default="both",
                    choices=["raw", "clean", "durable", "both"])
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--tight-ticks", type=float, default=2.0)
    a = ap.parse_args()

    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    M = pd.read_parquet(os.path.join(CACHE, "moves_H%d.parquet" % a.H))
    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    D = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
    D = D[D["valid_label_H%d" % a.H]].reset_index(drop=True)
    D = D.merge(M[["sid", "ts"] + list(MOVE_COLS)], on=["sid", "ts"], how="inner")

    clean = D.clean_move >= a.J
    durable = D.durable_move >= a.J
    y_all = (clean if a.label == "clean" else durable if a.label == "durable"
             else clean & durable if a.label == "both" else D.raw_move >= a.J)
    D["y"] = y_all.to_numpy(np.int8)
    D = D[(D.spread_ticks <= a.tight_ticks)].iloc[::a.stride].reset_index(drop=True)

    state, price, book, act = families(D.columns)
    cols = state + price + book + act
    split, _ = SP.assign(D, verbose=False)
    tr, te = split == "train", split == "test"
    X = D[cols].to_numpy(np.float32)
    y = D.y.to_numpy(np.int8)

    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                           min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8,
                           random_state=0, n_jobs=8, verbose=-1,
                           class_weight="balanced")
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]

    sub = D[te].copy()
    sub["p"] = p
    rows = []
    for mk, g in sub.groupby("market", observed=True):
        if g.y.sum() < 20 or g.y.nunique() < 2:
            continue
        prev = float(g.y.mean())
        pr = float(average_precision_score(g.y, g.p))
        rows.append(dict(market=mk, n=len(g), prevalence=prev, pr_auc=pr,
                         pr_auc_lift=pr / max(prev, 1e-9),
                         roc_auc=float(roc_auc_score(g.y, g.p))))
    R = pd.DataFrame(rows).sort_values("pr_auc_lift", ascending=False)
    f = os.path.join(RES, "per_market_clean_J%s_H%d_%s.csv" % (a.J, a.H, a.label))
    R.to_csv(f, index=False)
    print(R.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\nmarkets: %d | lift median %.2f  min %.2f  max %.2f | "
          "markets with lift>1.2: %d/%d | ROC>0.6: %d/%d"
          % (len(R), R.pr_auc_lift.median(), R.pr_auc_lift.min(),
             R.pr_auc_lift.max(), int((R.pr_auc_lift > 1.2).sum()), len(R),
             int((R.roc_auc > 0.6).sum()), len(R)))
    print("wrote", f)


if __name__ == "__main__":
    sys.exit(main())
