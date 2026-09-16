"""Is the 'jump' label measuring price movement, or measuring the spread?

The headline result of the first pass (results/jump_prediction/summary.md) was
PR-AUC 0.93 against a 0.62 prevalence floor. Two facts make that suspicious:

  * prevalence rises 0.42 -> 0.98 across spread buckets, and
  * median future_move rises 0.010 -> 0.245 across the same buckets.

future_move is defined on mid = (bid+ask)/2. In a book with a 50-tick spread a
single quote cancellation moves the mid by tens of ticks with no trade and no
information. So "jump" may be close to a deterministic function of spread, and
a model scoring well may only be reading book width.

This script tests that directly by adding a one-feature control:

  A         prevalence floor
  SPREAD    LightGBM on spread_ticks ALONE
  C         LightGBM on all current-state features
  D         LightGBM on state + trajectory features

If SPREAD ~= C ~= D then the study is measuring book width, not predicting
price moves, and the result has to be restated. Everything is run twice: on
all rows, and restricted to tight books where the mid is meaningful.

    python audit_label.py --J 0.02 --H 30
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import splits as SP  # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "jump_prediction")
TRAJ_PREFIX = ("d_", "ret_", "rv_", "upd_rate_")
DROP = {"row", "session", "sid", "ts", "series", "market", "market_type", "book_age_ms"}
LABEL_PREFIX = ("jump_", "dir_", "valid_", "future_move")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit_eval(Xtr, ytr, Xte, yte, name, seed=0):
    """LightGBM, train-only class weighting, PR-AUC/ROC-AUC on test."""
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    if Xtr.ndim == 1:
        Xtr, Xte = Xtr.reshape(-1, 1), Xte.reshape(-1, 1)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                           min_child_samples=200, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, n_jobs=4,
                           verbose=-1)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return dict(model=name, n_train=len(ytr), n_test=len(yte),
                prevalence=float(yte.mean()),
                pr_auc=float(average_precision_score(yte, p)),
                roc_auc=float(roc_auc_score(yte, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3, help="row subsample for speed")
    ap.add_argument("--tight-ticks", type=float, default=2.0)
    a = ap.parse_args()

    lab = f"jump_{a.J}_H{a.H}"
    val = f"valid_label_H{a.H}"
    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    log(f"loading {len(ps)} caches, stride {a.stride}")
    frames = []
    for p in ps:
        d = pd.read_parquet(p)
        d = d[d[val]].iloc[::a.stride]
        frames.append(d)
        log(f"  {os.path.basename(p)}: {len(d):,} rows kept")
    D = pd.concat(frames, ignore_index=True)
    del frames
    log(f"total {len(D):,} rows")

    feats = [c for c in D.columns if c not in DROP and not c.startswith(LABEL_PREFIX)]
    traj = [c for c in feats if c.startswith(TRAJ_PREFIX)]
    state = [c for c in feats if c not in traj]
    assert not [c for c in feats if c.startswith("future_")], "future column leaked"
    log(f"state features {len(state)}, trajectory features {len(traj)}")

    split, S = SP.assign(D, verbose=False)
    D["split"] = split
    y = D[lab].to_numpy(np.int8)
    tr, te = (split == "train"), (split == "test")
    log(f"train {tr.sum():,}  test {te.sum():,}")

    rows = []
    for tag, mask in (("all books", np.ones(len(D), bool)),
                      (f"tight (spread<={a.tight_ticks:g}t)",
                       (D.spread_ticks <= a.tight_ticks).to_numpy())):
        mtr, mte = tr & mask, te & mask
        if mte.sum() < 1000 or y[mte].sum() < 100:
            log(f"{tag}: too few rows/positives, skipping")
            continue
        log(f"--- {tag}: train {mtr.sum():,} test {mte.sum():,} "
            f"prevalence {y[mte].mean():.4f}")
        base = dict(model="A prevalence floor", n_train=int(mtr.sum()),
                    n_test=int(mte.sum()), prevalence=float(y[mte].mean()),
                    pr_auc=float(y[mte].mean()), roc_auc=0.5)
        rows.append(dict(regime=tag, **base))
        for name, cols in (("SPREAD only (1 feature)", ["spread_ticks"]),
                           ("C state", state),
                           ("D state+trajectory", state + traj)):
            X = D[cols].to_numpy(np.float32)
            r = fit_eval(X[mtr], y[mtr], X[mte], y[mte], name)
            rows.append(dict(regime=tag, **r))
            log(f"    {name:<26} PR-AUC {r['pr_auc']:.4f}  ROC-AUC {r['roc_auc']:.4f}")

    out = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    f = os.path.join(RES, f"audit_label_spread_J{a.J}_H{a.H}.csv")
    out.to_csv(f, index=False)
    log(f"wrote {f}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
