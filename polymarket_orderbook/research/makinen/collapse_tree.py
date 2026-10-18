"""Does the CNN earn its keep, or does a tree on the instantaneous book match it?

collapse_classify.py found that a logistic regression on the book at the single
collapse instant reached PR-AUC 0.593 against the CNN-LSTM-Attention's 0.614 --
i.e. the 20-second sequence and the attention machinery bought +0.021. This
checks the obvious follow-up: give a gradient-boosted tree the same
instantaneous book, and separately a small set of hand-made trajectory
summaries, and see whether either matches the neural models.

Same events, same chronological split, same test session, so the numbers drop
straight into the table in results/makinen/collapse/SUMMARY.md.

    python collapse_tree.py --min-move 0.02
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "jump_prediction"))
sys.path.insert(0, HERE)
import features as FE  # noqa: E402
from collapse_classify import TRACK, metrics, pick_thr, build_sequences  # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen", "collapse")


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-move", type=float, default=0.02)
    ap.add_argument("--pre-s", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    import lightgbm as lgb

    E = pd.read_parquet(os.path.join(CACHE, "collapse_events.parquet")).reset_index(drop=True)
    E["y"] = (E.durable_move >= a.min_move).astype(np.int8)
    X, keep = build_sequences(E, a.pre_s, a.stride)
    E = E.loc[keep].reset_index(drop=True)
    y = E.y.to_numpy(int)

    sess = sorted(E.session.unique())
    sp = np.where(E.session.isin(sess[:-2]), "train",
                  np.where(E.session.isin(sess[-2:-1]), "val", "test"))
    tr, va, te = sp == "train", sp == "val", sp == "test"
    log("train {:,} / val {:,} / test {:,}  base rate test {:.3f}"
        .format(int(tr.sum()), int(va.sum()), int(te.sum()), y[te].mean()))

    # --- A: the book at the collapse instant only
    inst = X[:, -1, :]
    names_i = ["%s_now" % c for c in TRACK]

    # --- B: instant + hand-made trajectory summaries over the same 20s window
    def summ(A, k):
        """level now, change over the last k steps, and window min/max/slope"""
        now = A[:, -1, :]
        chg = A[:, -1, :] - A[:, -k, :]
        return now, chg
    _, chg_1s = summ(X, 5)        # 5 steps * 2 * 200ms = 2s back
    _, chg_5s = summ(X, 13)
    _, chg_20s = summ(X, X.shape[1])
    wmin = X.min(axis=1); wmax = X.max(axis=1); wstd = X.std(axis=1)
    traj = np.concatenate([inst, chg_1s, chg_5s, chg_20s, wmin, wmax, wstd], axis=1)
    names_t = (names_i + ["%s_d1s" % c for c in TRACK] + ["%s_d5s" % c for c in TRACK]
               + ["%s_d20s" % c for c in TRACK] + ["%s_min" % c for c in TRACK]
               + ["%s_max" % c for c in TRACK] + ["%s_std" % c for c in TRACK])

    rows = []
    for tag, M, names in (("5 GBM (book at collapse only)", inst, names_i),
                          ("6 GBM (instant + trajectory summaries)", traj, names_t)):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                               min_child_samples=100, subsample=.8, subsample_freq=1,
                               colsample_bytree=.8, random_state=0, n_jobs=8,
                               verbose=-1, class_weight="balanced")
        m.fit(M[tr], y[tr])
        pv, pt = m.predict_proba(M[va])[:, 1], m.predict_proba(M[te])[:, 1]
        r = metrics(y[te], pt, pick_thr(y[va], pv))
        rows.append(dict(model=tag, **r))
        log("  %-40s PR-AUC %.4f (lift %.2fx) ROC %.4f"
            % (tag, r["pr_auc"], r["pr_auc_lift"], r["roc_auc"]))
        if "trajectory" in tag:
            imp = pd.DataFrame({"feature": names,
                                "gain": m.booster_.feature_importance("gain")})
            imp.sort_values("gain", ascending=False).head(25).to_csv(
                os.path.join(RES, "gbm_feature_importance.csv"), index=False)
            print("\n  top 12 features by gain:")
            print(imp.nlargest(12, "gain").to_string(index=False))

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "collapse_tree_%.2f.csv" % a.min_move), index=False)
    print()
    print(R[["model", "prevalence", "pr_auc", "pr_auc_lift", "roc_auc",
             "precision", "recall", "f1"]].to_string(
        index=False, float_format=lambda v: "%.4f" % v))


if __name__ == "__main__":
    sys.exit(main())
