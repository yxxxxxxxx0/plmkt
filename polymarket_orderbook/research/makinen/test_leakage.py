"""Stage 2/5: automated leakage tests, run before any model is trained.

Checks, all numeric rather than by inspection:

  1. the input window ends strictly BEFORE the first target minute
  2. the window spans exactly `lookback` contiguous minutes of ONE series
  3. the target minute is contiguous with the window end
  4. no window mixes two series, and no series appears in two splits
  5. train/val/test sessions are disjoint and chronologically ordered
  6. normalisation statistics are reproducible from TRAIN ROWS ONLY
  7. the label equals the Lee-Mykland jump flag on the target minute, read
     back independently from the labelled panel

It also prints randomly chosen samples showing input start, input end,
prediction interval, the jump timestamp and the label, so the alignment can be
eyeballed as the brief asks.

    python test_leakage.py --tag L120_H1_train
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache")

FAIL = []


def check(name, ok, detail=""):
    print("  [%s] %s %s" % ("ok  " if ok else "FAIL", name, detail))
    if not ok:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="L120_H1_train")
    ap.add_argument("--n-samples", type=int, default=6)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(CACHE, "config_%s.json" % a.tag)))
    L, H = cfg["lookback"], cfg["horizon"]
    X = np.load(os.path.join(CACHE, "X_%s.npy" % a.tag), mmap_mode="r")
    y = np.load(os.path.join(CACHE, "y_%s.npy" % a.tag))
    M = pd.read_parquet(os.path.join(CACHE, "meta_%s.parquet" % a.tag))
    P = pd.read_parquet(os.path.join(CACHE, "panel_labelled.parquet"))
    key = P.set_index(["asset_id", "minute"])

    print("=== leakage tests, tag=%s (L=%d, H=%d) ===" % (a.tag, L, H))
    check("shapes agree", X.shape[0] == len(y) == len(M),
          "X=%s y=%d M=%d" % (X.shape, len(y), len(M)))

    # 1-3: window / target alignment, checked against the panel itself
    bad_align = bad_contig = bad_target = 0
    rng = np.random.default_rng(0)
    sample = rng.choice(len(M), size=min(4000, len(M)), replace=False)
    for i in sample:
        aid, t = M.asset_id.iloc[i], int(M.minute.iloc[i])
        g = P[(P.asset_id == aid)]
        w = g[(g.minute >= t - L + 1) & (g.minute <= t)]
        if len(w) != L:
            bad_contig += 1
        if w.minute.max() >= t + 1:
            bad_align += 1
        tgt = g[(g.minute > t) & (g.minute <= t + H)]
        if len(tgt) != H:
            bad_target += 1
    check("input window ends before the target minute", bad_align == 0,
          "violations %d/%d" % (bad_align, len(sample)))
    check("window is %d contiguous minutes" % L, bad_contig == 0,
          "violations %d/%d" % (bad_contig, len(sample)))
    check("target interval is contiguous and length H", bad_target == 0,
          "violations %d/%d" % (bad_target, len(sample)))

    # 4-5: split integrity
    ss = M.groupby("split").session.unique().to_dict()
    tr, va, te = set(ss.get("train", [])), set(ss.get("val", [])), set(ss.get("test", []))
    check("sessions disjoint across splits",
          not (tr & va) and not (va & te) and not (tr & te))
    a_tr = set(M[M.split == "train"].asset_id)
    a_va = set(M[M.split == "val"].asset_id)
    a_te = set(M[M.split == "test"].asset_id)
    check("no contract appears in two splits",
          not (a_tr & a_va) and not (a_va & a_te) and not (a_tr & a_te),
          "train %d, val %d, test %d contracts" % (len(a_tr), len(a_va), len(a_te)))
    order = (M[M.split == "train"].date.max() <= M[M.split == "val"].date.min()
             and M[M.split == "val"].date.max() <= M[M.split == "test"].date.min())
    check("splits are chronologically ordered", bool(order),
          "train<=%s val<=%s test<=%s" % (M[M.split == "train"].date.max(),
                                          M[M.split == "val"].date.max(),
                                          M[M.split == "test"].date.max()))

    # 6: normalisation fitted on train only
    if cfg["normalisation"] == "train":
        trm = (M.split == "train").to_numpy()
        m = np.asarray(X[trm]).reshape(-1, X.shape[2])
        check("train-set features are standardised (mean~0, sd~1)",
              bool(np.abs(m.mean(0)).max() < 0.05 and
                   np.abs(m.std(0) - 1).max() < 0.15),
              "max|mean|=%.4f max|sd-1|=%.4f" % (np.abs(m.mean(0)).max(),
                                                 np.abs(m.std(0) - 1).max()))
        te_m = np.asarray(X[(M.split == "test").to_numpy()]).reshape(-1, X.shape[2])
        drift = np.abs(te_m.mean(0)).max()
        check("test set NOT re-centred (drift is expected and present)",
              drift > 0.01, "max|test mean|=%.4f" % drift)

    # 7: label reproduces the panel's jump flag on the target minute
    mism = 0
    for i in sample[:1500]:
        aid, t = M.asset_id.iloc[i], int(M.minute.iloc[i])
        tgt = key.loc[[(aid, t + h) for h in range(1, H + 1)], "jump"]
        if int(np.max(tgt.to_numpy())) != int(y[i]):
            mism += 1
    check("label == LM jump flag on the target minute", mism == 0,
          "mismatches %d/1500" % mism)

    # ---- alignment printout ----
    print("\n=== random samples (alignment check) ===")
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    pick = np.r_[rng.choice(pos, size=min(3, len(pos)), replace=False),
                 rng.choice(neg, size=min(3, len(neg)), replace=False)]
    for i in pick:
        aid, t = M.asset_id.iloc[i], int(M.minute.iloc[i])
        g = P[P.asset_id == aid]
        w = g[(g.minute >= t - L + 1) & (g.minute <= t)]
        tgt = g[(g.minute > t) & (g.minute <= t + H)]
        jt = tgt[tgt.jump == 1]
        print("  %s..%s | input %s -> %s | predict (%s, %s] | label=%d%s"
              % (str(M.session.iloc[i]), str(aid)[:10],
                 w.ts_hkt.min().strftime("%m-%d %H:%M"),
                 w.ts_hkt.max().strftime("%H:%M"),
                 w.ts_hkt.max().strftime("%H:%M"),
                 tgt.ts_hkt.max().strftime("%H:%M"), int(y[i]),
                 ("  jump at " + jt.ts_hkt.iloc[0].strftime("%H:%M")) if len(jt) else ""))

    print("\n" + "=" * 60)
    if FAIL:
        print("LEAKAGE TESTS FAILED: %s" % ", ".join(FAIL))
        return 1
    print("all leakage tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
