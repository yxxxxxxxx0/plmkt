"""What is actually carrying the trajectory signal, and does it anticipate?

audit_label.py established two things: the 'jump' label is heavily contaminated
by spread-driven mid noise (prevalence 0.42 -> 0.98 across spread buckets), and
once that is controlled for by restricting to tight books the trajectory
advantage survives and in fact grows (C 0.731 -> D 0.871 PR-AUC).

Two questions follow, and they decide what the study is allowed to claim.

1. FEATURE FAMILY. The trajectory block mixes three very different things:
     price path   ret_*, rv_*  -- realised volatility. Volatility clusters;
                                  predicting a big move from recent big moves
                                  is a well known statistical fact and says
                                  nothing about the order book.
     book path    d_depth, d_spread, d_hhi, d_entropy, d_l1, d_imbalance
                               -- the actual hypothesis: liquidity weakening
                                  or concentrating before moves.
     activity     upd_rate_*
   If state+price alone reproduces the full trajectory model, the headline
   "LOB trajectory predicts jumps" is really "volatility clusters".

2. ANTICIPATION VS REACTION. A point at t whose move began at t-2s is trivially
   labelled. The clean test is to predict the label as computed at t+g using
   only features at t: the model must call a move that has not started. Gaps
   g = 0/5/10/30s are built by shifting labels forward WITHIN a series on the
   1Hz grid, joining on exact timestamp so no interpolation is involved.
   Skill decaying to the floor as g grows means the model reacts rather than
   anticipates -- which is what a zero lead time would imply.

    python audit_mechanism.py --J 0.02 --H 30 --stride 4
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
DROP = {"row", "session", "sid", "ts", "series", "market", "market_type",
        "book_age_ms"}
LABEL_PREFIX = ("jump_", "dir_", "valid_", "future_move")
PRICE_PREFIX = ("ret_", "rv_")
ACT_PREFIX = ("upd_rate_",)
BOOK_PREFIX = ("d_",)


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def families(cols):
    feats = [c for c in cols
             if c not in DROP and not c.startswith(LABEL_PREFIX)
             and not c.startswith("y_gap") and c != "split"]
    leaked = [c for c in feats if c.startswith("future_")]
    assert not leaked, "future column leaked: %s" % leaked
    price = [c for c in feats if c.startswith(PRICE_PREFIX)]
    act = [c for c in feats if c.startswith(ACT_PREFIX)]
    book = [c for c in feats if c.startswith(BOOK_PREFIX)]
    state = [c for c in feats if c not in price + act + book]
    return state, price, book, act


def fit_eval(X, y, tr, te, name, regime, gap=0, seed=0):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                           min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, n_jobs=4, verbose=-1)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    r = dict(regime=regime, gap_s=gap, model=name, n_train=int(tr.sum()),
             n_test=int(te.sum()), prevalence=float(y[te].mean()),
             pr_auc=float(average_precision_score(y[te], p)),
             roc_auc=float(roc_auc_score(y[te], p)))
    log("    %-28s gap%3ds  PR-AUC %.4f  ROC %.4f"
        % (name, gap, r["pr_auc"], r["roc_auc"]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--tight-ticks", type=float, default=2.0)
    ap.add_argument("--gaps", type=int, nargs="*", default=[0, 5, 10, 30])
    a = ap.parse_args()

    lab = "jump_%s_H%d" % (a.J, a.H)
    val = "valid_label_H%d" % a.H
    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    log("loading %d caches" % len(ps))
    D = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
    D = D[D[val]].reset_index(drop=True)
    log("{:,} valid-label rows before stride".format(len(D)))

    # Label shifting needs the FULL 1Hz grid, so build the gap targets before
    # subsampling rows. The join is on exact (sid, ts) -- no interpolation.
    key = D[["sid", "ts"]].copy()
    key["y0"] = D[lab].to_numpy(np.int8)
    D["y_gap0"] = key["y0"].to_numpy()
    for g in a.gaps:
        if g == 0:
            continue
        col = "y_gap%d" % g
        src = key.rename(columns={"y0": col}).copy()
        src["ts"] = src["ts"] - g * 1000        # label at t+g attaches to row t
        merged = D[["sid", "ts"]].merge(src[["sid", "ts", col]],
                                        on=["sid", "ts"], how="left")
        D[col] = merged[col].to_numpy()
        log("  gap %ds: %.3f of rows have a target" % (g, D[col].notna().mean()))

    D = D.iloc[::a.stride].reset_index(drop=True)
    log("{:,} rows after stride".format(len(D)))

    state, price, book, act = families(D.columns)
    log("state %d | price-path %d | book-path %d | activity %d"
        % (len(state), len(price), len(book), len(act)))
    log("  price-path: %s" % price)
    log("  activity:   %s" % act)

    split, _ = SP.assign(D, verbose=False)
    tr0, te0 = (split == "train"), (split == "test")
    tight = (D["spread_ticks"] <= a.tight_ticks).to_numpy()

    sets = [("C state only", state),
            ("D state+price(rv)", state + price),
            ("D state+book", state + book),
            ("D state+book+activity", state + book + act),
            ("D state+all trajectory", state + price + book + act)]

    rows = []
    for regime, mask in (("all books", np.ones(len(D), bool)),
                         ("tight<=%gt" % a.tight_ticks, tight)):
        # ---- part 1: which feature family carries the gain, at gap 0
        y = D["y_gap0"].to_numpy(np.int8)
        tr, te = tr0 & mask, te0 & mask
        log("--- %s: train %s test %s prev %.4f"
            % (regime, "{:,}".format(int(tr.sum())),
               "{:,}".format(int(te.sum())), y[te].mean()))
        rows.append(dict(regime=regime, gap_s=0, model="A prevalence floor",
                         n_train=int(tr.sum()), n_test=int(te.sum()),
                         prevalence=float(y[te].mean()),
                         pr_auc=float(y[te].mean()), roc_auc=0.5))
        for name, cols in sets:
            rows.append(fit_eval(D[cols].to_numpy(np.float32), y, tr, te,
                                 name, regime))

        # ---- part 2: anticipation gaps, state vs full trajectory
        for g in a.gaps:
            if g == 0:
                continue
            yv = D["y_gap%d" % g]
            ok = yv.notna().to_numpy() & mask
            y = yv.fillna(0).to_numpy(np.int8)
            trg, teg = tr0 & ok, te0 & ok
            if teg.sum() < 5000 or y[teg].sum() < 200:
                continue
            rows.append(dict(regime=regime, gap_s=g, model="A prevalence floor",
                             n_train=int(trg.sum()), n_test=int(teg.sum()),
                             prevalence=float(y[teg].mean()),
                             pr_auc=float(y[teg].mean()), roc_auc=0.5))
            for name, cols in (("C state only", state),
                               ("D state+all trajectory",
                                state + price + book + act)):
                rows.append(fit_eval(D[cols].to_numpy(np.float32), y, trg, teg,
                                     name, regime, gap=g))

    out = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    f = os.path.join(RES, "audit_mechanism_J%s_H%d.csv" % (a.J, a.H))
    out.to_csv(f, index=False)
    log("wrote %s" % f)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
