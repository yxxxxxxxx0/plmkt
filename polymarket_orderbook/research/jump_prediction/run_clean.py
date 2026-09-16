"""The corrected headline experiment: jumps that are real repricings.

Why this file exists. The original label,

    future_move(t,H) = max over t<u<=t+H of |mid_u - mid_t|,

counts a mid excursion however briefly it existed and however broken the book
was when it happened. audit_durability.py measured the consequence: in the
tight-book regime only 24.5% of labelled jumps are ever observed while the book
is still tight, and only 42.8% are still displaced at the end of the window.
audit_durability's own recomputation is a strict lower bound (cached >=
recomputed in 100.0% of rows, because the cached label is computed on the
finer 200ms grid), so the artefact share is if anything larger.

The corrected labels, all built in audit_durability.py and cached in
moves_H30.parquet:

    clean    max |mid_u - mid_t| over u whose own book is tight
             (spread_u <= 2 ticks). A price that only exists while one side of
             the book is empty does not count.
    durable  the displacement must persist >= hold seconds.
    both     clean AND durable -- the strictest, and the primary label here.

Prediction points are additionally restricted to books that are tight at t, so
the model cannot win by detecting "this book is already broken".

Everything else -- whole-market chronological splits, train-only normalisation
and thresholds, the feature families, the anticipation gaps -- is unchanged
from audit_mechanism.py so the numbers are directly comparable.

    python run_clean.py --J 0.02 --H 30 --label both
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
MOVE_COLS = ("raw_move", "clean_move", "durable_move", "endpoint_move")


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def families(cols):
    feats = [c for c in cols
             if c not in DROP and not c.startswith(LABEL_PREFIX)
             and not c.startswith("y_gap") and c not in MOVE_COLS
             and c not in ("split", "y")]
    leaked = [c for c in feats
              if c.startswith("future_") or c.endswith("_move")]
    assert not leaked, "future-derived column reached features: %s" % leaked
    price = [c for c in feats if c.startswith(PRICE_PREFIX)]
    act = [c for c in feats if c.startswith(ACT_PREFIX)]
    book = [c for c in feats if c.startswith(BOOK_PREFIX)]
    state = [c for c in feats if c not in price + act + book]
    return state, price, book, act


def metrics(y, p, hours):
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 brier_score_loss)
    prev = float(y.mean())
    out = dict(prevalence=prev,
               pr_auc=float(average_precision_score(y, p)),
               pr_auc_lift=float(average_precision_score(y, p) / max(prev, 1e-9)),
               roc_auc=float(roc_auc_score(y, p)),
               brier=float(brier_score_loss(y, p)))
    out["brier_skill"] = 1.0 - out["brier"] / max(prev * (1 - prev), 1e-9)
    order = np.argsort(-p)
    for q in (0.01, 0.05, 0.10):
        k = max(int(len(p) * q), 1)
        out["prec_top%dpct" % int(q * 100)] = float(y[order[:k]].mean())
    return out


def fit_eval(X, y, tr, va, te, name, regime, gap, hours, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                           min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, n_jobs=4, verbose=-1,
                           class_weight="balanced")
    m.fit(X[tr], y[tr])
    pv, pt = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
    # class_weight="balanced" makes the classifier rank well but report
    # probabilities inflated towards the positive class, which shows up as a
    # NEGATIVE Brier skill -- worse than quoting the base rate. Isotonic
    # regression fitted on VALIDATION ONLY restores calibration without
    # touching the ranking (it is monotone, so PR-AUC and ROC-AUC are
    # unchanged). The uncalibrated Brier is kept alongside so the effect of
    # this step is visible rather than hidden.
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, f1_score
    brier_raw = float(brier_score_loss(y[te], pt))
    if len(np.unique(y[va])) > 1:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(pv, y[va])
        pv, pt = iso.predict(pv), iso.predict(pt)
    grid = np.quantile(pv, np.linspace(0.50, 0.999, 60))
    thr = float(max(grid, key=lambda t: f1_score(y[va], pv >= t, zero_division=0)))
    pred = pt >= thr
    tp = int((pred & (y[te] == 1)).sum())
    fp = int((pred & (y[te] == 0)).sum())
    fn = int((~pred & (y[te] == 1)).sum())
    r = dict(regime=regime, gap_s=gap, model=name, n_train=int(tr.sum()),
             n_test=int(te.sum()), threshold=thr,
             alert_rate=float(pred.mean()),
             precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))
    r["f1"] = 2 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-9)
    r["false_alerts_per_market_hour"] = fp / max(hours, 1e-9)
    r.update(metrics(y[te], pt, hours))
    r["brier_uncalibrated"] = brier_raw
    log("    %-28s gap%3ds  PR-AUC %.4f (lift %.2fx)  ROC %.4f  P@1%% %.3f"
        % (name, gap, r["pr_auc"], r["pr_auc_lift"], r["roc_auc"],
           r["prec_top1pct"]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--label", default="both",
                    choices=["raw", "clean", "durable", "both"])
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--tight-ticks", type=float, default=2.0)
    ap.add_argument("--gaps", type=int, nargs="*", default=[0, 5, 10, 30])
    ap.add_argument("--shuffle-control", action="store_true",
                    help="permute labels within market; skill must collapse")
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap training rows to match the deep models' budget. "
                         "The first pass trained trees on ~800k rows and the "
                         "neural models on 80k for IO reasons, which made the "
                         "trees-vs-deep comparison unfair in the trees' "
                         "favour; setting this to train.py's --max-train makes "
                         "it matched. 0 = use everything.")
    a = ap.parse_args()

    mv = os.path.join(CACHE, "moves_H%d.parquet" % a.H)
    if not os.path.exists(mv):
        raise SystemExit("run audit_durability.py --H %d first" % a.H)
    M = pd.read_parquet(mv)
    log("moves table {:,} rows".format(len(M)))

    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    D = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
    D = D[D["valid_label_H%d" % a.H]].reset_index(drop=True)
    D = D.merge(M[["sid", "ts"] + list(MOVE_COLS)], on=["sid", "ts"], how="inner")
    log("{:,} rows after joining decomposed moves".format(len(D)))

    if a.label == "raw":
        y_all = (D.raw_move >= a.J)
    elif a.label == "clean":
        y_all = (D.clean_move >= a.J)
    elif a.label == "durable":
        y_all = (D.durable_move >= a.J)
    else:
        y_all = (D.clean_move >= a.J) & (D.durable_move >= a.J)
    D["y"] = y_all.to_numpy(np.int8)
    log("label '%s' prevalence (all rows): %.4f" % (a.label, D.y.mean()))

    D = D.sort_values(["sid", "ts"]).reset_index(drop=True)
    key = D[["sid", "ts"]].copy()
    key["y0"] = D.y.to_numpy()
    D["y_gap0"] = key.y0.to_numpy()
    for g in a.gaps:
        if g == 0:
            continue
        col = "y_gap%d" % g
        src = key.rename(columns={"y0": col}).copy()
        src["ts"] = src["ts"] - g * 1000
        D[col] = D[["sid", "ts"]].merge(src[["sid", "ts", col]],
                                        on=["sid", "ts"], how="left")[col].to_numpy()

    D = D.iloc[::a.stride].reset_index(drop=True)
    log("{:,} rows after stride".format(len(D)))

    state, price, book, act = families(D.columns)
    log("state %d | price %d | book %d | activity %d"
        % (len(state), len(price), len(book), len(act)))

    split, _ = SP.assign(D, verbose=True)
    tr0 = split == "train"
    va0 = split == "val"
    te0 = split == "test"
    tight_t = (D.spread_ticks <= a.tight_ticks).to_numpy()

    sets = [("SPREAD only (1 feature)", ["spread_ticks"]),
            ("C state only", state),
            ("D state+price(rv)", state + price),
            ("D state+book", state + book),
            ("D state+book+activity", state + book + act),
            ("D state+all trajectory", state + price + book + act)]

    rows = []
    for regime, mask in (("tight at t (primary)", tight_t),
                         ("all books", np.ones(len(D), bool))):
        y = D["y_gap0"].to_numpy(np.int8)
        tr, va, te = tr0 & mask, va0 & mask, te0 & mask
        if a.max_train and tr.sum() > a.max_train:
            # subsample train rows only; val and test keep natural prevalence
            sel = np.where(tr)[0]
            drop = np.random.default_rng(0).choice(
                sel, size=len(sel) - a.max_train, replace=False)
            tr = tr.copy()
            tr[drop] = False
            log("  train capped to {:,} rows to match the deep budget"
                .format(int(tr.sum())))
        hours = int(te.sum()) / 3600.0
        log("--- %s | label=%s | train %s val %s test %s | prevalence %.4f"
            % (regime, a.label, "{:,}".format(int(tr.sum())),
               "{:,}".format(int(va.sum())), "{:,}".format(int(te.sum())),
               y[te].mean()))
        rows.append(dict(regime=regime, gap_s=0, model="A prevalence floor",
                         n_train=int(tr.sum()), n_test=int(te.sum()),
                         prevalence=float(y[te].mean()),
                         pr_auc=float(y[te].mean()), pr_auc_lift=1.0,
                         roc_auc=0.5))
        for name, cols in sets:
            rows.append(fit_eval(D[cols].to_numpy(np.float32), y, tr, va, te,
                                 name, regime, 0, hours))

        if a.shuffle_control and regime.startswith("tight"):
            rng = np.random.default_rng(0)
            ys = y.copy()
            for _, g in D.groupby("market", observed=True, sort=False):
                i = g.index.to_numpy()
                ys[i] = rng.permutation(ys[i])
            r = fit_eval(D[state + price + book + act].to_numpy(np.float32),
                         ys, tr, va, te, "SHUFFLED-LABEL control", regime, 0,
                         hours)
            rows.append(r)

        for g in a.gaps:
            if g == 0:
                continue
            yv = D["y_gap%d" % g]
            ok = yv.notna().to_numpy() & mask
            y = yv.fillna(0).to_numpy(np.int8)
            trg, vag, teg = tr0 & ok, va0 & ok, te0 & ok
            if teg.sum() < 5000 or y[teg].sum() < 100:
                continue
            hours = int(teg.sum()) / 3600.0
            rows.append(dict(regime=regime, gap_s=g, model="A prevalence floor",
                             n_train=int(trg.sum()), n_test=int(teg.sum()),
                             prevalence=float(y[teg].mean()),
                             pr_auc=float(y[teg].mean()), pr_auc_lift=1.0,
                             roc_auc=0.5))
            for name, cols in (("C state only", state),
                               ("D state+all trajectory",
                                state + price + book + act)):
                rows.append(fit_eval(D[cols].to_numpy(np.float32), y, trg, vag,
                                     teg, name, regime, g, hours))

    out = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    f = os.path.join(RES, "clean_metrics_%s_J%s_H%d%s.csv"
                     % (a.label, a.J, a.H,
                        "_matched" if a.max_train else ""))
    out.to_csv(f, index=False)
    log("wrote %s" % f)
    show = ["regime", "gap_s", "model", "prevalence", "pr_auc", "pr_auc_lift",
            "roc_auc", "precision", "recall", "prec_top1pct",
            "false_alerts_per_market_hour"]
    print()
    print(out[[c for c in show if c in out.columns]]
          .to_string(index=False, float_format=lambda v: "%.4f" % v))


if __name__ == "__main__":
    sys.exit(main())
