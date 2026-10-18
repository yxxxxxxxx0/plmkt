"""Phase 4: does temporal history help, before any neural network is built?

Four models, deliberately in increasing order of what they are allowed to see:

  A  prevalence        predict the training base rate for every row. The only
                       honest floor for a rare-event problem.
  B  logreg_state      logistic regression, CURRENT book geometry only
  C  tree_state        gradient-boosted trees, same current-state features
  D  tree_trajectory   same trees plus trailing changes over 5/10/30s

The comparison C -> D is the whole question in miniature. If trajectory
features do not beat current state here, a CNN+Transformer over the same
information is unlikely to, and that result should be reported rather than
explained away.

A label-shuffling control is run alongside: labels are permuted WITHIN each
market, which destroys the timing relationship while preserving prevalence and
market composition. Any model still scoring above baseline there is reading
something structural rather than predictive, and the pipeline is wrong.

    python baselines.py --J 0.02 --H 30
    python baselines.py --all-settings --tight-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import splits as SP          # noqa: E402
from evaluate import calibration, evaluate, pick_threshold  # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "jump_prediction")
GRID_MS = 200

TRAJ_PREFIX = ("d_", "ret_", "rv_", "upd_rate_")
DROP = {"row", "session", "sid", "ts", "series", "market", "market_type",
        "book_age_ms"}
# Anything derived from the future is banned from the feature set by PREFIX,
# not by exact name. The first version of this list held the literal string
# "future_move" while the cache actually stores future_move_H10/H30/H60 --
# so the models were handed a continuous version of their own label and
# scored ROC-AUC 1.0000. Prefix matching is the only safe form here.
LABEL_PREFIX = ("jump_", "dir_", "valid_", "future_move")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(sessions=None, columns=None):
    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    if sessions:
        ps = [p for p in ps if any(s in os.path.basename(p) for s in sessions)]
    if not ps:
        raise SystemExit("no cache; run build_dataset.py first")
    return pd.concat([pd.read_parquet(p, columns=columns) for p in ps],
                     ignore_index=True)


def feature_groups(cols):
    feats = [c for c in cols
             if c not in DROP and not c.startswith(LABEL_PREFIX)]
    traj = [c for c in feats if c.startswith(TRAJ_PREFIX)]
    state = [c for c in feats if c not in traj]
    leaked = [c for c in feats if c.startswith("future_")]
    assert not leaked, f"future-derived columns reached the feature set: {leaked}"
    return state, traj


def market_hours(D, mask):
    """Observation-hours represented by a mask, for false-alert rates."""
    return float(mask.sum()) / 3600.0 * 1.0      # points are 1 Hz


def fit_predict(kind, Xtr, ytr, Xva, Xte, seed=0):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced",
                               n_jobs=-1)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
        n_jobs=max(1, (os.cpu_count() or 4) - 2), verbose=-1)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m


def run_setting(D, split, J, H, state, traj, args, shuffled=False, rng=None):
    ycol = f"jump_{J:g}_H{H}"
    vcol = f"valid_label_H{H}"
    ok = D[vcol].to_numpy(bool)
    y = D[ycol].to_numpy(bool)
    if shuffled:
        # permute labels WITHIN each market: prevalence and market mix are
        # preserved, only the timing link to the book is destroyed
        y = y.copy()
        for _, idx in D.groupby("market", observed=True).indices.items():
            y[idx] = rng.permutation(y[idx])

    tr = ok & (split == "train")
    va = ok & (split == "val")
    te = ok & (split == "test")
    if min(tr.sum(), va.sum(), te.sum()) < 2000 or y[tr].sum() < 100:
        log(f"  J={J} H={H}: too few rows/positives, skipped")
        return []
    if args.max_train and tr.sum() > args.max_train:
        keep = rng.choice(np.where(tr)[0], args.max_train, replace=False)
        tr = np.zeros(len(D), bool)
        tr[keep] = True

    rows = []
    prev_tr = float(y[tr].mean())
    # ---- A: prevalence floor --------------------------------------------- #
    p_va = np.full(va.sum(), prev_tr)
    p_te = np.full(te.sum(), prev_tr)
    rows.append(evaluate("A_prevalence", y[te], p_te, prev_tr,
                         hours=market_hours(D, te),
                         extra=dict(J=J, H=H, shuffled=shuffled)))

    norm = SP.Normaliser(state + traj).fit(D[state + traj].to_numpy(), split)
    A = norm.transform(D[state + traj].to_numpy())
    si = [(state + traj).index(c) for c in state]
    ti = [(state + traj).index(c) for c in (state + traj)]

    for name, cols_idx, kind in (("B_logreg_state", si, "logreg"),
                                 ("C_tree_state", si, "tree"),
                                 ("D_tree_trajectory", ti, "tree")):
        if name.endswith("trajectory") and not traj:
            continue
        t0 = time.time()
        pv, pt, _ = fit_predict(kind, A[tr][:, cols_idx], y[tr],
                                A[va][:, cols_idx], A[te][:, cols_idx],
                                seed=args.seed)
        thr = pick_threshold(y[va], pv)          # validation only
        r = evaluate(name, y[te], pt, thr, hours=market_hours(D, te),
                     extra=dict(J=J, H=H, shuffled=shuffled,
                                n_features=len(cols_idx),
                                fit_s=round(time.time() - t0, 1)))
        rows.append(r)
        log(f"  {name:>18} J={J:g} H={H:>2}  PR-AUC {r['pr_auc']:.4f} "
            f"(lift {r['pr_auc_lift']:.2f}x)  ROC {r['roc_auc']:.4f}  "
            f"P@1% {r['prec_top1pct']:.3f}  {r['fit_s']}s"
            + ("   [SHUFFLED]" if shuffled else ""))
        if not shuffled and name == "D_tree_trajectory":
            calibration(y[te], pt).to_csv(
                os.path.join(RES, f"calibration_D_J{J:g}_H{H}.csv"), index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--J", type=float, default=None)
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--all-settings", action="store_true")
    ap.add_argument("--tight-only", action="store_true",
                    help="spread <= 2 ticks: where a mid move is a repricing "
                         "rather than an illiquid quote wobbling")
    ap.add_argument("--max-train", type=int, default=1_200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-shuffle", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    D = load(a.sessions)
    state, traj = feature_groups(D.columns)
    log(f"{len(D):,} prediction points, {len(state)} state + {len(traj)} "
        f"trajectory features")
    if a.tight_only:
        D = D[D.spread_ticks <= 2.0].reset_index(drop=True)
        log(f"tight books only: {len(D):,} rows")

    split, S = SP.assign(D)
    S.to_csv(os.path.join(RES, "split_markets.csv"), index=False)

    settings = ([(J, H) for H in (10, 30, 60) for J in (0.01, 0.02, 0.03, 0.05)]
                if a.all_settings else [(a.J or 0.02, a.H or 30)])
    rows = []
    for J, H in settings:
        rows += run_setting(D, split, J, H, state, traj, a, rng=rng)
    if not a.skip_shuffle:
        J, H = settings[len(settings) // 2]
        log("label-shuffle control (labels permuted within each market)...")
        rows += run_setting(D, split, J, H, state, traj, a,
                            shuffled=True, rng=rng)

    R = pd.DataFrame(rows)
    tag = "_tight" if a.tight_only else ""
    p = os.path.join(RES, f"baseline_metrics{tag}.csv")
    R.to_csv(p, index=False)
    log(f"wrote {p}")

    show = ["model", "J", "H", "prevalence", "pr_auc", "pr_auc_lift", "roc_auc",
            "precision", "recall", "f1", "prec_top1pct", "prec_top5pct",
            "brier_skill", "shuffled"]
    print("\n" + R[show].to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))

    # the headline question, stated plainly
    real = R[~R.shuffled]
    for (J, H), g in real.groupby(["J", "H"]):
        c = g[g.model == "C_tree_state"]
        d = g[g.model == "D_tree_trajectory"]
        if len(c) and len(d):
            dc = float(d.pr_auc.iloc[0] - c.pr_auc.iloc[0])
            print(f"\nJ={J:g} H={H}s  trajectory vs current state: "
                  f"PR-AUC {c.pr_auc.iloc[0]:.4f} -> {d.pr_auc.iloc[0]:.4f} "
                  f"({dc:+.4f}, {dc/max(c.pr_auc.iloc[0],1e-9):+.1%})")


if __name__ == "__main__":
    main()
