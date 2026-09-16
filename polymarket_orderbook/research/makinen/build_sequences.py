"""Stages 2-5: target, features, sequences, chronological split, leakage tests.

Target (Stage 2)
----------------
    y_t = 1 if a Lee-Mykland jump occurs in (t, t+H] on the SAME series
Input X_t covers minutes [t-L+1, t] inclusive. Minute t+1 onwards never enters
X_t. The two are built from disjoint index ranges and `test_leakage.py` checks
that numerically rather than by inspection.

Sequences (Stage 4)
-------------------
L=120 minutes at 1-minute spacing, matching the paper. Windows are built inside
a single series only and every minute in [t-L+1, t+H] must be contiguous; a
window spanning a data gap is discarded rather than bridged.

Split (Stage 5)
---------------
Whole SESSIONS in chronological order. A session is one night's recording and
its contracts exist nowhere else, so a session boundary is also a contract
boundary -- no window can straddle a split, and no contract appears twice.
With six sessions the usable split is 4 / 1 / 1, which is coarse; that is a
property of the data and is reported rather than papered over.

Normalisation (Stage 4)
-----------------------
A. leakage-safe (default): per-feature mean/sd from TRAIN ROWS ONLY.
B. paper-style: per-sample normalisation, each window standardised using only
   its own 120 minutes. Also causal, but a different experiment -- selected
   with --normalisation sample and written to a separate cache.

    python build_sequences.py --lookback 120 --horizon 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen")

# Feature set actually available from this data. Event-flow features from the
# paper (order arrivals, cancellations, market-order intensities) are NOT here
# and are deliberately absent rather than approximated -- see
# results/makinen/features_available.md.
AGG = ["mid", "spread_ticks", "l1_bid_usd", "l1_ask_usd",
       "log_bid_depth", "log_ask_depth", "log_total_depth",
       "imbalance", "imbalance_l1", "hhi_bid", "hhi_ask",
       "entropy_bid", "entropy_ask", "slope_bid", "slope_ask",
       "levels_bid", "levels_ask", "reach_bid", "reach_ask",
       "log_bid_usd_within_2t", "log_ask_usd_within_2t",
       "near_frac_bid_2t", "near_frac_ask_2t"]
LEVELS = ([f"bid_usd_{i}" for i in range(1, 11)]
          + [f"ask_usd_{i}" for i in range(1, 11)]
          + [f"bid_dist_{i}" for i in range(1, 6)]
          + [f"ask_dist_{i}" for i in range(1, 6)])
GAPS = [f"bid_gap_{i}" for i in range(1, 4)] + [f"ask_gap_{i}" for i in range(1, 4)]
DELTA_BASE = ["mid", "spread_ticks", "imbalance", "log_total_depth",
              "log_bid_depth", "log_ask_depth"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def add_derived(P):
    """1-minute changes, short-horizon realised volatility, time of day.

    Every quantity is computed from the current and PAST minutes of the same
    series only; `shift` is strictly positive so nothing looks forward.
    """
    P = P.sort_values(["asset_id", "minute"]).reset_index(drop=True)
    g = P.groupby("asset_id", sort=False)
    contig = g.minute.diff() == 1
    for c in DELTA_BASE:
        d = g[c].diff()
        P["d_" + c] = np.where(contig, d, np.nan)
    r = P["d_mid"]
    for w in (5, 15):
        P[f"rv_{w}"] = (r.groupby(P.asset_id).transform(
            lambda s: s.rolling(w, min_periods=3).std()))
    P["tod_sin"] = np.sin(2 * np.pi * P.tod_min / 1440.0)
    P["tod_cos"] = np.cos(2 * np.pi * P.tod_min / 1440.0)
    return P


def feature_columns():
    d = ["d_" + c for c in DELTA_BASE]
    return AGG + LEVELS + GAPS + d + ["rv_5", "rv_15", "tod_sin", "tod_cos"]


def build_windows(P, feats, L, H):
    """Index-based window construction. Returns X, y, and a metadata frame."""
    Xs, ys, meta = [], [], []
    F = P[feats].to_numpy(np.float32)
    for aid, g in P.groupby("asset_id", sort=False, observed=True):
        idx = g.index.to_numpy()
        mins = g.minute.to_numpy(np.int64)
        jump = g.jump.to_numpy(np.int8)
        tested = g.tested.to_numpy(bool)
        n = len(idx)
        if n < L + H:
            continue
        for e in range(L - 1, n - H):           # e = index of minute t
            s = e - L + 1
            # every minute from window start to the end of the target must be
            # contiguous; never bridge a gap
            if mins[e] - mins[s] != L - 1:
                continue
            if mins[e + H] - mins[e] != H:
                continue
            tgt = slice(e + 1, e + H + 1)
            if not tested[tgt].any():           # target minute never testable
                continue
            win = F[idx[s]:idx[e] + 1]
            if win.shape[0] != L or not np.isfinite(win).all():
                continue
            Xs.append(win)
            ys.append(int(jump[tgt].max()))
            meta.append((aid, int(mins[e]), int(idx[e])))
    X = np.stack(Xs).astype(np.float32) if Xs else np.zeros((0, L, len(feats)), np.float32)
    y = np.array(ys, np.int8)
    M = pd.DataFrame(meta, columns=["asset_id", "minute", "row"])
    return X, y, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--normalisation", default="train",
                    choices=["train", "sample"])
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    tag = a.tag or ("L%d_H%d_%s" % (a.lookback, a.horizon, a.normalisation))

    P = pd.read_parquet(os.path.join(CACHE, "panel_labelled.parquet"))
    P = add_derived(P)
    feats = feature_columns()
    missing = [c for c in feats if c not in P.columns]
    assert not missing, "feature columns absent from panel: %s" % missing
    log("%d features, %s panel rows" % (len(feats), "{:,}".format(len(P))))

    # drop rows with non-finite features so windows are clean
    P = P.reset_index(drop=True)
    X, y, M = build_windows(P, feats, a.lookback, a.horizon)
    log("windows: {:,}  positives {:,} ({:.3%})".format(len(y), int(y.sum()),
                                                        y.mean() if len(y) else 0))
    if not len(y):
        raise SystemExit("no windows built")

    M = M.merge(P[["asset_id", "minute", "session", "date", "ts_hkt"]],
                on=["asset_id", "minute"], how="left")

    # ---- Stage 5: chronological split by whole session ----
    sessions = sorted(P.session.unique())
    n_tr = max(int(round(len(sessions) * 0.70)), 1)
    n_va = max(int(round(len(sessions) * 0.15)), 1)
    tr_s, va_s = sessions[:n_tr], sessions[n_tr:n_tr + n_va]
    te_s = sessions[n_tr + n_va:]
    split = np.where(M.session.isin(tr_s), "train",
                     np.where(M.session.isin(va_s), "val", "test"))
    M["split"] = split
    print("\n=== chronological split (whole sessions) ===")
    for nm, ss in (("train", tr_s), ("val", va_s), ("test", te_s)):
        sub = M[M.split == nm]
        print("  %-5s sessions=%s  windows=%s  positives=%s (%.3f%%)  dates %s..%s"
              % (nm, ss, "{:,}".format(len(sub)),
                 "{:,}".format(int(y[M.split == nm].sum())),
                 100 * y[M.split == nm].mean() if len(sub) else 0,
                 sub.date.min(), sub.date.max()))
    assert not (set(tr_s) & set(va_s)) and not (set(va_s) & set(te_s))

    # ---- Stage 4: normalisation ----
    tr = M.split.to_numpy() == "train"
    if a.normalisation == "train":
        flat = X[tr].reshape(-1, X.shape[2])
        mu, sd = flat.mean(0), flat.std(0)
        sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
        Xn = ((X - mu) / sd).astype(np.float32)
        norm = dict(kind="train", mu=mu.tolist(), sd=sd.tolist())
    else:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd < 1e-8] = 1.0
        Xn = ((X - mu) / sd).astype(np.float32)
        norm = dict(kind="sample")
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(os.path.join(CACHE, "X_%s.npy" % tag), Xn)
    np.save(os.path.join(CACHE, "y_%s.npy" % tag), y)
    M.to_parquet(os.path.join(CACHE, "meta_%s.parquet" % tag), index=False)
    cfg = dict(tag=tag, lookback=a.lookback, horizon=a.horizon,
               normalisation=a.normalisation, features=feats,
               n_windows=int(len(y)), prevalence=float(y.mean()),
               train_sessions=tr_s, val_sessions=va_s, test_sessions=te_s,
               shape=list(Xn.shape))
    with open(os.path.join(CACHE, "config_%s.json" % tag), "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(RES, "config_%s.json" % tag), "w") as fh:
        json.dump({k: v for k, v in cfg.items() if k != "features"}, fh, indent=2)
    log("saved X %s  y %s  tag=%s" % (Xn.shape, y.shape, tag))


if __name__ == "__main__":
    sys.exit(main())
