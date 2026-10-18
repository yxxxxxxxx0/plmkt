"""Given a liquidity collapse, is it REAL or does the price snap back?

Candidates come from collapse_events.py. For each one the input is the book in
the `--pre-s` seconds ENDING AT the collapse instant, at 200 ms resolution.
Nothing after the collapse enters the input; the collapse detector itself is a
trailing-window rule, so the candidate set is causal too.

Target: `real` = the displacement reaches --min-move within the horizon AND
still holds --hold-s later. The negatives are dominated by the interesting case
-- the mid ticked mechanically because the book emptied, then liquidity
returned and the price reverted.

Split is chronological by whole session; a contract lives in exactly one
session, so no contract can straddle the boundary.

    python collapse_classify.py --min-move 0.02 --pre-s 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "jump_prediction"))
sys.path.insert(0, HERE)
import features as FE  # noqa: E402
import models as MD  # noqa: E402

JD = os.path.join(ROOT, "data", "jump")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen", "collapse")
GRID_MS = 200

TRACK = ["spread_ticks", "l1_bid_usd", "l1_ask_usd", "log_bid_depth",
         "log_ask_depth", "log_total_depth", "imbalance", "imbalance_l1",
         "hhi_bid", "hhi_ask", "entropy_bid", "entropy_ask", "slope_bid",
         "slope_ask", "levels_bid", "levels_ask", "reach_bid", "reach_ask",
         "log_bid_usd_within_2t", "log_ask_usd_within_2t",
         "near_frac_bid_2t", "near_frac_ask_2t"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def build_sequences(E, pre_s, stride):
    steps = int(pre_s * 1000 / GRID_MS / stride)
    offs = (np.arange(-steps + 1, 1) * stride)
    Xs, keep = [], []
    for sess, Es in E.groupby("session", sort=False):
        fp = os.path.join(JD, "feat_%s_trimmed.parquet" % sess)
        tp = os.path.join(JD, "lob_%s_trimmed.npy" % sess)
        F = pq.read_table(fp, columns=["sid", "mid", "spread_ticks"]).to_pandas()
        T = np.load(tp, mmap_mode="r")
        sid = F.sid.to_numpy(); mid = F.mid.to_numpy(np.float64)
        spt = F.spread_ticks.to_numpy(); n = len(F)
        rows = Es.row.to_numpy()
        grid = rows[:, None] + offs[None, :]
        ok = (grid >= 0) & (grid < n)
        g2 = np.clip(grid, 0, n - 1)
        ok &= (sid[g2] == sid[rows][:, None])
        good = ok.all(axis=1)
        g2 = g2[good]
        flat = g2.reshape(-1)
        d = FE.decode(np.asarray(T[flat]), mid[flat])
        S = pd.DataFrame(FE.state_features(d, spread_ticks=spt[flat]))
        A = S[TRACK].to_numpy(np.float32).reshape(g2.shape[0], len(offs), len(TRACK))
        Xs.append(A)
        keep.append(Es.index.to_numpy()[good])
        log("  %s: %s of %s events had a complete window"
            % (sess, "{:,}".format(int(good.sum())), "{:,}".format(len(Es))))
        del F, T, S, A
    return np.concatenate(Xs), np.concatenate(keep)


def metrics(y, p, thr):
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 confusion_matrix)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    ap = float(average_precision_score(y, p))
    return dict(precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12),
                pr_auc=ap, pr_auc_lift=ap / max(y.mean(), 1e-12),
                roc_auc=float(roc_auc_score(y, p)), prevalence=float(y.mean()),
                tp=int(tp), fp=int(fp), fn=int(fn))


def pick_thr(yv, pv):
    from sklearn.metrics import f1_score
    grid = np.unique(np.quantile(pv, np.linspace(.3, .999, 100)))
    best, bt = -1, .5
    for t in grid:
        f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-move", type=float, default=0.02)
    ap.add_argument("--pre-s", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    np.random.seed(a.seed); torch.manual_seed(a.seed)

    E = pd.read_parquet(os.path.join(CACHE, "collapse_events.parquet"))
    E = E.reset_index(drop=True)
    E["y"] = (E.durable_move >= a.min_move).astype(np.int8)
    log("{:,} collapse events, base rate {:.3f}".format(len(E), E.y.mean()))

    X, keep = build_sequences(E, a.pre_s, a.stride)
    E = E.loc[keep].reset_index(drop=True)
    y = E.y.to_numpy(int)
    log("sequences %s  base rate %.3f" % (str(X.shape), y.mean()))

    sess = sorted(E.session.unique())
    tr_s, va_s, te_s = sess[:-2], sess[-2:-1], sess[-1:]
    sp = np.where(E.session.isin(tr_s), "train",
                  np.where(E.session.isin(va_s), "val", "test"))
    tr, va, te = sp == "train", sp == "val", sp == "test"
    log("train %s (%s) / val %s / test %s (%s)"
        % ("{:,}".format(int(tr.sum())), tr_s, "{:,}".format(int(va.sum())),
           "{:,}".format(int(te.sum())), te_s))

    flat = X[tr].reshape(-1, X.shape[2])
    mu, sd = flat.mean(0), flat.std(0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    X = np.nan_to_num((X - mu) / sd, nan=0., posinf=0., neginf=0.).astype(np.float32)

    rows = []
    rows.append(dict(model="0 prevalence", prevalence=float(y[te].mean()),
                     pr_auc=float(y[te].mean()), pr_auc_lift=1.0, roc_auc=.5,
                     precision=np.nan, recall=np.nan, f1=np.nan))

    from sklearn.linear_model import LogisticRegression
    Xs = X[:, -1, :]
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs[tr], y[tr])
    pv, pt = lr.predict_proba(Xs[va])[:, 1], lr.predict_proba(Xs[te])[:, 1]
    rows.append(dict(model="1 logistic (book at collapse)",
                     **metrics(y[te], pt, pick_thr(y[va], pv))))

    from sklearn.metrics import average_precision_score
    for nm, label in (("cnn", "2 CNN"), ("cnn_lstm", "3 CNN-LSTM"),
                      ("cnn_lstm_attention", "4 CNN-LSTM-Attention")):
        torch.manual_seed(a.seed)
        m = MD.build(nm, X.shape[2], X.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        pw = torch.tensor([(tr.sum() - y[tr].sum()) / max(y[tr].sum(), 1)],
                          dtype=torch.float32)
        lf = nn.BCEWithLogitsLoss(pos_weight=pw)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X[tr]),
                torch.from_numpy(y[tr].astype(np.float32))),
            batch_size=a.batch, shuffle=True)
        Xv = torch.from_numpy(X[va])
        best, bs, bad = -1, None, 0
        for ep in range(a.epochs):
            m.train()
            for xb, yb in dl:
                opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pv = torch.sigmoid(m(Xv)).numpy()
            s = float(average_precision_score(y[va], pv))
            log("    %-22s ep%2d val PR-AUC %.4f" % (label, ep + 1, s))
            if s > best:
                best, bad = s, 0
                bs = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        m.load_state_dict(bs); m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(Xv)).numpy()
            pt = np.concatenate([torch.sigmoid(m(torch.from_numpy(X[te][i:i+1024])))
                                 .numpy() for i in range(0, int(te.sum()), 1024)])
        rows.append(dict(model=label, **metrics(y[te], pt, pick_thr(y[va], pv))))
        np.save(os.path.join(RES, "preds_%s.npy" % nm), pt)

    R = pd.DataFrame(rows)
    R["min_move"] = a.min_move; R["pre_s"] = a.pre_s
    R.to_csv(os.path.join(RES, "collapse_classify_%.2f.csv" % a.min_move), index=False)
    print()
    print(R[["model", "prevalence", "pr_auc", "pr_auc_lift", "roc_auc",
             "precision", "recall", "f1"]].to_string(
        index=False, float_format=lambda v: "%.4f" % v))
    E[te].to_parquet(os.path.join(RES, "test_events.parquet"), index=False)


if __name__ == "__main__":
    sys.exit(main())
