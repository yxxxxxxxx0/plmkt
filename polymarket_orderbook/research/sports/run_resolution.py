"""Label, sequence, train and evaluate at ONE bar resolution.

Run once per resolution; `sweep_resolutions.py` collects the results. The point
of the sweep is to answer whether a coarser bar gives a cleaner jump signal, so
everything except the bar size is held fixed:

  * K is constant in BARS (not in seconds). Bipower variation then has the same
    statistical properties at every resolution and the only thing that varies is
    the sampling rate -- which is what is under test. The wall-clock span of the
    window is reported so the change in meaning stays visible.
  * lookback is constant in BARS, horizon is 1 bar.
  * alpha = 0.01 throughout; the Lee-Mykland critical value is recomputed from n
    by the method itself and is never tuned.
  * chronological split over the 21 calendar dates, whole days, 70/15/15.

Features are the L1 + order-flow set this feed supports; there is no depth here
(see polymarket_sports/reports/DATA_AUDIT.md) and none is fabricated.

    python run_resolution.py --res 5 --K 60 --lookback 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "lee_mykland"))
sys.path.insert(0, os.path.join(ROOT, "research", "makinen"))
from detect_lee_mykland_jumps import lm_threshold, returns_for  # noqa: E402
import models as MD  # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "sports")

FEATS = ["mid", "spread", "n_updates", "n_buy", "n_sell", "flow_imb",
         "size_buy", "size_sell", "size_imb", "near_frac",
         "d_mid", "d_spread", "d_flow_imb", "rv_10", "rv_30",
         "tod_sin", "tod_cos"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def label_jumps(B, K, alpha, min_nz):
    parts = []
    for tok, g in B.groupby("token_id", sort=False, observed=True):
        g = g.sort_values("bar")
        bar = g.bar.to_numpy(np.int64)
        mid = g.mid.to_numpy(np.float64)
        adj = np.empty(len(g), bool)
        adj[0] = False
        adj[1:] = (bar[1:] - bar[:-1]) == 1
        r = returns_for(mid, "diff")
        r[~adj] = np.nan
        r = pd.Series(r)
        p = r.abs() * r.abs().shift(1)
        win = max(K - 2, 2)
        ps = p.shift(1)
        sig = np.sqrt(ps.rolling(win, min_periods=max(min_nz, 2)).mean())
        nz = ps.gt(0).rolling(win, min_periods=1).sum()
        sig[nz < min_nz] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            L = r / sig
        parts.append(pd.DataFrame({"token_id": tok, "bar": bar,
                                   "ret": r.to_numpy(), "sigma": sig.to_numpy(),
                                   "LM": L.to_numpy()}))
    D = pd.concat(parts, ignore_index=True)
    B = B.merge(D, on=["token_id", "bar"], how="left")
    tested = np.isfinite(B.LM.to_numpy())
    n = int(tested.sum())
    thr, c_n, s_n = lm_threshold(n, alpha)
    B["jump"] = ((tested) & (np.abs(B.LM.to_numpy()) > thr)).astype(np.int8)
    B["tested"] = tested
    return B, thr, n


def add_derived(B):
    B = B.sort_values(["token_id", "bar"]).reset_index(drop=True)
    g = B.groupby("token_id", sort=False)
    contig = g.bar.diff() == 1
    for c in ("mid", "spread", "flow_imb"):
        B["d_" + c] = np.where(contig, g[c].diff(), np.nan)
    for w in (10, 30):
        B["rv_%d" % w] = B.d_mid.groupby(B.token_id).transform(
            lambda s: s.rolling(w, min_periods=5).std())
    sec = (B.ts // 1000) % 86400
    B["tod_sin"] = np.sin(2 * np.pi * sec / 86400.0)
    B["tod_cos"] = np.cos(2 * np.pi * sec / 86400.0)
    return B


def build_windows(B, L, H, stride=1):
    F = B[FEATS].to_numpy(np.float32)
    Xs, ys, meta = [], [], []
    for tok, g in B.groupby("token_id", sort=False, observed=True):
        idx = g.index.to_numpy()
        bars = g.bar.to_numpy(np.int64)
        jump = g.jump.to_numpy(np.int8)
        tested = g.tested.to_numpy(bool)
        dates = g.date.to_numpy()
        n = len(idx)
        if n < L + H:
            continue
        for e in range(L - 1, n - H, stride):
            s = e - L + 1
            if bars[e] - bars[s] != L - 1 or bars[e + H] - bars[e] != H:
                continue
            if not tested[e + 1:e + H + 1].any():
                continue
            w = F[idx[s]:idx[e] + 1]
            if w.shape[0] != L or not np.isfinite(w).all():
                continue
            Xs.append(w)
            ys.append(int(jump[e + 1:e + H + 1].max()))
            meta.append((tok, int(bars[e]), dates[e]))
    if not Xs:
        return (np.zeros((0, L, len(FEATS)), np.float32), np.zeros(0, np.int8),
                pd.DataFrame(columns=["token_id", "bar", "date"]))
    return (np.stack(Xs).astype(np.float32), np.array(ys, np.int8),
            pd.DataFrame(meta, columns=["token_id", "bar", "date"]))


def evaluate(y, p, thr, bars, res_s):
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 confusion_matrix)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    hours = bars * res_s / 3600.0
    ap = float(average_precision_score(y, p))
    return dict(precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12),
                pr_auc=ap, pr_auc_lift=ap / max(y.mean(), 1e-12),
                roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
                fp_per_hour=fp / max(hours, 1e-9), tp=int(tp), fp=int(fp),
                fn=int(fn), prevalence=float(y.mean()))


def pick_threshold(yv, pv):
    from sklearn.metrics import f1_score
    grid = np.unique(np.quantile(pv, np.linspace(0.5, 0.9995, 100)))
    best, bt = -1.0, 0.5
    for t in grid:
        f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-nonzero-pairs", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=250000)
    ap.add_argument("--target-windows", type=int, default=250000,
                    help="windows are strided to approximately this many; at 1s "
                         "resolution the unstrided tensor would be ~98 GB")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    np.random.seed(a.seed); torch.manual_seed(a.seed)

    B = pd.read_parquet(os.path.join(CACHE, "bars_%ds.parquet" % a.res))
    log("res %ds: {:,} bars, {:,} tokens".format(len(B), B.token_id.nunique())
        % a.res)
    B = add_derived(B)
    B, thr_lm, n_tested = label_jumps(B, a.K, a.alpha, a.min_nonzero_pairs)
    prev_all = float(B.jump[B.tested].mean()) if B.tested.any() else np.nan
    log("  LM: n=%s threshold=%.3f  jumps=%s (%.3f%% of tested)  K spans %.1f min"
        % ("{:,}".format(n_tested), thr_lm, "{:,}".format(int(B.jump.sum())),
           100 * prev_all, a.K * a.res / 60.0))

    approx = max(int((B.tested.sum() - a.lookback)), 1)
    stride = max(1, int(round(approx / max(a.target_windows, 1))))
    log("  window stride %d (approx %s candidate windows)"
        % (stride, "{:,}".format(approx)))
    X, y, M = build_windows(B, a.lookback, a.horizon, stride)
    if len(y) < 2000 or y.sum() < 50:
        log("  too few windows/positives, skipping")
        return 0
    dates = sorted(M.date.unique())
    n_tr = max(int(round(len(dates) * .7)), 1)
    n_va = max(int(round(len(dates) * .15)), 1)
    tr_d, va_d = set(dates[:n_tr]), set(dates[n_tr:n_tr + n_va])
    te_d = set(dates[n_tr + n_va:])
    sp = np.where(M.date.isin(tr_d), "train",
                  np.where(M.date.isin(va_d), "val", "test"))
    tr, va, te = sp == "train", sp == "val", sp == "test"
    log("  windows {:,} (pos {:,}, {:.3%}) | train {:,} val {:,} test {:,}"
        .format(len(y), int(y.sum()), y.mean(), int(tr.sum()), int(va.sum()),
                int(te.sum())))
    if te.sum() < 500 or y[te].sum() < 20:
        log("  test split too small, skipping")
        return 0

    flat = X[tr].reshape(-1, X.shape[2])
    mu, sd = flat.mean(0), flat.std(0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    X = np.nan_to_num((X - mu) / sd, nan=0., posinf=0., neginf=0.).astype(np.float32)

    if tr.sum() > a.max_train:
        keep = np.where(tr)[0]
        drop = np.random.default_rng(0).choice(keep, len(keep) - a.max_train,
                                               replace=False)
        tr = tr.copy(); tr[drop] = False
        log("  train capped to {:,}".format(int(tr.sum())))

    rows = []
    # baseline: prevalence
    rows.append(dict(model="0 prevalence", **evaluate(
        y[te], np.full(te.sum(), y[tr].mean()), 1.1, int(te.sum()), a.res)))
    rows[-1].update(pr_auc=float(y[te].mean()), pr_auc_lift=1.0, roc_auc=.5)

    from sklearn.linear_model import LogisticRegression
    Xs = X[:, -1, :]
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs[tr], y[tr])
    pv, pt = lr.predict_proba(Xs[va])[:, 1], lr.predict_proba(Xs[te])[:, 1]
    rows.append(dict(model="2 logistic (snapshot)",
                     **evaluate(y[te], pt, pick_threshold(y[va], pv),
                                int(te.sum()), a.res)))

    from sklearn.metrics import average_precision_score
    for name, label in (("cnn_lstm", "7 CNN-LSTM"),
                        ("cnn_lstm_attention", "8 CNN-LSTM-Attention")):
        torch.manual_seed(a.seed)
        m = MD.build(name, X.shape[2], X.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        pw = torch.tensor([(tr.sum() - y[tr].sum()) / max(y[tr].sum(), 1)],
                          dtype=torch.float32)
        lf = nn.BCEWithLogitsLoss(pos_weight=pw)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X[tr]),
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
            pt = torch.sigmoid(m(torch.from_numpy(X[te]))).numpy()
        rows.append(dict(model=label, **evaluate(y[te], pt,
                                                 pick_threshold(y[va], pv),
                                                 int(te.sum()), a.res)))
        log("    %-22s test PR-AUC %.4f (lift %.2fx)"
            % (label, rows[-1]["pr_auc"], rows[-1]["pr_auc_lift"]))

    R = pd.DataFrame(rows)
    R["res_s"] = a.res; R["K_bars"] = a.K
    R["K_span_min"] = a.K * a.res / 60.0
    R["lookback_bars"] = a.lookback
    R["lookback_min"] = a.lookback * a.res / 60.0
    R["lm_threshold"] = thr_lm
    R["n_windows"] = len(y)
    R["stride"] = stride
    R.to_csv(os.path.join(RES, "results_%ds.csv" % a.res), index=False)
    print(R[["model", "prevalence", "pr_auc", "pr_auc_lift", "roc_auc",
             "precision", "recall", "f1"]].to_string(
        index=False, float_format=lambda v: "%.4f" % v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
