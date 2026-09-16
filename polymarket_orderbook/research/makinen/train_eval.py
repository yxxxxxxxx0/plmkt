"""Stages 6-10: train every model on identical splits, evaluate, compare.

Rules enforced here:
  * validation and test keep their NATURAL jump frequency -- no oversampling
    ever touches them; class imbalance is handled in training only, by
    weighting the positive class in the BCE loss
  * the decision threshold is chosen on VALIDATION and then frozen before the
    test set is scored once
  * early stopping and model selection use validation PR-AUC, never test
  * predicted probabilities are saved, not just labels
  * seeds are fixed and recorded; runs on CPU or GPU

    python train_eval.py --tag L120_H1_train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import models as MD  # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen")
NEURAL = ["mlp_snapshot", "mlp_history", "cnn", "lstm", "cnn_lstm",
          "cnn_lstm_attention"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def metrics(y, p, thr, minutes, days, name):
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 confusion_matrix, balanced_accuracy_score)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(
        model=name, threshold=float(thr),
        precision=prec, recall=rec,
        f1=2 * prec * rec / max(prec + rec, 1e-12),
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        accuracy=float((pred == y).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        jumps_caught=int(tp), jumps_missed=int(fn),
        fp_per_hour=fp / max(minutes / 60.0, 1e-9),
        fp_per_day=fp / max(days, 1e-9),
        predicted_jump_freq=float(pred.mean()),
        actual_jump_freq=float(y.mean()),
        pr_auc_lift=float(average_precision_score(y, p) / max(y.mean(), 1e-12)),
    )


def pick_threshold(yv, pv):
    """F1-maximising threshold on VALIDATION only."""
    from sklearn.metrics import f1_score
    grid = np.unique(np.quantile(pv, np.linspace(0.50, 0.9995, 120)))
    best, bt = -1.0, 0.5
    for t in grid:
        f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def train_neural(name, Xtr, ytr, Xva, yva, args, dev):
    from sklearn.metrics import average_precision_score
    seed_all(args.seed)
    m = MD.build(name, Xtr.shape[2], Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)],
                      dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr),
                                           torch.from_numpy(ytr.astype(np.float32)))
    dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    Xva_t = torch.from_numpy(Xva).to(dev)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(args.epochs):
        m.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            l = lossf(m(xb), yb)
            l.backward()
            opt.step()
            tot += l.item() * len(xb)
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(Xva_t)).cpu().numpy()
        ap = float(average_precision_score(yva, pv))
        vl = float(nn.BCEWithLogitsLoss()(
            torch.from_numpy(np.log(pv / (1 - pv + 1e-9) + 1e-9)),
            torch.from_numpy(yva.astype(np.float32))))
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(tr_ds),
                         val_loss=vl, val_pr_auc=ap))
        log("    %-20s ep%2d train_loss %.4f  val PR-AUC %.4f"
            % (name, ep + 1, tot / len(tr_ds), ap))
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log("    early stop (patience %d)" % args.patience)
                break
    if best_state:
        m.load_state_dict(best_state)
    return m, pd.DataFrame(hist)


def predict(m, X, dev, bs=512):
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(dev)
            out.append(torch.sigmoid(m(xb)).cpu().numpy())
    return np.concatenate(out)


def bootstrap_ci(y, p, groups, n=400, seed=0):
    """Day-level bootstrap CI for PR-AUC: resample whole days, not rows."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], p[idx]))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="L120_H1_train")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    OUT = a.outdir or RES
    os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "curves"), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device %s | tag %s" % (dev, a.tag))

    cfg = json.load(open(os.path.join(CACHE, "config_%s.json" % a.tag)))
    X = np.load(os.path.join(CACHE, "X_%s.npy" % a.tag))
    y = np.load(os.path.join(CACHE, "y_%s.npy" % a.tag)).astype(int)
    M = pd.read_parquet(os.path.join(CACHE, "meta_%s.parquet" % a.tag))
    feats = cfg["features"]
    tr, va, te = (M.split == "train").to_numpy(), (M.split == "val").to_numpy(), \
        (M.split == "test").to_numpy()
    log("train %s / val %s / test %s   prevalence test %.3f%%"
        % ("{:,}".format(tr.sum()), "{:,}".format(va.sum()),
           "{:,}".format(te.sum()), 100 * y[te].mean()))

    te_minutes = int(te.sum())
    te_days = M[te].date.nunique()
    te_groups = M[te].date.to_numpy()
    rows, curves, preds = [], {}, {}

    # ---- Model 0: prevalence ----
    p0 = np.full(te.sum(), y[tr].mean())
    r = metrics(y[te], p0, 0.5, te_minutes, te_days, "0 prevalence")
    r.update(pr_auc=float(y[te].mean()), pr_auc_lift=1.0, roc_auc=0.5)
    rows.append(r); preds["0_prevalence"] = p0

    # ---- Model 1: time of day only ----
    from sklearn.linear_model import LogisticRegression
    ti = [feats.index(c) for c in ("tod_sin", "tod_cos")]
    Xt = X[:, -1, :][:, ti]
    lr1 = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xt[tr], y[tr])
    pv, pt = lr1.predict_proba(Xt[va])[:, 1], lr1.predict_proba(Xt[te])[:, 1]
    thr = pick_threshold(y[va], pv)
    rows.append(metrics(y[te], pt, thr, te_minutes, te_days, "1 time-of-day only"))
    curves["1 time-of-day only"] = (y[te], pt); preds["1_time_of_day"] = pt

    # ---- Model 2: logistic on the latest snapshot ----
    Xs = X[:, -1, :]
    lr2 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs[tr], y[tr])
    pv, pt = lr2.predict_proba(Xs[va])[:, 1], lr2.predict_proba(Xs[te])[:, 1]
    thr = pick_threshold(y[va], pv)
    rows.append(metrics(y[te], pt, thr, te_minutes, te_days, "2 logistic (snapshot)"))
    curves["2 logistic (snapshot)"] = (y[te], pt); preds["2_logistic"] = pt

    # ---- Models 3-8: neural ----
    names = a.models or NEURAL
    for nm in names:
        log("  training %s" % nm)
        m, hist = train_neural(nm, X[tr], y[tr], X[va], y[va], a, dev)
        pv, pt = predict(m, X[va], dev), predict(m, X[te], dev)
        thr = pick_threshold(y[va], pv)
        label = {"mlp_snapshot": "3 MLP snapshot", "mlp_history": "4 MLP history",
                 "cnn": "5 CNN", "lstm": "6 LSTM", "cnn_lstm": "7 CNN-LSTM",
                 "cnn_lstm_attention": "8 CNN-LSTM-Attention"}[nm]
        r = metrics(y[te], pt, thr, te_minutes, te_days, label)
        r["n_params"] = MD.n_params(m)
        rows.append(r)
        curves[label] = (y[te], pt); preds[nm] = pt
        hist.to_csv(os.path.join(OUT, "curves", "history_%s.csv" % nm), index=False)
        torch.save(m.state_dict(), os.path.join(OUT, "preds", "model_%s.pt" % nm))
        if nm == "cnn_lstm_attention":
            with torch.no_grad():
                m(torch.from_numpy(X[te][:512]).to(dev))
            w = m.attention_weights().cpu().numpy().mean(0)
            pd.DataFrame({"feature": feats, "attention_weight": w}) \
                .sort_values("attention_weight", ascending=False) \
                .to_csv(os.path.join(OUT, "attention_weights.csv"), index=False)

    # ---- comparison table with day-level bootstrap CIs ----
    R = pd.DataFrame(rows)
    lo, hi = [], []
    for nm in R.model:
        key = {v: k for k, v in
               {"0_prevalence": "0 prevalence", "1_time_of_day": "1 time-of-day only",
                "2_logistic": "2 logistic (snapshot)",
                "mlp_snapshot": "3 MLP snapshot", "mlp_history": "4 MLP history",
                "cnn": "5 CNN", "lstm": "6 LSTM", "cnn_lstm": "7 CNN-LSTM",
                "cnn_lstm_attention": "8 CNN-LSTM-Attention"}.items()}.get(nm)
        if key in preds:
            l, h = bootstrap_ci(y[te], preds[key], te_groups)
        else:
            l, h = np.nan, np.nan
        lo.append(l); hi.append(h)
    R["pr_auc_ci_lo"], R["pr_auc_ci_hi"] = lo, hi
    R.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    np.savez(os.path.join(OUT, "preds", "test_predictions_%s.npz" % a.tag),
             y=y[te], **preds)
    M[te].to_parquet(os.path.join(OUT, "preds", "test_meta_%s.parquet" % a.tag),
                     index=False)

    show = ["model", "precision", "recall", "f1", "pr_auc", "pr_auc_lift",
            "roc_auc", "fp_per_hour", "jumps_caught", "jumps_missed"]
    print("\n=== TEST SET (prevalence %.3f%%, %d positives) ==="
          % (100 * y[te].mean(), int(y[te].sum())))
    print(R[show].to_string(index=False, float_format=lambda v: "%.4f" % v))

    # ---- PR curves ----
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for nm, (yy, pp) in curves.items():
        pr, rc, _ = precision_recall_curve(yy, pp)
        ax.plot(rc, pr, lw=1.3, label=nm)
    ax.axhline(y[te].mean(), color="k", ls="--", lw=1,
               label="prevalence (%.3f)" % y[te].mean())
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-recall, test set (%s)" % a.tag)
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pr_curves.png"), dpi=130)
    plt.close(fig)
    log("wrote %s" % os.path.join(OUT, "model_comparison.csv"))


if __name__ == "__main__":
    sys.exit(main())
