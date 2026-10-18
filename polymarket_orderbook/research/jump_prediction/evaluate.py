"""Metrics for a rare-event classifier, per section 11 of the brief.

Accuracy is deliberately absent. With a 10-40% positive rate and thresholds
that matter only in the tail, accuracy rewards predicting the majority class
and hides everything of interest. PR-AUC is primary; ROC-AUC is reported as
secondary because it is the number most readers expect.

Decision thresholds are chosen on VALIDATION and then applied unchanged to
test. `pick_threshold` is the only place a threshold is allowed to be chosen,
and it refuses to see test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)


def pick_threshold(y_val, p_val, objective="f1"):
    """Choose the operating point on validation only."""
    pr, rc, th = precision_recall_curve(y_val, p_val)
    pr, rc = pr[:-1], rc[:-1]
    if objective == "f1":
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
        i = int(np.nanargmax(f1))
    else:
        i = int(np.nanargmax(pr))
    return float(th[i])


def topk_precision(y, p, frac):
    k = max(1, int(round(len(p) * frac)))
    idx = np.argsort(-p)[:k]
    return float(np.mean(y[idx]))


def calibration(y, p, bins=12):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    if len(q) < 3:
        return pd.DataFrame(columns=["p_mean", "y_rate", "n"])
    b = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
    rows = []
    for i in range(len(q) - 1):
        m = b == i
        if m.sum() < 20:
            continue
        rows.append(dict(p_mean=float(p[m].mean()), y_rate=float(y[m].mean()),
                         n=int(m.sum())))
    return pd.DataFrame(rows)


def evaluate(name, y, p, thr, hours=None, extra=None):
    """One row of the central results table."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, np.float64)
    prev = float(y.mean())
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    row = dict(
        model=name, n=len(y), prevalence=prev,
        pr_auc=float(average_precision_score(y, p)) if 0 < prev < 1 else np.nan,
        # lift over the trivial "always predict the base rate" classifier
        pr_auc_lift=(float(average_precision_score(y, p)) / prev
                     if 0 < prev < 1 else np.nan),
        roc_auc=float(roc_auc_score(y, p)) if 0 < prev < 1 else np.nan,
        threshold=float(thr), alert_rate=float(pred.mean()),
        precision=float(prec), recall=float(rec),
        f1=float(2 * prec * rec / max(prec + rec, 1e-12)),
        prec_top1pct=topk_precision(y, p, 0.01),
        prec_top5pct=topk_precision(y, p, 0.05),
        prec_top10pct=topk_precision(y, p, 0.10),
        brier=float(brier_score_loss(y, p)) if 0 < prev < 1 else np.nan,
        brier_skill=(1 - brier_score_loss(y, p) / max(prev * (1 - prev), 1e-12)
                     if 0 < prev < 1 else np.nan),
    )
    if hours:
        row["false_alerts_per_market_hour"] = float(fp / max(hours, 1e-9))
        row["true_alerts_per_market_hour"] = float(tp / max(hours, 1e-9))
    if extra:
        row.update(extra)
    return row
