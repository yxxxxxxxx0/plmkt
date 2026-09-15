"""Phase 6: how much warning does a correct prediction actually give?

Section 12. A model that fires one second before a jump is a different
proposition from one that fires thirty seconds before, even at identical
PR-AUC, and the headline metrics cannot tell them apart.

Method. For every jump event, walk backwards from the moment the jump
occurred and find the FIRST time the model's probability crossed the
validation-selected threshold in an unbroken run up to the event. That run
length is the warning lead time. Events the model never flagged are reported
as misses rather than silently dropped, because a lead-time distribution
computed only over detections flatters any model with low recall.

The comparison that gives it meaning is the control: the same measurement on
matched non-jump windows tells you how long the model spends "warning" when
nothing is about to happen. A warning that is on most of the time is not a
warning.

    python leadtime.py --model D_tree_trajectory --J 0.02 --H 30
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

RES = os.path.join(ROOT, "results", "jump_prediction")
CACHE = os.path.join(HERE, "cache")
POINT_HZ = 1.0          # prediction points are 1 per second


def lead_times(ts, sid, y, p, thr, max_lead_s=60.0):
    """First moment of an unbroken above-threshold run ending at each jump."""
    order = np.lexsort((ts, sid))
    ts, sid, y, p = ts[order], sid[order], y[order], p[order]
    fired = p >= thr
    leads, missed = [], 0
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    ends = np.r_[starts[1:], len(sid)]
    for s, e in zip(starts, ends):
        f = fired[s:e]
        yy = y[s:e]
        tt = ts[s:e]
        for i in np.flatnonzero(yy):
            if not f[i]:
                missed += 1
                continue
            j = i
            while j - 1 >= 0 and f[j - 1] and (tt[i] - tt[j - 1]) <= max_lead_s * 1000:
                j -= 1
            leads.append((tt[i] - tt[j]) / 1000.0)
    return np.asarray(leads), missed


def rebuild_filtered(a):
    """Reproduce exactly the frame train.py indexed into.

    train.py keeps rows that (1) have a valid label at this horizon, (2) pass
    the tight-book filter if it was used, and (3) have a complete contiguous
    history window. All three are deterministic, so importing its own helpers
    reproduces the frame rather than approximating it.
    """
    import train as TR
    D = TR.load_points(None)
    D = D[D[f"valid_label_H{a.H}"].to_numpy(bool)].reset_index(drop=True)
    if a.tight_only:
        D = D[D.spread_ticks <= 2.0].reset_index(drop=True)
    offs = TR.frame_offsets()
    S = TR.Sessions(sorted(D.session.unique()))
    keep = np.zeros(len(D), bool)
    for nm, g in D.groupby("session", observed=True):
        keep[g.index.to_numpy()] = S.valid_rows(nm, g.row.to_numpy(), offs)
    return D[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--models", nargs="*",
                    default=["cnn", "cnn_lstm", "cnn_transformer"])
    ap.add_argument("--max-lead-s", type=float, default=60.0)
    ap.add_argument("--tight-only", action="store_true")
    ap.add_argument("--top-frac", type=float, default=None,
                    help="instead of the F1 threshold, alert on the top X of "
                         "predicted probabilities. At 61%% prevalence the F1 "
                         "threshold alerts two-thirds of the time, which makes "
                         "any 'lead time' meaningless; a strict operating "
                         "point is the only one worth quoting.")
    a = ap.parse_args()

    rows, dists = [], {}
    D_filtered = None
    for m in a.models:
        pp = os.path.join(RES, f"preds_{m}_J{a.J:g}_H{a.H}.npy")
        ip = os.path.join(RES, f"testidx_{m}_J{a.J:g}_H{a.H}.npy")
        if not (os.path.exists(pp) and os.path.exists(ip)):
            print(f"  {m}: no saved predictions, skipped")
            continue
        p = np.load(pp)
        idx = np.load(ip)
        # The saved index refers to the FILTERED frame train.py built, not the
        # raw cache. Rebuilding it here by importing train.py's own functions
        # is the only way to guarantee the rows line up; re-deriving the
        # filter by hand would silently drift the moment either changes.
        if D_filtered is None:
            D_filtered = rebuild_filtered(a)
        D = D_filtered
        if idx.max() >= len(D):
            print(f"  {m}: index out of range ({idx.max()} >= {len(D)}), skipped")
            continue
        sub = D.iloc[idx]
        y = sub[f"jump_{a.J:g}_H{a.H}"].to_numpy(bool)
        # sid is unique within a session only
        sid = (sub.session.astype(str) + "#" + sub.sid.astype(str)).to_numpy()
        _, sid = np.unique(sid, return_inverse=True)
        met = pd.read_csv(os.path.join(RES, f"deep_metrics_J{a.J:g}_H{a.H}.csv"))
        r = met[met.model == m]
        if not len(r):
            continue
        thr = (float(np.quantile(p, 1 - a.top_frac)) if a.top_frac
               else float(r.threshold.iloc[0]))
        L, miss = lead_times(sub.ts.to_numpy(np.int64), sid, y, p, thr,
                             a.max_lead_s)
        dists[m] = L
        rows.append(dict(model=m, J=a.J, H=a.H, threshold=thr,
                         n_jumps=int(y.sum()), n_detected=len(L),
                         detection_rate=len(L) / max(int(y.sum()), 1),
                         lead_p25=float(np.percentile(L, 25)) if len(L) else np.nan,
                         lead_median=float(np.median(L)) if len(L) else np.nan,
                         lead_p75=float(np.percentile(L, 75)) if len(L) else np.nan,
                         lead_p90=float(np.percentile(L, 90)) if len(L) else np.nan,
                         frac_lead_under_2s=float((L < 2).mean()) if len(L) else np.nan,
                         frac_lead_over_10s=float((L > 10).mean()) if len(L) else np.nan,
                         alert_duty_cycle=float((p >= thr).mean())))
        print(f"  {m}: {len(L):,}/{int(y.sum()):,} jumps flagged, median lead "
              f"{np.median(L) if len(L) else float('nan'):.1f}s, "
              f"alert on {(p>=thr).mean():.1%} of all moments")

    if not rows:
        raise SystemExit("no model predictions found; run train.py first")
    R = pd.DataFrame(rows)
    p = os.path.join(RES, f"lead_time_J{a.J:g}_H{a.H}.csv")
    R.to_csv(p, index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for m, L in dists.items():
        if len(L) < 50:
            continue
        ax.hist(L, bins=np.arange(0, a.max_lead_s + 1, 1), histtype="step",
                lw=1.6, density=True, label=f"{m} (median {np.median(L):.1f}s)")
    ax.set_xlabel("warning lead time before the jump (s)")
    ax.set_ylabel("density of detected jumps")
    ax.set_title(f"Warning lead time  |  J={a.J:g}  H={a.H}s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "plots", f"lead_time_J{a.J:g}_H{a.H}.png"),
                dpi=130)
    print(f"\nwrote {p}")
    print(R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nA lead time concentrated under 2s means the model is reacting to "
          "the start of the move, not anticipating it. Read it together with "
          "the alert duty cycle: a model alerting most of the time earns a "
          "long 'lead' for free.")


if __name__ == "__main__":
    main()
