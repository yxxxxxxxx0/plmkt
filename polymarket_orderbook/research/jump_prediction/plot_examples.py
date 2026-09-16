"""Look at the actual price series around labelled jumps.

A summary statistic cannot tell you whether a label marks a real price move or
an artefact of the mid. Plotting can, and on this project it has: the previous
series-key defect (FINDINGS_SERIES_KEY.md) was invisible in the aggregates and
obvious within seconds of drawing the price.

This draws randomly chosen labelled jumps from the TEST split only, in the
tight-book regime where the mid is meaningful, with the pre-jump window shaded.
Each panel shows mid, spread and near-touch depth on a common time axis so a
mid that "jumps" because a far-away quote was cancelled is visually
distinguishable from one that jumps because the market moved.

    python plot_examples.py --J 0.02 --H 30 --n 12
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import splits as SP  # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "jump_prediction")
PLOTS = os.path.join(RES, "plots")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--pre-s", type=int, default=60)
    ap.add_argument("--post-s", type=int, default=30)
    ap.add_argument("--tight-ticks", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    lab = "jump_%s_H%d" % (a.J, a.H)
    val = "valid_label_H%d" % a.H
    cols = ["sid", "ts", "series", "market", "mid", "spread_ticks",
            "log_bid_usd_within_2t", "log_ask_usd_within_2t",
            "future_move_H%d" % a.H, val, lab]
    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    D = pd.concat([pd.read_parquet(p, columns=cols) for p in ps],
                  ignore_index=True)
    D = D[D[val]].reset_index(drop=True)
    split, _ = SP.assign(D, verbose=False)
    D["split"] = split

    # candidate jump instants: test split, tight book at t, labelled jump
    cand = D[(D.split == "test") & (D.spread_ticks <= a.tight_ticks)
             & D[lab]].reset_index(drop=True)
    print("candidate tight-book test jumps: {:,}".format(len(cand)))
    rng = np.random.default_rng(a.seed)
    pick = cand.iloc[rng.choice(len(cand), size=min(a.n, len(cand)),
                                replace=False)]

    os.makedirs(PLOTS, exist_ok=True)
    by_sid = {s: g.sort_values("ts").reset_index(drop=True)
              for s, g in D.groupby("sid", observed=True)}

    ncol = 3
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, ev) in zip(axes, pick.iterrows()):
        g = by_sid[ev.sid]
        t0 = ev.ts
        w = g[(g.ts >= t0 - a.pre_s * 1000) & (g.ts <= t0 + a.post_s * 1000)]
        if len(w) < 10:
            ax.set_visible(False)
            continue
        rel = (w.ts - t0) / 1000.0
        ax.plot(rel, w.mid, lw=1.4, color="#1f77b4", label="mid")
        ax.axvline(0, color="#d62728", lw=1.2, ls="--")
        ax.axvspan(-a.pre_s, 0, color="#cccccc", alpha=0.25)
        ax.axhline(ev.mid, color="#888888", lw=0.7, ls=":")
        ax.axhline(ev.mid + a.J, color="#2ca02c", lw=0.7, ls=":")
        ax.axhline(ev.mid - a.J, color="#2ca02c", lw=0.7, ls=":")
        ax2 = ax.twinx()
        ax2.plot(rel, w.spread_ticks, lw=0.9, color="#ff7f0e", alpha=0.8)
        ax2.set_ylabel("spread (ticks)", fontsize=7, color="#ff7f0e")
        ax2.tick_params(labelsize=6, colors="#ff7f0e")
        ax.set_title("%s\n%s  move=%.3f" %
                     (str(ev.market)[:28], str(ev.series).split("|")[1][:18],
                      ev["future_move_H%d" % a.H]), fontsize=7)
        ax.set_xlabel("seconds relative to t", fontsize=7)
        ax.set_ylabel("mid", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(pick):]:
        ax.set_visible(False)
    fig.suptitle("Labelled jumps, tight books, TEST split "
                 "(J=%s, H=%ds). Shaded = model input window; "
                 "dashed red = prediction instant t." % (a.J, a.H),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f = os.path.join(PLOTS, "examples_jumps_tight_J%s_H%d.png" % (a.J, a.H))
    fig.savefig(f, dpi=120)
    print("wrote", f)


if __name__ == "__main__":
    sys.exit(main())
