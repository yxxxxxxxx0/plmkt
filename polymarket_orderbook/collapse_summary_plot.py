"""
One figure summarising what the event study actually found.

Built to make three specific claims visible, because the raw event-study plot
shows the first and hides the other two:

  A/B  liquidity does collapse before upward jumps -- and the BID collapses
       just as much as the ask, so the effect is not ask-specific.
  C    the paired difference C_A - C_B is centred on zero: within a single
       jump, neither side reliably leads.
  D    the contrast is not purely an artefact of comparing against dead books:
       it survives when controls are restricted to those with any activity,
       though it narrows.
  E    imbalance at t=0 straddles zero, so it does not separate events.

Reads aggregate.npz (stacked per-event series) and events.csv (per-event
scalars), both written by liquidity_collapse.py.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "collapse")

JUMP_C = "#cf222e"
CTRL_C = "#1f4e9c"
BID_C = "#1a7f37"


def band(ax, rel, A, colour, label, ls="-"):
    med = np.nanmedian(A, 0)
    ax.plot(rel, med, color=colour, lw=2.0, ls=ls, label=label)
    ax.fill_between(rel, np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0),
                    color=colour, alpha=0.13, lw=0)


def main():
    z = np.load(os.path.join(OUT, "aggregate.npz"))
    rel = z["rel"]
    rows = list(csv.DictReader(open(os.path.join(OUT, "events.csv"))))
    J = [r for r in rows if r["kind"] == "jump"]
    N = [r for r in rows if r["kind"] == "nojump"]
    col = lambda rs, k: np.array([float(r[k]) for r in rs])

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.22,
                          height_ratios=[1.15, 1.15, 1])

    # A. depth decay, ask vs bid, jump vs control
    ax = fig.add_subplot(gs[0, :])
    band(ax, rel, z["J_normDA"], JUMP_C, "ask, jump")
    band(ax, rel, z["J_normDB"], BID_C, "bid, jump")
    ax.plot(rel, np.nanmedian(z["N_normDA"], 0), color=CTRL_C, lw=1.6, ls="--",
            label="ask, no-jump")
    ax.plot(rel, np.nanmedian(z["N_normDB"], 0), color="#7aa4dd", lw=1.6, ls=":",
            label="bid, no-jump")
    ax.axvline(0, color="#444", lw=1.0, ls="--")
    ax.axhline(1, color="#888", lw=0.7)
    ax.set_ylabel("depth within 5c\n(relative to window start)")
    ax.set_title("A.  Depth collapses before the jump — and the bid collapses with the ask",
                 fontsize=11, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=4)
    ax.grid(alpha=0.2, lw=0.4)

    # B. collapse score, ask vs bid
    ax = fig.add_subplot(gs[1, :])
    band(ax, rel, z["J_CA"], JUMP_C, "$C_A(5c)$ jump")
    band(ax, rel, z["J_CB"], BID_C, "$C_B(5c)$ jump")
    ax.plot(rel, np.nanmedian(z["N_CA"], 0), color=CTRL_C, lw=1.6, ls="--",
            label="$C_A(5c)$ no-jump")
    ax.axvline(0, color="#444", lw=1.0, ls="--")
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_ylabel("collapse score")
    ax.set_xlabel("time relative to t=0 (s)")
    ax.set_title("B.  Collapse score — the two sides track each other almost exactly",
                 fontsize=11, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    ax.grid(alpha=0.2, lw=0.4)

    # C. paired ask-minus-bid
    ax = fig.add_subplot(gs[2, 0])
    d = col(J, "CA5_max_pre") - col(J, "CB5_max_pre")
    lim = np.nanpercentile(np.abs(d), 98)
    ax.hist(d, bins=60, range=(-lim, lim), color="#6f42c1", alpha=0.85)
    ax.axvline(0, color="#111", lw=1.2)
    ax.axvline(np.median(d), color="#f0a202", lw=1.6, ls="--",
               label=f"median {np.median(d):+.3f}")
    ax.set_xlabel("max $C_A(5c)$  −  max $C_B(5c)$   (per jump event)")
    ax.set_ylabel("events")
    ax.set_title("C.  Neither side leads\n(ask worse in %.0f%% of events)" % (100 * np.mean(d > 0)),
                 fontsize=10, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    # D. how far depth falls, three groups
    ax = fig.add_subplot(gs[2, 1])
    jr = col(J, "DA5_ratio")
    nr = col(N, "DA5_ratio")
    act = col(N, "CA5_max_pre") > 0.01
    bins = np.linspace(0, 1.5, 46)
    ax.hist(jr, bins=bins, color=JUMP_C, alpha=0.65, label=f"jump (n={len(jr)})", density=True)
    ax.hist(nr[act], bins=bins, color="#f0a202", alpha=0.55,
            label=f"no-jump, active (n={act.sum()})", density=True)
    ax.hist(nr, bins=bins, color=CTRL_C, alpha=0.35,
            label=f"no-jump, all (n={len(nr)})", density=True)
    ax.axvline(0.5, color="#111", lw=1.0, ls=":")
    ax.set_xlabel("$D_A(5c)$ at t=0  ÷  at window start")
    ax.set_ylabel("density")
    ax.set_title("D.  Survives against ACTIVE controls, but narrows",
                 fontsize=10, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Does ask liquidity collapse before upward price jumps?   "
                 "%d jump / %d no-jump events, 51 matches" % (len(J), len(N)),
                 fontsize=13, fontweight="bold", y=0.945)
    fig.text(0.5, 0.012,
             "Answer: liquidity collapses before jumps (A, B, D) but the effect is NOT ask-specific — "
             "the bid collapses equally (A, B, C).",
             ha="center", fontsize=10.5, color="#7a2312")

    path = os.path.join(OUT, "SUMMARY_findings.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
