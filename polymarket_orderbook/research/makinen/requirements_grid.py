"""
What selection precision and what direction accuracy would actually be needed?

Justin: what do I need to do to produce that $1,358, and what do my direction
and prediction accuracy have to be?

The $1,358 is an oracle: it knows which jumps pay, which way each one goes, and
when to get out. This turns that into the two dials a real system has, and
solves for where they have to sit.

Nothing here is approximated. Every jump carries BOTH outcomes -- what a long
would have made and what a short would have made, each priced at the touch on
entry and exit -- so "getting the direction wrong" is the other side's actual
P&L, not a sign flip of the winning one. That distinction matters: the two are
not symmetric, because both sides pay the spread and the fee.

    EV per trade = a * better_side + (1 - a) * worse_side

with a the directional accuracy. Selection precision p mixes the population:

    EV(p, a) = p * E[EV | jump pays] + (1 - p) * E[EV | jump does not]

Dollars, not ticks, because the share count depends on the price: $10 buys
10/entry_px shares of a long and 10/(1-entry_px) shares of the complement for a
short, and the marked set spans prices from a few cents to nearly a dollar.

The feasibility ceiling is reported alongside, and it binds harder than either
dial. Only 1,670 jumps pay out of 136,624, so a selection of n trades at
precision p needs n*p of them to exist. Wanting 1,000 trades a night at 80%
precision is not a modelling problem, it is arithmetic: the payers are not
there.

    python requirements_grid.py

Outputs, under results/makinen/oracle_jumps/
  requirements_grid.csv
  requirements_grid.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

TICK = 0.01
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
GRID_C = "#e9e8e4"
POS = "#2a78d6"
NEG = "#eb6834"

ACCS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
PRECS = [0.0122, 0.05, 0.10, 0.20, 0.30, 0.40, 0.529, 0.60, 0.716, 0.80, 0.90, 1.00]


def dollars(net_ticks, entry_px, side_is_long, stake):
    paid = np.where(side_is_long, entry_px, 1.0 - entry_px)
    return (stake / paid) * net_ticks * TICK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    ap.add_argument("--stake", type=float, default=10.0)
    args = ap.parse_args()
    outdir = Path(args.outdir)

    a = pd.read_parquet(outdir / "breakeven_exec_all.parquet")
    long_is_better = a.long_net_ticks >= a.short_net_ticks
    better = np.where(long_is_better, a.long_net_ticks, a.short_net_ticks)
    worse = np.where(long_is_better, a.short_net_ticks, a.long_net_ticks)
    # Entry price for each side: a long pays the ask, a short pays 1 - bid.
    better_px = np.where(long_is_better, a.entry_ask, a.entry_bid)
    worse_px = np.where(long_is_better, a.entry_bid, a.entry_ask)

    d_better = dollars(better, better_px, long_is_better, args.stake)
    d_worse = dollars(worse, worse_px, ~long_is_better, args.stake)

    pays = better > 0
    n_pay = int(pays.sum())
    print(f"{len(a):,} durable jumps, {n_pay:,} pay with perfect direction "
          f"({pays.mean():.2%})")
    print(f"  a payer, right side : ${d_better[pays].mean():+.3f}   "
          f"wrong side ${d_worse[pays].mean():+.3f}")
    print(f"  a non-payer, 'right': ${d_better[~pays].mean():+.3f}   "
          f"wrong side ${d_worse[~pays].mean():+.3f}")
    print()

    ev_pay_b, ev_pay_w = d_better[pays].mean(), d_worse[pays].mean()
    ev_non_b, ev_non_w = d_better[~pays].mean(), d_worse[~pays].mean()

    rows = []
    for p in PRECS:
        for acc in ACCS:
            ev_pay = acc * ev_pay_b + (1 - acc) * ev_pay_w
            ev_non = acc * ev_non_b + (1 - acc) * ev_non_w
            ev = p * ev_pay + (1 - p) * ev_non
            rows.append(dict(precision=p, accuracy=acc, ev_per_trade=ev,
                             max_trades=int(n_pay / p) if p > 0 else 0,
                             max_total=ev * (n_pay / p) if p > 0 else 0))
    g = pd.DataFrame(rows)
    g.to_csv(outdir / "requirements_grid.csv", index=False)

    print("=== EV per $10 trade, by selection precision x direction accuracy ===")
    piv = g.pivot(index="precision", columns="accuracy", values="ev_per_trade")
    print(piv.round(2).to_string())

    print()
    print("=== the break-even frontier: minimum direction accuracy at each precision ===")
    print(f"{'precision':>10} {'min accuracy':>13} {'max trades':>11} "
          f"{'total at that corner':>21}")
    front = []
    for p in PRECS:
        sub = g[(g.precision == p) & (g.ev_per_trade > 0)]
        if sub.empty:
            print(f"{p:>10.1%} {'impossible':>13} {'-':>11} {'-':>21}")
            front.append(dict(precision=p, min_accuracy=np.nan))
            continue
        acc = sub.accuracy.min()
        row = sub[sub.accuracy == acc].iloc[0]
        print(f"{p:>10.1%} {acc:>13.0%} {int(row.max_trades):>11,} "
              f"{'$' + format(row.max_total, ',.0f'):>21}")
        front.append(dict(precision=p, min_accuracy=acc))
    pd.DataFrame(front).to_csv(outdir / "requirements_frontier.csv", index=False)

    print()
    print("=== where the measured system actually sits ===")
    print("  selector precision 52.9% (top 0.1%, SELECTOR.md), direction ~55% "
          "(batting-team, moneyline)")
    ev = (0.529 * (0.55 * ev_pay_b + 0.45 * ev_pay_w)
          + 0.471 * (0.55 * ev_non_b + 0.45 * ev_non_w))
    print(f"  --> ${ev:+.2f} per $10 trade")
    print("  and at 52.9% precision with PERFECT direction:")
    ev2 = 0.529 * ev_pay_b + 0.471 * ev_non_b
    print(f"  --> ${ev2:+.2f} per $10 trade")

    draw(g, outdir / "requirements_grid.png", args.stake, n_pay, len(a))


def draw(g, path, stake, n_pay, n_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    piv = g.pivot(index="precision", columns="accuracy", values="ev_per_trade")
    fig, ax = plt.subplots(figsize=(12.5, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    v = np.nanmax(np.abs(piv.to_numpy()))
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
    im = ax.imshow(piv.to_numpy(), cmap="RdBu", norm=norm, aspect="auto",
                   origin="lower")

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in piv.columns], fontsize=9, color=INK2)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{i:.1%}" for i in piv.index], fontsize=9, color=INK2)
    ax.set_xlabel("direction accuracy", color=INK2, fontsize=10)
    ax.set_ylabel("selection precision", color=INK2, fontsize=10)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iat[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7.6,
                    color="#111111" if abs(val) < v * 0.55 else "#ffffff")

    # the break-even contour
    ax.contour(np.arange(piv.shape[1]), np.arange(piv.shape[0]),
               piv.to_numpy(), levels=[0.0], colors=[INK], linewidths=2.2)

    # where the measured system sits
    try:
        yi = list(piv.index).index(0.529)
        xi = float(np.interp(0.55, piv.columns, np.arange(len(piv.columns))))
        ax.plot([xi], [yi], marker="o", ms=13, mfc="none", mec=INK, mew=2.4)
        ax.annotate("measured today:\n53% precision, ~55% direction",
                    (xi, yi), xytext=(18, 16), textcoords="offset points",
                    fontsize=9, color=INK, fontweight="bold")
    except ValueError:
        pass

    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label(f"expected P&L per ${stake:.0f} trade", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=INK2, labelsize=8.5)

    fig.suptitle("What the two dials have to reach before any of this pays",
                 color=INK, fontsize=13.5, x=0.006, y=0.985, ha="left")
    fig.text(0.006, 0.928, f"{n_pay:,} of {n_all:,} durable jumps pay with perfect "
             f"direction. The black line is break-even; everything below and left "
             f"of it loses money.",
             color=INK2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
