"""
Elo for MLB, walk-forward by construction, scored against real baselines.

Elo is naturally causal: every rating at time t is built only from games
before t, so there is no train/test split to get wrong. Ratings update after
each game:

    E_home = 1 / (1 + 10^-((R_home + HFA - R_away)/400))
    R     += K * (result - E)

Two refinements from the published baseball work:

  * SEASON REGRESSION. Ratings carry 1 - `regress` of their deviation from
    1500 into the next season. Rosters change over a winter; a team is not
    the team it was in September.
  * MARGIN OF VICTORY. A 10-1 win is more evidence than a 2-1 win. The
    multiplier uses log(margin) damped by rating difference, which is the
    standard form used to stop favourites farming rating from blowouts.

The literature's headline is that these models reach ~55-60% accuracy and
still LOSE to the betting line, so the baselines here are the point: if Elo
cannot clear "always pick the home team" by a wide margin, it certainly
cannot clear a market price.

    python model_mlb.py
"""
import argparse
import collections
import csv
import math
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "data", "mlb.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--hfa", type=float, default=24.0, help="home rating bonus")
    ap.add_argument("--regress", type=float, default=0.30,
                    help="fraction of rating pulled back to 1500 each season")
    ap.add_argument("--mov", action="store_true", help="use margin of victory")
    ap.add_argument("--burn-in", type=int, default=2000)
    args = ap.parse_args()

    rows = []
    with open(args.src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: (r["date"], r["home"]))

    R = collections.defaultdict(lambda: 1500.0)
    season = None
    P, Y, W = [], [], []          # model prob, outcome, "always home" prob
    rec = collections.defaultdict(lambda: [0, 0])   # season win-loss

    for n, g in enumerate(rows):
        if g["season"] != season:
            season = g["season"]
            for t in list(R):
                R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - args.regress)
            rec.clear()
        h, a = g["home"], g["away"]
        e = 1.0 / (1.0 + 10 ** (-((R[h] + args.hfa - R[a]) / 400.0)))
        y = 1 if g["result"] == "H" else 0
        if n >= args.burn_in:
            P.append(e)
            Y.append(y)
            # record-based baseline: pick the better season win pct
            wh = rec[h][0] / max(sum(rec[h]), 1)
            wa = rec[a][0] / max(sum(rec[a]), 1)
            W.append(0.5 + 0.5 * np.tanh((wh - wa) * 3.0))
        margin = abs(int(g["hr"]) - int(g["ar"]))
        k = args.k
        if args.mov:
            diff = (R[h] + args.hfa - R[a]) * (1 if y else -1)
            k *= math.log(max(margin, 1) + 1.0) * (2.2 / (0.001 * diff + 2.2))
        upd = k * (y - e)
        R[h] += upd
        R[a] -= upd
        rec[h][0 if y else 1] += 1
        rec[a][1 if y else 0] += 1

    P = np.array(P)
    Y = np.array(Y)
    W = np.clip(np.array(W), 1e-6, 1 - 1e-6)
    base = float(Y.mean())

    def sc(p, name):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        acc = float(((p > 0.5).astype(int) == Y).mean())
        ll = float(-np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))
        br = float(np.mean((p - Y) ** 2))
        return {"name": name, "acc": acc, "ll": ll, "brier": br}

    res = [
        sc(P, f"Elo (K={args.k:g}, HFA={args.hfa:g}"
              + (", MOV)" if args.mov else ")")),
        sc(W, "season win-pct"),
        sc(np.full_like(P, base), "home base rate"),
        sc(np.full_like(P, 1 - 1e-6), "always Home"),
    ]

    print()
    print("=" * 78)
    print(f"MLB: walk-forward, binary outcome (no draws) -- {len(Y):,} games scored")
    print(f"home win rate in the scored window: {base*100:.2f}%")
    print("=" * 78)
    print(f"  {'model':>28} {'accuracy':>10} {'log loss':>10} {'Brier':>9}")
    print("  " + "-" * 60)
    for r in sorted(res, key=lambda r: r["ll"]):
        print(f"  {r['name']:>28} {r['acc']*100:>9.2f}% {r['ll']:>10.4f} "
              f"{r['brier']:>9.4f}")

    print()
    print("  Elo reliability by confidence decile:")
    order = np.argsort(P)
    for d in range(10):
        lo, hi = d * len(order) // 10, (d + 1) * len(order) // 10
        idx = order[lo:hi]
        print(f"    decile {d+1:>2}: predicted {P[idx].mean()*100:>5.1f}%  "
              f"actual {Y[idx].mean()*100:>5.1f}%  n={len(idx):,}")

    spread = P.max() - P.min()
    print()
    print(f"  Elo probability range: {P.min()*100:.1f}% to {P.max()*100:.1f}% "
          f"(spread {spread*100:.1f} pp)")
    print("  A narrow range is the story of baseball: even a great team beats a")
    print("  poor one only ~60-40, so there is little room for a model to")
    print("  separate itself from the base rate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
