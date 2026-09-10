"""How far must the price move for a taker to break even -- and how often does it?

Two questions, in order, because the second decides whether a model is worth
training at all.

1. THE BREAK-EVEN MOVE. A taker crosses the spread to get in, crosses it again
   to get out, and pays a fee on each leg:

       cost_ticks = half_spread(t) + half_spread(t+H)
                  + fee(p_t) + fee(p_{t+H})
       fee(p)     = feeRate * p * (1-p) / tick        feeRate 0.05 for sports

   With certain direction you need move > cost. With directional accuracy q,
   E[pnl] = m(2q-1) - cost, so the required move is

       m* = cost / (2q - 1)

   which blows up fast as q falls toward a coin flip. The fee term is the
   interesting part: it peaks at p = 0.50 and nearly vanishes at the extremes,
   so break-even is far cheaper on a 0.05 contract than a 0.50 one.

2. HOW MANY SUCH JUMPS EXIST, where you could actually trade them. The last
   round established that directional P&L was concentrated entirely in touches
   holding under $50 -- trivially cancellable and too thin to take size in --
   and that on touches above $5,000 the mean move collapsed to 0.76 ticks
   against a ~3.5-tick round trip. So the population that matters is:

       |move| >= break-even   AND   the touch is well backed

   If that population is tiny, no architecture recovers it, and knowing the
   count is much cheaper than training to find out.

    python diag_breakeven.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from jump_model import frame_offsets
from jump_split import TICK, load
from polymarket_fees import taker_fee_ticks
from taker_signal import prepare
from diag_slices import touch_dollars


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="sports")
    ap.add_argument("--max-spread", type=float, default=2.0)
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    _, _, _, lookback = frame_offsets(24, 24, 8)
    F, T, tr, va, te = load(["moneyline", "total", "spread"],
                            lookback=lookback, log=log, max_book_age_s=5.0)
    F, _ = prepare(F, 5.0, a.category, a.max_spread)

    idx = np.concatenate([va, te])
    idx = idx[F.tight.to_numpy()[idx]]
    signed = F.signed_ticks.to_numpy(np.float64)[idx]
    cost = F.cost_taker.to_numpy(np.float64)[idx]
    mid = F.mid.to_numpy(np.float64)[idx]
    sp = F.spread_ticks.to_numpy(np.float64)[idx]
    mt = F.mt.to_numpy().astype(str)[idx]

    log(f"\n{'='*104}\n1. THE BREAK-EVEN MOVE  (tight books, spread <= "
        f"{a.max_spread:g} ticks, n={len(idx):,})\n{'='*104}")
    log(f"  mean round-trip cost {cost.mean():.2f} ticks  "
        f"= spread {sp.mean()/2 + sp.mean()/2:.2f} + fees "
        f"{2*taker_fee_ticks(mid, TICK, a.category).mean():.2f}")

    log(f"\n  cost by contract price -- the fee's p(1-p) shape means the "
        f"extremes are much cheaper")
    log(f"  {'price':>12} {'n':>9} {'spread':>8} {'fee x2':>8} "
        f"{'cost':>8} {'m* q=1.0':>9} {'m* q=.90':>9} {'m* q=.85':>9}")
    for lo, hi in [(0, .1), (.1, .25), (.25, .4), (.4, .6), (.6, .75),
                   (.75, .9), (.9, 1.01)]:
        s = (mid >= lo) & (mid < hi)
        if s.sum() < 1000:
            continue
        c = cost[s].mean()
        log(f"  {lo:.2f}-{hi:<6.2f} {s.sum():>9,} {sp[s].mean():>8.2f} "
            f"{2*taker_fee_ticks(mid[s], TICK, a.category).mean():>8.2f} "
            f"{c:>8.2f} {c:>9.2f} {c/0.8:>9.2f} {c/0.7:>9.2f}")

    log(f"\n  required move m* = cost/(2q-1) at the pooled mean cost "
        f"{cost.mean():.2f} ticks:")
    for q in (1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60):
        log(f"    directional accuracy {q:.2f}  ->  "
            f"{cost.mean()/(2*q-1):>6.2f} ticks "
            f"({cost.mean()/(2*q-1)*TICK*100:>5.1f} cents/share)")

    # ---- 2. how often, and where ----------------------------------------- #
    tb, ta = touch_dollars(T, idx)
    backing = np.minimum(tb, ta)
    log(f"\n{'='*104}\n2. HOW MANY QUALIFYING JUMPS EXIST\n{'='*104}")

    log(f"\n  P(|move| >= K) by touch backing -- the population a taker could "
        f"actually trade")
    ks = [2, 3, 4, 5, 8]
    hdr = "  " + f"{'backing $':>14} {'n':>9} " + "".join(
        f"{'K='+str(k):>9}" for k in ks) + f"{'mean|mv|':>10}"
    log(hdr)
    edges = [0, 50, 200, 1000, 5000, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (backing >= lo) & (backing < hi)
        if s.sum() < 1000:
            continue
        lbl = f"{lo:,.0f}-{hi:,.0f}" if np.isfinite(hi) else f">{lo:,.0f}"
        row = "".join(f"{(np.abs(signed[s]) >= k).mean():>9.4f}" for k in ks)
        log(f"  {lbl:>14} {s.sum():>9,} {row}{np.abs(signed[s]).mean():>10.2f}")

    log(f"\n  counts of |move| >= 4 ticks (a defensible break-even target) "
        f"by backing AND market type")
    log(f"  {'backing $':>14} " + "".join(f"{m:>13}" for m in
                                          ["moneyline", "total", "spread"]))
    for lo, hi in zip(edges[:-1], edges[1:]):
        s0 = (backing >= lo) & (backing < hi)
        if s0.sum() < 1000:
            continue
        lbl = f"{lo:,.0f}-{hi:,.0f}" if np.isfinite(hi) else f">{lo:,.0f}"
        cells = []
        for m in ["moneyline", "total", "spread"]:
            s = s0 & (mt == m)
            n = int((np.abs(signed[s]) >= 4).sum())
            cells.append(f"{n:>7,}/{s.sum():>5,}" if s.sum() else "  -")
        log(f"  {lbl:>14} " + "".join(f"{c:>13}" for c in cells))

    # the decisive cell: tradeable size AND a move that clears costs
    for bmin in (200, 1000, 5000):
        for k in (3, 4, 5):
            s = (backing >= bmin) & (np.abs(signed) >= k)
            per_type = {m: int((s & (mt == m)).sum())
                        for m in ["moneyline", "total", "spread"]}
            log(f"\n  backing >= ${bmin:,} and |move| >= {k} ticks: "
                f"{s.sum():,} of {len(idx):,} rows ({s.mean():.4%})")
            log(f"    by type: " + "  ".join(f"{k2} {v:,}"
                                             for k2, v in per_type.items()))
            if s.sum():
                log(f"    mean |move| there {np.abs(signed[s]).mean():.2f} "
                    f"ticks vs cost {cost[s].mean():.2f} -> "
                    f"gross edge {np.abs(signed[s]).mean() - cost[s].mean():+.2f} "
                    f"ticks if direction were certain")


if __name__ == "__main__":
    main()
