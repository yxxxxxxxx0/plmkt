"""Is there a tradeable DIRECTIONAL signal? Run before building a taker model.

The jump study predicts |move|, which is what a maker needs -- it is picked off
whichever way the market runs. A taker must choose a side and pays the full
spread instead of earning half of it:

    maker:  pnl = half_spread - path_max|excursion|
    taker:  pnl = side * signed_move - half_spread(t) - half_spread(t+H) - fee

Two controls this file exists to enforce, because the first version of it
violated both and reported 81.6% directional accuracy in the top confidence
decile against a 52.8% break-even -- a spectacular result that was entirely an
artifact:

  1. NO CONDITIONING ON THE OUTCOME. It is tempting to evaluate only on rows
     where a move happened (`signed_ticks != 0`), but that selects on the
     future. 51.5% of rows do not move at all, and a taker who enters one of
     those pays the spread for nothing. Every row is scored here.

  2. EXECUTABLE PRICES ONLY. The mid is not a price you can trade in a wide
     book. Mean spread is 4.40 ticks against a 1.00 median, and the tail is
     stale books where bid 0.10 / ask 0.50 puts the "mid" at 0.30 with nothing
     resting near it. Mid-to-mid P&L books gains nobody could capture, so
     entry crosses the spread at t and exit crosses it again at t+H.

Endpoint vs path: `signed_ticks` (the endpoint at t+H) is the right measure
for a taker holding to the horizon, which is the mirror image of the maker
case where the path maximum is right. The same column that was wrong there is
correct here.

    python diag_direction.py --max-spread 2
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jump_split import GRID_MS, HAND, TICK, load
from polymarket_fees import (break_even_accuracy, round_trip_cost_ticks,
                             taker_fee_ticks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-s", type=float, default=5.0)
    ap.add_argument("--max-spread", type=float, default=2.0,
                    help="ticks; books wider than this have no executable mid")
    ap.add_argument("--category", default="sports",
                    help="Polymarket fee category; MLB is 'sports' (0.05)")
    a = ap.parse_args()

    pd.set_option("display.width", 200)
    F, T, tr, va, te = load(["moneyline", "total", "spread"], lookback=200,
                            max_book_age_s=5.0)

    h = int(a.horizon_s * 1000 / GRID_MS)
    # the spread you must cross to EXIT, h rows ahead in the same series
    F["spread_fwd"] = F.groupby("sid", sort=False)["spread_ticks"].shift(-h)
    F["spread_fwd"] = F["spread_fwd"].fillna(F["spread_ticks"])

    signed = F.signed_ticks.to_numpy(np.float64)
    spread = F.spread_ticks.to_numpy(np.float64)
    spread_f = F.spread_fwd.to_numpy(np.float64)
    mid = F.mid.to_numpy(np.float64)
    mid_f = mid + signed * TICK                  # price at the exit
    # Real schedule: fee = feeRate * p * (1-p) per share, per taker leg.
    fee_ticks = (taker_fee_ticks(mid, TICK, a.category)
                 + taker_fee_ticks(mid_f, TICK, a.category))
    cost = spread / 2.0 + spread_f / 2.0
    # optimistic bound: leave passively, earning the far half-spread and
    # paying no exit fee. Requires a fill this data cannot verify.
    cost_maker_exit = round_trip_cost_ticks(mid, mid_f, spread, spread_f,
                                            TICK, a.category,
                                            exit_as_maker=True)

    print("\n=== book width: where is the mid even executable? ===")
    for lim in (1, 2, 3, 5, 10, 1e9):
        m = spread <= lim
        lbl = f"<= {lim:g}" if lim < 1e9 else "all"
        print(f"  spread {lbl:>6} ticks: {m.mean():>6.1%} of rows   "
              f"mean |move| {np.abs(signed[m]).mean():>6.2f} ticks   "
              f"mean round-trip cost {cost[m].mean():>6.2f} ticks")

    tight = spread <= a.max_spread
    print(f"\n  restricting to spread <= {a.max_spread:g} ticks "
          f"({tight.mean():.1%} of rows)")

    # ---- break-even, on executable terms --------------------------------- #
    print("\n=== the taker's arithmetic (tight books) ===")
    st = signed[tight]
    print(f"  P(move == 0):      {(st == 0).mean():.1%}")
    print(f"  P(up | move != 0): {(st > 0)[st != 0].mean():.3f}")
    print(f"  mean |move|:       {np.abs(st).mean():.2f} ticks")
    print(f"  mean round trip:   {cost[tight].mean():.2f} ticks "
          f"(spread only)")
    print(f"  taker fee, 2 legs: {fee_ticks[tight].mean():.2f} ticks "
          f"({a.category} rate, mean p {mid[tight].mean():.3f})")
    m_bar = np.abs(st).mean()
    for nm, c in (("spread only", cost[tight].mean()),
                  ("spread + fee", (cost + fee_ticks)[tight].mean()),
                  ("maker exit", cost_maker_exit[tight].mean())):
        print(f"  break-even accuracy, {nm:>13}: "
              f"{break_even_accuracy(m_bar, c):.3f}")

    # ---- directional model, scored on EVERY row -------------------------- #
    import lightgbm as lgb
    cols = [c for c in HAND if c in F.columns]
    tr2 = tr[tight[tr]]
    te2 = te[tight[te]]
    # label is the sign of the move; flat rows are labelled by sign 0 and
    # excluded from FITTING only (they carry no directional information), but
    # they are kept in SCORING because a taker cannot avoid them
    s_tr = signed[tr2]
    fit = s_tr != 0
    g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                           min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=5.0, verbose=-1, n_jobs=10,
                           random_state=0)
    Xtr = np.nan_to_num(F.loc[tr2, cols].to_numpy(np.float32))
    Xte = np.nan_to_num(F.loc[te2, cols].to_numpy(np.float32))
    g.fit(Xtr[fit], (s_tr[fit] > 0).astype(int))
    p = g.predict_proba(Xte)[:, 1]

    s_te = signed[te2]
    cost_te = cost[te2]
    fee_te = fee_ticks[te2]
    moved = s_te != 0
    print(f"\n=== directional GBM, tight books "
          f"(fit {fit.sum():,} / score {len(te2):,}) ===")
    print(f"  AUC on sign, moves only:  "
          f"{roc_auc_score((s_te[moved] > 0).astype(int), p[moved]):.4f}")

    # ---- taker P&L by confidence decile, every row scored ---------------- #
    side = np.where(p > 0.5, 1.0, -1.0)
    gross = side * s_te                            # ticks, before costs
    net_sp = gross - cost_te                       # spread only
    net_all = gross - cost_te - fee_te             # taker exit: the real case
    net_mk = gross - cost_maker_exit[te2]          # passive exit: optimistic
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)

    print(f"\n  taker P&L in TICKS by confidence decile, all rows scored")
    print(f"  {'decile':>7} {'n':>8} {'hit%':>7} {'flat%':>7} {'gross':>8} "
          f"{'-spread':>9} {'TAKER EXIT':>11} {'maker exit':>11}")
    for d in range(10):
        sl = order[d * len(order) // 10:(d + 1) * len(order) // 10]
        mv = s_te[sl] != 0
        hit = ((side[sl] > 0) == (s_te[sl] > 0))[mv].mean() if mv.any() else np.nan
        print(f"  {d+1:>7} {len(sl):>8,} {hit:>7.3f} {1-mv.mean():>7.3f} "
              f"{gross[sl].mean():>8.3f} {net_sp[sl].mean():>9.3f} "
              f"{net_all[sl].mean():>11.3f} {net_mk[sl].mean():>11.3f}")

    print(f"\n  ALL rows: gross {gross.mean():+.3f}  "
          f"net of spread {net_sp.mean():+.3f}  "
          f"TAKER EXIT {net_all.mean():+.3f}  "
          f"maker exit {net_mk.mean():+.3f}  ticks/trade")
    print("  The TAKER EXIT column is the one this data can support. The "
          "maker-exit column assumes a passive fill that trade prints would "
          "be needed to verify, so read it as an upper bound only.")


if __name__ == "__main__":
    main()
