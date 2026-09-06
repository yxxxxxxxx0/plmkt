"""Why does taking 500 shares cost ~10c when the quoted spread is only 2.6c?

Prints the depth actually resting at the swing points sim_jump_capture trades,
so its slippage figure can be believed or discarded on evidence rather than
taken on trust. The answer for this data: the MEDIAN swing has 500 shares at
the touch and pays zero slippage, but the 10th percentile has 8, and sweeping
those thin moments is what drags the mean cost to several cents a leg.

Usage:
    python sim_depth_report.py mlb-kc-tor-2026-08-27
"""
import sys

import numpy as np

import rebuild_duckdb as rd
import rebuild_multi as rm
from sim_jump_capture import (best_price_column, side_total, walk_ladder,
                              zigzag_up_legs)

slug = sys.argv[1] if len(sys.argv) > 1 else "mlb-kc-tor-2026-08-27"
theta = 0.01

w = rm.load_game_windows().get(slug, {})
asset_ids, ts, bpx, bsz, boff, apx, asz, aoff, meta = rd.load_columns(slug)
t_lo, t_hi = w["kickoff_ts"], w["game_end_ts"]

touch_sz, tot_sz, qs, slips, n_levels_hit, full_sweep = [], [], [], [], [], 0
for aid, lo, hi in rd.asset_boundaries(asset_ids):
    t = rd.dedup_nudge(ts[lo:hi])
    keep = (t >= t_lo) & (t <= t_hi)
    if keep.sum() < 2:
        continue
    bb = best_price_column(lo, hi, boff, bpx)[keep]
    ba = best_price_column(lo, hi, aoff, apx)[keep]
    rows = np.arange(lo, hi)[keep]
    two = ~(np.isnan(bb) | np.isnan(ba))
    mid = np.where(two, (bb + ba) / 2.0, np.nan)
    for i, j in zigzag_up_legs(mid, theta):
        if not (two[i] and two[j]):
            continue
        if (ba[i] - bb[i] > 0.05) or (ba[j] - bb[j] > 0.05):
            continue
        r = rows[i]
        a0, a1 = aoff[r], aoff[r + 1]
        if a1 <= a0:
            continue
        avail = side_total(r, aoff, asz)
        if avail < 5:
            continue
        touch_sz.append(float(asz[a0]))
        tot_sz.append(avail)
        q = min(500.0, avail, side_total(rows[j], boff, bsz))
        qs.append(q)
        filled, notional, _ = walk_ladder(r, aoff, apx, asz, q)
        if filled > 0:
            slips.append(notional / filled - ba[i])
            csum = np.cumsum(asz[a0:a1])
            n_levels_hit.append(int(np.searchsorted(csum, q) + 1))
        if q >= avail - 1e-9:
            full_sweep += 1

for name, v in (("size at best ask", touch_sz), ("total ask side", tot_sz),
                ("q actually traded", qs)):
    a = np.array(v)
    print(f"{name:>20}: n={a.size:,}  p10={np.percentile(a,10):>10,.0f} "
          f"p50={np.percentile(a,50):>10,.0f} p90={np.percentile(a,90):>10,.0f} "
          f"mean={a.mean():>10,.0f}")
s = np.array(slips) * 100
print(f"{'buy-leg slippage':>20}: p50={np.median(s):+.2f}c  mean={s.mean():+.2f}c  "
      f"p90={np.percentile(s,90):+.2f}c")
lv = np.array(n_levels_hit)
print(f"{'levels consumed':>20}: p50={np.median(lv):.0f}  p90={np.percentile(lv,90):.0f}  "
      f"max={lv.max()}")
print(f"{'swept whole side':>20}: {full_sweep:,} of {len(qs):,} "
      f"({full_sweep/max(len(qs),1)*100:.1f}%)")
