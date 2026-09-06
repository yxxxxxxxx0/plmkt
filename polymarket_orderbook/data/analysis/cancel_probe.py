"""How much of the liquidity that leaves this book is CANCELLED, not traded?

A price level shrinking is either a fill or a cancellation, and the book alone
cannot tell you which. Subtracting the trade tape from the observed size
removal leaves the cancellations. If that residual is large, then most of what
moves this book is makers pulling quotes -- which is a signal you can watch in
real time, and one that needs no trade feed to compute once it is calibrated.
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import rebuild_duckdb as rd          # noqa: E402
import rebuild_multi as rm           # noqa: E402

slug = sys.argv[1] if len(sys.argv) > 1 else "mlb-ari-sf-2026-08-27"

trades = collections.defaultdict(float)     # asset -> traded shares (in window)
tr_rows = 0
with open(os.path.join("data", "trades", f"{slug}.jsonl"), encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        trades[t["asset"]] += float(t["size"])
        tr_rows += 1

w = rm.load_game_windows().get(slug, {})
t_lo, t_hi = w["kickoff_ts"], w["game_end_ts"]
asset_ids, ts, bpx, bsz, boff, apx, asz, aoff, meta = rd.load_columns(slug)

print(f"{slug}: {tr_rows:,} trades on tape, window "
      f"{(t_hi-t_lo)/60:.0f} min")
print(f"{'market':>16} {'outcome':>6} {'snapshots':>10} {'added':>13} "
      f"{'removed':>13} {'traded':>12} {'cancelled':>13} {'cxl share':>10}")
print("-" * 100)

tot_rm = tot_tr = 0.0
for aid, lo, hi in rd.asset_boundaries(asset_ids):
    mt = (meta.get(aid) or {}).get("market_type") or "?"
    if mt not in ("moneyline", "spread", "total"):
        continue
    t = rd.dedup_nudge(ts[lo:hi])
    keep = (t >= t_lo) & (t <= t_hi)
    if keep.sum() < 3:
        continue
    idx = np.arange(lo, hi)[keep]

    # Total resting size per snapshot, both sides, via segment sums over the
    # ragged level arrays -- no per-row Python.
    def totals(off, sz):
        starts, ends = off[idx], off[idx + 1]
        out = np.zeros(idx.size)
        nz = ends > starts
        if nz.any():
            cs = np.concatenate(([0.0], np.cumsum(sz)))
            out[nz] = cs[ends[nz]] - cs[starts[nz]]
        return out

    depth = totals(boff, bsz) + totals(aoff, asz)
    d = np.diff(depth)
    added = float(d[d > 0].sum())
    removed = float(-d[d < 0].sum())
    traded = trades.get(aid, 0.0)
    cancelled = removed - traded
    tot_rm += removed
    tot_tr += min(traded, removed)
    print(f"{mt:>16} {(meta.get(aid) or {}).get('outcome','?'):>6} "
          f"{idx.size:>10,} {added:>13,.0f} {removed:>13,.0f} {traded:>12,.0f} "
          f"{cancelled:>13,.0f} {cancelled/removed*100 if removed else 0:>9.1f}%")

print("-" * 100)
print(f"across these tokens: {tot_rm:,.0f} shares of liquidity removed, "
      f"{tot_tr:,.0f} traded -> {(1-tot_tr/tot_rm)*100:.1f}% cancelled")
