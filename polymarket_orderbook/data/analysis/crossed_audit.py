"""How often is a reconstructed book CROSSED (best bid >= best ask)?

A crossed book is impossible in a real order book, so wherever it appears the
reconstruction is wrong. It matters far beyond tidiness:

  * it manufactures risk-free "arbitrage" -- buy the token's own ask below its
    own bid. That is what produced a fake 8-16c complete-set edge.
  * it makes `spread = ask - bid` NEGATIVE, so any filter written as
    `spread <= threshold` silently ADMITS the worst rows instead of removing
    them. sim_jump_capture.py and analyse_jumps.py both gate that way, so this
    audit also says how much those earlier results were affected.

Reports the rate overall and by market type, the size of the crossing, and how
long crossed states persist.
"""
import collections
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

files = sorted(glob.glob(os.path.join("data", "panel", "*.parquet")))
if not files:
    raise SystemExit("no panel files")

tot = collections.Counter()
by_type = collections.defaultdict(lambda: [0, 0])
depth_c = []
dur = []

for path in files:
    tab = pq.read_table(path, columns=["t", "bb_c", "ba_c", "market_type", "asset_id"])
    d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in tab.column_names}
    two = np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"])
    crossed = two & (d["ba_c"] <= d["bb_c"])
    locked = two & (d["ba_c"] == d["bb_c"])
    tot["rows"] += int(two.sum())
    tot["crossed"] += int(crossed.sum())
    tot["locked"] += int(locked.sum())
    if crossed.any():
        depth_c.append((d["bb_c"] - d["ba_c"])[crossed])
    for mt in np.unique(d["market_type"]):
        m = (d["market_type"] == mt) & two
        by_type[str(mt)][0] += int(m.sum())
        by_type[str(mt)][1] += int((m & crossed).sum())
    # persistence, per token
    for a in np.unique(d["asset_id"]):
        m = d["asset_id"] == a
        c = crossed[m]
        if not c.any():
            continue
        t = d["t"][m]
        idx = np.flatnonzero(c)
        brk = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([0], brk + 1))
        ends = np.concatenate((brk, [idx.size - 1]))
        dur.append(t[idx[ends]] - t[idx[starts]])

print(f"{len(files)} game(s)")
print(f"  two-sided rows      {tot['rows']:>14,}")
print(f"  CROSSED (ask<bid)   {tot['crossed']:>14,}   {tot['crossed']/max(tot['rows'],1)*100:.4f}%")
print(f"  locked  (ask==bid)  {tot['locked']:>14,}   {tot['locked']/max(tot['rows'],1)*100:.4f}%")
if depth_c:
    v = np.concatenate(depth_c)
    print(f"  crossing size       p50 {np.median(v):.2f}c  p90 {np.percentile(v,90):.2f}c  max {v.max():.2f}c")
if dur:
    v = np.concatenate(dur)
    print(f"  persistence         p50 {np.median(v)*1000:.0f}ms  p90 {np.percentile(v,90):.1f}s  max {v.max():.1f}s")
print()
print(f"  {'market type':>14} {'rows':>14} {'crossed':>12} {'rate':>9}")
for mt, (n, c) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
    print(f"  {mt:>14} {n:>14,} {c:>12,} {c/max(n,1)*100:>8.4f}%")
