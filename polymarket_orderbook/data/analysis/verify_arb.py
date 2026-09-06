"""Verify a claimed complete-set arbitrage against the RAW recording.

The panel is a derived artifact. Before believing an 8-16c risk-free edge,
find the largest instances and print the actual recorded book records for both
tokens at that moment, straight from data/live/by_game. If the arb is real the
two raw books will show it; if it is an artifact of the panel or the pairing,
that will be obvious here.
"""
import bisect
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))

slug = sys.argv[1] if len(sys.argv) > 1 else "mlb-tor-kc-2026-09-04"
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 3

tm = json.load(open(os.path.join("data", "token_map.json"), encoding="utf-8"))
tokens = tm["tokens"]

tab = pq.read_table(os.path.join("data", "panel", f"{slug}.parquet"),
                    columns=["t", "bb_c", "ba_c", "ba_sz", "asset_id", "market_type"])
d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in tab.column_names}
aid = d["asset_id"]
uniq = {a: np.flatnonzero(aid == a) for a in np.unique(aid)}


def ffill(src_t, src_v, grid):
    i = np.searchsorted(src_t, grid, side="right") - 1
    out = np.full(grid.size, np.nan)
    out[i >= 0] = src_v[i[i >= 0]]
    return out


cands = []
seen = set()
for a, idx in uniq.items():
    info = tokens.get(a)
    if not info:
        continue
    comp = info.get("complement")
    if comp not in uniq or (a, comp) in seen or (comp, a) in seen:
        continue
    seen.add((a, comp))
    j = uniq[comp]
    ta, tb = d["t"][idx], d["t"][j]
    grid = np.union1d(ta, tb)
    aY = ffill(ta, d["ba_c"][idx], grid)
    aN = ffill(tb, d["ba_c"][j], grid)
    ok = np.isfinite(aY) & np.isfinite(aN)
    edge = np.where(ok, 100.0 - (aY + aN), -np.inf)
    k = int(np.argmax(edge))
    if np.isfinite(edge[k]) and edge[k] > 0:
        cands.append((edge[k], float(grid[k]), a, comp,
                      float(aY[k]), float(aN[k]), str(d["market_type"][idx[0]])))

cands.sort(reverse=True)
print(f"{slug}: {len(cands)} pairs with a positive-edge moment; showing top {TOP}\n")

# raw records for the two tokens, read straight from the split recording
path = os.path.join("data", "live", "by_game", f"{slug}.jsonl")
want = {c[2] for c in cands[:TOP]} | {c[3] for c in cands[:TOP]}
times = {a: [] for a in want}
recs = {a: [] for a in want}
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if '"asset_id":"' not in line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        a = r.get("asset_id")
        if a in want:
            times[a].append(r["ts"] / 1000.0)
            recs[a].append(r)

for edge, t, a, b, aY, aN, mt in cands[:TOP]:
    print("=" * 88)
    print(f"{mt}  t={t:.6f}  panel says ask_Y={aY:.1f}c ask_N={aN:.1f}c "
          f"sum={aY+aN:.1f}c edge={edge:+.1f}c")
    for tag, tok in (("YES-side", a), ("NO-side ", b)):
        ts = times[tok]
        i = bisect.bisect_right(ts, t) - 1
        if i < 0:
            print(f"  {tag}: no record at or before t")
            continue
        r = recs[tok][i]
        asks = r.get("asks") or []
        bids = r.get("bids") or []
        print(f"  {tag} {tok[:18]}... ts={r['ts']/1000.0:.3f} (age {t-ts[i]:.3f}s) "
              f"et={r['et']}")
        print(f"      question: {str(r.get('market_question'))[:64]}  "
              f"outcome={r.get('outcome')} ({r.get('outcome_name')})")
        print(f"      best bid {bids[0] if bids else None}   best ask {asks[0] if asks else None}")
        print(f"      asks[:4] {asks[:4]}")
    print(f"  --> raw check: buy both at their best asks costs "
          f"{aY + aN:.1f}c for a set worth 100c")
