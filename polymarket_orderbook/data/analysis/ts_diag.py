"""Why does a trade not appear as a matching decrement in the book?

Prints concrete cases: the trade, whether its price exists as a level at all
nearby, and the largest decrements actually seen at that price. Distinguishes
a timestamp problem (price never present / no activity) from a mechanism
problem (present, but sizes never line up).
"""
import bisect
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
w = rm.load_game_windows().get(slug, {})
t_lo, t_hi = w["kickoff_ts"], w["game_end_ts"]

by_asset = collections.defaultdict(list)
with open(os.path.join("data", "trades", f"{slug}.jsonl"), encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        ts = float(t["timestamp"])
        if t_lo <= ts <= t_hi:
            by_asset[t["asset"]].append((ts, float(t["price"]), float(t["size"]),
                                         t["side"], t.get("outcome")))

asset_ids, ts, bpx, bsz, boff, apx, asz, aoff, meta = rd.load_columns(slug)

def lv(off, px, sz, row, price):
    i0, i1 = off[row], off[row + 1]
    for k in range(i0, i1):
        if abs(px[k] - price) < 5e-7:
            return float(sz[k])
    return None

present = collections.Counter()
shown = 0
for aid, lo, hi in rd.asset_boundaries(asset_ids):
    trades = by_asset.get(aid)
    if not trades or (meta.get(aid) or {}).get("market_type") != "moneyline":
        continue
    t = rd.dedup_nudge(ts[lo:hi])
    keep = (t >= t_lo) & (t <= t_hi)
    times = t[keep]; rows = np.arange(lo, hi)[keep]
    tl = times.tolist()

    for tt, price, size, side, outcome in trades[:400]:
        i0 = bisect.bisect_left(tl, tt - 1.0)
        i1 = bisect.bisect_right(tl, tt + 2.0)
        if i1 - i0 < 2:
            continue
        seen_bid = any(lv(boff, bpx, bsz, rows[k], price) is not None
                       for k in range(i0, i1))
        seen_ask = any(lv(aoff, apx, asz, rows[k], price) is not None
                       for k in range(i0, i1))
        comp = round(1.0 - price, 6)
        seen_cb = any(lv(boff, bpx, bsz, rows[k], comp) is not None
                      for k in range(i0, i1))
        key = (f"bid@p={seen_bid}", f"ask@p={seen_ask}", f"bid@1-p={seen_cb}")
        present[key] += 1

        if shown < 6 and (seen_bid or seen_ask):
            drops = []
            for k in range(i0, i1 - 1):
                for nm, off, px, sz in (("bid", boff, bpx, bsz), ("ask", aoff, apx, asz)):
                    a = lv(off, px, sz, rows[k], price)
                    b = lv(off, px, sz, rows[k + 1], price)
                    if a is not None and b is not None and a - b > 0:
                        drops.append((nm, round(a - b, 4)))
                    elif a is not None and b is None:
                        drops.append((nm + "*wiped", round(a, 4)))
            drops.sort(key=lambda x: -x[1])
            print(f"\ntrade {side} {size:>10,.2f} @ {price} ({outcome[:16]}) "
                  f"window {i1-i0} updates")
            print(f"   decrements at {price}: {drops[:6]}")
            shown += 1
    break

print("\nprice presence in the book near the trade (moneyline):")
for k, v in present.most_common(6):
    print(f"  {v:>6}  {k}")
