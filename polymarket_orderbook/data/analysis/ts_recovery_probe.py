"""What does the retroactive tape's SECOND-resolution timestamp actually cost,
and can the missing milliseconds be recovered from the book?

Two questions:

 1. AMBIGUITY. How many book updates share the second a trade is stamped
    with? That is the size of the haystack -- the number of distinct book
    states the trade could have hit.

 2. RECOVERY. A trade of `size` at `price` should appear in the book as a
    decrement of exactly that size at exactly that price. If exactly one
    transition in the second matches, the millisecond is recovered and the
    retroactive tape becomes as good as a live capture for that trade. This
    measures how often that works.

Matching on both price AND exact size is what makes this plausible despite
~96% of touch removal being cancellations: a cancel has to coincidentally be
for the same size at the same price to create a false match.
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
MAX_TRADES = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
TOL = 1e-6

w = rm.load_game_windows().get(slug, {})
t_lo, t_hi = w["kickoff_ts"], w["game_end_ts"]

by_asset = collections.defaultdict(list)
with open(os.path.join("data", "trades", f"{slug}.jsonl"), encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        ts = float(t["timestamp"])
        if t_lo <= ts <= t_hi:
            by_asset[t["asset"]].append((ts, float(t["price"]), float(t["size"])))

asset_ids, ts, bpx, bsz, boff, apx, asz, aoff, meta = rd.load_columns(slug)

def level_size(off, px, sz, row, price):
    i0, i1 = off[row], off[row + 1]
    for k in range(i0, i1):
        if abs(px[k] - price) < 5e-7:
            return sz[k]
    return 0.0

upd_per_sec = []
res = collections.Counter()
checked = 0

for aid, lo, hi in rd.asset_boundaries(asset_ids):
    trades = by_asset.get(aid)
    if not trades:
        continue
    mt = (meta.get(aid) or {}).get("market_type")
    if mt not in ("moneyline", "spread", "total"):
        continue
    t = rd.dedup_nudge(ts[lo:hi])
    keep = (t >= t_lo) & (t <= t_hi)
    times = t[keep]
    rows = np.arange(lo, hi)[keep]
    if times.size < 10:
        continue
    tl = times.tolist()

    for tt, price, size in trades:
        if checked >= MAX_TRADES:
            break
        # every book transition stamped inside the trade's second, plus a
        # second of slack either way for clock skew between the two feeds
        i0 = bisect.bisect_left(tl, tt - 1.0)
        i1 = bisect.bisect_right(tl, tt + 2.0)
        n = i1 - i0
        if n < 2:
            res["no_book_coverage"] += 1
            continue
        upd_per_sec.append(n)
        checked += 1

        hits = 0
        for k in range(i0, min(i1, len(tl)) - 1):
            r0, r1 = rows[k], rows[k + 1]
            for off, px, sz in ((boff, bpx, bsz), (aoff, apx, asz)):
                d = level_size(off, px, sz, r0, price) - level_size(off, px, sz, r1, price)
                if d > 0 and abs(d - size) < max(TOL, size * 1e-6):
                    hits += 1
        if hits == 1:
            res["unique"] += 1
        elif hits == 0:
            res["none"] += 1
        else:
            res["ambiguous"] += 1
    if checked >= MAX_TRADES:
        break

u = np.array(upd_per_sec)
print(f"{slug}: {checked:,} in-game trades checked")
print()
print("1. AMBIGUITY -- book updates inside the trade's timestamp window (+/-1s)")
if u.size:
    print(f"   p10={np.percentile(u,10):.0f}  p50={np.percentile(u,50):.0f}  "
          f"p90={np.percentile(u,90):.0f}  max={u.max():.0f}  mean={u.mean():.1f}")
print()
print("2. RECOVERY -- can the exact (price, size) decrement be found?")
tot = sum(res[k] for k in ("unique", "none", "ambiguous"))
for k in ("unique", "ambiguous", "none"):
    print(f"   {k:>10}: {res[k]:>7,}  {res[k]/tot*100 if tot else 0:>5.1f}%")
if res["no_book_coverage"]:
    print(f"   (no book coverage for {res['no_book_coverage']:,} trades)")
