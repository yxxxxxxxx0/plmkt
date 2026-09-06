"""Trade-vs-cancel share of size removal AT THE TOUCH, not across the whole book.

The 99.95%-cancelled figure covered every level, including the 0.01/0.99 stubs
that churn constantly and can never trade. That number answers "what moves the
book" but NOT "what moves a queue", because a resting order only ever sits at
one price -- usually the touch -- and only the removal at THAT price decides
whether it fills or gets cancelled around.

If trades are a meaningful share of removal at the touch, a queue model needs
the trade/cancel decomposition to be correct. If they are still negligible
there, the queue advances almost entirely through cancels whose position in
the queue is unobservable, and no decomposition can rescue it.
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

trades = collections.defaultdict(float)
with open(os.path.join("data", "trades", f"{slug}.jsonl"), encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        trades[t["asset"]] += float(t["size"])

w = rm.load_game_windows().get(slug, {})
t_lo, t_hi = w["kickoff_ts"], w["game_end_ts"]
asset_ids, ts, bpx, bsz, boff, apx, asz, aoff, meta = rd.load_columns(slug)


def touch(off, px, sz, idx):
    """(price, size) of level 0 per snapshot; NaN where the side is empty."""
    a, b = off[idx], off[idx + 1]
    has = b > a
    p = np.full(idx.size, np.nan)
    s = np.zeros(idx.size)
    p[has] = px[a[has]]
    s[has] = sz[a[has]]
    return p, s


print(f"{slug}")
print(f"{'market':>12} {'out':>4} {'touch removed':>15} {'level wiped':>13} "
      f"{'traded':>11} {'trade share of':>15}")
print(f"{'':>12} {'':>4} {'(same price)':>15} {'(price moved)':>13} "
      f"{'(tape)':>11} {'touch removal':>15}")
print("-" * 78)

for aid, lo, hi in rd.asset_boundaries(asset_ids):
    mt = (meta.get(aid) or {}).get("market_type") or "?"
    if mt not in ("moneyline", "spread", "total"):
        continue
    t = rd.dedup_nudge(ts[lo:hi])
    keep = (t >= t_lo) & (t <= t_hi)
    if keep.sum() < 3:
        continue
    idx = np.arange(lo, hi)[keep]

    same_px_removed = 0.0
    wiped = 0.0
    for off, px, sz in ((boff, bpx, bsz), (aoff, apx, asz)):
        p, s = touch(off, px, sz, idx)
        same = (p[1:] == p[:-1]) & ~np.isnan(p[1:])
        d = s[1:] - s[:-1]
        # Size shrinking while the best price is unchanged: someone at the
        # front either traded or cancelled. This is the only removal a
        # resting order at that price competes with.
        same_px_removed += float(-d[same & (d < 0)].sum())
        # Best price changed: whatever sat there is gone entirely.
        moved = ~same & ~np.isnan(p[:-1])
        wiped += float(s[:-1][moved].sum())

    tr = trades.get(aid, 0.0)
    denom = same_px_removed + wiped
    print(f"{mt:>12} {(meta.get(aid) or {}).get('outcome','?'):>4} "
          f"{same_px_removed:>15,.0f} {wiped:>13,.0f} {tr:>11,.0f} "
          f"{tr/denom*100 if denom else 0:>14.2f}%")
