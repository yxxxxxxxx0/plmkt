"""
Is passive market making profitable here, once adverse selection is paid?

A maker's arithmetic is one line:

    edge per fill  =  half-spread captured  -  adverse mid move after the fill

Makers pay no fee on Polymarket, and they earn the spread instead of paying
it, which is why every taker result in this repo is negative and the maker
question stays open. But a resting order does not fill at random: it fills
when someone insists on trading against it, which is disproportionately when
the price is about to keep going. That is adverse selection, and it is the
entire risk.

This measures it directly rather than assuming a value. For every trade on the
tape:

    taker BUY  lifted an offer -> a maker was SHORT at the ask
    taker SELL hit a bid       -> a maker was LONG at the bid

so the maker's post-fill PnL is the mid move signed AGAINST the taker. If the
mid drifts the taker's way, the maker loses exactly that much, and it comes
straight out of the half-spread they captured.

Data note: this needs trade timestamps aligned to the book. The live
millisecond tape (2026-09-04) supports it properly. The backfilled tape is
second-resolution, and one second holds a median of ~85 book updates here, so
running this on backfilled games measures something much blurrier -- the
source is reported per game and mixed results are never pooled silently.

    python maker_adverse.py --slate 2026-09-04
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")
HOR = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_trades(slug):
    live = os.path.join(BASE, "data", "trades_live", f"{slug}.jsonl")
    back = os.path.join(BASE, "data", "trades", f"{slug}.jsonl")
    rows = collections.defaultdict(list)
    if os.path.exists(live):
        src, path, ms = "live-ms", live, True
    elif os.path.exists(back):
        src, path, ms = "backfill-s", back, False
    else:
        return {}, "none"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            aid = t["asset_id"] if ms else t["asset"]
            ts = float(t["ts"]) / 1000.0 if ms else float(t["timestamp"])
            rows[aid].append((ts, float(t["size"]),
                              1.0 if (t.get("side") or "").upper() == "BUY" else -1.0))
    out = {}
    for aid, v in rows.items():
        v.sort()
        out[aid] = (np.array([x[0] for x in v]), np.array([x[1] for x in v]),
                    np.array([x[2] for x in v]))
    return out, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--slate", default=None)
    ap.add_argument("--market-type", default=None)
    ap.add_argument("--max-spread", type=float, default=None)
    ap.add_argument("--require-live", action="store_true",
                    help="skip games whose only tape is the second-resolution backfill")
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pat = f"*-{args.slate}.parquet" if args.slate else "*.parquet"
    files = ([os.path.join(args.panel_dir, f"{g}.parquet") for g in args.games]
             if args.games else sorted(glob.glob(os.path.join(args.panel_dir, pat))))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit("no panel files")

    # acc[(market_type, horizon)] -> [n, sum_adverse_c, sum_halfspread_c, sum_size]
    acc = collections.defaultdict(lambda: np.zeros(4))
    srcs = collections.Counter()
    n_tr = 0

    for path in files:
        slug = os.path.basename(path)[:-8]
        trades, src = load_trades(slug)
        srcs[src] += 1
        if not trades or (args.require_live and src != "live-ms"):
            continue
        cols = ["t", "mid_c", "spread_c", "bb_c", "ba_c", "asset_id", "market_type"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        ok = (np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"])
              & (d["ba_c"] > d["bb_c"]))       # crossed books excluded

        for aid in np.unique(d["asset_id"]):
            sel = ok & (d["asset_id"] == aid)
            if sel.sum() < 2000:
                continue
            mt = str(d["market_type"][np.flatnonzero(sel)[0]])
            if args.market_type and mt != args.market_type:
                continue
            tr = trades.get(aid)
            if tr is None:
                continue
            tt = d["t"][sel]
            mid = d["mid_c"][sel]
            spr = d["spread_c"][sel]
            ttime, tsize, tsign = tr

            # book state in force at each trade
            k = np.searchsorted(tt, ttime, side="right") - 1
            good = k >= 0
            if args.max_spread is not None:
                good &= np.isfinite(spr[np.maximum(k, 0)]) & (spr[np.maximum(k, 0)] <= args.max_spread)
            if not good.any():
                continue
            k = k[good]
            sgn = tsign[good]
            sz = tsize[good]
            m0 = mid[k]
            half = spr[k] / 2.0
            n_tr += int(good.sum())

            for h in HOR:
                j = np.minimum(np.searchsorted(tt, ttime[good] + h, side="left"),
                               tt.size - 1)
                real = (tt[j] - ttime[good]) >= h * 0.5
                dm = mid[j] - m0
                # adverse move = mid drift in the TAKER's direction; that is
                # precisely what the maker on the other side gives back.
                adv = dm * sgn
                v = np.isfinite(adv) & np.isfinite(half) & real
                if not v.any():
                    continue
                a = acc[(mt, h)]
                a[0] += int(v.sum())
                a[1] += float(adv[v].sum())
                a[2] += float(half[v].sum())
                a[3] += float(sz[v].sum())
        log(f"  {slug} ({src}): {n_tr:,} fills so far")

    print()
    print("=" * 100)
    print("MAKER ECONOMICS: half-spread captured vs adverse selection paid")
    print(f"{len(files)} game(s), tape sources: {dict(srcs)}, {n_tr:,} fills examined")
    print("adverse = mid drift in the taker's direction after the fill "
          "(what the maker gives back)")
    print("net = half-spread - adverse.  Positive net means passive quoting pays.")
    print("=" * 100)
    print(f"{'market':>12} {'horizon':>9} {'fills':>10} {'half-spread':>13} "
          f"{'adverse':>11} {'NET c/share':>13}")
    print("-" * 100)
    out = {}
    for mt in sorted({k[0] for k in acc}):
        for h in HOR:
            a = acc.get((mt, h))
            if a is None or a[0] < 200:
                continue
            n, adv, hs = a[0], a[1] / a[0], a[2] / a[0]
            net = hs - adv
            out[f"{mt}|{h}"] = {"fills": int(n), "half_spread_c": hs,
                                "adverse_c": adv, "net_c": net}
            print(f"{mt:>12} {h:>8.2f}s {int(n):>10,} {hs:>+12.4f}c "
                  f"{adv:>+10.4f}c {net:>+12.4f}c")
        print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"games": len(files), "fills": n_tr,
                       "sources": dict(srcs), "rows": out}, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
