"""
Does jump-timing skill improve the market-making strategy?

The jump models forecast WHEN a large move happens (Hawkes lift up to 12.8x)
but not WHICH WAY (signed move ~0.000c). That is useless for a taker, who
needs direction. It is potentially valuable for a MAKER, who needs the
opposite thing: to know when NOT to be quoting.

A maker's edge is

    net = half-spread captured - adverse selection paid

and adverse selection is exactly what a jump does to a resting order. So the
test is direct: bucket every observed fill by the jump intensity prevailing
at that moment (computed causally, from past jumps only) and compare the
adverse selection paid in each bucket.

If adverse selection rises sharply with intensity, then a filter -- stop
quoting, or widen, when intensity is high -- raises the net edge of strategy
1 without needing any directional forecast at all.

Everything is causal: the Hawkes intensity at a fill uses only jumps strictly
before it, and the Lee-Mykland detector that feeds it uses only trailing
bipower variation.

    python maker_jump_filter.py --slate 2026-09-04 --market-type moneyline
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

from jump_models import hawkes_intensity, lee_mykland

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")
DECAY = 0.5          # best-performing kernel from jump_models.py
HOR = (1.0, 5.0, 10.0)
NBUCKET = 5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_trades(slug):
    live = os.path.join(BASE, "data", "trades_live", f"{slug}.jsonl")
    if not os.path.exists(live):
        return {}, "none"
    rows = collections.defaultdict(list)
    with open(live, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            rows[t["asset_id"]].append(
                (float(t["ts"]) / 1000.0, float(t["size"]),
                 1.0 if (t.get("side") or "").upper() == "BUY" else -1.0))
    out = {}
    for aid, v in rows.items():
        v.sort()
        out[aid] = (np.array([x[0] for x in v]), np.array([x[1] for x in v]),
                    np.array([x[2] for x in v]))
    return out, "live-ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", default="2026-09-04")
    ap.add_argument("--market-type", default="moneyline")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.panel_dir, f"*-{args.slate}.parquet")))
    if args.max_games:
        files = files[:args.max_games]
    if not files:
        raise SystemExit("no panel files")

    # acc[(bucket, H)] -> [n, sum_adverse, sum_halfspread, sum_net, sum_net^2]
    # The second moment is the point of this pass. A higher mean edge per fill
    # is worthless if it is simply payment for higher variance -- that is a
    # risk premium, not an edge, and concentrating capital into it raises
    # return and risk together while leaving the ratio unchanged.
    acc = collections.defaultdict(lambda: np.zeros(5))
    gaps = collections.defaultdict(list)
    n_fill = 0

    for path in files:
        slug = os.path.basename(path)[:-8]
        trades, src = load_trades(slug)
        if src != "live-ms":
            continue
        cols = ["t", "mid_c", "spread_c", "bb_c", "ba_c", "asset_id", "market_type"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        ok = (np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"])
              & (d["ba_c"] > d["bb_c"]))
        if args.market_type:
            ok &= (d["market_type"] == args.market_type)

        for aid in np.unique(d["asset_id"][ok]):
            sel = ok & (d["asset_id"] == aid)
            if sel.sum() < 5000 or aid not in trades:
                continue
            t, mid, spr = d["t"][sel], d["mid_c"][sel], d["spread_c"][sel]
            L, r, sig = lee_mykland(mid)
            nn = max(mid.size, 2)
            Cn = (2 * np.log(nn)) ** 0.5 / 0.7979 - (
                np.log(np.pi) + np.log(np.log(nn))) / (2 * 0.7979 * (2 * np.log(nn)) ** 0.5)
            Sn = 1.0 / (0.7979 * (2 * np.log(nn)) ** 0.5)
            jump = np.abs(L) > (Cn + Sn * (-np.log(-np.log(0.99))))
            lam = hawkes_intensity(t, jump, DECAY)

            ttime, tsz, tsign = trades[aid]
            k = np.searchsorted(t, ttime, side="right") - 1
            good = k >= 0
            if not good.any():
                continue
            k = k[good]
            sgn = tsign[good]
            lam_at = lam[k]
            half = spr[k] / 2.0
            n_fill += int(good.sum())

            # buckets from the token's OWN first-half intensity distribution,
            # so the rule is implementable in real time
            ref = lam[: max(lam.size // 2, 1000)]
            edges = np.quantile(ref, np.linspace(0, 1, NBUCKET + 1)[1:-1])
            b = np.searchsorted(edges, lam_at)

            for H in HOR:
                j = np.minimum(np.searchsorted(t, ttime[good] + H, side="left"),
                               t.size - 1)
                real = (t[j] - ttime[good]) >= H * 0.5
                adv = (mid[j] - mid[k]) * sgn
                v = np.isfinite(adv) & np.isfinite(half) & real
                for bi in range(NBUCKET):
                    m = v & (b == bi)
                    if not m.any():
                        continue
                    net = half[m] - adv[m]
                    a = acc[(bi, H)]
                    a[0] += int(m.sum())
                    a[1] += float(adv[m].sum())
                    a[2] += float(half[m].sum())
                    a[3] += float(net.sum())
                    a[4] += float((net ** 2).sum())
                    if H == HOR[0] and m.sum() > 2:
                        # spacing between fills in this bucket: clustered fills
                        # are not independent bets, so the effective sample is
                        # smaller than the raw count suggests
                        gaps[bi].append(np.diff(ttime[good][m]))
        log(f"  {slug}: {n_fill:,} fills")

    print()
    print("=" * 96)
    print(f"MAKER EDGE BY JUMP INTENSITY  ({args.market_type}, "
          f"{len(files)} game(s), {n_fill:,} fills)")
    print(f"Hawkes intensity (decay {DECAY:g}s) at the moment of the fill, "
          f"bucketed into {NBUCKET} quintiles.")
    print("Q1 = calmest, Q5 = most jump-clustered. All causal.")
    print("=" * 96)
    out = {}
    for H in HOR:
        print(f"\n  horizon {H:g}s")
        print(f"    {'bucket':>8} {'fills':>9} {'half-spread':>13} {'adverse':>11} "
              f"{'NET c/share':>13} {'sd':>9} {'mean/sd':>9} {'IR*sqrtN':>10}")
        rows = []
        for bi in range(NBUCKET):
            a = acc.get((bi, H))
            if a is None or a[0] < 50:
                continue
            n, adv, hs = a[0], a[1] / a[0], a[2] / a[0]
            mu = a[3] / a[0]
            var = max(a[4] / a[0] - mu * mu, 1e-12)
            sd = var ** 0.5
            ir = mu / sd
            rows.append((bi, int(n), hs, adv, hs - adv, sd, ir, ir * n ** 0.5))
            print(f"    {'Q'+str(bi+1):>8} {int(n):>9,} {hs:>+12.4f}c "
                  f"{adv:>+10.4f}c {hs-adv:>+12.4f}c {sd:>8.3f}c {ir:>+9.4f} "
                  f"{ir*n**0.5:>+10.2f}")
        if H == HOR[0]:
            print()
            print("    per-fill information ratio (mean/sd) is the number that decides")
            print("    whether Q5 is edge or risk premium. If IR is flat across buckets,")
            print("    Q5 pays more only because it risks more.")
        if len(rows) >= 2:
            alln = sum(r[1] for r in rows)
            allnet = sum(r[1] * r[4] for r in rows) / alln
            keep = [r for r in rows if r[0] < NBUCKET - 1]
            if keep:
                kn = sum(r[1] for r in keep)
                knet = sum(r[1] * r[4] for r in keep) / kn
                print(f"    {'-'*56}")
                print(f"    quote always      : {allnet:+.4f}c/share over {alln:,} fills")
                print(f"    skip top quintile : {knet:+.4f}c/share over {kn:,} fills "
                      f"({kn/alln*100:.0f}% of flow)")
                print(f"    improvement       : {knet-allnet:+.4f}c/share "
                      f"({(knet-allnet)/abs(allnet)*100:+.1f}%)")
            out[str(H)] = {"rows": rows, "all_net": allnet,
                           "filtered_net": knet if keep else None}

    if gaps:
        print()
        print("  fill clustering (seconds between consecutive fills in a bucket):")
        for bi in sorted(gaps):
            g = np.concatenate(gaps[bi])
            if g.size:
                print(f"    Q{bi+1}: median {np.median(g):>8.2f}s  "
                      f"p10 {np.percentile(g,10):>7.2f}s   n={g.size:,}")
        print("  (short gaps in Q5 mean fills arrive together -- correlated bets,")
        print("   so the effective sample is smaller than the fill count implies)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"market_type": args.market_type, "fills": n_fill,
                       "decay": DECAY, "by_horizon": out}, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
