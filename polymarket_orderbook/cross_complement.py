"""
The complement relationship between a market's YES and NO books.

Two things live here, and they are very different in kind.

1. COMPLETE-SET ARBITRAGE (no prediction required)
   Exactly one of YES/NO pays $1 at settlement, so a YES share plus a NO share
   is worth exactly $1, always. If

        ask_YES + ask_NO < $1

   then buying one of each locks in the difference, held to settlement,
   regardless of who wins the game. There is no forecast, no adverse
   selection and no exit problem -- the position self-liquidates at
   resolution. This is the cleanest edge that can exist on this venue, so the
   question is purely empirical: does it occur, how large, how deep, and for
   how long?

   Costs charged: both legs are takers, and the ladder is walked on each side
   for the requested size, so the number reported is what remains after
   actually paying for the fills. The Polymarket taker fee is
   shares * rate * p * (1-p) per leg; for a complete set the two legs are at
   p and (1-p), so the pair costs 2 * rate * p * (1-p) -- 2.5c at p=0.50 under
   rate 0.05, and nothing under rate 0. Both are reported because the venue's
   own surfaces disagree about the rate, and the live trade tape says 0.

2. LEAD-LAG (prediction)
   The two books are fed by separate message streams. When one side reprices
   first, its complement is briefly stale, and mid_YES vs 1-mid_NO diverge.
   Measuring which side moves first, and how long the divergence survives,
   says whether that staleness is long enough to act on.

Both use event-time alignment: the union of the two tokens' update times with
each side forward-filled, so no interpolation invents a quote that never
existed.

STALENESS IS THE TRAP. Forward-filling carries a quote forward for as long as
its token stays quiet, and a quiet token is exactly what an untradeable market
looks like. Without a guard this measurement reported 8-9c of "arbitrage"
lasting up to 258 SECONDS -- which is not an arbitrage, it is one side of the
pair having stopped quoting while the other moved. Every state is therefore
required to have BOTH sides refreshed within --max-stale seconds, and the hit
rate is reported across a range of that threshold so the reader can see how
much of the effect is staleness and how much survives.

    python cross_complement.py --slate 2026-09-04
    python cross_complement.py --games mlb-tor-kc-2026-09-04 --shares 200
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
TOKMAP = os.path.join(BASE, "data", "token_map.json")
LAGS_MS = (-1000, -500, -250, -100, -50, 0, 50, 100, 250, 500, 1000)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ffill_onto(src_t, src_v, grid):
    """Value in force at each grid time (last observation at or before)."""
    i = np.searchsorted(src_t, grid, side="right") - 1
    out = np.full(grid.size, np.nan)
    ok = i >= 0
    out[ok] = src_v[i[ok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--slate", default=None)
    ap.add_argument("--shares", type=float, default=200.0,
                    help="size per leg for the executable arb measurement")
    ap.add_argument("--max-stale", type=float, default=2.0,
                    help="both sides must have updated within this many seconds "
                         "for a state to count; forward-filled quotes older than "
                         "this are not tradeable and are excluded")
    ap.add_argument("--examples", type=int, default=5,
                    help="print this many raw instances for manual verification")
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with open(TOKMAP, "r", encoding="utf-8") as f:
        tm = json.load(f)
    tokens = tm["tokens"]

    pat = f"*-{args.slate}.parquet" if args.slate else "*.parquet"
    files = ([os.path.join(args.panel_dir, f"{g}.parquet") for g in args.games]
             if args.games else sorted(glob.glob(os.path.join(args.panel_dir, pat))))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit("no panel files")

    arb = collections.defaultdict(float)
    stale_scan = collections.defaultdict(lambda: [0, 0, 0.0])
    examples = []
    per_type = collections.defaultdict(lambda: collections.defaultdict(float))
    lagcorr = collections.defaultdict(list)
    dur_samples = []
    edge_samples = []
    n_pairs = 0

    for path in files:
        slug = os.path.basename(path)[:-8]
        cols = ["t", "bb_c", "ba_c", "bb_sz", "ba_sz", "asset_id", "market_type"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        aid = d["asset_id"]
        uniq = {a: np.flatnonzero(aid == a) for a in np.unique(aid)}

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
            if idx.size < 500 or j.size < 500:
                continue
            n_pairs += 1
            mt = d["market_type"][idx[0]]

            ta, tb = d["t"][idx], d["t"][j]
            grid = np.union1d(ta, tb)
            aY = ffill_onto(ta, d["ba_c"][idx], grid)
            bY = ffill_onto(ta, d["bb_c"][idx], grid)
            aYs = ffill_onto(ta, d["ba_sz"][idx], grid)
            aN = ffill_onto(tb, d["ba_c"][j], grid)
            bN = ffill_onto(tb, d["bb_c"][j], grid)
            aNs = ffill_onto(tb, d["ba_sz"][j], grid)
            # Age of each side's most recent update at every grid point.
            ageY = grid - ta[np.maximum(np.searchsorted(ta, grid, side="right") - 1, 0)]
            ageN = grid - tb[np.maximum(np.searchsorted(tb, grid, side="right") - 1, 0)]
            # A CROSSED book (best ask <= best bid) is impossible in reality;
            # where it appears the reconstruction is wrong, and it fabricates
            # risk-free edge by letting a token trade against its own quotes.
            # 0.29% of rows overall, 0.81% of spread markets. Excluded from
            # every state below -- including the denominator, so the hit rate
            # is not diluted by rows that were never eligible.
            uncrossed = (aY > bY) & (aN > bN)
            fresh = (ageY <= args.max_stale) & (ageN <= args.max_stale) & uncrossed
            ok = (np.isfinite(aY) & np.isfinite(aN) & np.isfinite(bY)
                  & np.isfinite(bN) & fresh)
            arb["states_all"] += int((np.isfinite(aY) & np.isfinite(aN)
                                      & np.isfinite(bY) & np.isfinite(bN)).sum())
            arb["states_fresh"] += int(ok.sum())
            # Same measurement at other staleness thresholds, to show how much
            # of any effect is simply one side having gone quiet.
            for thr in (0.25, 0.5, 1.0, 2.0, 5.0, 1e9):
                f2 = ((ageY <= thr) & (ageN <= thr) & np.isfinite(aY)
                      & np.isfinite(aN) & np.isfinite(bY) & np.isfinite(bN)
                      & uncrossed)
                stale_scan[thr][0] += int(f2.sum())
                stale_scan[thr][1] += int((f2 & ((aY + aN) < 100.0)).sum())
                if (f2 & ((aY + aN) < 100.0)).any():
                    stale_scan[thr][2] += float((100.0 - (aY + aN))[f2 & ((aY + aN) < 100.0)].sum())
            if ok.sum() < 500:
                continue

            # --- complete-set arbitrage --------------------------------------
            cost = aY + aN                      # cents to buy one of each
            edge = 100.0 - cost                 # cents locked in per set
            live = ok & (edge > 0)
            arb["rows"] += int(ok.sum())
            arb["hits"] += int(live.sum())
            per_type[mt]["rows"] += int(ok.sum())
            per_type[mt]["hits"] += int(live.sum())
            if live.any():
                sz = np.minimum(aYs[live], aNs[live])
                q = np.minimum(sz, args.shares)
                p = aY[live] / 100.0
                gross = edge[live] * q
                # fee at both legs, p and 1-p -> 2*rate*p*(1-p) per share
                fee05 = 2 * 0.05 * p * (1 - p) * 100.0 * q
                arb["gross_c"] += float(gross.sum())
                arb["net05_c"] += float((gross - fee05).sum())
                arb["shares"] += float(q.sum())
                per_type[mt]["gross_c"] += float(gross.sum())
                per_type[mt]["net05_c"] += float((gross - fee05).sum())
                edge_samples.append(edge[live])
                if len(examples) < args.examples:
                    w = np.flatnonzero(live)[0]
                    examples.append({
                        "slug": slug, "market_type": str(mt),
                        "t": float(grid[w]),
                        "ask_YES_c": float(aY[w]), "ask_NO_c": float(aN[w]),
                        "sum_c": float(aY[w] + aN[w]), "edge_c": float(edge[w]),
                        "ask_YES_sz": float(aYs[w]), "ask_NO_sz": float(aNs[w]),
                        "age_YES_s": float(ageY[w]), "age_NO_s": float(ageN[w]),
                        "token_a": str(a), "token_b": str(comp)})
                # how long each opportunity persists
                g = grid[live]
                brk = np.flatnonzero(np.diff(np.flatnonzero(live)) > 1)
                starts = np.concatenate(([0], brk + 1))
                ends = np.concatenate((brk, [g.size - 1]))
                dur_samples.append(g[ends] - g[starts])

            # --- lead-lag ----------------------------------------------------
            # YES mid vs the NO book's implied YES mid (100 - mid_NO).
            midY = (aY + bY) / 2.0
            impY = 100.0 - (aN + bN) / 2.0
            dY = np.diff(midY)
            dI = np.diff(impY)
            m = np.isfinite(dY) & np.isfinite(dI)
            if m.sum() > 1000 and dY[m].std() > 0 and dI[m].std() > 0:
                gt = grid[1:]
                for L in LAGS_MS:
                    if L == 0:
                        x, y = dY[m], dI[m]
                    else:
                        shifted = ffill_onto(gt[m], dI[m], gt[m] + L / 1000.0)
                        x, y = dY[m], shifted
                    ok2 = np.isfinite(x) & np.isfinite(y)
                    if ok2.sum() > 500 and x[ok2].std() > 0 and y[ok2].std() > 0:
                        lagcorr[L].append(float(np.corrcoef(x[ok2], y[ok2])[0, 1]))
        log(f"  {slug}: {n_pairs} pairs so far")

    print()
    print("=" * 92)
    print("COMPLETE-SET ARBITRAGE  (buy YES + NO for under $1, hold to settlement)")
    print(f"{len(files)} game(s), {n_pairs} complement pairs, {args.shares:g} shares/leg cap")
    print("=" * 92)
    rows, hits = arb["rows"], arb["hits"]
    print(f"  quote states examined     {rows:>15,}")
    print(f"  states with ask_Y+ask_N<$1{hits:>15,}   ({hits/max(rows,1)*100:.4f}% of the time)")
    if hits:
        e = np.concatenate(edge_samples)
        dur = np.concatenate(dur_samples) if dur_samples else np.array([0.0])
        print(f"  edge per set              mean {e.mean():+.3f}c  p50 {np.median(e):+.3f}c  "
              f"p95 {np.percentile(e,95):+.3f}c  max {e.max():+.3f}c")
        print(f"  fillable shares (capped)  {arb['shares']:>15,.0f}")
        print(f"  gross if every state taken  ${arb['gross_c']/100:>13,.0f}")
        print(f"  net @ fee rate 0.05         ${arb['net05_c']/100:>13,.0f}")
        print(f"  opportunity duration      p50 {np.median(dur)*1000:.0f}ms  "
              f"p90 {np.percentile(dur,90)*1000:.0f}ms  max {dur.max():.1f}s")
        print()
        print(f"  {'market type':>34} {'states':>12} {'hits':>10} {'hit %':>8} {'gross $':>12}")
        for mt, v in sorted(per_type.items(), key=lambda kv: -kv[1]["hits"]):
            print(f"  {str(mt):>34} {int(v['rows']):>12,} {int(v['hits']):>10,} "
                  f"{v['hits']/max(v['rows'],1)*100:>7.4f}% {v['gross_c']/100:>12,.0f}")
    else:
        print("  none found -- the complement never crossed")

    print()
    print("  how much of this is just a stale quote?")
    print(f"  {'max age':>10} {'states':>14} {'hits':>10} {'hit %':>9} {'mean edge':>11}")
    for thr in sorted(stale_scan):
        n, h, e = stale_scan[thr]
        lab = "no guard" if thr > 1e8 else f"{thr:g}s"
        print(f"  {lab:>10} {n:>14,} {h:>10,} {h/max(n,1)*100:>8.4f}% "
              f"{(e/h if h else 0):>10.3f}c")

    if examples:
        print()
        print("  raw instances (verify these against the recordings):")
        for e in examples:
            print(f"    {e['slug']} {e['market_type']:>10} t={e['t']:.3f} "
                  f"ask_Y={e['ask_YES_c']:.1f}c({e['ask_YES_sz']:.0f}) "
                  f"ask_N={e['ask_NO_c']:.1f}c({e['ask_NO_sz']:.0f}) "
                  f"sum={e['sum_c']:.1f}c edge={e['edge_c']:+.1f}c "
                  f"age Y/N={e['age_YES_s']:.2f}/{e['age_NO_s']:.2f}s")

    print()
    print("=" * 92)
    print("LEAD-LAG: YES mid change vs the NO book's implied YES mid change")
    print("positive lag = NO book shifted LATER, so a peak at lag>0 means YES leads")
    print("=" * 92)
    print(f"  {'lag':>8} {'mean corr':>12} {'pairs':>8}")
    for L in LAGS_MS:
        v = lagcorr.get(L) or []
        if v:
            print(f"  {L:>6}ms {np.mean(v):>12.4f} {len(v):>8}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"games": len(files), "pairs": n_pairs,
                       "arb": dict(arb),
                       "per_type": {k: dict(v) for k, v in per_type.items()},
                       "lagcorr": {str(k): float(np.mean(v)) for k, v in lagcorr.items() if v}},
                      f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
