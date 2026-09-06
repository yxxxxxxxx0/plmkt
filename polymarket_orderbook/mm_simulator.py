"""
Two-sided market-making simulator with inventory, skew, limits and SETTLEMENT.

Everything before this measured edge PER FILL. That is the right way to find
an edge but the wrong way to size one, because the two risks that can actually
end a market-making operation never appear in a per-fill average:

  INVENTORY   fills arrive in bursts (median 3.0s apart in the jumpiest
              quintile vs 38.7s in the calmest), so a passive quoter
              accumulates a one-sided position exactly while the price runs
              away from it.
  SETTLEMENT  these are BINARY contracts. Inventory carried to resolution does
              not mark out at the mid -- it pays 0 or 1. A book that looks
              flat in cents can be a large directional bet in dollars, and the
              loss is not bounded by any spread.

So this simulates quoting rather than scoring fills:

  * quote both sides continuously; the strategy is maker-in AND maker-out,
    because a taker exit was measured at -1.85c/share and loses in 41/41 games
  * on each observed trade, capture a share of the flow (--capture) capped by
    quote size -- the fill-rate assumption, stated as a parameter because it
    is the one thing this data cannot settle
  * skew quotes against inventory: long inventory shifts both quotes down, so
    the sell side fills first and the position mean-reverts to flat
  * refuse to add to a position beyond --max-inv
  * TAPER the inventory limit to zero over the last --flatten-frac of the
    game, and during that window quote ONLY the side that reduces inventory.
    Without this the simulator ends games with a mean |inventory| of ~1,150
    shares, and every worst-case token was short into a contract that
    resolved at 100c. A binary market walks to its outcome, so a symmetric
    quoter is systematically left holding the losing side; risk has to be
    retired on a clock, not left to mean-revert.
  * mark terminal inventory at SETTLEMENT (0 or 1), inferred from where the
    token's price ends, not at the last mid

Latency is applied as in validate_realistic.py: the quote in force at a fill
is the one derived from the book --latency ago, so a stale quote is picked off
and a quote the market has run past simply misses.

    python mm_simulator.py --slate 2026-09-04 --capture 0.10 --max-inv 2000
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
from validate_strategy import load_trades

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")
WARMUP = 0.20


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def taper_cap(frac_done, args):
    """Inventory cap as a function of how far through the game we are."""
    if args.flatten_frac <= 0:
        return args.max_inv
    start = 1.0 - args.flatten_frac
    if frac_done <= start:
        return args.max_inv
    left = max(0.0, (1.0 - frac_done) / args.flatten_frac)
    return args.max_inv * left


def simulate_token(t, mid, bb, ba, spr, trades_at, args, lam):
    """One token, one pass. Returns a dict of results, or None."""
    n = t.size
    ttime, tsize, tsign = trades_at
    if ttime.size < 20:
        return None

    warm = int(n * WARMUP)
    t_start = t[warm]

    # settlement: where the token actually ends. Binary contracts resolve to
    # 0 or 1, so the last observed mid is rounded to the nearer outcome. If it
    # never resolves cleanly the token is skipped rather than guessed at.
    final = mid[-1] / 100.0
    if final > 0.90:
        settle = 100.0
    elif final < 0.10:
        settle = 0.0
    else:
        return None

    inv = 0.0
    cash = 0.0
    fills = 0
    inv_track = []
    pnl_track = []
    skipped_inv = 0
    lam_ref = np.quantile(lam[warm:max(warm + 5000, warm + 1)], 0.8) if n > warm + 5000 else 1.0

    idx = np.searchsorted(t, ttime, side="right") - 1
    for m in range(ttime.size):
        k = idx[m]
        if k < warm or k >= n:
            continue
        ft = ttime[m]
        # quote derived from the book `latency` ago
        ks = max(np.searchsorted(t, ft - args.latency, side="right") - 1, 0)
        q_bid, q_ask = bb[ks], ba[ks]
        if not (np.isfinite(q_bid) and np.isfinite(q_ask)) or q_ask <= q_bid:
            continue

        # inventory skew: shift both quotes by a fraction of the spread per
        # unit of inventory relative to the cap. Long -> quotes move DOWN, so
        # the ask is easier to hit and the bid harder, pulling us back to flat.
        sk = args.skew * (inv / max(args.max_inv, 1.0)) * (q_ask - q_bid)  # noqa
        q_bid -= sk
        q_ask -= sk

        # how far through the game are we, and what is the cap now
        frac = (ft - t_start) / max(t[-1] - t_start, 1e-9)
        cap = taper_cap(frac, args)
        reducing_only = args.flatten_frac > 0 and frac > (1.0 - args.flatten_frac)

        # size, optionally scaled down when jump intensity is high
        size = args.size
        if args.intensity_sizing and lam[k] > lam_ref:
            size *= args.intensity_floor

        taker_buy = tsign[m] > 0          # taker lifts my ask -> I sell
        if taker_buy:
            # selling reduces inventory only when we are long
            if reducing_only and inv <= 0:
                continue
            if q_ask > ba[k] + 1e-9:      # my stale ask is behind the touch
                continue
            fill = min(size, tsize[m] * args.capture)
            if inv - fill < -cap:
                fill = max(0.0, inv + cap)
                skipped_inv += 1
            if fill <= 0:
                continue
            inv -= fill
            cash += fill * q_ask
        else:
            # buying reduces inventory only when we are short
            if reducing_only and inv >= 0:
                continue
            if q_bid < bb[k] - 1e-9:
                continue
            fill = min(size, tsize[m] * args.capture)
            if inv + fill > cap:
                fill = max(0.0, cap - inv)
                skipped_inv += 1
            if fill <= 0:
                continue
            inv += fill
            cash -= fill * q_bid
        fills += 1
        inv_track.append(inv)
        pnl_track.append(cash + inv * mid[k])

    if fills < 20:
        return None
    inv_track = np.array(inv_track)
    pnl_track = np.array(pnl_track)
    mtm_final = cash + inv * mid[-1]
    settled = cash + inv * settle
    peak = np.maximum.accumulate(pnl_track)
    return {
        "fills": fills, "shares": float(np.abs(np.diff(np.concatenate(([0.0], inv_track)))).sum()),
        "mtm_c": float(mtm_final), "settled_c": float(settled),
        "settle_delta_c": float(settled - mtm_final),
        "final_inv": float(inv), "max_abs_inv": float(np.abs(inv_track).max()),
        "mean_abs_inv": float(np.abs(inv_track).mean()),
        "maxdd_c": float((peak - pnl_track).max()),
        "inv_cap_hits": skipped_inv,
        "settle_price": settle,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", default="2026-09-04")
    ap.add_argument("--market-type", default="moneyline")
    ap.add_argument("--capture", type=float, default=0.10,
                    help="fraction of each trade's size you win (fill-rate assumption)")
    ap.add_argument("--size", type=float, default=500.0, help="max shares per quote")
    ap.add_argument("--max-inv", type=float, default=2000.0, help="hard inventory cap, shares")
    ap.add_argument("--skew", type=float, default=1.0,
                    help="quote shift per unit of inventory/cap, in spreads")
    ap.add_argument("--latency", type=float, default=0.5)
    ap.add_argument("--intensity-sizing", action="store_true",
                    help="shrink size when Hawkes intensity is high")
    ap.add_argument("--intensity-floor", type=float, default=0.3)
    ap.add_argument("--flatten-frac", type=float, default=0.0,
                    help="fraction of the game at the end over which the "
                         "inventory cap tapers to zero and only "
                         "inventory-reducing quotes are shown")
    ap.add_argument("--tape", default="auto")
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    pat = f"*-{args.slate}.parquet" if args.slate else "*.parquet"
    files = sorted(glob.glob(os.path.join(args.panel_dir, pat)))
    res = []
    for path in files:
        slug = os.path.basename(path)[:-8]
        trades, tsrc = load_trades(slug, args.tape)
        if not trades:
            continue
        cols = ["t", "mid_c", "spread_c", "bb_c", "ba_c", "asset_id", "market_type"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        ok = (np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"]) & (d["ba_c"] > d["bb_c"]))
        if args.market_type:
            ok &= (d["market_type"] == args.market_type)
        for aid in np.unique(d["asset_id"][ok]):
            sel = ok & (d["asset_id"] == aid)
            if sel.sum() < 20000 or aid not in trades:
                continue
            t, mid = d["t"][sel], d["mid_c"][sel]
            L, r, sig = lee_mykland(mid)
            nn = max(t.size, 2)
            Cn = (2 * np.log(nn)) ** 0.5 / 0.7979 - (
                np.log(np.pi) + np.log(np.log(nn))) / (2 * 0.7979 * (2 * np.log(nn)) ** 0.5)
            Sn = 1.0 / (0.7979 * (2 * np.log(nn)) ** 0.5)
            lam = hawkes_intensity(t, np.abs(L) > (Cn + Sn * (-np.log(-np.log(0.99)))), 0.5)
            out = simulate_token(t, mid, d["bb_c"][sel], d["ba_c"][sel],
                                 d["spread_c"][sel], trades[aid], args, lam)
            if out:
                out["slug"] = slug
                out["tape"] = tsrc
                res.append(out)
        if not args.quiet:
            log(f"  {slug}: {len([r for r in res if r['slug']==slug])} tokens simulated")

    if not res:
        raise SystemExit("nothing simulated")

    mtm = np.array([r["mtm_c"] for r in res]) / 100.0        # dollars
    stl = np.array([r["settled_c"] for r in res]) / 100.0
    dlt = np.array([r["settle_delta_c"] for r in res]) / 100.0
    sh = np.array([r["shares"] for r in res])
    fin = np.array([r["final_inv"] for r in res])
    mx = np.array([r["max_abs_inv"] for r in res])
    dd = np.array([r["maxdd_c"] for r in res]) / 100.0

    print()
    print("=" * 92)
    print(f"MARKET-MAKING SIMULATION  ({args.market_type}, {len(res)} tokens)")
    print(f"capture {args.capture:.0%} of flow | quote {args.size:g} sh | "
          f"inv cap +/-{args.max_inv:g} | skew {args.skew:g} | latency "
          f"{args.latency*1000:.0f}ms" + (" | intensity-sized" if args.intensity_sizing else ""))
    print("=" * 92)
    print(f"  shares traded            {sh.sum():>14,.0f}")
    print(f"  PnL marked to mid        ${mtm.sum():>13,.0f}   "
          f"({mtm.sum()/max(sh.sum(),1)*100:>+7.4f} c/share)")
    print(f"  PnL after SETTLEMENT     ${stl.sum():>13,.0f}   "
          f"({stl.sum()/max(sh.sum(),1)*100:>+7.4f} c/share)")
    print(f"  settlement impact        ${dlt.sum():>13,.0f}   "
          f"<- terminal inventory paid 0 or 1, not the mid")
    print(f"  tokens profitable        {int((stl>0).sum()):>7}/{len(res)}")
    print()
    print(f"  final inventory   |mean| {np.abs(fin).mean():>10,.0f} sh   "
          f"p95 {np.percentile(np.abs(fin),95):>10,.0f}   max {np.abs(fin).max():>10,.0f}")
    print(f"  peak inventory    |mean| {mx.mean():>10,.0f} sh   "
          f"p95 {np.percentile(mx,95):>10,.0f}   max {mx.max():>10,.0f}")
    print(f"  max drawdown (mid) mean  ${dd.mean():>9,.0f}   "
          f"p95 ${np.percentile(dd,95):>9,.0f}   worst ${dd.max():>9,.0f}")
    print(f"  inventory cap binding    {sum(r['inv_cap_hits'] for r in res):>10,} times")
    print()
    worst = sorted(res, key=lambda r: r["settled_c"])[:5]
    print("  worst tokens after settlement:")
    for r in worst:
        print(f"    {r['slug']:>26}  settled ${r['settled_c']/100:>9,.0f}   "
              f"mtm ${r['mtm_c']/100:>9,.0f}   final inv {r['final_inv']:>+8,.0f} sh   "
              f"resolved {r['settle_price']:.0f}c")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "tokens": len(res),
                       "shares": float(sh.sum()), "mtm_usd": float(mtm.sum()),
                       "settled_usd": float(stl.sum()),
                       "settle_delta_usd": float(dlt.sum())}, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
