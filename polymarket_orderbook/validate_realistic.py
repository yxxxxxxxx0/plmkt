"""
The market-making strategy under LATENCY and a REAL EXIT.

Two things every earlier number in this repo assumed away.

1. LATENCY (--latency, default 500ms)
   A resting quote is priced from a book you saw `latency` ago. So the price
   you are showing at the moment someone trades against you is bb(t-lat), not
   bb(t). That asymmetry is the whole cost:

     * price moved AGAINST you during the delay -> your stale quote is now the
       most aggressive in the book, you get picked off, and you capture
       mid(t) - bb(t-lat), which can be negative.
     * price moved FOR you -> your stale quote is behind the new touch, so you
       simply do not get filled. The good fills are the ones you miss.

   So the fill condition is explicit: a stale bid only trades if it is still
   at or above the current best bid. Nothing is assumed filled.

2. THE EXIT
   "net = half-spread - adverse" is a MARK-TO-MID figure. It is the right
   per-fill measure of edge, but it is not money until the position is closed,
   and closing it is not free. Three exits are priced:

     mark-to-mid   the old optimistic number, kept only for comparison
     taker         cross the spread at t+H and pay the taker fee. The
                   pessimistic bound, and the one a flat-by-design maker
                   actually pays when it cannot find the other side.
     maker         rest on the opposite side and earn the spread again.
                   Requires a second fill, so it is a best case.

   Polymarket charges makers nothing (takerOnly: true), so a fee appears only
   on a taker exit: fee = shares * rate * p * (1-p). The rate is swept because
   Polymarket's own surfaces disagree -- the market config says 0.05 while all
   16,305 live-captured MLB trades reported fee_rate_bps = 0.

Everything else keeps the guarantees from validate_strategy.py: causal
parameters, crossed books removed, feed-gap and warmup rows excluded,
walk-forward by game.

    python validate_realistic.py --latency 0.5 --fee-rate 0.05
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


def token_eval(d, sel, trades, aid, lat, horizon, max_gap, gap_buffer,
               fee_rate, q):
    t = d["t"][sel]
    mid, spr = d["mid_c"][sel], d["spread_c"][sel]
    bb, ba = d["bb_c"][sel], d["ba_c"][sel]
    n = t.size
    if n < 20000 or aid not in trades:
        return None

    dt = np.diff(t, prepend=t[0])
    bad = dt > max_gap
    if bad.any():
        for g in np.flatnonzero(bad.copy()):
            bad[g:np.searchsorted(t, t[g] + gap_buffer, side="right")] = True
    warm = int(n * WARMUP)
    usable = ~bad
    usable[:warm] = False
    if usable.sum() < 5000:
        return None

    ttime, tsz, tsign = trades[aid]
    k = np.searchsorted(t, ttime, side="right") - 1
    ok = (k >= 0)
    k, sgn, ft = k[ok], tsign[ok], ttime[ok]
    keep = usable[k]
    k, sgn, ft = k[keep], sgn[keep], ft[keep]
    if k.size < 30:
        return None
    q["fills_before_latency"] += int(k.size)

    # --- the stale quote -----------------------------------------------------
    ks = np.searchsorted(t, ft - lat, side="right") - 1
    ks = np.maximum(ks, 0)
    stale_bid = bb[ks]
    stale_ask = ba[ks]

    buy = sgn < 0            # taker SELL hits my bid -> I am the buyer
    sell = ~buy
    # A stale quote only trades if it is still at least as aggressive as the
    # current touch. Otherwise the market has moved away from it and someone
    # else is at the front.
    filled = np.zeros(k.size, dtype=bool)
    filled[buy] = stale_bid[buy] >= bb[k][buy] - 1e-9
    filled[sell] = stale_ask[sell] <= ba[k][sell] + 1e-9
    q["fills_after_latency"] += int(filled.sum())
    if filled.sum() < 20:
        return None

    k, sgn, ft = k[filled], sgn[filled], ft[filled]
    price = np.where(sgn < 0, stale_bid[filled], stale_ask[filled])
    m0 = mid[k]
    # edge captured at the moment of the fill, versus the CURRENT mid
    edge = np.where(sgn < 0, m0 - price, price - m0)

    j = np.minimum(np.searchsorted(t, ft + horizon, side="left"), n - 1)
    real = (t[j] - ft) >= horizon * 0.5
    adv = (mid[j] - m0) * (-np.sign(sgn))    # signed against my position
    # my position is +1 when I bought (taker sold), -1 when I sold
    pos = np.where(sgn < 0, 1.0, -1.0)
    drift = (mid[j] - m0) * pos              # mark-to-mid move in my favour

    half_exit = spr[j] / 2.0
    p_exit = np.clip(mid[j] / 100.0, 1e-6, 1 - 1e-6)
    fee_exit = fee_rate * p_exit * (1 - p_exit) * 100.0     # cents/share
    p_in = np.clip(price / 100.0, 1e-6, 1 - 1e-6)

    v = np.isfinite(edge) & np.isfinite(drift) & real & np.isfinite(half_exit)
    if v.sum() < 20:
        return None

    # three exits
    mark = edge[v] + drift[v]                                  # mark-to-mid
    taker = edge[v] + drift[v] - half_exit[v] - fee_exit[v]     # cross out + fee
    maker = edge[v] + drift[v] + half_exit[v]                   # rest the other side

    L, r, sig = lee_mykland(mid)
    nn = max(n, 2)
    Cn = (2 * np.log(nn)) ** 0.5 / 0.7979 - (
        np.log(np.pi) + np.log(np.log(nn))) / (2 * 0.7979 * (2 * np.log(nn)) ** 0.5)
    Sn = 1.0 / (0.7979 * (2 * np.log(nn)) ** 0.5)
    jump = np.abs(L) > (Cn + Sn * (-np.log(-np.log(0.99))))
    lam = hawkes_intensity(t, jump, 0.5)
    return lam[k][v], mark, taker, maker, edge[v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", default=None)
    ap.add_argument("--market-type", default="moneyline")
    ap.add_argument("--latency", type=float, default=0.5)
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--fee-rate", type=float, default=0.05)
    ap.add_argument("--max-gap", type=float, default=30.0)
    ap.add_argument("--gap-buffer", type=float, default=5.0)
    ap.add_argument("--tape", default="auto")
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pat = f"*-{args.slate}.parquet" if args.slate else "*.parquet"
    files = sorted(glob.glob(os.path.join(args.panel_dir, pat)))
    q = collections.Counter()
    per_game = {}

    for path in files:
        slug = os.path.basename(path)[:-8]
        trades, tsrc = load_trades(slug, args.tape)
        if not trades:
            continue
        cols = ["t", "mid_c", "spread_c", "bb_c", "ba_c", "asset_id", "market_type"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        ok = (np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"])
              & (d["ba_c"] > d["bb_c"]))
        if args.market_type:
            ok &= (d["market_type"] == args.market_type)
        acc = []
        for aid in np.unique(d["asset_id"][ok]):
            sel = ok & (d["asset_id"] == aid)
            r = token_eval(d, sel, trades, aid, args.latency, args.horizon,
                           args.max_gap, args.gap_buffer, args.fee_rate, q)
            if r is not None:
                acc.append(r)
        if not acc:
            continue
        per_game[slug] = {
            "lam": np.concatenate([a[0] for a in acc]),
            "mark": np.concatenate([a[1] for a in acc]),
            "taker": np.concatenate([a[2] for a in acc]),
            "maker": np.concatenate([a[3] for a in acc]),
            "edge": np.concatenate([a[4] for a in acc]),
            "tape": tsrc,
        }
        log(f"  {slug} [{tsrc}]: {per_game[slug]['mark'].size:,} fills survive latency")

    if not per_game:
        raise SystemExit("no usable games")

    fr = q["fills_after_latency"] / max(q["fills_before_latency"], 1)
    print()
    print("=" * 96)
    print(f"REALISTIC EXECUTION  ({args.market_type}, latency {args.latency*1000:.0f}ms, "
          f"mark-out {args.horizon:g}s, taker fee rate {args.fee_rate:g})")
    print("=" * 96)
    print(f"  fills available (instant quoting) {q['fills_before_latency']:>10,}")
    print(f"  fills a {args.latency*1000:.0f}ms-stale quote still wins  "
          f"{q['fills_after_latency']:>10,}   ({fr*100:.1f}%)")
    print(f"  -> latency costs {100-fr*100:.1f}% of the flow outright, and the "
          f"survivors are adversely selected")
    print()
    allm = np.concatenate([v["mark"] for v in per_game.values()])
    allt = np.concatenate([v["taker"] for v in per_game.values()])
    allk = np.concatenate([v["maker"] for v in per_game.values()])
    alle = np.concatenate([v["edge"] for v in per_game.values()])
    print(f"  {'exit assumption':>26} {'c/share':>10} {'positive games':>16} "
          f"{'t-stat':>9}")
    print("-" * 70)
    out = {}
    for name, arr in (("capture at fill only", alle),
                      ("mark-to-mid (optimistic)", allm),
                      ("exit as MAKER (best case)", allk),
                      ("exit as TAKER + fee", allt)):
        per = [np.mean(v[{"capture at fill only": "edge",
                          "mark-to-mid (optimistic)": "mark",
                          "exit as MAKER (best case)": "maker",
                          "exit as TAKER + fee": "taker"}[name]])
               for v in per_game.values()]
        per = np.array(per)
        se = per.std(ddof=1) / np.sqrt(per.size) if per.size > 1 else np.nan
        tstat = per.mean() / se if se and se > 0 else np.nan
        out[name] = {"mean_c": float(arr.mean()), "pos_games": int((per > 0).sum()),
                     "games": int(per.size), "t": float(tstat)}
        print(f"  {name:>26} {arr.mean():>+9.4f}c {int((per>0).sum()):>10}/{per.size:<5} "
              f"{tstat:>8.2f}")

    print()
    print("  by tape source (backfill-s overstates edge -- second-resolution fills):")
    for src in sorted({v["tape"] for v in per_game.values()}):
        g = [v for v in per_game.values() if v["tape"] == src]
        a = np.concatenate([x["taker"] for x in g])
        b = np.concatenate([x["maker"] for x in g])
        print(f"    {src:>12}  {len(g):>2} games  taker-exit {a.mean():>+8.4f}c   "
              f"maker-exit {b.mean():>+8.4f}c")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "fill_retention": fr,
                       "results": out}, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
