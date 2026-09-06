"""
Is Polymarket miscalibrated? A buy-and-hold-to-settlement test.

Why this is the right question for a small account
--------------------------------------------------
Every taker strategy in this repo died on the ROUND TRIP: you pay the spread
going in and again coming out, and the moneyline round trip costs ~1.5c
against moves averaging 1.5c. But a position held to RESOLUTION pays the
spread exactly once, because settlement is free -- the contract simply pays
$1 or $0.

That removes the cost that killed everything else, and it needs no
infrastructure, no latency, no quoting, and no inventory management. It is
the only category of edge a $50 account can realistically access.

So the question becomes purely: at a quoted price of p, does the contract
actually resolve YES p of the time? If contracts priced at 5c resolve YES
only 2% of the time, selling them is profitable even after crossing the
spread once. That pattern -- longshots overpriced, favourites underpriced --
is the favourite-longshot bias, one of the most replicated effects in betting
markets, and it has never been checked on this data.

Method
------
For every token, settlement is read off where its price ends (>90c -> YES,
<10c -> NO; anything ambiguous is discarded rather than guessed). Prices are
then sampled at fixed fractions through the game, and each sample is bucketed
by the price a trader would actually PAY:

    buying  costs the ASK    -> profit at settlement = 100 - ask   (or -ask)
    selling receives the BID -> profit at settlement = bid         (or bid - 100)

so the spread is charged once, on entry, exactly as it would be live.

Sample-size honesty: 52 games give ~880 independent binary outcomes, and
YES/NO are the same contract mirrored, so they are NOT two observations.
Outcomes inside one game are correlated too (the moneyline, the run line and
the totals all resolve off one result). Wide intervals are expected and are
reported rather than hidden.

    python calibration.py
    python calibration.py --market-type moneyline --at 0.25 0.5 0.75
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")
TOKMAP = os.path.join(BASE, "data", "token_map.json")

# Buckets in cents. Deliberately finer in the tails, because that is where a
# favourite-longshot bias lives and where a $50 account can take a position at
# all (5 shares at 3c is 15 cents of risk).
EDGES = [0, 2, 5, 10, 20, 35, 50, 65, 80, 90, 95, 98, 100]


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct for proportions near 0 and 1, where a
    normal interval would run outside [0,1] and mislead exactly in the tails
    this study is about."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-type", default=None)
    ap.add_argument("--at", nargs="*", type=float, default=[0.25, 0.5, 0.75],
                    help="fractions through the game at which to sample")
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tokens = {}
    if os.path.exists(TOKMAP):
        with open(TOKMAP, "r", encoding="utf-8") as f:
            tokens = json.load(f)["tokens"]

    files = sorted(glob.glob(os.path.join(args.panel_dir, "*.parquet")))
    if not files:
        raise SystemExit("no panel files")

    # obs[(mtype, bucket)] -> [n, yes_count, sum_buy_pnl, sum_sell_pnl, sum_ask, sum_bid]
    obs = collections.defaultdict(lambda: np.zeros(6))
    n_tok = n_amb = 0
    seen_pairs = set()

    for path in files:
        cols = ["t", "mid_c", "bb_c", "ba_c", "asset_id", "market_type"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        ok = (np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"])
              & (d["ba_c"] > d["bb_c"]))
        for aid in np.unique(d["asset_id"][ok]):
            info = tokens.get(aid) or {}
            comp = info.get("complement")
            # YES and NO are one contract mirrored: count each market once.
            key = tuple(sorted([str(aid), str(comp)])) if comp else (str(aid),)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            sel = ok & (d["asset_id"] == aid)
            if sel.sum() < 2000:
                continue
            mid = d["mid_c"][sel]
            bb, ba = d["bb_c"][sel], d["ba_c"][sel]
            t = d["t"][sel]
            mt = str(d["market_type"][np.flatnonzero(sel)[0]])
            if args.market_type and mt != args.market_type:
                continue

            final = mid[-1]
            if final > 90.0:
                outcome = 1
            elif final < 10.0:
                outcome = 0
            else:
                n_amb += 1
                continue
            n_tok += 1

            t0, t1 = t[0], t[-1]
            for frac in args.at:
                k = int(np.searchsorted(t, t0 + frac * (t1 - t0)))
                k = min(max(k, 0), t.size - 1)
                a, b = ba[k], bb[k]
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                # bucket on the MID, but score on executable prices
                p = mid[k]
                bi = int(np.searchsorted(EDGES, p, side="right")) - 1
                bi = min(max(bi, 0), len(EDGES) - 2)
                rec = obs[(mt, bi)]
                rec[0] += 1
                rec[1] += outcome
                # buy at ask, hold: pays 100 if YES else 0
                rec[2] += (100.0 - a) if outcome else (-a)
                # sell at bid, hold: pays bid if NO else bid-100
                rec[3] += (b - 100.0) if outcome else b
                rec[4] += a
                rec[5] += b

    print()
    print("=" * 104)
    print("CALIBRATION: does a contract priced at p resolve YES p of the time?")
    print(f"{n_tok} settled contracts (YES/NO counted once), sampled at "
          f"{', '.join(f'{f:.0%}' for f in args.at)} through each game")
    if n_amb:
        print(f"{n_amb} contracts discarded as ambiguous at the close")
    print("buy = pay the ASK and hold to settlement | sell = receive the BID and hold")
    print("The spread is charged ONCE, on entry. Settlement is free.")
    print("=" * 104)

    for mt in sorted({k[0] for k in obs}):
        rows = [(bi, obs[(mt, bi)]) for bi in range(len(EDGES) - 1)
                if obs[(mt, bi)][0] >= 20]
        if not rows:
            continue
        tot = sum(r[1][0] for r in rows)
        print(f"\n### {mt}   ({int(tot):,} observations)")
        print(f"  {'price band':>12} {'n':>6} {'implied':>9} {'actual':>9} "
              f"{'95% CI':>16} {'buy&hold':>10} {'sell&hold':>10}")
        print("  " + "-" * 84)
        for bi, r in rows:
            n, yes = r[0], r[1]
            lo, hi = EDGES[bi], EDGES[bi + 1]
            implied = (r[4] + r[5]) / 2.0 / n / 100.0   # mean mid as probability
            actual = yes / n
            cl, ch = wilson(yes, n)
            buy = r[2] / n
            sell = r[3] / n
            flag = ""
            if cl > implied:
                flag = "  UNDERpriced"
            elif ch < implied:
                flag = "  OVERpriced"
            print(f"  {lo:>4}-{hi:<3}c {int(n):>8,} {implied*100:>8.1f}% "
                  f"{actual*100:>8.1f}% {f'[{cl*100:.1f},{ch*100:.1f}]':>16} "
                  f"{buy:>+9.2f}c {sell:>+9.2f}c{flag}")

    print()
    print("  buy&hold / sell&hold are cents per share after paying the spread once.")
    print("  A band is only tradeable if the profitable side is positive AND the")
    print("  confidence interval excludes the implied probability.")

    if args.json:
        out = {f"{k[0]}|{k[1]}": v.tolist() for k, v in obs.items()}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"contracts": n_tok, "ambiguous": n_amb,
                       "edges": EDGES, "obs": out}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
