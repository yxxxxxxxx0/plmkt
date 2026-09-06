"""
Calibration on the wide resolved-market sample.

Tests the favourite-longshot hypothesis on thousands of settled Polymarket
contracts rather than the 259 the recorded games could supply.

    price -> does it resolve YES that often?

One row per MARKET (never per token: YES and NO are the same contract). The
price is the median trade in the middle third of the market's life, which
avoids both the illiquid opening and the near-resolution end where the answer
is already known.

What this can and cannot settle
-------------------------------
CAN: whether a systematic pricing bias exists, and how large it is in cents.
     That is the whole question for a buy-and-hold-to-settlement strategy,
     because settlement costs nothing -- the spread is paid once, on entry.

CANNOT: the spread you would actually pay, because the trade tape carries no
     quotes. So the gross bias is reported first, then charged a spread you
     supply. From the recorded order books: moneyline ~1.5c, run-line ~9.1c,
     totals ~10.5c. Half of that is the entry cost against the mid.

Independence caveat, stated rather than buried: markets inside one event
resolve off one underlying result, so the effective sample is smaller than
the row count. Intervals are Wilson, which stay inside [0,1] near the tails
where a normal interval would not.

    python calibration_wide.py --spread 3.0
    python calibration_wide.py --sports-only --spread 1.5
"""
import argparse
import collections
import json
import math
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "data", "resolved", "resolved.jsonl")
EDGES = [0, 2, 5, 10, 20, 35, 50, 65, 80, 90, 95, 98, 100]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--spread", type=float, default=3.0,
                    help="assumed quoted spread in cents; half is charged on entry")
    ap.add_argument("--sports-only", action="store_true")
    ap.add_argument("--min-volume", type=float, default=0.0)
    ap.add_argument("--min-n", type=int, default=25)
    args = ap.parse_args()

    rows = []
    with open(args.src, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.sports_only and not r.get("sports_market_type"):
                continue
            if r.get("volume", 0) < args.min_volume:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit("no rows")

    half = args.spread / 2.0
    # both orientations: a market priced at p is also a market priced at 1-p on
    # the other side, so folding them in doubles nothing -- it is the SAME
    # observation viewed from the complement, and including both would
    # double-count. Only outcome0 is used.
    buckets = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        p = r["price0_mid_life"] * 100.0
        bi = min(max(int(sorted(EDGES).index(next(e for e in EDGES if e >= p)) - 1
                        if p > 0 else 0), 0), len(EDGES) - 2)
        # simpler, robust bucketing
        bi = 0
        for k in range(len(EDGES) - 1):
            if EDGES[k] <= p < EDGES[k + 1]:
                bi = k
                break
        else:
            bi = len(EDGES) - 2
        b = buckets[bi]
        b[0] += 1
        b[1] += r["resolved0"]
        b[2] += p

    print()
    print("=" * 100)
    print("CALIBRATION ON RESOLVED MARKETS")
    print(f"{len(rows):,} settled markets"
          + (" (sports only)" if args.sports_only else "")
          + f" | spread charged: {args.spread:.1f}c ({half:.2f}c on entry)")
    print("buy&hold = pay ask, collect $1 if YES.  sell&hold = receive bid, keep it if NO.")
    print("=" * 100)
    print(f"  {'band':>10} {'n':>6} {'priced':>8} {'actual':>8} {'95% CI':>15} "
          f"{'gross':>9} {'buy net':>9} {'sell net':>9}")
    print("  " + "-" * 84)
    tradeable = []
    for bi in range(len(EDGES) - 1):
        b = buckets.get(bi)
        if not b or b[0] < args.min_n:
            continue
        n, yes, psum = b
        priced = psum / n
        actual = yes / n * 100.0
        cl, ch = wilson(yes, n)
        gross = actual - priced          # cents of mispricing, + = underpriced
        buy = gross - half
        sell = -gross - half
        sig = ""
        if cl * 100 > priced:
            sig = " UNDER"
        elif ch * 100 < priced:
            sig = " OVER"
        if sig and max(buy, sell) > 0:
            tradeable.append((EDGES[bi], EDGES[bi + 1], int(n), priced, actual,
                              buy, sell, sig.strip()))
        print(f"  {EDGES[bi]:>3}-{EDGES[bi+1]:<3}c {int(n):>8,} {priced:>7.1f}% "
              f"{actual:>7.1f}% {f'[{cl*100:.1f},{ch*100:.1f}]':>15} "
              f"{gross:>+8.2f}c {buy:>+8.2f}c {sell:>+8.2f}c{sig}")

    print()
    if tradeable:
        print("  BANDS THAT CLEAR BOTH TESTS (significant AND net-positive after spread):")
        for lo, hi, n, pr, ac, buy, sell, s in tradeable:
            side = "BUY " if buy > sell else "SELL"
            print(f"    {lo:>3}-{hi:<3}c  n={n:<5}  priced {pr:.1f}%  actual {ac:.1f}%  "
                  f"{side} and hold -> {max(buy,sell):+.2f}c/share  [{s}priced]")
    else:
        print("  No band is both statistically significant and net-positive after "
              "the spread.")
    print()
    print("  A band needs BOTH: the interval must exclude the priced probability")
    print("  (the bias is real) and the net must be positive (it survives the spread).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
