"""Why the corrupted dataset paid, and what the model was really learning.

The merged-asset bug did not just make moves BIGGER. It made them almost
perfectly PREDICTABLE, and that is the part that turned into fake P&L.

Two legs of a spread bet trade at two unrelated levels -- say 0.195 and 0.545.
Keyed together, the forward-filled "mid" alternates between them. The series
is not a price, it is a square wave. And a square wave is the easiest thing in
the world to trade: if you are sitting on the low level the next move is up,
if you are on the high level the next move is down. One bit of information.

The CNN was handed that bit for free. Its scalar inputs include `rel`, the mid
relative to the prediction instant, and `dmid`, the recent mid change -- so it
could always tell which level it was standing on. It never needed the order
book at all.

This script tests that claim directly: it runs a rule that uses NOTHING but
the current level -- buy if the mid sits below the series midpoint, sell if
above -- and compares its P&L to what the CNN reported. If a one-line rule
with no book, no depth, no training and no parameters reproduces the model's
edge, then the model was learning the artifact and nothing else.

    python why_it_looked_profitable.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from polymarket_fees import round_trip_cost_ticks

BASE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.join(BASE, "data", "jump")
OLD = os.path.join(JD, "_seriesbug_20260915")
TICK = 0.01
GRID_MS = 200
COLS = ["mid", "spread_ticks", "signed_ticks", "valid", "book_age_ms",
        "series", "sid"]


def load(path, market_type="spread"):
    F = pq.read_table(path, columns=COLS).to_pandas()
    F["mt"] = F["series"].astype(str).str.split("|").str[1]
    F = F[(F.mt == market_type) & F.valid].reset_index(drop=True)
    return F


def bimodality(F):
    """How two-level is each series? Reported as the share of rows sitting
    near one of the two extremes rather than anywhere in between."""
    out = []
    for sid, g in F.groupby("sid", sort=False):
        m = g.mid.to_numpy(np.float64)
        if len(m) < 5000:
            continue
        lo, hi = np.quantile(m, 0.05), np.quantile(m, 0.95)
        if hi - lo < 0.05:              # a genuinely quiet market, not a flip
            out.append((sid, hi - lo, np.nan))
            continue
        span = hi - lo
        near = ((m < lo + 0.15 * span) | (m > hi - 0.15 * span)).mean()
        out.append((sid, span, near))
    return pd.DataFrame(out, columns=["sid", "range", "near_extremes"])


def level_rule(F, horizon_s=5.0, max_spread=2.0, max_age_s=5.0):
    """Trade purely on which level you are standing on. No book, no model."""
    h = int(horizon_s * 1000 / GRID_MS)
    sp_f = (F.groupby("sid", sort=False)["spread_ticks"]
            .shift(-h).fillna(F["spread_ticks"]).to_numpy(np.float64))
    mid = F.mid.to_numpy(np.float64)
    signed = F.signed_ticks.to_numpy(np.float64)
    sp = F.spread_ticks.to_numpy(np.float64)
    cost = round_trip_cost_ticks(mid, mid + signed * TICK, sp, sp_f,
                                 TICK, "sports", exit_as_maker=False)

    # the series midpoint, computed per series on that series' own history --
    # deliberately generous to the rule, but the levels are so far apart that
    # any reasonable split gives the same answer
    centre = F.groupby("sid", sort=False)["mid"].transform("median").to_numpy()
    side = np.where(mid < centre, 1.0, -1.0)     # low -> expect up, high -> down

    ok = (sp <= max_spread) & (F.book_age_ms.to_numpy() <= max_age_s * 1000)
    moved = signed != 0
    m = ok & moved
    if m.sum() < 1000:
        return None
    hit = ((side[m] > 0) == (signed[m] > 0)).mean()
    pnl = side[ok] * signed[ok] - cost[ok]
    return dict(n=int(ok.sum()), hit=float(hit), pnl=float(pnl.mean()),
                move=float(np.abs(signed[ok]).mean()),
                cost=float(cost[ok].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="books_2026-09-10")
    ap.add_argument("--cnn-reported", type=float, default=0.3208,
                    help="the headline the CNN produced on the broken data")
    a = ap.parse_args()

    pd.set_option("display.width", 200)
    for tag, d in (("BEFORE (merged assets)", OLD), ("AFTER  (fixed)", JD)):
        p = os.path.join(d, f"feat_{a.session}.parquet")
        if not os.path.exists(p):
            print(f"{tag}: {p} missing")
            continue
        print(f"\n{'=' * 74}\n{tag} -- {a.session}, spread markets\n{'=' * 74}")
        F = load(p)
        b = bimodality(F)
        print(f"  series examined: {len(b)}")
        print(f"  median price range within a series : "
              f"{b['range'].median():.3f}  ({b['range'].median() / TICK:.0f} ticks)")
        print(f"  median share of rows at the extremes: "
              f"{b['near_extremes'].median():.1%}   "
              f"(a real price wanders; a two-level flip sits at the ends)")
        r = level_rule(F)
        if r:
            print(f"\n  the level-only rule (no order book, no model, no fit):")
            print(f"    rows traded        {r['n']:,}")
            print(f"    direction correct  {r['hit']:.1%}")
            print(f"    avg move           {r['move']:.3f} ticks")
            print(f"    avg cost           {r['cost']:.3f} ticks")
            print(f"    P&L per trade      {r['pnl']:+.4f} ticks")
            print(f"    CNN reported       {a.cnn_reported:+.4f} ticks "
                  f"on this same broken data")
        del F, b

    print(f"\n{'=' * 74}")
    print("If the level-only rule matches or beats the CNN on the broken data "
          "and collapses on the fixed data, the CNN was not reading the order "
          "book. It was reading which of two tokens the last message came "
          "from, and betting the next message came from the other one.")


if __name__ == "__main__":
    main()
