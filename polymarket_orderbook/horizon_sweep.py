"""Does waiting longer than 5 seconds make the trade pay?

The 5-second taker case fails because the move (0.44 ticks on average) is a
fraction of the round trip (about 3.5 ticks, and far worse right after a big
move when the book gaps). The obvious question is whether a longer horizon
fixes that: a price wanders further in 30s than in 5s, while the fee half of
the cost does not change at all.

There are two effects pulling in opposite directions, and only measurement
settles which wins:

  FOR  the move grows with the horizon, roughly as its square root when the
       price is diffusive, and faster when news arrives
  FOR  exiting later can be CHEAPER than exiting at +5s, because +5s often
       lands inside the liquidity gap the move itself opened; give the book
       20-30s and the spread frequently recovers
  AGAINST  every extra second is more time for the position to be wrong, and
       the number of independent opportunities falls as the horizon grows

Reported per horizon:
  move      mean |endpoint move| in ticks
  cost      mean round trip, entry crossing plus exit crossing plus both fees
  oracle    mean(|move| - cost): what perfect direction foresight earns. This
            bounds every possible model, so a negative number closes the
            horizon off entirely regardless of how good a predictor is.
  worth     share of opportunities where |move| exceeds the cost at all
  need      directional accuracy required to break even on an average move

Only rows whose forward window is CONTIGUOUS are used: the grid leaves real
gaps where a market went quiet past the fill cap, and shifting by h rows
across such a gap would silently compare prices minutes apart.

    python horizon_sweep.py
    python horizon_sweep.py --sessions 2026-09-10 2026-09-13
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from polymarket_fees import break_even_accuracy, round_trip_cost_ticks

BASE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.join(BASE, "data", "jump")
TICK = 0.01
GRID_MS = 200
HORIZONS_S = [5, 10, 20, 30, 60, 120, 300, 600]


def sweep(name, max_spread=2.0, max_age_s=5.0):
    p = os.path.join(JD, f"feat_{name}_trimmed.parquet")
    if not os.path.exists(p):
        p = os.path.join(JD, f"feat_{name}.parquet")
    F = pq.read_table(p, columns=["mid", "spread_ticks", "valid", "ts",
                                  "book_age_ms", "sid"]).to_pandas()
    sid = F.sid.to_numpy()
    ts = F.ts.to_numpy(np.int64)
    mid = F.mid.to_numpy(np.float64)
    sp = F.spread_ticks.to_numpy(np.float64)
    age = F.book_age_ms.to_numpy()
    n = len(F)
    del F

    # entry eligibility: the rows a taker would actually be allowed to hit
    entry = (sp <= max_spread) & (age <= max_age_s * 1000)

    rows = []
    for hs in HORIZONS_S:
        h = int(hs * 1000 / GRID_MS)
        if h >= n:
            continue
        j = np.arange(n - h) + h
        i = np.arange(n - h)
        # contiguous: same series AND exactly h grid steps of elapsed time
        good = (sid[j] == sid[i]) & (ts[j] - ts[i] == h * GRID_MS) & entry[i]
        if good.sum() < 5000:
            continue
        i, j = i[good], j[good]
        move = (mid[j] - mid[i]) / TICK
        cost = round_trip_cost_ticks(mid[i], mid[j], sp[i], sp[j],
                                     TICK, "sports", exit_as_maker=False)
        a = np.abs(move)
        rows.append(dict(
            horizon_s=hs, n=len(i), move=a.mean(), exit_spread=sp[j].mean(),
            cost=cost.mean(), oracle=(a - cost).mean(),
            worth=float((a > cost).mean()),
            need=break_even_accuracy(a.mean(), cost.mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*",
                    default=["books_2026-09-10", "books_2026-09-13"])
    a = ap.parse_args()

    pd.set_option("display.width", 200)
    allr = []
    for s in a.sessions:
        name = s if s.startswith("books_") else f"books_{s}"
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        R = sweep(name)
        if R.empty:
            print("  no usable rows")
            continue
        R.insert(0, "session", name)
        allr.append(R)
        sh = R.drop(columns=["session"]).copy()
        sh["need"] = sh["need"].map(lambda v: f"{v:.1%}" if v <= 1 else "impossible")
        for c in ("move", "exit_spread", "cost", "oracle"):
            sh[c] = sh[c].map(lambda v: f"{v:8.2f}")
        sh["worth"] = sh["worth"].map(lambda v: f"{v:.2%}")
        sh["n"] = sh["n"].map(lambda v: f"{v:,}")
        print(sh.to_string(index=False))

    if not allr:
        raise SystemExit("nothing measured")
    R = pd.concat(allr, ignore_index=True)
    out = os.path.join(JD, "horizon_sweep.csv")
    R.to_csv(out, index=False)

    print(f"\n{'=' * 78}\nORACLE P&L PER OPPORTUNITY BY HORIZON (ticks)\n{'=' * 78}")
    print(R.pivot(index="horizon_s", columns="session", values="oracle")
          .to_string(float_format=lambda v: f"{v:+.2f}"))
    print(f"\nwrote {out}")
    best = R.loc[R.oracle.idxmax()]
    print(f"\nbest cell: {best.session} at {best.horizon_s:.0f}s -> "
          f"{best.oracle:+.2f} ticks with perfect foresight")
    if best.oracle <= 0:
        print("Every horizon is negative even with perfect foresight, so no "
              "model and no amount of patience makes the taker case work.")


if __name__ == "__main__":
    main()
