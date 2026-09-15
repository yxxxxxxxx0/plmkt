"""How much did the series-key fix change the data, and the money?

The bug: `extract()` keyed a series on (slug, market_type, line), which is not
unique -- both legs of a spread bet share all three and differ only by
asset_id. Their updates interleaved into one forward-filled book, so the "mid"
alternated between two unrelated price levels and manufactured a large move on
every switch. This compares the datasets built before and after the fix.

The headline is not a model score, it is an ORACLE bound: the P&L per
opportunity of a trader who knows the direction of every move in advance and
still has to pay the spread and both taker fees.

    oracle = E[ |move| - round-trip cost ]

No model can beat it. If it is negative, the strategy cannot work no matter
how good the predictor is, and the only honest thing to do is say so. Running
it on both datasets separates "the model was wrong" from "the data was wrong".

    python compare_seriesfix.py
    python compare_seriesfix.py --sessions 2026-09-10 --tradeable-only
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

COLS = ["mid", "spread_ticks", "signed_ticks", "path_exc_ticks",
        "valid", "book_age_ms", "series", "sid"]


def load(path):
    F = pq.read_table(path, columns=COLS).to_pandas()
    F["mt"] = F["series"].astype(str).str.split("|").str[1]
    F.drop(columns=["series"], inplace=True)
    return F


def stats(F, horizon_s=5.0, max_spread=2.0, max_age_s=5.0, tradeable_only=False):
    """Oracle economics on one dataset, pooled and by market type."""
    h = int(horizon_s * 1000 / GRID_MS)
    # forward spread, per series, exactly as taker_signal.prepare does
    sp_f = (F.groupby("sid", sort=False)["spread_ticks"]
            .shift(-h).fillna(F["spread_ticks"]).to_numpy(np.float64))
    mid = F.mid.to_numpy(np.float64)
    signed = F.signed_ticks.to_numpy(np.float64)
    sp = F.spread_ticks.to_numpy(np.float64)
    cost = round_trip_cost_ticks(mid, mid + signed * TICK, sp, sp_f,
                                 TICK, "sports", exit_as_maker=False)

    keep = np.array(F.valid.to_numpy(bool), copy=True)
    if tradeable_only:
        keep &= (sp <= max_spread) & (F.book_age_ms.to_numpy() <= max_age_s * 1000)

    out = []
    mt = F.mt.to_numpy().astype(str)
    for label in ["ALL", "moneyline", "total", "spread", "first_inning_run"]:
        m = keep if label == "ALL" else (keep & (mt == label))
        if m.sum() < 1000:
            continue
        a = np.abs(signed[m])
        c = cost[m]
        out.append(dict(
            slice=label, n=int(m.sum()),
            move=a.mean(), path=F.path_exc_ticks.to_numpy()[m].mean(),
            cost=c.mean(), oracle=(a - c).mean(),
            pct_worth=float((a > c).mean()),
            big4=float((a >= 4).mean())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--tradeable-only", action="store_true",
                    help="restrict to fresh book and spread <= 2 ticks, i.e. "
                         "the rows the model is actually allowed to trade")
    a = ap.parse_args()

    names = sorted(f[len("feat_"):-len(".parquet")]
                   for f in os.listdir(JD) if f.startswith("feat_books_")
                   and f.endswith(".parquet") and "_trimmed" not in f)
    if a.sessions:
        names = [n for n in names if any(s in n for s in a.sessions)]

    pd.set_option("display.width", 200)
    rows = []
    for n in names:
        new_p = os.path.join(JD, f"feat_{n}.parquet")
        old_p = os.path.join(OLD, f"feat_{n}.parquet")
        print(f"\n{'=' * 78}\n{n}\n{'=' * 78}")
        for tag, p in (("BEFORE (merged assets)", old_p), ("AFTER  (fixed)", new_p)):
            if not os.path.exists(p):
                print(f"  {tag}: not present")
                continue
            F = load(p)
            s = stats(F, tradeable_only=a.tradeable_only)
            del F
            s.insert(0, "dataset", tag)
            s.insert(0, "session", n)
            rows.append(s)
            print(f"\n  {tag}")
            print(s.drop(columns=["session", "dataset"]).to_string(
                index=False, float_format=lambda v: f"{v:9.4f}"))

    if not rows:
        raise SystemExit("nothing to compare")
    R = pd.concat(rows, ignore_index=True)
    out = os.path.join(JD, "seriesfix_comparison.csv")
    R.to_csv(out, index=False)

    print(f"\n{'=' * 78}\nORACLE P&L PER OPPORTUNITY, ticks  "
          f"(perfect direction foresight, still pays spread + both fees)"
          f"\n{'=' * 78}")
    piv = R[R.slice == "ALL"].pivot(index="session", columns="dataset",
                                    values="oracle")
    print(piv.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\nBy market type, AFTER the fix:")
    print(R[R.dataset.str.startswith("AFTER")]
          .pivot(index="session", columns="slice", values="oracle")
          .to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nwrote {out}")
    print("\nA negative oracle means a trader who knew every move in advance "
          "would still lose money after costs. That is a property of the "
          "market and the fee schedule, not of the model, and no amount of "
          "model work changes it.")


if __name__ == "__main__":
    main()
