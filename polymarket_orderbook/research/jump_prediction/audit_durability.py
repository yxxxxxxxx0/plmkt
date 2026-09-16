"""How many labelled 'jumps' are real repricings and how many are quote vacuums?

plots/examples_jumps_tight_J0.02_H30.png shows the problem: in most labelled
jumps the mid spike coincides exactly with the spread exploding from ~2 ticks
to 20-80 ticks, and the mid snaps straight back afterwards. That is not a price
move. It is one side of the book being briefly emptied: the best quote is
cancelled, the next resting order is far away, the midpoint mechanically jumps,
and it reverts as soon as a quote returns. No trade need occur at any point.

The existing label, future_move(t,H) = max over t<u<=t+H of |mid_u - mid_t|,
counts every one of those as a jump. Restricting to tight books does not help,
because the tightness test is applied at t while the artefact happens later,
inside the label window.

This script decomposes the label without fitting any model:

  raw_move      max |mid_u - mid_t|                      (the current label)
  clean_move    same, but only over u whose book is tight (spread_u <= S).
                A move that only exists while the book is blown out is
                excluded.
  durable_move  same as raw, but the displacement must persist for at least
                `hold` seconds. A transient spike cannot qualify.
  endpoint_move |mid_{t+H} - mid_t|, the move that actually stuck.

The gap between raw and the other three is the size of the artefact, and it
bounds how much of the previously reported skill was predicting quote flicker.

    python audit_durability.py --J 0.02 --H 30
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import splits as SP  # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "jump_prediction")


def per_series_moves(mid, spread, ts, H, hold, tight_ticks):
    """Vectorised move decomposition for one contiguous 1Hz series.

    Returns raw/clean/durable/endpoint move arrays and a validity mask. Only
    forward offsets on an exactly-1Hz contiguous grid are used; any row whose
    forward window is broken by a gap is marked invalid, never bridged.
    """
    n = len(mid)
    ok = np.zeros(n, bool)
    raw = np.full(n, np.nan, np.float32)
    clean = np.full(n, np.nan, np.float32)
    dur = np.full(n, np.nan, np.float32)
    endp = np.full(n, np.nan, np.float32)
    if n < H + 1:
        return raw, clean, dur, endp, ok

    # contiguity: row i has a usable window iff ts[i+H] - ts[i] == H*1000
    idx = np.arange(n - H)
    ok[idx] = (ts[idx + H] - ts[idx]) == H * 1000

    disp = np.empty((H, n - H), np.float32)       # |mid_{t+k} - mid_t|
    tightu = np.empty((H, n - H), bool)
    for k in range(1, H + 1):
        d = np.abs(mid[k:k + (n - H)] - mid[:n - H])
        disp[k - 1] = d
        tightu[k - 1] = spread[k:k + (n - H)] <= tight_ticks

    raw[idx] = disp.max(axis=0)
    masked = np.where(tightu, disp, -np.inf)
    cm = masked.max(axis=0)
    clean[idx] = np.where(np.isfinite(cm), cm, 0.0)

    # durable: displacement must hold for `hold` consecutive seconds
    if hold <= 1:
        dur[idx] = raw[idx]
    else:
        held = disp.copy()
        for j in range(1, hold):
            held[:H - j] = np.minimum(held[:H - j], disp[j:])
        held = held[:max(H - hold + 1, 1)]
        dur[idx] = held.max(axis=0)

    endp[idx] = disp[H - 1]
    return raw, clean, dur, endp, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--hold", type=int, default=3, help="seconds a move must persist")
    ap.add_argument("--tight-ticks", type=float, default=2.0)
    a = ap.parse_args()

    val = "valid_label_H%d" % a.H
    cols = ["sid", "ts", "market", "mid", "spread_ticks", val,
            "future_move_H%d" % a.H]
    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    D = pd.concat([pd.read_parquet(p, columns=cols) for p in ps],
                  ignore_index=True)
    D = D[D[val]].reset_index(drop=True)
    print("{:,} valid-label rows".format(len(D)))

    D = D.sort_values(["sid", "ts"]).reset_index(drop=True)
    raw = np.full(len(D), np.nan, np.float32)
    clean = np.full(len(D), np.nan, np.float32)
    dur = np.full(len(D), np.nan, np.float32)
    endp = np.full(len(D), np.nan, np.float32)
    okm = np.zeros(len(D), bool)
    for _, g in D.groupby("sid", observed=True, sort=False):
        i = g.index.to_numpy()
        r, c, u, e, ok = per_series_moves(
            g.mid.to_numpy(np.float32), g.spread_ticks.to_numpy(np.float32),
            g.ts.to_numpy(np.int64), a.H, a.hold, a.tight_ticks)
        raw[i], clean[i], dur[i], endp[i], okm[i] = r, c, u, e, ok

    D["raw_move"], D["clean_move"] = raw, clean
    D["durable_move"], D["endpoint_move"] = dur, endp
    D = D[okm].reset_index(drop=True)
    print("{:,} rows with a contiguous forward window".format(len(D)))

    # sanity: our recomputed raw move must match the cached label
    cached = D["future_move_H%d" % a.H].to_numpy()
    agree = np.nanmean(np.abs(cached - D.raw_move.to_numpy()) < 1e-6)
    print("recomputed raw_move matches cached future_move: %.4f" % agree)

    tight_t = D.spread_ticks.to_numpy() <= a.tight_ticks
    rows = []
    for regime, m in (("all books", np.ones(len(D), bool)),
                      ("tight at t", tight_t)):
        sub = D[m]
        r = dict(regime=regime, n=int(m.sum()))
        for nm, col in (("raw", "raw_move"), ("clean", "clean_move"),
                        ("durable", "durable_move"), ("endpoint", "endpoint_move")):
            r["prev_" + nm] = float((sub[col] >= a.J).mean())
            r["median_" + nm] = float(sub[col].median())
        jr = (sub.raw_move >= a.J).to_numpy()
        r["frac_raw_jumps_that_are_clean"] = float(
            (sub.clean_move.to_numpy()[jr] >= a.J).mean()) if jr.any() else np.nan
        r["frac_raw_jumps_that_are_durable"] = float(
            (sub.durable_move.to_numpy()[jr] >= a.J).mean()) if jr.any() else np.nan
        r["frac_raw_jumps_that_stick"] = float(
            (sub.endpoint_move.to_numpy()[jr] >= a.J).mean()) if jr.any() else np.nan
        rows.append(r)

    out = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    f = os.path.join(RES, "audit_durability_J%s_H%d.csv" % (a.J, a.H))
    out.to_csv(f, index=False)
    print("\n" + out.to_string(index=False))
    print("\nwrote", f)

    # keep the decomposed labels for the re-run
    keep = D[["sid", "ts", "market", "mid", "spread_ticks", "raw_move",
              "clean_move", "durable_move", "endpoint_move"]]
    g = os.path.join(CACHE, "moves_H%d.parquet" % a.H)
    keep.to_parquet(g, index=False)
    print("wrote", g)


if __name__ == "__main__":
    sys.exit(main())
