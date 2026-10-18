"""Detect liquidity-collapse events, then ask which of them are REAL.

Reframing (Justin's, and it is the right one). Instead of scanning every instant
for a rare jump, do it in two stages:

  1. DETECT  every collapse of near-touch liquidity. This is cheap, mechanical
     and needs no model -- it is the thing that physically precedes the mid
     moving.
  2. CLASSIFY  which collapses are followed by a DURABLE price move and which
     are noise: the book empties, the mid ticks mechanically, then liquidity
     returns and the price snaps back.

Stage 2 is the interesting problem and the one worth a CNN. Stage 1 turns a
~1-2% prevalence needle-hunt into a candidate set with a sane base rate.

Definitions, all strictly causal
--------------------------------
near_bid(t) = dollars resting within 2 ticks of the mid on the bid, likewise
near_ask. The baseline is a TRAILING rolling median over `--baseline-s` seconds,
so nothing after t is used.

A collapse on a side starts at the first slot where

    near_side(t) < drop_frac * baseline_side(t)     and baseline_side >= min_base

and it is debounced: once an event fires, no new event on that side for
`--refractory-s` seconds.

Label: from the collapse instant t, the move over the next `--horizon-s`
seconds is DURABLE if the displacement is >= `--min-move` and still holds
`--hold-s` seconds later. A mid excursion that reverts is labelled 0 -- that is
precisely the "not real" case.

    python collapse_events.py --drop-frac 0.25 --horizon-s 10 --min-move 0.02
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "jump_prediction"))
import features as FE  # noqa: E402

JD = os.path.join(ROOT, "data", "jump")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen", "collapse")
GRID_MS = 200


def log(m):
    import time
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def detect_and_label(sess, a):
    fp = os.path.join(JD, "feat_%s_trimmed.parquet" % sess)
    tp = os.path.join(JD, "lob_%s_trimmed.npy" % sess)
    if not (os.path.exists(fp) and os.path.exists(tp)):
        return None
    F = pq.read_table(fp, columns=["ts", "sid", "series", "mid", "spread_ticks",
                                   "valid"]).to_pandas()
    F["row"] = np.arange(len(F), dtype=np.int64)
    F = F[F.valid.to_numpy(bool)]
    T = np.load(tp, mmap_mode="r")

    base_slots = int(a.baseline_s * 1000 / GRID_MS)
    refr = int(a.refractory_s * 1000 / GRID_MS)
    hor = int(a.horizon_s * 1000 / GRID_MS)
    hold = int(a.hold_s * 1000 / GRID_MS)

    out = []
    for sid, g in F.groupby("sid", sort=False):
        rows = g.row.to_numpy()
        if len(rows) < base_slots + hor + hold + 10:
            continue
        mid = g.mid.to_numpy(np.float64)
        spt = g.spread_ticks.to_numpy()
        d = FE.decode(np.asarray(T[rows]), mid)
        S = pd.DataFrame(FE.state_features(d, spread_ticks=spt))
        nb = np.expm1(S.log_bid_usd_within_2t.to_numpy())
        na = np.expm1(S.log_ask_usd_within_2t.to_numpy())

        for side, near in (("bid", nb), ("ask", na)):
            s = pd.Series(near)
            base = s.rolling(base_slots, min_periods=base_slots // 2).median().shift(1)
            b = base.to_numpy()
            fired = (near < a.drop_frac * b) & (b >= a.min_base)
            last = -10**9
            idx = np.where(fired)[0]
            for i in idx:
                if i - last < refr:
                    continue
                if i + hor + hold >= len(mid):
                    continue
                last = i
                m0 = mid[i]
                fwd = mid[i + 1:i + hor + 1]
                disp = np.abs(fwd - m0)
                k = int(np.argmax(disp))
                peak = float(disp[k])
                # durable: still displaced hold_s after the peak
                j = i + 1 + k
                later = mid[j:j + hold + 1]
                dur = float(np.min(np.abs(later - m0))) if len(later) else 0.0
                out.append(dict(
                    session=sess, sid=int(sid), series=str(g.series.iloc[0]),
                    row=int(rows[i]), ts=int(g.ts.to_numpy()[i]), side=side,
                    mid=m0, spread_ticks=float(spt[i]),
                    near_bid=float(nb[i]), near_ask=float(na[i]),
                    baseline=float(b[i]),
                    peak_move=peak, durable_move=dur,
                    signed_peak=float(fwd[k] - m0),
                ))
    del F, T
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-s", type=float, default=60.0)
    ap.add_argument("--drop-frac", type=float, default=0.25)
    ap.add_argument("--min-base", type=float, default=200.0)
    ap.add_argument("--refractory-s", type=float, default=30.0)
    ap.add_argument("--horizon-s", type=float, default=10.0)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--min-move", type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)

    P = pd.read_parquet(os.path.join(CACHE, "panel_labelled.parquet"),
                        columns=["session"])
    sessions = sorted(P.session.unique())
    frames = []
    for s in sessions:
        log("scanning %s" % s)
        D = detect_and_label(s, a)
        if D is not None and len(D):
            frames.append(D)
            log("  %s collapse events" % "{:,}".format(len(D)))
    E = pd.concat(frames, ignore_index=True)
    E["real"] = (E.durable_move >= a.min_move).astype(np.int8)
    E["any_move"] = (E.peak_move >= a.min_move).astype(np.int8)
    E["reverted"] = ((E.any_move == 1) & (E.real == 0)).astype(np.int8)
    E.to_parquet(os.path.join(CACHE, "collapse_events.parquet"), index=False)

    print("\n=== collapse events ===")
    print("  total                              : {:,}".format(len(E)))
    print("  by side                            : %s" % E.side.value_counts().to_dict())
    print("  events per session (median)        : %.0f"
          % E.groupby("session").size().median())
    print("\n=== which are REAL? (durable move >= %.2f within %.0fs) ==="
          % (a.min_move, a.horizon_s))
    print("  REAL (durable move)                : {:,} ({:.1%})"
          .format(int(E.real.sum()), E.real.mean()))
    print("  moved but REVERTED (the fake case) : {:,} ({:.1%})"
          .format(int(E.reverted.sum()), E.reverted.mean()))
    print("  never moved at all                 : {:,} ({:.1%})"
          .format(int((E.any_move == 0).sum()), (E.any_move == 0).mean()))
    for thr in (0.02, 0.05, 0.10, 0.20):
        r = (E.durable_move >= thr).mean()
        print("    base rate at min-move %.2f: %.3f  (n_pos {:,})"
              .format(int((E.durable_move >= thr).sum())) % (thr, r))
    print("\n=== by side ===")
    print(E.groupby("side").agg(n=("real", "size"), real_rate=("real", "mean"),
                                revert_rate=("reverted", "mean"),
                                med_peak=("peak_move", "median")).round(4).to_string())
    E.groupby("session").agg(n=("real", "size"), real=("real", "sum")).to_csv(
        os.path.join(RES, "collapse_by_session.csv"))
    print("\nwrote", os.path.join(CACHE, "collapse_events.parquet"))


if __name__ == "__main__":
    sys.exit(main())
