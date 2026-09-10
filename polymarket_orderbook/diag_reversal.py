"""Was the jump a real relocation of price, or the mid oscillating?

The directional CNN reaches an 89% hit rate on >=4-tick moves and a positive
CI even on touches backed by >$5,000. Before treating that as an edge, one
alternative has to be ruled out.

The mid is the average of two quotes, not a price anyone transacts at. If a
book flickers between two configurations, the mid oscillates deterministically
and mean reversion predicts it beautifully -- `dmid_25` already scores AUC
0.374 on the sign of the next move, i.e. 0.626 inverted, and the plain
momentum control earns positive P&L on its own. A model that is mostly a
better mean-reversion detector would look exactly like what we measured.

Two things separate a relocation from a bounce:

  PERSISTENCE   If the price genuinely moved, it is still moved at t+2H.
                If it bounced, mid(t+2H) has returned toward mid(t). The
                retained fraction (mid(t+2H) - mid(t)) / (mid(t+H) - mid(t))
                is ~1 for a relocation and ~0 for an oscillation.

  EXIT TIMING   This is the decisive one, because it is what you would
                actually suffer. The P&L assumes an exit at exactly t+H. If
                the move is a bounce, exiting a second late gives the gain
                back, and the strategy needs timing precision it cannot have.
                A genuine relocation is insensitive to holding slightly
                longer.

Gaps matter here: jump_data.py caps forward-fill, so a shift of 2H rows can
cross a discontinuity. Every horizon is checked for exact grid spacing and
rows that fail are dropped rather than compared across a time jump.

    python diag_reversal.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from jump_model import frame_offsets
from jump_split import GRID_MS, JD, TICK, load
from polymarket_fees import taker_fee_ticks
from taker_signal import prepare, touch_backing


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-s", type=float, default=5.0)
    ap.add_argument("--category", default="sports")
    ap.add_argument("--max-spread", type=float, default=2.0)
    ap.add_argument("--min-backing", type=float, default=200.0)
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    _, _, _, lookback = frame_offsets(24, 24, 8)
    F, T, tr, va, te = load(["moneyline", "total", "spread"],
                            lookback=lookback, log=log, max_book_age_s=5.0)
    F, h = prepare(F, a.horizon_s, a.category, a.max_spread)

    # forward path at 1H, 2H and 4H, with exact-spacing checks
    lags = {"1H": h, "2H": 2 * h, "4H": 4 * h}
    g = F.groupby("sid", sort=False)
    for k, lag in lags.items():
        F[f"mid_{k}"] = g["mid"].shift(-lag)
        F[f"sp_{k}"] = g["spread_ticks"].shift(-lag)
        F[f"ok_{k}"] = (g["ts"].shift(-lag) - F["ts"]) == lag * GRID_MS

    log(f"\nloading the trained signal...")
    ep = os.path.join(JD, "takeredge_cnn_direction.npy")
    ip = os.path.join(JD, "takeridx_cnn_direction.npy")
    if not (os.path.exists(ep) and os.path.exists(ip)):
        raise SystemExit("run taker_signal.py first (it saves edges + index)")
    e = np.load(ep)
    idx = np.load(ip)
    tcsv = os.path.join(JD, "taker_comparison.csv")
    tau = 0.90
    if os.path.exists(tcsv):
        t = pd.read_csv(tcsv)
        r = t[t.model == "cnn_direction"]
        if len(r):
            tau = float(r.iloc[0].tau)
    log(f"  {len(idx):,} test rows, tau = {tau:.2f}")

    mid = F.mid.to_numpy(np.float64)
    m1 = F.mid_1H.to_numpy(np.float64)
    m2 = F.mid_2H.to_numpy(np.float64)
    m4 = F.mid_4H.to_numpy(np.float64)
    good = (F.ok_1H.to_numpy() & F.ok_2H.to_numpy() & F.ok_4H.to_numpy()
            & np.isfinite(m1) & np.isfinite(m2) & np.isfinite(m4))

    side = np.where(e > tau, 1.0, np.where(e < -tau, -1.0, 0.0))
    traded = (side != 0) & good[idx]
    log(f"  {traded.sum():,} traded rows with a clean 4H forward path "
        f"({traded.mean():.1%} of test rows)")
    if traded.sum() < 500:
        raise SystemExit("too few clean traded rows to conclude anything")

    i = idx[traded]
    sd = side[traded]
    d1 = (m1[i] - mid[i]) / TICK          # the move the strategy trades
    d2 = (m2[i] - mid[i]) / TICK
    d4 = (m4[i] - mid[i]) / TICK
    mt = F.mt.to_numpy().astype(str)[i]
    bk = touch_backing(T)[i]

    # ---- PERSISTENCE ----------------------------------------------------- #
    log(f"\n{'='*104}\nPERSISTENCE: is the move still there later?\n{'='*104}")
    big = np.abs(d1) >= 1.0               # ratios are meaningless on ~0 moves
    log(f"  on {big.sum():,} traded rows whose 1H move was >= 1 tick")
    log(f"  {'slice':>18} {'n':>8} {'move 1H':>9} {'kept 2H':>9} "
        f"{'kept 4H':>9} {'frac 2H':>9} {'frac 4H':>9}")

    def prow(lbl, s):
        s = s & big
        if s.sum() < 300:
            return
        f2 = np.median(d2[s] / d1[s])
        f4 = np.median(d4[s] / d1[s])
        log(f"  {lbl:>18} {s.sum():>8,} {np.abs(d1[s]).mean():>9.2f} "
            f"{np.abs(d2[s]).mean():>9.2f} {np.abs(d4[s]).mean():>9.2f} "
            f"{f2:>9.3f} {f4:>9.3f}")

    prow("ALL traded", np.ones(len(d1), bool))
    for m in ["moneyline", "total", "spread"]:
        prow(m, mt == m)
    for lo, hi in [(200, 1000), (1000, 5000), (5000, np.inf)]:
        lbl = f"${lo:,.0f}-{hi:,.0f}" if np.isfinite(hi) else f">${lo:,.0f}"
        prow(lbl, (bk >= lo) & (bk < hi))
    log(f"\n  frac 2H / 4H are median (move at that horizon)/(move at 1H).")
    log(f"  ~1.0 means the price relocated and stayed. ~0.0 means it came "
        f"back, i.e. the mid oscillated and there was never a new level.")

    # ---- EXIT TIMING ----------------------------------------------------- #
    log(f"\n{'='*104}\nEXIT TIMING: what if you cannot exit at exactly t+H?"
        f"\n{'='*104}")
    fee0 = taker_fee_ticks(mid[i], TICK, a.category)
    sp0 = F.spread_ticks.to_numpy(np.float64)[i]

    def pnl_at(dk, mk, spk):
        """Executable taker P&L exiting at that horizon."""
        return (sd * dk - sp0 / 2.0 - spk / 2.0 - fee0
                - taker_fee_ticks(mk, TICK, a.category))

    p1 = pnl_at(d1, m1[i], F.sp_1H.to_numpy(np.float64)[i])
    p2 = pnl_at(d2, m2[i], F.sp_2H.to_numpy(np.float64)[i])
    p4 = pnl_at(d4, m4[i], F.sp_4H.to_numpy(np.float64)[i])

    log(f"  ticks per TRADE, exiting at each horizon")
    log(f"  {'slice':>18} {'n':>8} {'exit 1H':>10} {'exit 2H':>10} "
        f"{'exit 4H':>10} {'retained':>10}")

    def erow(lbl, s):
        if s.sum() < 300:
            return
        v1, v2, v4 = p1[s].mean(), p2[s].mean(), p4[s].mean()
        log(f"  {lbl:>18} {s.sum():>8,} {v1:>10.3f} {v2:>10.3f} {v4:>10.3f} "
            f"{(v2 / v1 if abs(v1) > 1e-9 else np.nan):>10.3f}")

    erow("ALL traded", np.ones(len(d1), bool))
    for m in ["moneyline", "total", "spread"]:
        erow(m, mt == m)
    for lo, hi in [(200, 1000), (1000, 5000), (5000, np.inf)]:
        lbl = f"${lo:,.0f}-{hi:,.0f}" if np.isfinite(hi) else f">${lo:,.0f}"
        erow(lbl, (bk >= lo) & (bk < hi))
    log(f"\n  'retained' is (exit 2H)/(exit 1H). A genuine relocation keeps "
        f"most of its P&L when held longer.")
    log(f"  If it collapses toward zero or negative, the edge requires "
        f"exiting within a 5s window to the tick -- which is a latency "
        f"strategy, not a prediction strategy, and this data cannot support "
        f"one (delivery lag dispersion on this file is 4.6s).")


if __name__ == "__main__":
    main()
