"""Single events: where does the book break, and how long before the price moves?

For each chosen large jump this draws, at the native 200 ms resolution:

    panel 1   mid price, with the jump marked
    panel 2   spread in ticks
    panel 3   near-touch dollars, bid and ask separately

and marks two instants on all three panels:

    BREAKDOWN  detected by an explicit rule, not by eye -- the first moment at
               which near-touch dollars (bid+ask within 2 ticks) fall below
               `--break-frac` of their own pre-event baseline and STAY below for
               at least `--sustain-s` seconds. The baseline is the median over
               [-30s, -15s], i.e. well before anything happens.
    JUMP       the 200 ms slot in which the mid actually moves most.

The gap between the two lines is the warning time for that event. Printing it
per event, rather than only the pooled median, shows how variable it is.

    python plot_breakdown_examples.py --n 8 --min-move 0.20
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "jump_prediction"))
import features as FE  # noqa: E402

JD = os.path.join(ROOT, "data", "jump")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen", "breakdown_examples")
GRID_MS = 200


def find_breakdown(taus, near_tot, base_lo, base_hi, frac, sustain_slots):
    """First sustained collapse of near-touch liquidity relative to baseline."""
    base = np.nanmedian(near_tot[(taus >= base_lo) & (taus <= base_hi)])
    if not np.isfinite(base) or base <= 0:
        return np.nan, np.nan
    below = near_tot < frac * base
    for i in range(len(taus)):
        if taus[i] < base_hi:
            continue
        if below[i:i + sustain_slots].all() and len(below[i:i + sustain_slots]):
            return taus[i], base
    return np.nan, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-move", type=float, default=0.20)
    ap.add_argument("--pre-s", type=float, default=30.0)
    ap.add_argument("--post-s", type=float, default=3.0)
    ap.add_argument("--quiet-min", type=int, default=5)
    ap.add_argument("--break-frac", type=float, default=0.5)
    ap.add_argument("--sustain-s", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    P = pd.read_parquet(os.path.join(CACHE, "panel_labelled.parquet"))
    jm = {aid: set(g.minute[g.jump == 1]) for aid, g in P.groupby("asset_id")}
    big = P[(P.jump == 1) & (P.ret.abs() >= a.min_move)].copy()
    big["clean"] = [not any((m - k) in jm[aid] for k in range(1, a.quiet_min + 1))
                    for aid, m in zip(big.asset_id, big.minute)]
    B = big[big.clean]
    # spread the examples over sessions and over both signs
    rng = np.random.default_rng(a.seed)
    picks = []
    for sign in (1, -1):
        sub = B[np.sign(B.ret) == sign]
        for sess, g in sub.groupby("session"):
            if len(picks) >= a.n * 3:
                break
            picks.append(g.iloc[rng.integers(len(g))])
    picks = picks[:a.n * 2]

    offs = np.arange(-int(a.pre_s * 1000 / GRID_MS),
                     int(a.post_s * 1000 / GRID_MS) + 1)
    taus = offs * GRID_MS / 1000.0
    sustain = max(int(a.sustain_s * 1000 / GRID_MS), 1)

    rows, made = [], 0
    for ev in picks:
        if made >= a.n:
            break
        sess = ev.session
        fp = os.path.join(JD, "feat_%s_trimmed.parquet" % sess)
        tp = os.path.join(JD, "lob_%s_trimmed.npy" % sess)
        if not (os.path.exists(fp) and os.path.exists(tp)):
            continue
        F = pq.read_table(fp, columns=["ts", "sid", "series", "mid",
                                       "spread_ticks"]).to_pandas()
        F["row"] = np.arange(len(F), dtype=np.int64)
        F["asset_id"] = F.series.astype(str).str.rsplit("|", n=1).str[-1]
        g = F[F.asset_id == ev.asset_id]
        gr = g[(g.ts >= ev.minute * 60000) & (g.ts < (ev.minute + 1) * 60000)]
        if len(gr) < 5:
            continue
        gm = gr.mid.to_numpy(np.float64)
        st = np.abs(np.diff(gm))
        if not len(st) or not np.isfinite(st).any():
            continue
        anchor = int(gr.row.to_numpy()[int(np.nanargmax(st))])
        rows_idx = anchor + offs
        n = len(F)
        ok = (rows_idx >= 0) & (rows_idx < n)
        r2 = np.clip(rows_idx, 0, n - 1)
        sid = F.sid.to_numpy()
        ok &= (sid[r2] == sid[anchor])
        if ok.sum() < len(offs) * 0.8:
            continue
        T = np.load(tp, mmap_mode="r")
        mid = F.mid.to_numpy(np.float64)
        d = FE.decode(np.asarray(T[r2]), mid[r2])
        S = pd.DataFrame(FE.state_features(
            d, spread_ticks=F.spread_ticks.to_numpy()[r2]))
        nb = np.expm1(S.log_bid_usd_within_2t.to_numpy())
        na = np.expm1(S.log_ask_usd_within_2t.to_numpy())
        sp = S.spread_ticks.to_numpy().astype(float)
        mm = mid[r2].copy()
        for arr in (nb, na, sp, mm):
            arr[~ok] = np.nan
        near = nb + na
        tb, base = find_breakdown(taus, near, -a.pre_s, -15.0,
                                  a.break_frac, sustain)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(taus, mm, color="#1f77b4", lw=1.6)
        axes[0].set_ylabel("mid price")
        axes[1].plot(taus, sp, color="#ff7f0e", lw=1.5)
        axes[1].set_ylabel("spread (ticks)")
        axes[2].plot(taus, nb, color="#2ca02c", lw=1.5, label="bid $ within 2 ticks")
        axes[2].plot(taus, na, color="#d62728", lw=1.5, label="ask $ within 2 ticks")
        axes[2].axhline(a.break_frac * base, color="#888888", ls=":", lw=1,
                        label="%.0f%% of baseline" % (100 * a.break_frac))
        axes[2].set_ylabel("near-touch $")
        axes[2].set_xlabel("seconds relative to the price move")
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.axvline(0, color="#c1121f", lw=1.6, ls="--")
            if np.isfinite(tb):
                ax.axvline(tb, color="#6a4c93", lw=1.6, ls="-.")
            ax.grid(alpha=.25)
        lead = (0 - tb) if np.isfinite(tb) else np.nan
        axes[0].set_title(
            "%s\n%s  |  move %+.3f  |  BREAKDOWN at %s  ->  JUMP at 0  "
            "(warning %s)"
            % (str(ev.series)[:62],
               pd.to_datetime(ev.minute * 60000 + 8 * 3600000, unit="ms")
                 .strftime("%Y-%m-%d %H:%M HKT"),
               ev.ret,
               ("%.1fs" % tb) if np.isfinite(tb) else "not detected",
               ("%.1f s" % lead) if np.isfinite(lead) else "n/a"),
            fontsize=9)
        fig.tight_layout()
        made += 1
        fig.savefig(os.path.join(RES, "breakdown_%02d.png" % made), dpi=130)
        plt.close(fig)
        rows.append(dict(file="breakdown_%02d.png" % made,
                         series=str(ev.series)[:60], move=float(ev.ret),
                         breakdown_tau_s=float(tb) if np.isfinite(tb) else np.nan,
                         warning_s=float(lead) if np.isfinite(lead) else np.nan,
                         spread_at_breakdown=float(sp[np.argmin(np.abs(taus - tb))])
                         if np.isfinite(tb) else np.nan,
                         spread_at_jump=float(sp[np.argmin(np.abs(taus - 0))]),
                         near_baseline=float(base)))
        del F, T

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "breakdown_examples.csv"), index=False)
    print(R.to_string(index=False, float_format=lambda v: "%.3f" % v))
    if len(R) and R.warning_s.notna().any():
        print("\nwarning time across these examples: median %.1fs  range %.1f-%.1fs"
              % (R.warning_s.median(), R.warning_s.min(), R.warning_s.max()))
    print("\nwrote", RES)


if __name__ == "__main__":
    sys.exit(main())
