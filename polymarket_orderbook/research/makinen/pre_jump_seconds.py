"""Does the book change in the SECONDS before a large jump?

Motivation: at 1-minute resolution every lead-time measurement in this project
collapses towards zero, which is consistent with the precursor -- if there is
one -- living a few seconds ahead of the move, far below the bar size. This
script drops to the native 200 ms grid (5 samples/sec; the underlying feed
updates every ~27 ms) and asks the question at that resolution.

Design
------
tau = 0 is located PRECISELY, not taken as the minute stamp. Within each
flagged minute the script finds the 200 ms slot where the mid actually moved
most, and anchors on the slot immediately BEFORE that move begins, so tau < 0
is strictly pre-move.

Events are CLEAN large jumps only: |move| >= --min-move, and no other jump on
the same contract in the preceding minutes, so nothing already under way is
counted as anticipation.

Controls are drawn from the same contract and the same spread/price regime, at
least 15 minutes from any jump, so a "control" is not the run-up to one.

Effect size is Cliff's delta at each tau. The sign convention is jump vs
control, so a negative delta means the quantity is LOWER before a jump.

    python pre_jump_seconds.py --min-move 0.20 --pre-s 30 --post-s 5
"""
from __future__ import annotations

import argparse
import glob
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
RES = os.path.join(ROOT, "results", "makinen", "pre_jump_seconds")
GRID_MS = 200

TRACK = ["spread_ticks", "l1_bid_usd", "l1_ask_usd", "log_total_depth",
         "log_bid_depth", "log_ask_depth", "imbalance", "imbalance_l1",
         "hhi_bid", "hhi_ask", "entropy_bid", "entropy_ask",
         "slope_bid", "slope_ask", "levels_bid", "levels_ask",
         "log_bid_usd_within_2t", "log_ask_usd_within_2t", "upd_rate"]


def cliffs(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 20 or len(b) < 20:
        return np.nan
    from scipy.stats import mannwhitneyu
    try:
        u = mannwhitneyu(a, b, alternative="two-sided").statistic
    except Exception:
        return np.nan
    return 2.0 * (u / (len(a) * len(b))) - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-move", type=float, default=0.20)
    ap.add_argument("--pre-s", type=float, default=30.0)
    ap.add_argument("--post-s", type=float, default=5.0)
    ap.add_argument("--quiet-min", type=int, default=5)
    ap.add_argument("--max-events", type=int, default=1200)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    P = pd.read_parquet(os.path.join(CACHE, "panel_labelled.parquet"))
    jm = {aid: set(g.minute[g.jump == 1]) for aid, g in P.groupby("asset_id")}
    big = P[(P.jump == 1) & (P.ret.abs() >= a.min_move)].copy()
    big["clean"] = [not any((m - k) in jm[aid] for k in range(1, a.quiet_min + 1))
                    for aid, m in zip(big.asset_id, big.minute)]
    B = big[big.clean]
    print("clean large jumps (|move|>=%.2f, quiet %d min before): %d"
          % (a.min_move, a.quiet_min, len(B)))

    offs = np.arange(-int(a.pre_s * 1000 / GRID_MS),
                     int(a.post_s * 1000 / GRID_MS) + 1)
    taus = offs * GRID_MS / 1000.0
    jt = {k: [[] for _ in offs] for k in TRACK}
    ct = {k: [[] for _ in offs] for k in TRACK}
    n_ev = n_ct = 0
    rng = np.random.default_rng(0)

    for sess, Bs in B.groupby("session"):
        fp = os.path.join(JD, "feat_%s_trimmed.parquet" % sess)
        tp = os.path.join(JD, "lob_%s_trimmed.npy" % sess)
        if not (os.path.exists(fp) and os.path.exists(tp)):
            continue
        F = pq.read_table(fp, columns=["ts", "sid", "series", "mid",
                                       "spread_ticks", "valid", "book_age_ms"]).to_pandas()
        F["row"] = np.arange(len(F), dtype=np.int64)
        F["asset_id"] = F.series.astype(str).str.rsplit("|", n=1).str[-1]
        T = np.load(tp, mmap_mode="r")
        ts = F.ts.to_numpy(np.int64); sid = F.sid.to_numpy()
        mid = F.mid.to_numpy(np.float64)
        n = len(F)

        def grab(rows, store):
            """book features at a set of absolute grid rows"""
            ok = (rows >= 0) & (rows < n)
            r2 = np.clip(rows, 0, n - 1)
            d = FE.decode(np.asarray(T[r2]), mid[r2])
            S = FE.state_features(d, spread_ticks=F.spread_ticks.to_numpy()[r2])
            S = pd.DataFrame(S)
            # update intensity: fraction of the last second with a fresh book
            age = F.book_age_ms.to_numpy()[r2]
            S["upd_rate"] = 1000.0 / np.maximum(age, 1.0)
            for k in TRACK:
                v = S[k].to_numpy(np.float64).copy()
                v[~ok] = np.nan
                store.append(v)
            return ok

        by_asset = {k: g for k, g in F.groupby("asset_id", sort=False)}
        for _, ev in Bs.iterrows():
            g = by_asset.get(ev.asset_id)
            if g is None:
                continue
            gr = g[(g.ts >= ev.minute * 60000) & (g.ts < (ev.minute + 1) * 60000)]
            if len(gr) < 5:
                continue
            gm = gr.mid.to_numpy(np.float64)
            # the 200ms slot where the move actually happens
            step = np.abs(np.diff(gm))
            if not len(step) or not np.isfinite(step).any():
                continue
            k0 = int(np.nanargmax(step))          # move occurs between k0 and k0+1
            anchor = int(gr.row.to_numpy()[k0])   # last slot BEFORE the move
            rows = anchor + offs
            same = (sid[np.clip(rows, 0, n - 1)] == sid[anchor])
            tmp = []
            ok = grab(rows, tmp)
            for i, k in enumerate(TRACK):
                v = tmp[i].copy(); v[~(ok & same)] = np.nan
                jt[k][0] = jt[k][0]  # keep structure
            for i, k in enumerate(TRACK):
                v = tmp[i].copy(); v[~(ok & same)] = np.nan
                for j in range(len(offs)):
                    jt[k][j].append(v[j])
            n_ev += 1

            # Matched control -- match on the state at the START of the window
            # (tau = -pre_s), NOT at the anchor. Matching at the anchor selects
            # control moments that are themselves depleted, which conditions
            # away the effect being measured and makes the two curves identical.
            back = int(a.pre_s * 1000 / GRID_MS)
            a_start = anchor - back
            if a_start < 0 or sid[a_start] != sid[anchor]:
                continue
            sp0 = float(F.spread_ticks.to_numpy()[a_start])
            pr0 = float(mid[a_start])
            crow = g.row.to_numpy()
            cstart = crow - back
            okc = (cstart >= 0) & (sid[np.clip(cstart, 0, n - 1)] == sid[np.clip(crow, 0, n - 1)])
            sp_c = F.spread_ticks.to_numpy()[np.clip(cstart, 0, n - 1)]
            pr_c = mid[np.clip(cstart, 0, n - 1)]
            cand = g[okc & (np.abs(sp_c - sp0) <= 1.0) & (np.abs(pr_c - pr0) <= 0.05)]
            if len(cand) > 0:
                cm = cand.ts.to_numpy() // 60000
                far = np.array([not any(abs(int(x) - q) < 15 for q in jm[ev.asset_id])
                                for x in cm])
                cand = cand[far]
            if len(cand) == 0:
                continue
            canchor = int(cand.row.to_numpy()[rng.integers(len(cand))])
            rows = canchor + offs
            same = (sid[np.clip(rows, 0, n - 1)] == sid[canchor])
            tmp = []
            ok = grab(rows, tmp)
            for i, k in enumerate(TRACK):
                v = tmp[i].copy(); v[~(ok & same)] = np.nan
                for j in range(len(offs)):
                    ct[k][j].append(v[j])
            n_ct += 1
        del F, T
    print("aligned %d jump windows and %d matched control windows" % (n_ev, n_ct))

    rows = []
    for k in TRACK:
        for j, t in enumerate(taus):
            jv = np.array(jt[k][j], float)
            cv = np.array(ct[k][j], float)
            rows.append(dict(feature=k, tau_s=t,
                             median_jump=np.nanmedian(jv) if len(jv) else np.nan,
                             median_ctrl=np.nanmedian(cv) if len(cv) else np.nan,
                             cliffs_delta=cliffs(jv, cv),
                             n_jump=int(np.isfinite(jv).sum()),
                             n_ctrl=int(np.isfinite(cv).sum())))
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(RES, "pre_jump_seconds_effects.csv"), index=False)

    key = [-30.0, -10.0, -5.0, -2.0, -1.0, -0.4, -0.2, 0.0]
    piv = (E[E.tau_s.isin(key)].pivot_table(index="feature", columns="tau_s",
                                            values="cliffs_delta"))
    piv = piv.reindex(piv.abs().max(axis=1).sort_values(ascending=False).index)
    print("\n=== Cliff's delta vs matched controls, SECONDS before a clean large jump ===")
    print("    (|d| < 0.15 negligible, 0.15-0.33 small, > 0.33 large)")
    print(piv.round(3).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, k in zip(axes.ravel(), piv.index[:6]):
        s = E[E.feature == k].sort_values("tau_s")
        ax.plot(s.tau_s, s.median_jump, color="#c1121f", lw=1.6,
                label="before large jump")
        ax.plot(s.tau_s, s.median_ctrl, color="#1d3557", lw=1.6,
                label="matched control")
        ax.axvline(0, color="k", ls="--", lw=.9)
        ax.set_title(k, fontsize=10)
        ax.set_xlabel("seconds relative to the move (0 = last book before it)")
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.suptitle("Order book in the %.0f seconds before a clean large jump "
                 "(|move| >= %.2f), n=%d events, 200 ms resolution"
                 % (a.pre_s, a.min_move, n_ev), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(os.path.join(RES, "pre_jump_seconds.png"), dpi=130)
    plt.close(fig)
    print("\nwrote", RES)


if __name__ == "__main__":
    sys.exit(main())
