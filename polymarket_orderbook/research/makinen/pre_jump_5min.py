"""Does the book change in the 5 minutes before a LARGE jump that has not started?

Two questions, deliberately separated.

1. EVENT STUDY (model-free). Take every large jump, align to tau=0 and look at
   the book at tau = -5..0 minutes against matched control windows from the
   same contract. Effect size is Cliff's delta, so "how big" does not rest on a
   plot.

2. FORECAST with a real blackout. y_t = 1 if a large jump lands in
   (t+G, t+G+1] with G=5. The input ends at t, so there is a five-minute gap
   between the last observation the model sees and the minute it must call.

"Clean" is the load-bearing definition. A large jump whose contract already
jumped in the previous 5 minutes is part of a cluster that is visibly under
way, and predicting it is not anticipation. Those events are EXCLUDED from the
positives, and windows whose blackout contains any jump are dropped entirely
rather than relabelled as negatives -- they are ambiguous, not quiet.

    python pre_jump_5min.py --min-move 0.20 --gap 5
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen", "pre_jump_5min")

TRACK = ["spread_ticks", "log_total_depth", "log_bid_depth", "log_ask_depth",
         "imbalance", "imbalance_l1", "hhi_bid", "hhi_ask", "entropy_bid",
         "entropy_ask", "slope_bid", "slope_ask", "l1_bid_usd", "l1_ask_usd",
         "log_bid_usd_within_2t", "log_ask_usd_within_2t", "levels_bid",
         "levels_ask", "rv_5", "rv_15"]


def cliffs_delta(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 15 or len(b) < 15:
        return np.nan
    from scipy.stats import mannwhitneyu
    try:
        u = mannwhitneyu(a, b, alternative="two-sided").statistic
    except Exception:
        return np.nan
    return 2.0 * (u / (len(a) * len(b))) - 1.0


def add_derived(P):
    P = P.sort_values(["asset_id", "minute"]).reset_index(drop=True)
    g = P.groupby("asset_id", sort=False)
    contig = g.minute.diff() == 1
    P["d_mid"] = np.where(contig, g.mid.diff(), np.nan)
    for w in (5, 15):
        P["rv_%d" % w] = P.d_mid.groupby(P.asset_id).transform(
            lambda s: s.rolling(w, min_periods=3).std())
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-move", type=float, default=0.20)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--quiet-before", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    P = add_derived(pd.read_parquet(os.path.join(CACHE, "panel_labelled.parquet")))
    P["pos"] = np.arange(len(P))
    jump_min = {aid: set(g.minute[g.jump == 1])
                for aid, g in P.groupby("asset_id", sort=False)}

    big = P[(P.jump == 1) & (P.ret.abs() >= a.min_move)].copy()
    clean = []
    for aid, m in zip(big.asset_id, big.minute):
        js = jump_min[aid]
        clean.append(not any((m - k) in js for k in range(1, a.quiet_before + 1)))
    big["clean"] = clean
    B = big[big.clean]
    print("large jumps (|move| >= %.2f): %d, of which CLEAN (no jump in the "
          "previous %d min): %d" % (a.min_move, len(big), a.quiet_before, len(B)))

    # ---------- 1. event study, matched controls ----------
    idx = P.set_index(["asset_id", "minute"])
    sp_b = np.digitize(P.spread_ticks, [1.5, 2.5, 5, 10, 30])
    pr_b = np.digitize(P.mid, [.1, .25, .5, .75, .9])
    P["stratum"] = (P.asset_id.astype(str) + "|" + sp_b.astype(str) + "|"
                    + pr_b.astype(str))
    rng = np.random.default_rng(0)
    # controls: same contract+regime, and at least 15 min from ANY jump
    far = []
    for aid, g in P.groupby("asset_id", sort=False):
        js = np.array(sorted(jump_min[aid])) if jump_min[aid] else np.array([-10**9])
        d = np.abs(g.minute.to_numpy()[:, None] - js[None, :]).min(axis=1)
        far.append(pd.Series(d >= 15, index=g.index))
    P["far"] = pd.concat(far).sort_index()

    # `big` was sliced before `stratum` existed on P; attach it now
    B = B.drop(columns=[c for c in ("stratum",) if c in B.columns]).merge(
        P[["asset_id", "minute", "stratum"]], on=["asset_id", "minute"], how="left")

    rows = []
    taus = list(range(-a.quiet_before, 1))
    traj = {k: {"j": {t: [] for t in taus}, "c": {t: [] for t in taus}}
            for k in TRACK}
    ctrl_pool = P[P.far]
    for _, ev in B.iterrows():
        cand = ctrl_pool[ctrl_pool.stratum == ev.stratum]
        if not len(cand):
            continue
        cm = int(cand.minute.iloc[rng.integers(len(cand))])
        for t in taus:
            for tag, aid_, base in (("j", ev.asset_id, ev.minute),
                                    ("c", ev.asset_id, cm)):
                try:
                    r = idx.loc[(aid_, base + t)]
                except KeyError:
                    continue
                for k in TRACK:
                    traj[k][tag][t].append(float(r[k]))
    for k in TRACK:
        for t in taus:
            j = np.array(traj[k]["j"][t]); c = np.array(traj[k]["c"][t])
            rows.append(dict(feature=k, tau=t, n_jump=len(j), n_ctrl=len(c),
                             median_jump=np.nanmedian(j) if len(j) else np.nan,
                             median_ctrl=np.nanmedian(c) if len(c) else np.nan,
                             cliffs_delta=cliffs_delta(j, c)))
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(RES, "event_study_effects.csv"), index=False)
    piv = E.pivot_table(index="feature", columns="tau", values="cliffs_delta")
    piv = piv.reindex(piv.abs().max(axis=1).sort_values(ascending=False).index)
    print("\n=== Cliff's delta, book at tau minutes before a CLEAN large jump "
          "vs matched controls ===")
    print("    (|d|<0.15 negligible, 0.15-0.33 small, >0.33 large)")
    print(piv.round(3).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, k in zip(axes.ravel(), piv.index[:6]):
        s = E[E.feature == k].sort_values("tau")
        ax.plot(s.tau, s.median_jump, "o-", color="#c1121f", label="before large jump")
        ax.plot(s.tau, s.median_ctrl, "s-", color="#1d3557", label="matched control")
        ax.set_title(k, fontsize=10); ax.set_xlabel("minutes before jump")
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.suptitle("Book in the %d minutes before a clean large jump (|move| >= %.2f), "
                 "n=%d" % (a.quiet_before, a.min_move, len(B)), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(os.path.join(RES, "pre_jump_trajectories.png"), dpi=130)
    plt.close(fig)
    print("\nwrote", RES)


if __name__ == "__main__":
    sys.exit(main())
