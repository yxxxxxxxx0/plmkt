"""Phase 3: does the book visibly deteriorate before a jump? Look first.

Before any model, align the book to the moment of each jump (tau = 0) and ask
what the preceding 60 seconds look like, against matched control windows drawn
from the same markets at moments where no jump followed.

Matching matters more here than anywhere else. Jumps do not happen at random
moments: they cluster in volatile phases, in wide-spread books, and at extreme
prices. A naive comparison of "jump windows" against "all other windows" would
mostly rediscover that, and would show a large spurious effect for any feature
correlated with the regime. Controls are therefore matched within the same
market on:

  spread regime      bucketed, since spread drives both jump rate and depth
  price regime       bucketed mid, since a book at 0.05 is not a book at 0.50
  game phase         quartile of elapsed time within the match

and drawn only from moments at least `separation_s` away from any jump, so a
"control" is not simply the run-up to a jump a few seconds later.

Outputs, per (J, H) and per feature:
  * median trajectory with bootstrap bands, jump vs control, tau in [-60, 0]
  * an effect size at tau = 0 and at tau = -10s: rank-biserial (equivalently
    the Mann-Whitney common-language effect) plus a robust standardised
    difference, so "how big" does not rest on eyeballing a plot

    python event_study.py --sessions 2026-09-10 --J 0.02 --H 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import features as FE  # noqa: E402

JD = os.path.join(ROOT, "data", "jump")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "jump_prediction")
GRID_MS = 200

# Features tracked through the window. Chosen to cover the mechanisms in the
# brief: liquidity level, liquidity shape, imbalance, and activity.
TRACK = ["spread_ticks", "l1_bid_usd", "l1_ask_usd", "log_bid_depth",
         "log_ask_depth", "log_total_depth", "imbalance", "imbalance_l1",
         "hhi_bid", "hhi_ask", "entropy_bid", "entropy_ask",
         "slope_bid", "slope_ask", "levels_bid", "levels_ask",
         "log_bid_usd_within_2t", "log_ask_usd_within_2t"]
NICE = {"spread_ticks": "spread (ticks)", "log_total_depth": "log total depth",
        "imbalance": "depth imbalance", "hhi_bid": "bid concentration (HHI)",
        "entropy_bid": "bid depth entropy", "levels_bid": "bid levels present"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_points(sessions=None):
    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    if sessions:
        ps = [p for p in ps if any(s in os.path.basename(p) for s in sessions)]
    if not ps:
        raise SystemExit("no cache; run build_dataset.py first")
    return pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)


def session_arrays(name):
    """The full-resolution grid for one session, for walking back from tau=0."""
    import pyarrow.parquet as pq
    fp = os.path.join(JD, f"feat_{name}_trimmed.parquet")
    tp = os.path.join(JD, f"lob_{name}_trimmed.npy")
    F = pq.read_table(fp, columns=["mid", "spread_ticks", "ts", "sid",
                                   "book_age_ms"]).to_pandas()
    T = np.load(tp, mmap_mode="r")
    return F, T


def bucket(x, edges):
    return np.digitize(np.asarray(x), edges)


def pick_controls(D, jmask, sep_rows, rng, ratio=1):
    """Matched non-jump windows from the same market and regime.

    Matching is exact on (market, spread bucket, price bucket, phase quartile).
    Controls must also be at least `sep_rows` grid rows from any jump in the
    same series, so the run-up to a jump cannot be sampled as a control.
    """
    d = D.reset_index(drop=True)
    sp_b = bucket(d.spread_ticks, [1.5, 2.5, 5, 10, 30])
    pr_b = bucket(d.mid, [0.1, 0.25, 0.5, 0.75, 0.9])
    ph = d.groupby("series", observed=True)["ts"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False,
                          duplicates="drop"))
    strata = (d.market.astype(str) + "|" + sp_b.astype(str) + "|"
              + pr_b.astype(str) + "|" + ph.astype(str))

    # distance (in rows of the same series) to the nearest jump
    near = np.zeros(len(d), bool)
    for _, g in d.groupby("sid", observed=True, sort=False):
        ii = g.index.to_numpy()
        jj = ii[jmask[ii]]
        if len(jj) == 0:
            continue
        pos = np.searchsorted(jj, ii)
        lo = np.abs(ii - jj[np.clip(pos - 1, 0, len(jj) - 1)])
        hi = np.abs(ii - jj[np.clip(pos, 0, len(jj) - 1)])
        near[ii] = np.minimum(lo, hi) < sep_rows
    eligible = (~jmask) & (~near)

    out = []
    js = pd.Series(strata[jmask]).value_counts()
    pool = pd.DataFrame({"s": strata[eligible],
                         "i": np.where(eligible)[0]})
    for stratum, k in js.items():
        cand = pool.i[pool.s == stratum].to_numpy()
        if len(cand) == 0:
            continue
        take = min(len(cand), int(k * ratio))
        out.append(rng.choice(cand, size=take, replace=False))
    if not out:
        return np.array([], np.int64), 0.0
    ctrl = np.concatenate(out)
    matched = len(ctrl) / max(jmask.sum(), 1)
    return ctrl, matched


def trajectories(name, rows, F, T, back_s, step_s):
    """Feature values at tau = -back_s .. 0, for a set of anchor rows."""
    offs = -np.arange(0, int(back_s / step_s) + 1)[::-1] * int(step_s * 1000 / GRID_MS)
    sid = F.sid.to_numpy()
    ts = F.ts.to_numpy(np.int64)
    mid = F.mid.to_numpy(np.float64)
    sp = F.spread_ticks.to_numpy()
    n = len(F)
    out = {k: np.full((len(rows), len(offs)), np.nan, np.float32) for k in TRACK}
    for c, o in enumerate(offs):
        j = rows + o
        ok = (j >= 0) & (j < n)
        j2 = np.clip(j, 0, n - 1)
        # same series and exactly |o| grid steps earlier -- no bridging gaps
        ok &= (sid[j2] == sid[rows]) & (ts[rows] - ts[j2] == -o * GRID_MS)
        if not ok.any():
            continue
        jj = j2[ok]
        d = FE.decode(np.asarray(T[jj]), mid[jj])
        S = FE.state_features(d, spread_ticks=sp[jj])
        for k in TRACK:
            out[k][ok, c] = S[k].to_numpy()
    return out, offs * GRID_MS / 1000.0


def med_band(A, rng, B=300):
    """Median trajectory with a bootstrap band over WINDOWS, not time points."""
    med = np.nanmedian(A, axis=0)
    n = A.shape[0]
    if n < 30:
        return med, med, med
    idx = rng.integers(0, n, size=(B, n))
    bs = np.nanmedian(A[idx], axis=1)
    return med, np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)


def effect(a, b):
    """Rank-biserial effect size, and a robust standardised difference."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 20 or len(b) < 20:
        return dict(auc=np.nan, cliffs=np.nan, robust_d=np.nan, n_j=len(a), n_c=len(b))
    from scipy.stats import mannwhitneyu
    try:
        u = mannwhitneyu(a, b, alternative="two-sided").statistic
        auc = u / (len(a) * len(b))
    except Exception:
        auc = np.nan
    mad = lambda x: np.median(np.abs(x - np.median(x))) * 1.4826
    s = np.sqrt((mad(a) ** 2 + mad(b) ** 2) / 2) or np.nan
    return dict(auc=float(auc), cliffs=float(2 * auc - 1),
                robust_d=float((np.median(a) - np.median(b)) / s),
                n_j=int(len(a)), n_c=int(len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--back-s", type=float, default=60.0)
    ap.add_argument("--step-s", type=float, default=1.0)
    ap.add_argument("--max-events", type=int, default=6000)
    ap.add_argument("--tight-only", action="store_true",
                    help="restrict to books with spread <= 2 ticks, where a "
                         "mid move is a repricing rather than an illiquid "
                         "quote wobbling")
    ap.add_argument("--label", default="raw",
                    choices=["raw", "clean", "durable", "both"],
                    help="'raw' is the original max-|mid| label; the others "
                         "use the artefact-free decomposition built by "
                         "audit_durability.py")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(RES, "plots"), exist_ok=True)
    rng = np.random.default_rng(a.seed)

    D = load_points(a.sessions)
    col = f"jump_{a.J:g}_H{a.H}"
    if col not in D.columns:
        raise SystemExit(f"{col} not in cache")
    D = D[D[f"valid_label_H{a.H}"].to_numpy(bool)].reset_index(drop=True)
    if a.label != "raw":
        # See audit_durability.py: the raw max-|mid| label is dominated by
        # transient quote vacuums, so an event study built on it is largely an
        # event study of the mid breaking rather than of the price moving.
        mv = os.path.join(CACHE, f"moves_H{a.H}.parquet")
        if not os.path.exists(mv):
            raise SystemExit(f"run audit_durability.py --H {a.H} first")
        M = pd.read_parquet(mv)
        D = D.merge(M[["sid", "ts", "clean_move", "durable_move"]],
                    on=["sid", "ts"], how="inner")
        clean = D.clean_move >= a.J
        durable = D.durable_move >= a.J
        D[col] = (clean if a.label == "clean" else
                  durable if a.label == "durable" else clean & durable)
        D = D.drop(columns=["clean_move", "durable_move"])
        log(f"using '{a.label}' label: {len(D):,} rows after move join")
    if a.tight_only:
        D = D[D.spread_ticks <= 2.0].reset_index(drop=True)
    log(f"{len(D):,} valid prediction points, "
        f"{int(D[col].sum()):,} jumps ({D[col].mean():.3%})")

    jm = D[col].to_numpy(bool)
    sep_rows = int((a.back_s + a.H) * 1000 / GRID_MS / 5)   # points are 1 Hz
    ctrl_idx, matched = pick_controls(D, jm, sep_rows, rng)
    log(f"matched {len(ctrl_idx):,} control windows ({matched:.2f} per jump)")

    jump_idx = np.where(jm)[0]
    if len(jump_idx) > a.max_events:
        jump_idx = rng.choice(jump_idx, a.max_events, replace=False)
    if len(ctrl_idx) > a.max_events:
        ctrl_idx = rng.choice(ctrl_idx, a.max_events, replace=False)

    # walk back through the full-resolution grid, per session
    J_all = {k: [] for k in TRACK}
    C_all = {k: [] for k in TRACK}
    taus = None
    for name, g in D.iloc[np.concatenate([jump_idx, ctrl_idx])].groupby(
            "session", observed=True):
        F, T = session_arrays(name)
        isj = np.isin(g.index.to_numpy(), jump_idx)
        for rows, store in ((g.row.to_numpy()[isj], J_all),
                            (g.row.to_numpy()[~isj], C_all)):
            if len(rows) == 0:
                continue
            tr, taus = trajectories(name, rows, F, T, a.back_s, a.step_s)
            for k in TRACK:
                store[k].append(tr[k])
        del F, T
        log(f"  {name}: {int(isj.sum()):,} jump / {int((~isj).sum()):,} control")

    rows_out = []
    for k in TRACK:
        if not J_all[k] or not C_all[k]:
            continue
        A = np.concatenate(J_all[k], axis=0)
        B = np.concatenate(C_all[k], axis=0)
        mj, lj, hj = med_band(A, rng)
        mc, lc, hc = med_band(B, rng)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(taus, mj, color="#c1121f", lw=1.8, label=f"jump (n={A.shape[0]:,})")
        ax.fill_between(taus, lj, hj, color="#c1121f", alpha=0.20, lw=0)
        ax.plot(taus, mc, color="#1d3557", lw=1.8, label=f"matched control (n={B.shape[0]:,})")
        ax.fill_between(taus, lc, hc, color="#1d3557", alpha=0.20, lw=0)
        ax.axvline(0, color="#888", ls="--", lw=0.9)
        ax.set_xlabel("seconds relative to jump (tau = 0)")
        ax.set_ylabel(NICE.get(k, k.replace("_", " ")))
        ax.set_title(f"{NICE.get(k, k)}  |  J={a.J:g}  H={a.H}s"
                     + ("  tight books" if a.tight_only else ""))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        tag = (f"J{a.J:g}_H{a.H}" + ("_tight" if a.tight_only else "")
               + ("" if a.label == "raw" else f"_{a.label}"))
        fig.savefig(os.path.join(RES, "plots", f"event_{k}_{tag}.png"), dpi=130)
        plt.close(fig)

        # effect sizes at the moment itself and 10s before
        i0 = int(np.argmin(np.abs(taus - 0)))
        i10 = int(np.argmin(np.abs(taus + 10)))
        i30 = int(np.argmin(np.abs(taus + 30)))
        for lbl, i in (("tau=0", i0), ("tau=-10s", i10), ("tau=-30s", i30)):
            e = effect(A[:, i], B[:, i])
            e.update(feature=k, when=lbl, J=a.J, H=a.H,
                     tight_only=bool(a.tight_only))
            rows_out.append(e)

    R = pd.DataFrame(rows_out)
    tag = (f"J{a.J:g}_H{a.H}" + ("_tight" if a.tight_only else "")
               + ("" if a.label == "raw" else f"_{a.label}"))
    p = os.path.join(RES, f"event_study_effects_{tag}.csv")
    R.to_csv(p, index=False)
    log(f"wrote {p} and {len(TRACK)} plots")

    print(f"\nEffect sizes, jump vs matched control (|Cliff's delta| > 0.15 is "
          f"a visible separation; > 0.33 is large):")
    piv = (R.pivot(index="feature", columns="when", values="cliffs")
           .reindex(columns=["tau=-30s", "tau=-10s", "tau=0"]))
    piv["|max|"] = piv.abs().max(axis=1)
    print(piv.sort_values("|max|", ascending=False)
          .to_string(float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
