"""Stage 1 / 1A: Lee-Mykland labels on the minute panel, plus the visual check.

The detector itself is the one already validated in `research/lee_mykland/`;
`lm_threshold` and `returns_for` are imported rather than reimplemented, so
there is exactly one Lee-Mykland implementation in this repository.

Method (full derivation in results/makinen/lee_mykland_method.md):

    r_i                = m_i - m_{i-1}            (arithmetic, see below)
    sigma_hat^2(i)     = 1/(K-2) * sum_{j=i-K+2}^{i-1} |r_j| |r_{j-1}|
    L(i)               = r_i / sigma_hat(i)
    threshold          = C_n + S_n * (-log(-log(1-alpha)))
    C_n = (2 log n)^.5 / c - (log pi + log log n) / (2c (2 log n)^.5)
    S_n = 1 / (c (2 log n)^.5),      c = sqrt(2/pi)

Two deliberate departures from the paper's settings, both forced by the data
and both measured rather than assumed (see results/lee_mykland/ for the
supporting sweeps):

  K = 30, not 600.  The paper studies continuously traded equities where 600
  minutes is about a day and a half. A Polymarket game contract lives a median
  of ~161 in-game minutes here, so K=600 would leave no testable observations
  at all. K=30 was picked from a sweep on the volatility-tracking diagnostic.

  Arithmetic returns, not log.  These are probability contracts bounded in
  (0,1). In log space one half-tick is 0.0055 at p=0.90 but 0.41 at p=0.01, so
  the test would spend its power on near-resolved contracts. In probability
  points a half-tick is 0.005 everywhere. Two checks confirm the change: the
  positive/negative jump counts become symmetric, and the YES and NO legs of a
  market produce exactly mirrored statistics.

alpha stays at 0.01 and the critical value is recomputed from n by the method.
The threshold was never tuned.

    python label_jumps.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "lee_mykland"))
from detect_lee_mykland_jumps import lm_threshold, returns_for, C_BP  # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen")
JD = os.path.join(RES, "jumps")


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def lm_for_series(g, K, min_nz, scale):
    g = g.sort_values("minute")
    minute = g.minute.to_numpy(np.int64)
    mid = g.mid.to_numpy(np.float64)
    adjacent = np.empty(len(g), bool)
    adjacent[0] = False
    adjacent[1:] = (minute[1:] - minute[:-1]) == 1
    r = returns_for(mid, scale)
    r[~adjacent] = np.nan
    r = pd.Series(r)
    p = r.abs() * r.abs().shift(1)
    win = max(K - 2, 2)
    ps = p.shift(1)                                   # strictly trailing
    sigma = np.sqrt(ps.rolling(win, min_periods=max(min_nz, 2)).mean())
    n_nz = ps.gt(0).rolling(win, min_periods=1).sum()
    sigma[n_nz < min_nz] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        L = r / sigma
    return pd.DataFrame({"minute": minute, "ret": r.to_numpy(),
                         "sigma": sigma.to_numpy(), "LM": L.to_numpy()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-nonzero-pairs", type=int, default=7)
    ap.add_argument("--return-scale", default="diff",
                    choices=["diff", "log", "logit"])
    a = ap.parse_args()
    os.makedirs(os.path.join(JD, "examples"), exist_ok=True)

    P = pd.read_parquet(os.path.join(CACHE, "minute_panel.parquet"))
    log("panel {:,} bars, {:,} series".format(len(P), P.asset_id.nunique()))

    parts = []
    for aid, g in P.groupby("asset_id", sort=False, observed=True):
        o = lm_for_series(g, a.K, a.min_nonzero_pairs, a.return_scale)
        o["asset_id"] = aid
        parts.append(o)
    D = pd.concat(parts, ignore_index=True)
    P = P.merge(D, on=["asset_id", "minute"], how="left")

    tested = np.isfinite(P.LM.to_numpy())
    n = int(tested.sum())
    thr, c_n, s_n = lm_threshold(n, a.alpha)
    P["threshold"] = thr
    P["jump"] = ((tested) & (np.abs(P.LM.to_numpy()) > thr)).astype(np.int8)
    P["jump_direction"] = 0
    P.loc[(P.jump == 1) & (P.ret > 0), "jump_direction"] = 1
    P.loc[(P.jump == 1) & (P.ret < 0), "jump_direction"] = -1
    P["tested"] = tested
    log("n=%s  C_n=%.4f  S_n=%.4f  threshold=%.4f"
        % ("{:,}".format(n), c_n, s_n, thr))

    P.to_parquet(os.path.join(CACHE, "panel_labelled.parquet"), index=False)
    cols = ["ts_hkt", "minute", "date", "session", "asset_id", "series",
            "mid", "ret", "sigma", "LM", "threshold", "jump", "jump_direction",
            "tested", "spread_ticks", "tod_min"]
    out = P[cols].rename(columns={"mid": "mid_price", "ret": "return",
                                  "LM": "LM_statistic", "sigma": "local_volatility"})
    out.to_csv(os.path.join(JD, "lee_mykland_jumps.csv"), index=False)

    # ---------------- sanity-check summary ---------------------------------
    J = P[P.jump == 1]
    days = P.loc[tested, "date"].nunique()
    st = {
        "total_minute_bars": len(P),
        "tested_observations": n,
        "warmup_or_untestable": int(len(P) - n),
        "total_jumps": len(J),
        "positive_jumps": int((J.jump_direction == 1).sum()),
        "negative_jumps": int((J.jump_direction == -1).sum()),
        "pct_of_tested_minutes": 100.0 * len(J) / max(n, 1),
        "distinct_days": days,
        "jumps_per_day": len(J) / max(days, 1),
        "median_abs_jump_return": float(J.ret.abs().median()) if len(J) else np.nan,
        "mean_abs_jump_return": float(J.ret.abs().mean()) if len(J) else np.nan,
        "p90_abs_jump_return": float(J.ret.abs().quantile(.9)) if len(J) else np.nan,
        "series": int(P.asset_id.nunique()),
        "K": a.K, "alpha": a.alpha, "threshold": thr,
        "return_scale": a.return_scale,
    }
    pd.Series(st).to_csv(os.path.join(JD, "jump_statistics.csv"))
    print("\n=== Stage 1A sanity check ===")
    for k, v in st.items():
        print("  %-26s %s" % (k, ("%.6g" % v) if isinstance(v, float) else v))
    rate = st["pct_of_tested_minutes"]
    verdict = ("PLAUSIBLE" if rate < 5 else
               "TOO HIGH - diagnose before modelling")
    print("  %-26s %s (%.2f%% of tested minutes)" % ("VERDICT", verdict, rate))

    top = (J.assign(absL=J.LM.abs()).nlargest(20, "absL")
           [["ts_hkt", "series", "mid", "ret", "jump_direction", "LM", "sigma"]])
    top.to_csv(os.path.join(JD, "top20_jumps.csv"), index=False)
    print("\n=== 20 strongest jumps by |LM| ===")
    print(top.to_string(index=False))

    # ---------------- FIGURE 1: full series with jumps ---------------------
    cand = (P[tested].groupby("asset_id")
            .agg(n=("LM", "size"), j=("jump", "sum")).reset_index())
    cand = cand[cand.j > 0].sort_values(["j", "n"], ascending=False)
    aid = cand.asset_id.iloc[0]
    s = P[P.asset_id == aid].sort_values("minute")
    lbl = str(s.series.iloc[0])
    up, dn = s[s.jump_direction == 1], s[s.jump_direction == -1]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(s.ts_hkt, s.mid, lw=1.0, color="#1f77b4", label="mid price")
    ax.scatter(up.ts_hkt, up.mid, marker="^", s=90, color="#2ca02c",
               edgecolor="black", lw=.4, zorder=3,
               label="positive jump (%d)" % len(up))
    ax.scatter(dn.ts_hkt, dn.mid, marker="v", s=90, color="#d62728",
               edgecolor="black", lw=.4, zorder=3,
               label="negative jump (%d)" % len(dn))
    ax.set_title("Lee-Mykland jumps | %s\n+%d / -%d / %d total  (K=%d, alpha=%.3f,"
                 " threshold=%.3f)" % (lbl, len(up), len(dn), len(up) + len(dn),
                                       a.K, a.alpha, thr), fontsize=10)
    ax.set_xlabel("time (HKT)"); ax.set_ylabel("mid price")
    ax.legend(); ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(JD, "lee_mykland_full_series.png"), dpi=130)
    plt.close(fig)

    # ---------------- FIGURE 2: the statistic ------------------------------
    s2 = s[np.isfinite(s.LM.to_numpy())]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(s2.ts_hkt, s2.LM, lw=.9, color="#444444", label="$L(i)$")
    ax.axhline(thr, color="#d62728", ls="--", lw=1.2,
               label="threshold $\\pm$%.3f" % thr)
    ax.axhline(-thr, color="#d62728", ls="--", lw=1.2)
    ax.scatter(s2[s2.jump == 1].ts_hkt, s2[s2.jump == 1].LM, s=45,
               color="#ff7f0e", edgecolor="black", lw=.4, zorder=3,
               label="detected jump")
    ax.set_title("Lee-Mykland statistic | %s" % lbl, fontsize=10)
    ax.set_xlabel("time (HKT)"); ax.set_ylabel("$L(i)$")
    ax.legend(); ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(JD, "lee_mykland_statistic.png"), dpi=130)
    plt.close(fig)

    # ---------------- FIGURE 3: zoomed examples ----------------------------
    pick = pd.concat([J.nlargest(5, "LM"), J.nsmallest(5, "LM")])
    for i, (_, ev) in enumerate(pick.iterrows()):
        w = P[(P.asset_id == ev.asset_id) & (P.minute >= ev.minute - 30)
              & (P.minute <= ev.minute + 30)].sort_values("minute")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(w.ts_hkt, w.mid, lw=1.3, marker="o", ms=2.5, color="#1f77b4")
        mk = "^" if ev.jump_direction == 1 else "v"
        col = "#2ca02c" if ev.jump_direction == 1 else "#d62728"
        ax.scatter([ev.ts_hkt], [ev.mid], marker=mk, s=200, color=col,
                   edgecolor="black", zorder=4)
        ax.axvline(ev.ts_hkt, color=col, ls="--", lw=.9, alpha=.6)
        ax.set_title("%s\n%s | dir %+d | r=%+.4f | LM=%+.2f (thr %.2f)"
                     % (str(ev.series)[:64],
                        ev.ts_hkt.strftime("%Y-%m-%d %H:%M HKT"),
                        int(ev.jump_direction), ev.ret, ev.LM, thr), fontsize=8)
        ax.set_xlabel("time (HKT)"); ax.set_ylabel("mid"); ax.grid(alpha=.25)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(os.path.join(JD, "examples", "jump_%03d.png" % (i + 1)), dpi=120)
        plt.close(fig)

    # ---------------- FIGURE 4: jump frequency -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
    bins = np.arange(0, 24 * 60 + 15, 15)
    tot, _ = np.histogram(P.loc[tested, "tod_min"], bins=bins)
    jm, _ = np.histogram(J.tod_min, bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_tod = np.where(tot > 0, jm / tot, np.nan)
    ctr = (bins[:-1] + 7.5) / 60
    axes[0].bar(ctr, rate_tod * 100, width=.22, color="#1f77b4")
    axes[0].set_xlabel("hour of day (HKT) — continuous market, no open/close")
    axes[0].set_ylabel("jump rate (% of tested minutes)")
    axes[0].set_title("Jump rate by time of day"); axes[0].grid(alpha=.25)
    perday = J.groupby("date").size()
    axes[1].bar(range(len(perday)), perday.values, color="#ff7f0e")
    axes[1].set_xticks(range(len(perday)))
    axes[1].set_xticklabels(perday.index, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("jumps"); axes[1].set_title("Jumps per day")
    axes[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(JD, "jump_frequency.png"), dpi=130)
    plt.close(fig)
    log("figures + csv written to %s" % JD)


if __name__ == "__main__":
    sys.exit(main())
