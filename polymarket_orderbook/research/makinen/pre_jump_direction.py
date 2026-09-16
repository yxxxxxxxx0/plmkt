"""Is the pre-jump book asymmetric, i.e. does it say WHICH WAY the price will go?

pre_jump_seconds.py established that near-touch liquidity drains and the spread
walks out in the ~6 seconds before a large jump. That is a MAGNITUDE signal: it
was measured on |move| with up and down jumps pooled, so it cannot support a
directional trade.

This asks the separate question. Split the clean large jumps by sign and test,
at each tau, whether the BID side and the ASK side behave differently:

    up-jumps   vs  down-jumps      (Cliff's delta between the two populations)

and specifically the bid/ask asymmetry features, which are what a directional
rule would have to key on:

    l1_imbalance        (bid_L1 - ask_L1) / (bid_L1 + ask_L1)
    near_imbalance      same, using dollars within 2 ticks
    depth_imbalance     same, using total book depth
    bid_minus_ask_drain log change in bid depth minus log change in ask depth

A large |delta| means the book leans, before the move, in the direction the
price is about to take. A delta near zero means the book tells you a jump is
coming but not which way -- which would make a directional trade impossible
from this signal alone.

It then runs the honest end-to-end check: at a chosen lead time, use ONLY the
book asymmetry to call the sign, and score it against the 50% coin flip.

    python pre_jump_direction.py --min-move 0.20 --lead-s 5
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
RES = os.path.join(ROOT, "results", "makinen", "pre_jump_direction")
GRID_MS = 200


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
    ap.add_argument("--pre-s", type=float, default=15.0)
    ap.add_argument("--lead-s", type=float, default=5.0)
    ap.add_argument("--quiet-min", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    P = pd.read_parquet(os.path.join(CACHE, "panel_labelled.parquet"))
    jm = {aid: set(g.minute[g.jump == 1]) for aid, g in P.groupby("asset_id")}
    big = P[(P.jump == 1) & (P.ret.abs() >= a.min_move)].copy()
    big["clean"] = [not any((m - k) in jm[aid] for k in range(1, a.quiet_min + 1))
                    for aid, m in zip(big.asset_id, big.minute)]
    B = big[big.clean]
    print("clean large jumps: %d  (up %d / down %d)"
          % (len(B), int((B.ret > 0).sum()), int((B.ret < 0).sum())))

    offs = np.arange(-int(a.pre_s * 1000 / GRID_MS), 1)
    taus = offs * GRID_MS / 1000.0
    KEYS = ["l1_imbalance", "near_imbalance", "depth_imbalance",
            "l1_bid_usd", "l1_ask_usd", "log_bid_usd_within_2t",
            "log_ask_usd_within_2t", "log_bid_depth", "log_ask_depth"]
    up = {k: [[] for _ in offs] for k in KEYS}
    dn = {k: [[] for _ in offs] for k in KEYS}
    lead_rows = []

    for sess, Bs in B.groupby("session"):
        fp = os.path.join(JD, "feat_%s_trimmed.parquet" % sess)
        tp = os.path.join(JD, "lob_%s_trimmed.npy" % sess)
        if not (os.path.exists(fp) and os.path.exists(tp)):
            continue
        F = pq.read_table(fp, columns=["ts", "sid", "series", "mid",
                                       "spread_ticks"]).to_pandas()
        F["row"] = np.arange(len(F), dtype=np.int64)
        F["asset_id"] = F.series.astype(str).str.rsplit("|", n=1).str[-1]
        T = np.load(tp, mmap_mode="r")
        sid = F.sid.to_numpy(); mid = F.mid.to_numpy(np.float64); n = len(F)
        spt = F.spread_ticks.to_numpy()
        by = {k: g for k, g in F.groupby("asset_id", sort=False)}

        for _, ev in Bs.iterrows():
            g = by.get(ev.asset_id)
            if g is None:
                continue
            gr = g[(g.ts >= ev.minute * 60000) & (g.ts < (ev.minute + 1) * 60000)]
            if len(gr) < 5:
                continue
            gm = gr.mid.to_numpy(np.float64)
            st = np.abs(np.diff(gm))
            if not len(st) or not np.isfinite(st).any():
                continue
            anchor = int(gr.row.to_numpy()[int(np.nanargmax(st))])
            rows = anchor + offs
            ok = (rows >= 0) & (rows < n)
            r2 = np.clip(rows, 0, n - 1)
            ok &= (sid[r2] == sid[anchor])
            d = FE.decode(np.asarray(T[r2]), mid[r2])
            S = pd.DataFrame(FE.state_features(d, spread_ticks=spt[r2]))
            eps = 1e-9
            lb, la = S.l1_bid_usd.to_numpy(), S.l1_ask_usd.to_numpy()
            nb = np.expm1(S.log_bid_usd_within_2t.to_numpy())
            na = np.expm1(S.log_ask_usd_within_2t.to_numpy())
            db = np.expm1(S.log_bid_depth.to_numpy())
            da = np.expm1(S.log_ask_depth.to_numpy())
            vals = {
                "l1_imbalance": (lb - la) / (lb + la + eps),
                "near_imbalance": (nb - na) / (nb + na + eps),
                "depth_imbalance": (db - da) / (db + da + eps),
                "l1_bid_usd": lb, "l1_ask_usd": la,
                "log_bid_usd_within_2t": S.log_bid_usd_within_2t.to_numpy(),
                "log_ask_usd_within_2t": S.log_ask_usd_within_2t.to_numpy(),
                "log_bid_depth": S.log_bid_depth.to_numpy(),
                "log_ask_depth": S.log_ask_depth.to_numpy(),
            }
            store = up if ev.ret > 0 else dn
            for k in KEYS:
                v = vals[k].copy(); v[~ok] = np.nan
                for j in range(len(offs)):
                    store[k][j].append(v[j])
            # the lead-time snapshot used for the directional test
            j_lead = int(round((-a.lead_s * 1000 / GRID_MS) - offs[0]))
            if 0 <= j_lead < len(offs) and ok[j_lead]:
                lead_rows.append(dict(up=int(ev.ret > 0),
                                      **{k: float(vals[k][j_lead]) for k in KEYS}))
        del F, T

    rows = []
    for k in KEYS:
        for j, t in enumerate(taus):
            u = np.array(up[k][j], float); w = np.array(dn[k][j], float)
            rows.append(dict(feature=k, tau_s=t,
                             median_up=np.nanmedian(u) if len(u) else np.nan,
                             median_down=np.nanmedian(w) if len(w) else np.nan,
                             cliffs_up_vs_down=cliffs(u, w),
                             n_up=int(np.isfinite(u).sum()),
                             n_down=int(np.isfinite(w).sum())))
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(RES, "direction_effects.csv"), index=False)
    key = [-15.0, -10.0, -5.0, -3.0, -2.0, -1.0, -0.4, 0.0]
    piv = E[E.tau_s.isin(key)].pivot_table(index="feature", columns="tau_s",
                                           values="cliffs_up_vs_down")
    piv = piv.reindex(piv.abs().max(axis=1).sort_values(ascending=False).index)
    print("\n=== Cliff's delta, UP-jumps vs DOWN-jumps (does the book lean?) ===")
    print("    (|d| < 0.15 negligible, 0.15-0.33 small, > 0.33 large)")
    print(piv.round(3).to_string())

    # ---- the honest directional test at the chosen lead ----
    L = pd.DataFrame(lead_rows)
    if len(L) > 60:
        print("\n=== directional call using ONLY book asymmetry at tau = -%.0fs "
              "(n=%d) ===" % (a.lead_s, len(L)))
        base = max(L.up.mean(), 1 - L.up.mean())
        print("  majority-class baseline: %.4f" % base)
        from sklearn.metrics import roc_auc_score
        for k in ("l1_imbalance", "near_imbalance", "depth_imbalance"):
            v = L[k].to_numpy()
            m = np.isfinite(v)
            if m.sum() < 60 or L.up[m].nunique() < 2:
                continue
            auc = roc_auc_score(L.up[m], v[m])
            acc = max((( v[m] > 0).astype(int) == L.up[m]).mean(),
                      ((v[m] <= 0).astype(int) == L.up[m]).mean())
            print("  %-18s ROC-AUC %.4f | best sign-rule accuracy %.4f"
                  % (k, auc, acc))
        L.to_csv(os.path.join(RES, "lead_snapshot.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, k in zip(axes, ["l1_imbalance", "near_imbalance", "depth_imbalance"]):
        s = E[E.feature == k].sort_values("tau_s")
        ax.plot(s.tau_s, s.median_up, color="#2ca02c", lw=1.8, label="before UP jump")
        ax.plot(s.tau_s, s.median_down, color="#d62728", lw=1.8, label="before DOWN jump")
        ax.axhline(0, color="k", lw=.8, ls=":")
        ax.axvline(-a.lead_s, color="#888888", lw=1.0, ls="--",
                   label="lead = -%.0fs" % a.lead_s)
        ax.set_title(k); ax.set_xlabel("seconds before the move")
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.suptitle("Does the book lean in the direction of the coming jump? "
                 "(clean large jumps, |move| >= %.2f)" % a.min_move, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(os.path.join(RES, "direction_asymmetry.png"), dpi=130)
    plt.close(fig)
    print("\nwrote", RES)


if __name__ == "__main__":
    sys.exit(main())
