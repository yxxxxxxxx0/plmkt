"""Two questions about the directional signal: WHEN does the move happen, and
is the model overfitting?

PART A -- timing inside the horizon.
    The signal fires at t and the move completes by t+5s, but a strategy needs
    to know WHERE inside that window. If most of the move is already done at
    t+0.5s, this is a latency race rather than a prediction, and it cannot be
    executed on infrastructure whose delivery-lag dispersion is 4.6s. If the
    move builds steadily to t+5s, there are seconds to act.

PART B -- overfitting, measured three ways, because they fail differently.

    1. Train-vs-test gap. The saved checkpoint is scored on TRAIN rows it was
       fit on and on TEST rows it never saw. A large AUC gap is classic
       parameter overfitting. Cheap, and it needs no retraining.

    2. Series concentration. 79 of the 85 series appear in BOTH train and
       test -- the split is temporal, not by market, so the model sees the
       same games at different times and can fit game-specific idiosyncrasy.
       There is no market-level holdout anywhere in this study. If the edge
       comes from a handful of series, it will not transfer to new games
       whatever the pooled numbers say.

    3. Regime drift. Base rates run 0.045 train -> 0.081 val -> 0.096 test,
       so val and test resemble each other more than either resembles train.
       That makes "val approx test" weak evidence: it shows the model carries
       from one late-day block to the next, not that it carries to a new day.

    python diag_timing_fit.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jump_model import frame_offsets
from jump_split import GRID_MS, JD, TICK, load, subsample
from polymarket_fees import taker_fee_ticks
from taker_signal import (batcher, build_dircnn, label3, prepare, taker_pnl,
                          touch_backing)


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="sports")
    ap.add_argument("--max-spread", type=float, default=2.0)
    ap.add_argument("--min-backing", type=float, default=200.0)
    ap.add_argument("--label-ticks", type=int, default=4)
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    OFF, STR, _, lookback = frame_offsets(24, 24, 8)
    F, T, tr, va, te = load(["moneyline", "total", "spread"],
                            lookback=lookback, log=log, max_book_age_s=5.0)
    F, h = prepare(F, 5.0, a.category, a.max_spread)

    # rebuild the exact split taker_signal.main() used
    tight = F.tight.to_numpy()
    tr, va, te = tr[tight[tr]], va[tight[va]], te[tight[te]]
    log("computing touch backing...")
    bk_all = touch_backing(T)
    F["backing"] = bk_all
    tr = tr[bk_all[tr] >= a.min_backing]
    va = va[bk_all[va] >= a.min_backing]
    te = te[bk_all[te] >= a.min_backing]
    tr = subsample(tr, 300_000); va = subsample(va, 150_000)
    te = subsample(te, 150_000)

    e = np.load(os.path.join(JD, "takeredge_cnn_direction.npy"))
    idx = np.load(os.path.join(JD, "takeridx_cnn_direction.npy"))
    assert len(idx) == len(te) and (idx == te).all(), \
        "reconstructed test split does not match the saved one"
    tcsv = os.path.join(JD, "taker_comparison.csv")
    tau = float(pd.read_csv(tcsv).query("model == 'cnn_direction'").iloc[0].tau)
    log(f"  loaded signal: {len(idx):,} test rows, tau {tau:.2f}")

    # ---- PART A: timing -------------------------------------------------- #
    log(f"\n{'='*104}\nPART A -- WHEN inside the 5s window does the move "
        f"happen?\n{'='*104}")
    steps = {"0.2s": 1, "0.6s": 3, "1.0s": 5, "2.0s": 10, "3.0s": 15,
             "5.0s": h}
    g = F.groupby("sid", sort=False)
    mid = F.mid.to_numpy(np.float64)
    side = np.where(e > tau, 1.0, np.where(e < -tau, -1.0, 0.0))
    traded = side != 0
    i = idx[traded]
    sd = side[traded]
    mt = F.mt.to_numpy().astype(str)[i]
    bkt = bk_all[i]

    prof = {}
    for lbl, lag in steps.items():
        m = g["mid"].shift(-lag).to_numpy(np.float64)
        okk = (g["ts"].shift(-lag).to_numpy() - F.ts.to_numpy()) == lag * GRID_MS
        d = np.where(okk, (m - mid) / TICK, np.nan)
        prof[lbl] = sd * d[i]                      # signed in the trade's favour

    log(f"  mean SIGNED move in the trade's direction, ticks "
        f"({traded.sum():,} traded rows)")
    log(f"  {'slice':>18} {'n':>8} " + "".join(f"{k:>9}" for k in steps))
    log(f"  {'':>18} {'':>8} " + "".join(f"{'':>9}" for k in steps))

    def prow(lbl, s):
        if s.sum() < 300:
            return
        vals = [np.nanmean(prof[k][s]) for k in steps]
        log(f"  {lbl:>18} {s.sum():>8,} " + "".join(f"{v:>9.2f}" for v in vals))
        return vals

    all_v = prow("ALL traded", np.ones(len(i), bool))
    for m_ in ["moneyline", "total", "spread"]:
        prow(m_, mt == m_)
    for lo, hi in [(200, 1000), (1000, 5000), (5000, np.inf)]:
        lbl = f"${lo:,.0f}-{hi:,.0f}" if np.isfinite(hi) else f">${lo:,.0f}"
        prow(lbl, (bkt >= lo) & (bkt < hi))

    if all_v:
        tot = all_v[-1]
        log(f"\n  fraction of the 5s move already realised, ALL traded:")
        log("    " + "  ".join(f"{k} {v/tot:.0%}" for k, v in
                               zip(steps, all_v)))
        log(f"\n  If a large share is realised by 1.0s, the signal is "
            f"detecting a move already underway and execution has to win a "
            f"race. This file's delivery-lag dispersion is 4.6s sustained, "
            f"so that race is not winnable with the stack that produced it.")

    # ---- PART B: overfitting --------------------------------------------- #
    log(f"\n{'='*104}\nPART B -- IS IT OVERFITTING?\n{'='*104}")
    ck = os.path.join(JD, "cnn_direction.pt")
    if not os.path.exists(ck):
        log("  no checkpoint; skipping the train-vs-test gap")
    else:
        import torch
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = blob.get("config", {})
        log(f"  checkpoint config: {cfg}")
        hist = blob.get("loss_history", [])
        if hist and isinstance(hist[0], dict):
            log(f"  per-epoch train/val loss:")
            for r in hist:
                log(f"    epoch {r['epoch']}: train {r['train']:.4f}  "
                    f"val {r['val']:.4f}  gap {r['val']-r['train']:+.4f}")
        else:
            log(f"  train loss history {['%.4f' % v for v in hist]} "
                f"(this run predates val-loss tracking)")

        model = build_dircnn(len(OFF), d=cfg.get("d_model", 64))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        y3 = label3(F.signed_ticks.to_numpy(np.float64), a.label_ticks)
        mb = batcher(T, F, OFF, STR, y3)

        def edge_of(ii):
            outp = []
            with torch.no_grad():
                for s0 in range(0, len(ii), 1024):
                    img, sc, _ = mb(ii[s0:s0 + 1024])
                    q = torch.softmax(model(img, sc), dim=-1).numpy()
                    outp.append(q[:, 2] - q[:, 0])
            return np.concatenate(outp)

        log(f"\n  scoring the checkpoint on each block "
            f"(train rows it was FIT on vs test rows it never saw)")
        log(f"  {'block':>8} {'n':>8} {'signAUC':>9} {'trade%':>8} "
            f"{'hit':>7} {'pnl/opp':>9}")
        rows = {}
        for nm, ii in (("train", subsample(tr, 40_000, seed=2)),
                       ("val", subsample(va, 40_000, seed=2)),
                       ("test", subsample(te, 40_000, seed=2))):
            ee = edge_of(ii)
            s = F.signed_ticks.to_numpy(np.float64)[ii]
            mv = s != 0
            pn, sdd, td = taker_pnl(ee, F, ii, tau)
            auc = (roc_auc_score((s[mv] > 0).astype(int), ee[mv])
                   if mv.sum() > 100 and len(np.unique(s[mv] > 0)) > 1
                   else np.nan)
            hit = (((sdd > 0) == (s > 0))[td & mv].mean()
                   if (td & mv).any() else np.nan)
            rows[nm] = (auc, pn.mean())
            log(f"  {nm:>8} {len(ii):>8,} {auc:>9.4f} {td.mean():>8.3f} "
                f"{hit:>7.4f} {pn.mean():>9.4f}")
        if "train" in rows and "test" in rows:
            log(f"\n  train->test AUC gap: "
                f"{rows['train'][0] - rows['test'][0]:+.4f}   "
                f"P&L gap: {rows['train'][1] - rows['test'][1]:+.4f} ticks")
            log(f"  A large positive gap is parameter overfitting. A gap near "
                f"zero rules that out but says nothing about the two "
                f"structural risks below.")

    # ---- series concentration -------------------------------------------- #
    log(f"\n  SERIES CONCENTRATION -- there is no market-level holdout, so "
        f"the edge could be a few games")
    sid = F.sid.to_numpy()[i]
    pnl, _, _ = taker_pnl(e, F, idx, tau)
    pnl_tr = pnl[traded]
    df = pd.DataFrame(dict(sid=sid, pnl=pnl_tr))
    agg = df.groupby("sid").pnl.agg(["size", "mean", "sum"]).sort_values(
        "sum", ascending=False)
    tot = agg["sum"].sum()
    log(f"  {len(agg)} series carry trades; total {tot:,.0f} ticks")
    for k in (1, 3, 5, 10):
        if len(agg) >= k:
            log(f"    top {k:>2} series = {agg['sum'].head(k).sum()/tot:>6.1%} "
                f"of all P&L")
    log(f"  {(agg['sum'] > 0).sum()} of {len(agg)} series are profitable")
    log(f"\n  top 5 series by P&L:")
    log(agg.head(5).to_string())
    log(f"\n  If a small number of series carry most of the P&L, the pooled "
        f"CI is far too narrow -- it treats correlated rows from one game as "
        f"independent evidence.")


if __name__ == "__main__":
    main()
