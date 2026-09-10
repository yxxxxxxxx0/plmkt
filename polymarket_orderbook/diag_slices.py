"""Two falsification tests for the jump result, from saved predictions.

Both ask whether the headline AUC is measuring what it appears to measure.
Neither needs retraining -- the completed runs saved their test predictions
and indices, which is the whole reason for saving them.

TEST 1 -- market-type mixing.
    Base rates run 0.021 on moneyline to 0.423 on spread markets, and spread
    is roughly a third of all rows. Any classifier gets free discrimination
    from telling those apart, and that discrimination is worth nothing: you
    cannot trade "this is a spread market". If the AUC holds WITHIN a single
    market type it was never the explanation. This is the cheapest possible
    way to be wrong.

TEST 2 -- is it value movement or quote withdrawal?
    Two very different events move a mid: someone trades and the price walks,
    or the best quote is pulled and the mid re-centres on what was behind it.
    Only the first is tradeable -- you cannot sell into a bid that is
    cancelled rather than consumed.

    A touch holding $8 is trivially withdrawable; one holding $5,000 is not.
    So if the edge lives only where the touch is thin, the model is predicting
    quote maintenance and the result is mechanical rather than economic. If it
    survives on well-backed touches, it is closer to real value discovery.

    This is a proxy for the trade-print test, not a substitute: books_2026-08-30
    has no prints, so consumption and cancellation remain formally
    indistinguishable here. Depth is the best available stand-in.

    python diag_slices.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jump_model import frame_offsets
from jump_split import JD, TICK, bootstrap_pnl_opp, econ_eval, load, subsample
from taker_signal import prepare, taker_pnl


def log(m):
    print(m, flush=True)


def load_saved(name, kind="preds"):
    p = os.path.join(JD, f"{kind}_{name}.npy")
    i = os.path.join(JD, f"{'testidx' if kind == 'preds' else 'takeridx'}_{name}.npy")
    if not (os.path.exists(p) and os.path.exists(i)):
        return None, None
    return np.load(p), np.load(i)


def touch_dollars(T, idx, chunk=50_000):
    """Dollars resting at the best bid and best ask for these rows.

    Tensor channels are [bid_px_ticks, bid_size_log$, ask_px_ticks,
    ask_size_log$], so level 0 of the size channels is log1p(dollars).
    """
    b = np.empty(len(idx), np.float64)
    a = np.empty(len(idx), np.float64)
    for s in range(0, len(idx), chunk):
        e = min(s + chunk, len(idx))
        w = T[idx[s:e]]
        b[s:e] = np.expm1(w[:, 1, 0].astype(np.float64))
        a[s:e] = np.expm1(w[:, 3, 0].astype(np.float64))
    return b, a


# --------------------------------------------------------------------------- #
def slice_magnitude(F, T, name="cnn_depth_raw"):
    p, idx = load_saved(name)
    if p is None:
        log(f"  no saved predictions for {name}; skipping")
        return
    log(f"\n{'='*100}\nMAGNITUDE model '{name}'  (n={len(idx):,})\n{'='*100}")
    y = F.y.to_numpy()[idx]
    R, base = econ_eval(p, F, idx)
    best = R.loc[R.pnl_per_opportunity.idxmax()]
    th = float(best.threshold)
    log(f"  pooled: AUC {roc_auc_score(y, p):.4f}   base rate {y.mean():.4f}   "
        f"always-quote {base:+.5f}   gated {best.pnl_per_opportunity:+.5f}")

    # ---- TEST 1: within market type -------------------------------------- #
    mt = F.mt.to_numpy().astype(str)[idx]
    log(f"\n  TEST 1 -- AUC within each market type "
        f"(pooled AUC gets free credit for telling them apart)")
    log(f"  {'market':>12} {'n':>9} {'base':>7} {'AUC':>8} "
        f"{'always':>9} {'gated':>9} {'gain':>9}")
    for m in ["moneyline", "total", "spread"]:
        s = mt == m
        if s.sum() < 2000 or len(np.unique(y[s])) < 2:
            continue
        Rm, bm = econ_eval(p[s], F, idx[s], thresholds=np.array([th]))
        log(f"  {m:>12} {s.sum():>9,} {y[s].mean():>7.3f} "
            f"{roc_auc_score(y[s], p[s]):>8.4f} {bm:>9.4f} "
            f"{Rm.iloc[0].pnl_per_opportunity:>9.4f} "
            f"{Rm.iloc[0].pnl_per_opportunity - bm:>9.4f}")

    # ---- TEST 2: by how well-backed the touch is ------------------------- #
    tb, ta = touch_dollars(T, idx)
    backing = np.minimum(tb, ta)          # a solid touch is thick on BOTH sides
    log(f"\n  TEST 2 -- AUC and P&L by touch backing "
        f"min($ at best bid, $ at best ask)")
    log(f"  a thin touch is trivially withdrawable; a thick one is not")
    log(f"  {'backing $':>14} {'n':>9} {'base':>7} {'AUC':>8} "
        f"{'always':>9} {'gated':>9} {'CI lo':>9} {'CI hi':>9}")
    edges = [0, 50, 200, 1000, 5000, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (backing >= lo) & (backing < hi)
        if s.sum() < 2000 or len(np.unique(y[s])) < 2:
            continue
        Rm, bm = econ_eval(p[s], F, idx[s], thresholds=np.array([th]))
        clo, chi = bootstrap_pnl_opp(p[s], F, idx[s], th)
        lbl = f"{lo:,.0f}-{hi:,.0f}" if np.isfinite(hi) else f">{lo:,.0f}"
        log(f"  {lbl:>14} {s.sum():>9,} {y[s].mean():>7.3f} "
            f"{roc_auc_score(y[s], p[s]):>8.4f} {bm:>9.4f} "
            f"{Rm.iloc[0].pnl_per_opportunity:>9.4f} {clo:>9.4f} {chi:>9.4f}")


# --------------------------------------------------------------------------- #
def slice_direction(F, T, te_recon, name="cnn_direction"):
    e, idx = load_saved(name, kind="takeredge")
    if e is None:
        p = os.path.join(JD, f"takeredge_{name}.npy")
        if not os.path.exists(p):
            log(f"  no saved edges for {name}; skipping")
            return
        # That run predated saving the index, so rebuild it the way
        # taker_signal.main() did: tight filter then subsample(seed=0), both
        # deterministic. The length check is the guard -- if the split ever
        # changes shape this refuses rather than silently misaligning.
        e = np.load(p)
        idx = te_recon
        if len(e) != len(idx):
            log(f"  reconstructed index length {len(idx):,} != saved edges "
                f"{len(e):,}; refusing to align. Re-run taker_signal.py "
                f"(it now saves the index).")
            return
        log(f"\n  reconstructed the test index for {name} "
            f"({len(idx):,} rows, lengths agree)")
    log(f"\n{'='*100}\nDIRECTION model '{name}'  (n={len(idx):,})\n{'='*100}")
    s_te = F.signed_ticks.to_numpy(np.float64)[idx]
    moved = s_te != 0
    tau = 0.80
    tcsv = os.path.join(JD, "taker_comparison.csv")
    if os.path.exists(tcsv):
        t = pd.read_csv(tcsv)
        r = t[t.model == name]
        if len(r):
            tau = float(r.iloc[0].tau)
    log(f"  using tau = {tau:.2f} (chosen on validation in the original run)")

    pnl, side, traded = taker_pnl(e, F, idx, tau)
    hit = ((side > 0) == (s_te > 0))[traded & moved].mean()
    log(f"  pooled: sign AUC "
        f"{roc_auc_score((s_te[moved] > 0).astype(int), e[moved]):.4f}   "
        f"trade frac {traded.mean():.3f}   hit {hit:.4f}   "
        f"pnl/opp {pnl.mean():+.4f} ticks")

    mt = F.mt.to_numpy().astype(str)[idx]
    log(f"\n  TEST 1 -- direction within each market type")
    log(f"  {'market':>12} {'n':>9} {'signAUC':>9} {'trade%':>8} "
        f"{'hit':>7} {'pnl/opp':>9}")
    for m in ["moneyline", "total", "spread"]:
        s = mt == m
        if s.sum() < 2000:
            continue
        mv = moved[s]
        if mv.sum() < 500 or len(np.unique((s_te[s][mv] > 0))) < 2:
            continue
        pn, sd, td = taker_pnl(e[s], F, idx[s], tau)
        h = ((sd > 0) == (s_te[s] > 0))[td & mv].mean() if (td & mv).any() else np.nan
        log(f"  {m:>12} {s.sum():>9,} "
            f"{roc_auc_score((s_te[s][mv] > 0).astype(int), e[s][mv]):>9.4f} "
            f"{td.mean():>8.3f} {h:>7.4f} {pn.mean():>9.4f}")

    tb, ta = touch_dollars(T, idx)
    backing = np.minimum(tb, ta)
    log(f"\n  TEST 2 -- direction by touch backing")
    log(f"  {'backing $':>14} {'n':>9} {'signAUC':>9} {'trade%':>8} "
        f"{'hit':>7} {'pnl/opp':>9} {'meanmove':>9}")
    edges = [0, 50, 200, 1000, 5000, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (backing >= lo) & (backing < hi)
        if s.sum() < 2000:
            continue
        mv = moved[s]
        if mv.sum() < 500 or len(np.unique((s_te[s][mv] > 0))) < 2:
            continue
        pn, sd, td = taker_pnl(e[s], F, idx[s], tau)
        h = ((sd > 0) == (s_te[s] > 0))[td & mv].mean() if (td & mv).any() else np.nan
        lbl = f"{lo:,.0f}-{hi:,.0f}" if np.isfinite(hi) else f">{lo:,.0f}"
        log(f"  {lbl:>14} {s.sum():>9,} "
            f"{roc_auc_score((s_te[s][mv] > 0).astype(int), e[s][mv]):>9.4f} "
            f"{td.mean():>8.3f} {h:>7.4f} {pn.mean():>9.4f} "
            f"{np.abs(s_te[s]).mean():>9.2f}")


# --------------------------------------------------------------------------- #
def main():
    pd.set_option("display.width", 220)
    _, _, _, lookback = frame_offsets(24, 24, 8)
    F, T, tr, va, te = load(["moneyline", "total", "spread"],
                            lookback=lookback, log=log, max_book_age_s=5.0)
    F, _ = prepare(F, 5.0, "sports", 2.0)
    slice_magnitude(F, T)
    # same construction as taker_signal.main(), for the index reconstruction
    tight = F.tight.to_numpy()
    te_recon = subsample(te[tight[te]], 150_000)
    slice_direction(F, T, te_recon)


if __name__ == "__main__":
    main()
