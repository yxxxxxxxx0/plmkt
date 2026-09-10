"""Directional CNN as a TAKER signal: can it pick the side, net of real costs?

This is a different question from the rest of the jump study, and the change
is economic rather than architectural.

    maker:  pnl = half_spread - path_max|excursion|     magnitude is everything
    taker:  pnl = side * signed_move - costs            sign is everything

A maker is picked off whichever way the market runs, so |move| is exactly what
it needs and direction is irrelevant. A taker must choose a side, crosses the
spread to get in, and pays a fee per taker leg. Note the mirror image on the
label: `signed_ticks` -- the ENDPOINT move at t+H -- is the right measure for
a taker holding to the horizon, whereas the maker case needs the path maximum
because a resting quote is picked off on the way.

Costs, from the verified schedule in polymarket_fees.py:

    fee = feeRate * p * (1 - p) per share, per taker leg; sports feeRate 0.05

At a 0.425 mid that is 1.22 ticks a leg, 2.13 ticks for a taker round trip,
against a 1.25-tick spread cost. An earlier version of this analysis read
`taker_fee_rate: 0.07` from a config as "7% of notional", which put the fee at
2.98 ticks against a 1.78-tick mean move and made every taker strategy look
arithmetically impossible. It is not; the real fee is roughly half that and
falls toward zero at extreme prices, because of the p(1-p) term.

Three controls this file enforces, because dropping any of them manufactures
an edge:

  1. NO CONDITIONING ON THE OUTCOME. Flat rows -- 86.4% of tight-book rows --
     are scored, not filtered. A taker who enters one pays the round trip for
     nothing. Only the FIT excludes them, since they carry no directional
     information.
  2. EXECUTABLE PRICES. Entry crosses the spread, exit crosses it again.
     The mid is not tradeable in a wide book, so only spread <= --max-spread
     ticks is used.
  3. THRESHOLD CHOSEN ON VALIDATION. The trade/no-trade edge cutoff is
     selected on val and applied once to test, and every figure carries a
     moving-block bootstrap CI.

The unresolved caveat, stated up front: the profitable confidence deciles are
dominated by very large moves, which look like a touch level VANISHING rather
than a price walking. A level that vanishes was either consumed (you were
racing a taker for it) or cancelled (there was nothing to trade against).
books_2026-08-30.jsonl has zero trade prints and cannot tell those apart, and
FINDINGS.md already measured 45.3% of one-tick touch moves as reprices. So a
positive number here is an upper bound until the same run is repeated on a
recording that has prints.

Usage:
    python taker_signal.py --epochs 4
    python taker_signal.py --skip-deep          # baselines only
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from collapse_cnn import NB, depth_image
from jump_model import frame_offsets
from jump_split import GRID_MS, HAND, JD, TICK, load, save_row, subsample
from polymarket_fees import round_trip_cost_ticks, taker_fee_ticks


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
def prepare(F, horizon_s, category, max_spread):
    """Attach exit prices, costs and the 3-class direction label."""
    h = int(horizon_s * 1000 / GRID_MS)
    F["spread_fwd"] = (F.groupby("sid", sort=False)["spread_ticks"]
                       .shift(-h).fillna(F["spread_ticks"]))
    signed = F.signed_ticks.to_numpy(np.float64)
    mid = F.mid.to_numpy(np.float64)
    mid_f = mid + signed * TICK
    sp = F.spread_ticks.to_numpy(np.float64)
    sp_f = F.spread_fwd.to_numpy(np.float64)

    F["cost_taker"] = round_trip_cost_ticks(mid, mid_f, sp, sp_f, TICK,
                                            category, exit_as_maker=False)
    F["cost_maker_exit"] = round_trip_cost_ticks(mid, mid_f, sp, sp_f, TICK,
                                                 category, exit_as_maker=True)
    F["tight"] = sp <= max_spread
    return F, h


def label3(signed, k):
    """0 = down, 1 = flat, 2 = up, at a k-tick threshold."""
    y = np.ones(len(signed), np.int64)
    y[signed <= -k] = 0
    y[signed >= k] = 2
    return y


def touch_backing(T, chunk=1_000_000):
    """min(dollars at best bid, dollars at best ask) for every row.

    The filter that matters. Directional P&L was previously concentrated
    entirely in touches holding under $50 -- trivially cancellable and too
    thin to take size in -- so a signal that survives only there is not a
    signal. Requiring both sides to be well backed is the cheapest available
    proxy for "this level would have to be traded, not merely pulled".
    """
    n = len(T)
    out = np.empty(n, np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        w = T[np.arange(s, e)]
        b = np.expm1(w[:, 1, 0].astype(np.float64))
        a = np.expm1(w[:, 3, 0].astype(np.float64))
        out[s:e] = np.minimum(b, a)
    return out


# --------------------------------------------------------------------------- #
def taker_pnl(edge, F, idx, tau, maker_exit=False):
    """P&L per OPPORTUNITY in ticks at trade threshold `tau`.

    `edge` is P(up) - P(down). Trade long above +tau, short below -tau, else
    stand aside and earn exactly zero. Standing aside is free, which is why
    per-opportunity is the only denominator that compares strategies fairly --
    per-trade flatters a strategy that trades rarely.
    """
    signed = F.signed_ticks.to_numpy(np.float64)[idx]
    cost = F["cost_maker_exit" if maker_exit else "cost_taker"].to_numpy(np.float64)[idx]
    side = np.where(edge > tau, 1.0, np.where(edge < -tau, -1.0, 0.0))
    traded = side != 0
    pnl = side * signed - traded * cost
    return pnl, side, traded


def sweep(edge, F, idx, taus, maker_exit=False):
    rows = []
    for t in taus:
        pnl, side, traded = taker_pnl(edge, F, idx, t, maker_exit)
        s = F.signed_ticks.to_numpy()[idx]
        mv = traded & (s != 0)
        rows.append(dict(
            tau=float(t), trade_frac=float(traded.mean()),
            pnl_per_opp=float(pnl.mean()),
            pnl_per_trade=float(pnl[traded].mean()) if traded.any() else 0.0,
            hit=float(((side > 0) == (s > 0))[mv].mean()) if mv.any() else np.nan,
            flat_frac=float((s[traded] == 0).mean()) if traded.any() else np.nan))
    return pd.DataFrame(rows)


def boot_ci(edge, F, idx, tau, maker_exit=False, n_boot=400, block=1000, seed=0):
    pnl, _, _ = taker_pnl(edge, F, idx, tau, maker_exit)
    n = len(pnl)
    block = min(block, max(1, n // 4))
    nb = max(1, n // block)
    starts = np.arange(0, max(1, n - block + 1))
    rng = np.random.default_rng(seed)
    off = np.arange(block)
    b = np.empty(n_boot)
    for i in range(n_boot):
        b[i] = pnl[(rng.choice(starts, nb)[:, None] + off).ravel()].mean()
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def report(name, e_val, e_te, F, va, te, out, extra=None):
    taus = np.round(np.arange(0.0, 0.96, 0.05), 3)
    Rv = sweep(e_val, F, va, taus)
    best = Rv.loc[Rv.pnl_per_opp.idxmax()]
    tau = float(best.tau)

    Rt = sweep(e_te, F, te, [tau]).iloc[0]
    lo, hi = boot_ci(e_te, F, te, tau)
    mlo, mhi = boot_ci(e_te, F, te, tau, maker_exit=True)
    pnl_mk, _, _ = taker_pnl(e_te, F, te, tau, maker_exit=True)

    s = F.signed_ticks.to_numpy()[te]
    moved = s != 0
    row = dict(model=name, tau=tau,
               trade_frac=float(Rt.trade_frac), hit=float(Rt.hit),
               pnl_per_opp=float(Rt.pnl_per_opp),
               pnl_lo=lo, pnl_hi=hi, profitable=bool(lo > 0),
               pnl_per_trade=float(Rt.pnl_per_trade),
               pnl_maker_exit=float(pnl_mk.mean()),
               pnl_mk_lo=mlo, pnl_mk_hi=mhi,
               sign_auc=float(roc_auc_score((s[moved] > 0).astype(int),
                                            e_te[moved])),
               n_test=len(te), val_pnl_per_opp=float(best.pnl_per_opp))
    if extra:
        row.update(extra)
    out.append(row)
    save_row(row, os.path.join(JD, "taker_comparison.csv"))
    np.save(os.path.join(JD, f"takeredge_{name}.npy"), e_te.astype(np.float32))
    # the index too -- without it the saved edges cannot be re-sliced by
    # market type or book depth later without reconstructing the split
    np.save(os.path.join(JD, f"takeridx_{name}.npy"), te)
    return row


# --------------------------------------------------------------------------- #
def run_baselines(F, tr, va, te, out, k):
    import lightgbm as lgb
    cols = [c for c in HAND if c in F.columns]
    s_tr = F.signed_ticks.to_numpy()[tr]
    fit = s_tr != 0                       # flat rows carry no direction

    log("  momentum control (mean reversion of dmid_25)...")
    # dmid_25 scored AUC 0.374 on sign, i.e. recent up-moves precede
    # down-moves. Inverted and rank-scaled it is the cheapest possible
    # directional signal, and any model must beat it to be worth anything.
    v = -F.dmid_25.to_numpy(np.float64)
    r = lambda x: 2.0 * pd.Series(x).rank(pct=True).to_numpy() - 1.0
    report("momentum_dmid25", r(v[va]), r(v[te]), F, va, te, out)

    log("  directional GBM on hand features...")
    g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                           min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=5.0, verbose=-1, n_jobs=10,
                           random_state=0)
    X = lambda i: np.nan_to_num(F.loc[i, cols].to_numpy(np.float32))
    g.fit(X(tr)[fit], (s_tr[fit] > 0).astype(int))
    ev = 2.0 * g.predict_proba(X(va))[:, 1] - 1.0
    et = 2.0 * g.predict_proba(X(te))[:, 1] - 1.0
    report("gbm_direction", ev, et, F, va, te, out)


# --------------------------------------------------------------------------- #
def build_dircnn(L, in_ch=2, d=64, n_scalar=3):
    """Depth-image trunk with a three-way head, at module level.

    Defined here rather than inside run_deep so a saved checkpoint can
    actually be reloaded -- the earlier closure-scoped model classes made
    their own .pt files unusable for follow-up analysis, which is how the
    first CNN-Transformer result got lost.

    The trunk is identical to cnn_depth_raw deliberately: that reached 0.930
    AUC on magnitude, so a poor directional result implicates the target
    rather than the representation.
    """
    import torch
    import torch.nn as nn

    class DirCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(in_ch, 24, (3, 5), padding=(1, 2)), nn.GELU(),
                nn.Conv2d(24, 32, (3, 5), padding=(1, 2)), nn.GELU(),
                nn.AdaptiveAvgPool2d((L, 4)))
            self.proj = nn.Linear(32 * 4 + n_scalar, d)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            enc = nn.TransformerEncoderLayer(d, 4, d * 4, batch_first=True,
                                             dropout=0.1, activation="gelu")
            self.tr = nn.TransformerEncoder(enc, 2)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))

        def forward(self, img, s):
            h = self.cnn(img).permute(0, 2, 1, 3).flatten(2)
            h = self.proj(torch.cat([h, s], dim=-1)) + self.pos[:, :h.shape[1]]
            return self.head(self.tr(h)[:, -1])

    return DirCNN()


def batcher(T, F, OFF, STR, y3):
    """Builds the (depth image, scalars, label) batch for a row index array."""
    import torch
    MID = F.mid.to_numpy(np.float32)

    def make_batch(j):
        J = j[:, None] + OFF[None, :]
        X = T[J]
        M = MID[J]
        rel = (M - MID[j][:, None]) / TICK
        img = depth_image(X, rel)
        dmid = (M - MID[J - STR[None, :]]) / TICK
        spr = X[:, :, 2, 0] - X[:, :, 0, 0]
        S = np.stack([np.clip(dmid, -30, 30) / 5.0,
                      np.clip(rel, -30, 30) / 5.0,
                      np.clip(spr, 0, 30) / 5.0], axis=-1)
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.from_numpy(S.astype(np.float32)),
                torch.from_numpy(y3[j]))
    return make_batch


def run_deep(T, F, tr, va, te, out, OFF, STR, k, epochs=4, batch=384,
             lr=1e-3, seed=0, d_model=64):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 4))
    L = len(OFF)
    y3 = label3(F.signed_ticks.to_numpy(np.float64), k)
    make_batch = batcher(T, F, OFF, STR, y3)
    model = build_dircnn(L, d=d_model)
    nparam = sum(p.numel() for p in model.parameters())
    cnt = np.bincount(y3[tr], minlength=3).astype(np.float64)
    log(f"  DirCNN: {nparam:,} params; train class mix "
        f"down {cnt[0]/cnt.sum():.3f} flat {cnt[1]/cnt.sum():.3f} "
        f"up {cnt[2]/cnt.sum():.3f}")
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)), dtype=torch.float32)
    w = w / w.mean()
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * (len(tr) // batch + 1),
        pct_start=0.25)
    rng = np.random.default_rng(seed)
    hist = []

    for ep in range(epochs):
        model.train(); tot = n = 0; t0 = time.time()
        order = rng.permutation(len(tr))
        for s0 in range(0, len(order), batch):
            img, sc, yb = make_batch(tr[order[s0:s0 + batch]])
            opt.zero_grad()
            loss = lossf(model(img, sc), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except ValueError:
                pass
            tot += loss.item() * len(yb); n += len(yb)
        # Validation loss per epoch. Without this the train curve alone says
        # nothing about overfitting -- a loss falling 0.395 -> 0.218 is
        # equally consistent with learning and with memorising. Measured on a
        # fixed subsample so the cost stays small.
        model.eval()
        vsub = subsample(va, 40_000, seed=1)
        vt = vn = 0
        with torch.no_grad():
            for s0 in range(0, len(vsub), 1024):
                img, sc, yb = make_batch(vsub[s0:s0 + 1024])
                vt += lossf(model(img, sc), yb).item() * len(yb); vn += len(yb)
        hist.append(dict(epoch=ep + 1, train=tot / n, val=vt / max(vn, 1)))
        log(f"    epoch {ep+1}/{epochs} train {tot/n:.4f} "
            f"val {vt/max(vn,1):.4f} "
            f"gap {vt/max(vn,1) - tot/n:+.4f} ({time.time()-t0:.0f}s)")

    def edge(idx):
        model.eval()
        outp = []
        with torch.no_grad():
            for s0 in range(0, len(idx), 1024):
                img, sc, _ = make_batch(idx[s0:s0 + 1024])
                q = torch.softmax(model(img, sc), dim=-1).numpy()
                outp.append(q[:, 2] - q[:, 0])          # P(up) - P(down)
        return np.concatenate(outp)

    torch.save(dict(state_dict=model.state_dict(), nparam=nparam,
                    loss_history=hist,
                    config=dict(L=L, NB=NB, k=k, epochs=epochs, lr=lr,
                                batch=batch, seed=seed)),
               os.path.join(JD, "cnn_direction.pt"))
    report("cnn_direction", edge(va), edge(te), F, va, te, out,
           extra=dict(nparam=nparam))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-types", nargs="*",
                    default=["moneyline", "total", "spread"])
    ap.add_argument("--fine", type=int, default=24)
    ap.add_argument("--coarse", type=int, default=24)
    ap.add_argument("--coarse-stride", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=300_000)
    ap.add_argument("--max-val", type=int, default=150_000)
    ap.add_argument("--max-test", type=int, default=150_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--horizon-s", type=float, default=5.0)
    ap.add_argument("--label-ticks", type=int, default=2,
                    help="a move of at least this many ticks counts as "
                         "directional; smaller ones are labelled flat")
    ap.add_argument("--max-spread", type=float, default=2.0)
    ap.add_argument("--min-backing", type=float, default=0.0,
                    help="require at least this many dollars at BOTH touches. "
                         "0 disables. The previous run's entire edge sat in "
                         "sub-$50 touches, which are cancellable and untakeable")
    ap.add_argument("--category", default="sports")
    ap.add_argument("--max-book-age-s", type=float, default=5.0)
    ap.add_argument("--skip-deep", action="store_true")
    a = ap.parse_args()

    pd.set_option("display.width", 250)
    OFF, STR, _, lookback = frame_offsets(a.fine, a.coarse, a.coarse_stride)
    F, T, tr, va, te = load(a.market_types, lookback=lookback, log=log,
                            max_book_age_s=(None if a.max_book_age_s < 0
                                            else a.max_book_age_s))
    F, h = prepare(F, a.horizon_s, a.category, a.max_spread)

    tight = F.tight.to_numpy()
    tr, va, te = tr[tight[tr]], va[tight[va]], te[tight[te]]
    log(f"tight books only (spread <= {a.max_spread:g} ticks): "
        f"train {len(tr):,} val {len(va):,} test {len(te):,}")

    if a.min_backing > 0:
        log(f"computing touch backing...")
        F["backing"] = touch_backing(T)
        bk = F.backing.to_numpy()
        tr, va, te = (tr[bk[tr] >= a.min_backing],
                      va[bk[va] >= a.min_backing],
                      te[bk[te] >= a.min_backing])
        log(f"  touch backed >= ${a.min_backing:,.0f} on both sides: "
            f"train {len(tr):,} val {len(va):,} test {len(te):,}")
        sg = F.signed_ticks.to_numpy()
        for nm, ii in (("train", tr), ("val", va), ("test", te)):
            log(f"    {nm}: P(|move| >= {a.label_ticks}) = "
                f"{(np.abs(sg[ii]) >= a.label_ticks).mean():.4f}")
    else:
        F["backing"] = np.nan

    tr = subsample(tr, a.max_train)
    va = subsample(va, a.max_val)
    te = subsample(te, a.max_test)

    s_te = F.signed_ticks.to_numpy()[te]
    log(f"  test: P(flat) {(s_te == 0).mean():.3f}  "
        f"mean |move| {np.abs(s_te).mean():.2f} ticks  "
        f"taker round trip {F.cost_taker.to_numpy()[te].mean():.2f} ticks "
        f"(fee {taker_fee_ticks(F.mid.to_numpy()[te], TICK, a.category).mean():.2f}/leg)")

    out = []
    run_baselines(F, tr, va, te, out, a.label_ticks)
    if not a.skip_deep:
        log("directional depth-image CNN:")
        run_deep(T, F, tr, va, te, out, OFF, STR, a.label_ticks,
                 epochs=a.epochs)

    R = pd.DataFrame(out).sort_values("pnl_per_opp", ascending=False)
    print("\n" + "=" * 132)
    print("TAKER SIGNAL  (P&L in TICKS per opportunity; 1 tick = $0.01/share)")
    print("  tau            trade threshold on P(up)-P(down), chosen on VAL")
    print("  pnl_per_opp    net of spread crossed twice AND both taker fees")
    print("  [lo, hi]       95% moving-block bootstrap CI")
    print("  pnl_maker_exit optimistic: passive exit, no exit fee (unverified)")
    print("=" * 132)
    cols = ["model", "sign_auc", "tau", "trade_frac", "hit", "pnl_per_opp",
            "pnl_lo", "pnl_hi", "profitable", "pnl_per_trade",
            "pnl_maker_exit", "val_pnl_per_opp"]
    print(R[[c for c in cols if c in R.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    win = R[R.profitable]
    print(f"\n  {len(win)} of {len(R)} model(s) have a CI strictly above zero "
          f"on full taker costs.")

    # Per-slice, because the previous headline of +2.84 ticks/opp turned out to
    # be 100% spread markets and sub-$50 touches. A pooled number cannot show
    # that; this can.
    best = R.iloc[0].model
    ep = os.path.join(JD, f"takeredge_{best}.npy")
    if os.path.exists(ep):
        e = np.load(ep)
        tau = float(R.iloc[0].tau)
        s_te = F.signed_ticks.to_numpy(np.float64)[te]
        moved = s_te != 0
        mt = F.mt.to_numpy().astype(str)[te]
        bk = F.backing.to_numpy()[te]
        print(f"\n  '{best}' broken out (tau={tau:.2f}) -- does it survive "
              f"slicing?")
        print(f"  {'slice':>20} {'n':>8} {'signAUC':>9} {'trade%':>8} "
              f"{'hit':>7} {'pnl/opp':>9} {'CI lo':>9} {'CI hi':>9}")

        def row(lbl, s):
            if s.sum() < 1500 or (moved & s).sum() < 300:
                return
            mv = moved[s]
            if len(np.unique(s_te[s][mv] > 0)) < 2:
                return
            pn, sd, td = taker_pnl(e[s], F, te[s], tau)
            lo, hi = boot_ci(e[s], F, te[s], tau)
            h = (((sd > 0) == (s_te[s] > 0))[td & mv].mean()
                 if (td & mv).any() else np.nan)
            print(f"  {lbl:>20} {s.sum():>8,} "
                  f"{roc_auc_score((s_te[s][mv] > 0).astype(int), e[s][mv]):>9.4f} "
                  f"{td.mean():>8.3f} {h:>7.4f} {pn.mean():>9.4f} "
                  f"{lo:>9.4f} {hi:>9.4f}")

        for m in ["moneyline", "total", "spread"]:
            row(m, mt == m)
        for lo_, hi_ in [(0, 200), (200, 1000), (1000, 5000), (5000, np.inf)]:
            lbl = (f"${lo_:,.0f}-{hi_:,.0f}" if np.isfinite(hi_)
                   else f">${lo_:,.0f}")
            row(lbl, (bk >= lo_) & (bk < hi_))
        print("  A signal that is real should hold in more than one row here.")
    print("  Caveat that bounds all of this: the profitable rows are "
          "concentrated in very large moves, which look like a touch level "
          "vanishing. Without trade prints this data cannot say whether those "
          "levels were consumed (tradeable) or cancelled (not).")

    with open(os.path.join(JD, "taker_config.json"), "w") as fh:
        json.dump(dict(vars(a), lookback=lookback, n_train=len(tr),
                       n_val=len(va), n_test=len(te),
                       ts=time.strftime("%Y-%m-%dT%H:%M:%S")), fh, indent=2)


if __name__ == "__main__":
    main()
