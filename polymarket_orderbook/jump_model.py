"""Does a CNN-Transformer on raw order books beat simple baselines -- economically?

The research question is not "can a neural net predict jumps" but whether what
it extracts from order-book dynamics is worth more than hand-crafted order-flow
features fed to logistic regression or a GBM. The deciding metric is not AUC:

    pnl(t) = half_spread(t) - max_{s<=H} |mid(t+s) - mid(t)|

Earn the half-spread when the market sits still, lose the excursion when it
runs. So jump probability is directly actionable -- pull the quote when it is
high -- and a model is useful only if gating on its output raises realised P&L.
That is strictly harder than AUC, because a model can rank well and still add
nothing over what quoting was worth anyway.

History of this comparison, since the numbers moved a lot
--------------------------------------------------------
The first version reported the CNN-Transformer at AUC 0.874 / +0.00291 per
quote, behind a single rank-transformed rv_150 feature, and concluded that deep
models add nothing here. Four defects were found afterwards, three of them in
the data rather than the model:

1. **The mid path was invisible to the CNN.** Levels are expressed relative to
   each row's own mid, so a window of frames is translation invariant and the
   mid's trajectory is absent by construction. rv_150 and rv_25 were 65% of the
   GBM's gain and the CNN could not compute either. Fixed: per-frame scalars
   (step change in mid, displacement vs. the prediction point, spread) join
   each frame's CNN embedding before the Transformer. `--no-mid-path` ablates
   them, reproducing the original input at identical compute.

2. **The receptive field was 4.7x too short.** seq_len=32 at a 200ms grid is
   6.4s; rv_150 spans 30s. Fixed with a dual-resolution window -- coarse frames
   at stride 8 plus fine frames at stride 1, 43.2s total, with a learned scale
   embedding. Parameters stay at ~127k so the comparison isolates the
   representation, not capacity.

3. **The grid back-filled future books into past rows** (jump_data.py), which
   made `stale` a near-deterministic predictor as an artifact. Fixed and now
   covered by test_jump_data.py.

4. **The label and the P&L measured different windows** -- the label used a
   path maximum over [t, t+H-1] while the P&L used the endpoint at t+H, so the
   "max" did not even contain the point it was compared against. Both now use
   the path maximum over [t, t+H].

Also: there was no validation set, and the gating threshold was chosen by
argmax P&L on test. See jump_split.py.

Usage:
    python jump_model.py --epochs 4
    python jump_model.py --no-mid-path --tag _nomid --skip-trivial
    python jump_model.py --pnl-weight
    python jump_model.py --target econ --pnl-weight
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from jump_split import (BASE, HAND, JD, TICK, econ_eval, evaluate, load,
                        pnl_weights, show, subsample)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
def run_trivial(F, va, te, out):
    """Single-feature gates, the control that decides the research question.

    If gating on realised volatility alone captures most of the economic gain,
    neither the GBM nor the deep model is contributing much: the strategy is
    just "do not quote when the market is moving", and no learned
    representation is needed for that.

    Ranks are computed within each block so the threshold means the same
    quantile in both -- these gates have no fitted parameters, so there is
    nothing for validation to leak.
    """
    rank = lambda v: pd.Series(v).rank(pct=True).to_numpy()
    for col, name in (("rv_25", "gate_rv25_only"),
                      ("rv_150", "gate_rv150_only"),
                      ("spread_ticks", "gate_spread_only"),
                      ("stale", "gate_staleness_only"),
                      ("book_age_ms", "gate_bookage_only")):
        if col not in F.columns:
            continue
        v = F[col].to_numpy(np.float64)
        evaluate(name, rank(v[va]), rank(v[te]), F, va, te, out)


def run_baselines(F, tr, va, te, out, suffix="", weights=None):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    cols = [c for c in HAND if c in F.columns]
    Xtr = F.loc[tr, cols].to_numpy(np.float32)
    Xva = F.loc[va, cols].to_numpy(np.float32)
    Xte = F.loc[te, cols].to_numpy(np.float32)
    ytr = F.y.to_numpy()[tr]

    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr, Xva, Xte = imp.transform(Xtr), imp.transform(Xva), imp.transform(Xte)

    log("  logistic regression...")
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.1, max_iter=2000)
    lr.fit(sc.transform(Xtr), ytr, sample_weight=weights)
    pv = lr.predict_proba(sc.transform(Xva))[:, 1]
    pt = lr.predict_proba(sc.transform(Xte))[:, 1]
    evaluate(f"logreg_hand{suffix}", pv, pt, F, va, te, out)

    log("  lightgbm...")
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                             num_leaves=63, min_child_samples=200,
                             subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=5.0,
                             verbose=-1, n_jobs=10, random_state=0)
    gbm.fit(Xtr, ytr, sample_weight=weights)
    gv = gbm.predict_proba(Xva)[:, 1]
    gt = gbm.predict_proba(Xte)[:, 1]
    evaluate(f"lightgbm_hand{suffix}", gv, gt, F, va, te, out)

    imp_df = pd.DataFrame({"feature": cols,
                           "gain": gbm.booster_.feature_importance("gain")})
    imp_df["gain"] /= max(imp_df.gain.sum(), 1e-12)
    imp_df = imp_df.sort_values("gain", ascending=False)
    imp_df.to_csv(os.path.join(JD, f"gbm_importance{suffix}.csv"), index=False)
    return imp_df


# --------------------------------------------------------------------------- #
def frame_offsets(fine, coarse, stride):
    """Dual-resolution window, oldest first, ending at the prediction point.

    `coarse` frames at `stride` cover the long horizon rv_150 needs
    (24 x 8 x 200ms = 38.4s); `fine` frames at stride 1 keep full resolution
    where it matters most. Returns the offsets, the per-frame stride needed to
    difference the mid, a scale id for the embedding, and the total lookback.
    """
    coarse_off = np.arange(-coarse * stride, 0, stride)
    fine_off = np.arange(-(fine - 1), 1)
    off = np.concatenate([coarse_off, fine_off]).astype(np.int64)
    strides = np.concatenate([np.full(coarse, stride),
                              np.full(fine, 1)]).astype(np.int64)
    scale = np.concatenate([np.zeros(coarse), np.ones(fine)]).astype(np.int64)
    return off, strides, scale, int(-(off - strides).min())


def build_multiscale(L, scale_ids, k=10, c=4, n_scalar=3, d=64, nhead=4,
                     layers=2):
    """CNN over price levels, Transformer over time, mid path as scalars.

    At module level so a checkpoint can be reloaded. The earlier version was
    scoped inside run_deep, which is how the first CNN-Transformer result
    became unrecoverable -- the weights were never saved, and even once they
    were there was no class to load them into.
    """
    import torch
    import torch.nn as nn

    scale_t = torch.as_tensor(scale_ids, dtype=torch.long)

    class MultiScaleCNNTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(c, 32, 3, padding=1), nn.GELU(),
                nn.Conv1d(32, 64, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool1d(4),
            )
            # the mid path joins here, per frame, before any mixing over time
            self.proj = nn.Linear(64 * 4 + n_scalar, d)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            self.scale = nn.Parameter(torch.zeros(2, d))
            enc = nn.TransformerEncoderLayer(d, nhead, d * 4, batch_first=True,
                                             dropout=0.1, activation="gelu")
            self.tr = nn.TransformerEncoder(enc, layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        def forward(self, x, s):
            B, Ln, C, K = x.shape
            h = self.cnn(x.reshape(B * Ln, C, K)).reshape(B * Ln, -1)
            h = torch.cat([h, s.reshape(B * Ln, -1)], dim=-1)
            h = self.proj(h).reshape(B, Ln, -1)
            h = h + self.pos[:, :Ln] + self.scale[scale_t[:Ln]]
            return self.head(self.tr(h)[:, -1]).squeeze(-1)

    return MultiScaleCNNTransformer()


def run_deep(T, F, tr, va, te, out, fine=24, coarse=24, stride=8, epochs=4,
             d_model=64, batch=512, lr=1e-3, seed=0, suffix="",
             weights=None, mid_path=True):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 4))

    OFF, STR, SCALE, _ = frame_offsets(fine, coarse, stride)
    L = len(OFF)
    Tm = T                    # a TensorView: indexing it reads from the memmap
    MID = F.mid.to_numpy(np.float32)
    y_all = F.y.to_numpy().astype(np.float32)
    W = np.ones(len(F), np.float32)
    if weights is not None:
        W[tr] = weights.astype(np.float32)
    scale_t = torch.from_numpy(SCALE)

    def make_batch(j):
        """Vectorised gather -- per-item indexing of a multi-GB array is the
        bottleneck otherwise, and num_workers>0 on Windows would pickle it."""
        J = j[:, None] + OFF[None, :]
        X = Tm[J]
        M = MID[J]
        dmid = (M - MID[J - STR[None, :]]) / TICK      # step change
        rel = (M - MID[j][:, None]) / TICK             # path vs. now
        spr = X[:, :, 2, 0] - X[:, :, 0, 0]            # spread, from level 0
        S = np.stack([np.clip(dmid, -30, 30) / 5.0,
                      np.clip(rel, -30, 30) / 5.0,
                      np.clip(spr, 0, 30) / 5.0], axis=-1)
        # The ablation: identical architecture, receptive field, compute and
        # seed, but the mid path zeroed -- the input the original comparison
        # actually ran on. The gap between the two runs is what the
        # representation fix is worth, with nothing else varying.
        if not mid_path:
            S = np.zeros_like(S)
        return (torch.from_numpy(np.ascontiguousarray(X)),
                torch.from_numpy(S.astype(np.float32)),
                torch.from_numpy(y_all[j]), torch.from_numpy(W[j]))

    def batches(idx, shuffle, rng, bs):
        order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
        for s in range(0, len(order), bs):
            yield make_batch(idx[order[s:s + bs]])

    model = build_multiscale(L, SCALE, d=d_model)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"  CNN-Transformer: {nparam:,} params, L={L} "
        f"(coarse {coarse}x{stride} + fine {fine}), "
        f"span {(coarse * stride + fine) * 0.2:.1f}s, mid_path={mid_path}")

    pos_w = float((1 - y_all[tr].mean()) / max(y_all[tr].mean(), 1e-6))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w),
                                 reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * (len(tr) // batch + 1),
        pct_start=0.25)
    rng = np.random.default_rng(seed)
    hist = []

    for ep in range(epochs):
        model.train(); tot = n = 0; t0 = time.time()
        for xb, sb, yb, wb in batches(tr, True, rng, batch):
            opt.zero_grad()
            loss = (lossf(model(xb, sb), yb) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except ValueError:
                pass
            tot += loss.item() * len(yb); n += len(yb)
        hist.append(tot / n)
        log(f"    epoch {ep+1}/{epochs} loss {tot/n:.4f} ({time.time()-t0:.0f}s)")

    def predict(idx):
        model.eval()
        ps = []
        with torch.no_grad():
            for xb, sb, _, _ in batches(idx, False, rng, 1024):
                ps.append(torch.sigmoid(model(xb, sb)).numpy())
        return np.concatenate(ps)

    pv, pt = predict(va), predict(te)
    name = f"cnn_transformer{suffix}"
    ckpt = os.path.join(JD, f"{name}.pt")
    torch.save(dict(state_dict=model.state_dict(), nparam=nparam,
                    loss_history=hist,
                    config=dict(fine=fine, coarse=coarse, stride=stride, L=L,
                                d_model=d_model, epochs=epochs, lr=lr,
                                batch=batch, seed=seed, mid_path=mid_path,
                                pnl_weighted=weights is not None)), ckpt)
    log(f"  checkpoint -> {os.path.relpath(ckpt, BASE)}")
    evaluate(name, pv, pt, F, va, te, out, extra=dict(nparam=nparam))
    return pv, pt


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-types", nargs="*",
                    default=["moneyline", "total", "spread"])
    ap.add_argument("--fine", type=int, default=24)
    ap.add_argument("--coarse", type=int, default=24)
    ap.add_argument("--coarse-stride", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=400_000)
    ap.add_argument("--max-val", type=int, default=200_000)
    ap.add_argument("--max-test", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--target", choices=["jump", "econ"], default="jump")
    ap.add_argument("--pnl-weight", action="store_true",
                    help="weight every sample by |half_spread - excursion|")
    ap.add_argument("--no-mid-path", action="store_true",
                    help="ablation: zero the per-frame mid-path scalars, "
                         "reproducing the original (defective) input")
    ap.add_argument("--skip-deep", action="store_true")
    ap.add_argument("--skip-trivial", action="store_true")
    ap.add_argument("--max-book-age-s", type=float, default=5.0,
                    help="a row may be a prediction point only if its carried "
                         "book is at most this stale; pass -1 to disable")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    pd.set_option("display.width", 240)
    _, _, _, lookback = frame_offsets(a.fine, a.coarse, a.coarse_stride)
    F, T, tr, va, te = load(a.market_types, lookback=lookback,
                            target=a.target, log=log,
                            max_book_age_s=(None if a.max_book_age_s < 0
                                            else a.max_book_age_s))
    tr = subsample(tr, a.max_train)
    va = subsample(va, a.max_val)
    te = subsample(te, a.max_test)
    log(f"using train {len(tr):,} val {len(va):,} test {len(te):,}")

    suffix = a.tag or ("" if (a.target == "jump" and not a.pnl_weight)
                       else f"_{a.target}{'_w' if a.pnl_weight else ''}")
    w = pnl_weights(F, tr) if a.pnl_weight else None
    if w is not None:
        log(f"  pnl weights: mean {w.mean():.3f} p50 {np.median(w):.3f} "
            f"p99 {np.percentile(w, 99):.3f} max {w.max():.3f}")

    out = []
    if not a.skip_trivial:
        log("trivial single-feature gates:")
        run_trivial(F, va, te, out)
    log("baselines on hand-crafted features:")
    imp = run_baselines(F, tr, va, te, out, suffix=suffix, weights=w)
    print("\n=== top hand-crafted features by GBM gain ===")
    print(imp.head(12).to_string(index=False, float_format=lambda v: f"{v:,.5f}"))

    if not a.skip_deep:
        log(f"deep model"
            f"{' (mid path ABLATED)' if a.no_mid_path else ''}:")
        run_deep(T, F, tr, va, te, out, fine=a.fine, coarse=a.coarse,
                 stride=a.coarse_stride, epochs=a.epochs, suffix=suffix,
                 weights=w, mid_path=not a.no_mid_path)

    show(out)
    with open(os.path.join(JD, f"run_config{suffix or '_base'}.json"), "w") as fh:
        json.dump(dict(vars(a), lookback=lookback, suffix=suffix,
                       n_train=len(tr), n_val=len(va), n_test=len(te),
                       ts=time.strftime("%Y-%m-%dT%H:%M:%S")), fh, indent=2)
    full = pd.read_csv(os.path.join(JD, "model_comparison.csv"))
    print(f"\nfull table ({len(full)} rows incl. earlier runs) -> "
          f"data/jump/model_comparison.csv")


if __name__ == "__main__":
    main()
