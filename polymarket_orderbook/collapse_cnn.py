"""Can a CNN find the liquidity-collapse -> price-jump relationship by itself?

Two separate questions, and they need different experiments.

Q1: "Can a CNN directly find the relationship?"
    A representation question. The earlier CNN could not, and the reason was
    not capacity -- its input made the answer unreachable. Levels were
    expressed relative to each frame's own mid, so a level at index i in frame
    t and index i in frame t+1 are DIFFERENT PRICES whenever the mid moved.
    Differencing them over time measures nothing in particular, and
    "withdrawal" is exactly a difference over time at a fixed price.

    So this file re-expresses the window on a common price axis anchored at
    the mid at the prediction instant: a (time x tick) depth image, where
    column p is the same price in every frame. On that axis a plain temporal
    convolution CAN express "size at this price fell", and the question
    becomes a fair one.

Q2: "Will the CNN figure out what contributes most, on its own?"
    An attribution question, and it is answered by measurement rather than
    assertion. Three things are done to every trained model:

      channel permutation  shuffle one input channel across the batch and
                           watch AUC and P&L fall -- the direct analogue of
                           the GBM's gain ranking, on the same scale
      price-band occlusion zero the touch / near / deep bands in turn, to see
                           WHERE in the book it is reading
      time occlusion       zero recent / mid / distant frames, to see the
                           horizon it actually uses vs. the 43.2s it is given

    And the decisive comparison is an ablation pair:

      cnn_depth_raw   depth image only. To use withdrawal it must construct
                      the temporal difference itself.
      cnn_depth_flow  the same, plus the per-price depth DELTA handed to it
                      precomputed, at matched parameter count.

    If `flow` beats `raw`, the CNN did not derive on its own a feature that is
    one subtraction away from its input -- which is the honest, quantitative
    answer to "will it figure it out itself?" If they tie, it did.

    `gbm_collapse` is the third leg: a GBM on explicit collapse features
    (two-sided withdrawal, touch-size change, asymmetry) computed from the
    same tensor. That is the "someone told it the answer" reference.

Background: liquidity_collapse.py found symmetric withdrawal activity at 0.924
AUC for detecting an impending repricing episode, but could not say WHICH WAY
the price would go, so it was logged as a maker risk-off filter rather than a
directional signal. Predicting |move| rather than sign is precisely what a
maker needs, so that is the target here.

Usage:
    python collapse_cnn.py --epochs 4
    python collapse_cnn.py --variants raw flow --epochs 4
    python collapse_cnn.py --skip-deep          # collapse features + GBM only
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from jump_split import (BASE, JD, TICK, evaluate, load, pnl_weights, save_row,
                        show, subsample)
from jump_model import frame_offsets

NB = 41           # tick bins in the depth image
HALF = NB // 2    # +/- 20 ticks around the reference mid

COLLAPSE = ["d_bid_1", "d_ask_1", "d_bid_5", "d_ask_5", "d_bid_25", "d_ask_25",
            "sym_wd_5", "sym_wd_25", "asym_wd_5", "d_touch_bid_1",
            "d_touch_ask_1", "touch_share_bid", "touch_share_ask"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
def add_collapse_features(F, T):
    """Explicit withdrawal features, computed from the tensor for the GBM leg.

    Derived here rather than in jump_data.py so the reference features and the
    CNN's input provably come from the same numbers -- if these were built in
    a separate pass, a discrepancy between them would be indistinguishable
    from the CNN failing to learn.
    """
    # tensor channels: 0 bid px (ticks from mid), 1 bid size (log1p $),
    #                  2 ask px, 3 ask size
    #
    # Read one channel at a time through the view's chunked reader and reduce
    # immediately. Slicing T[:, 1, :] materialises an 8.6M x 10 array per
    # channel, and with F.copy() on top that was ~4GB of avoidable peak.
    bch = T.channel(1)
    bid_usd = np.expm1(bch).sum(1, dtype=np.float64).astype(np.float32)
    tb = np.expm1(bch[:, 0])
    del bch
    ach = T.channel(3)
    ask_usd = np.expm1(ach).sum(1, dtype=np.float64).astype(np.float32)
    ta = np.expm1(ach[:, 0])
    del ach
    F["_lb"] = np.log1p(bid_usd)
    F["_la"] = np.log1p(ask_usd)
    F["_ltb"] = np.log1p(tb)
    F["_lta"] = np.log1p(ta)
    F["touch_share_bid"] = tb / np.maximum(bid_usd, 1e-9)
    F["touch_share_ask"] = ta / np.maximum(ask_usd, 1e-9)

    g = F.groupby("sid", sort=False)
    for w in (1, 5, 25):
        # log-ratio change in total displayed dollars: negative = withdrawal
        F[f"d_bid_{w}"] = F["_lb"] - g["_lb"].shift(w)
        F[f"d_ask_{w}"] = F["_la"] - g["_la"].shift(w)
    F["d_touch_bid_1"] = F["_ltb"] - g["_ltb"].shift(1)
    F["d_touch_ask_1"] = F["_lta"] - g["_lta"].shift(1)
    for w in (5, 25):
        # two-sided withdrawal: how much BOTH sides pulled at once. This is
        # the quantity liquidity_collapse.py found at 0.924 AUC.
        F[f"sym_wd_{w}"] = np.minimum(-F[f"d_bid_{w}"], -F[f"d_ask_{w}"]).clip(lower=0)
    F["asym_wd_5"] = F["d_bid_5"] - F["d_ask_5"]
    F.drop(columns=["_lb", "_la", "_ltb", "_lta"], inplace=True)
    for c in COLLAPSE:
        if c in F.columns:
            F[c] = F[c].astype(np.float32).fillna(0.0)
    return F


# --------------------------------------------------------------------------- #
def depth_image(X, rel):
    """(B,L,4,K) mid-centered levels -> (B,2,L,NB) depth on a COMMON price axis.

    `rel[b,l]` is the mid at frame l minus the mid at the prediction instant,
    in ticks. Adding it to each level's own mid-relative price puts every frame
    on one axis anchored at the prediction instant, so bin p is the same price
    throughout the window and a temporal difference means "size at this price
    changed". Without this step the CNN cannot express withdrawal at all.
    """
    B, L, _, K = X.shape
    b_idx = np.repeat(np.arange(B), L * K)
    l_idx = np.tile(np.repeat(np.arange(L), K), B)
    base = (b_idx * L + l_idx) * NB
    out = np.empty((B, 2, L, NB), np.float32)
    for s, (pch, sch) in enumerate(((0, 1), (2, 3))):
        off = X[:, :, pch, :] + rel[:, :, None]
        bins = np.clip(np.rint(off).astype(np.int64) + HALF, 0, NB - 1)
        # padded levels carry size 0, so they add nothing wherever they land
        img = np.bincount(base + bins.ravel(),
                          weights=X[:, :, sch, :].ravel().astype(np.float64),
                          minlength=B * L * NB)
        out[:, s] = img.reshape(B, L, NB).astype(np.float32)
    return out


def make_batcher(T, F, OFF, STR, variant, weights_row):
    Tm = T                    # a TensorView: indexing it reads from the memmap
    MID = F.mid.to_numpy(np.float32)
    y_all = F.y.to_numpy(np.float32)
    L = len(OFF)

    def make_batch(j):
        import torch
        J = j[:, None] + OFF[None, :]
        X = Tm[J]
        M = MID[J]
        rel = (M - MID[j][:, None]) / TICK
        img = depth_image(X, rel)                       # (B,2,L,NB)
        if variant == "flow":
            # the per-price delta, handed over precomputed. One subtraction
            # from `img` -- which is the point of the ablation.
            d = np.zeros_like(img)
            d[:, :, 1:] = img[:, :, 1:] - img[:, :, :-1]
            img = np.concatenate([img, d], axis=1)      # (B,4,L,NB)
        dmid = (M - MID[J - STR[None, :]]) / TICK
        spr = X[:, :, 2, 0] - X[:, :, 0, 0]
        S = np.stack([np.clip(dmid, -30, 30) / 5.0,
                      np.clip(rel, -30, 30) / 5.0,
                      np.clip(spr, 0, 30) / 5.0], axis=-1)
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.from_numpy(S.astype(np.float32)),
                torch.from_numpy(y_all[j]),
                torch.from_numpy(weights_row[j]))
    return make_batch, L


# --------------------------------------------------------------------------- #
def build_model(in_ch, L, d=64, n_scalar=3):
    import torch
    import torch.nn as nn

    class DepthImageCNN(nn.Module):
        """2D conv over (time x price), then a Transformer over time.

        The convolution is where withdrawal can be expressed: a 3-tall kernel
        spans three consecutive frames at one price, so "size here fell" is a
        single learnable filter. That was structurally impossible on the
        mid-relative level axis.
        """
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(in_ch, 24, (3, 5), padding=(1, 2)), nn.GELU(),
                nn.Conv2d(24, 32, (3, 5), padding=(1, 2)), nn.GELU(),
                nn.AdaptiveAvgPool2d((L, 4)),
            )
            self.proj = nn.Linear(32 * 4 + n_scalar, d)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            enc = nn.TransformerEncoderLayer(d, 4, d * 4, batch_first=True,
                                             dropout=0.1, activation="gelu")
            self.tr = nn.TransformerEncoder(enc, 2)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        def forward(self, img, s):
            h = self.cnn(img)                       # (B,32,L,4)
            h = h.permute(0, 2, 1, 3).flatten(2)    # (B,L,128)
            h = torch.cat([h, s], dim=-1)
            h = self.proj(h) + self.pos[:, :h.shape[1]]
            return self.head(self.tr(h)[:, -1]).squeeze(-1)

    return DepthImageCNN()


# --------------------------------------------------------------------------- #
def attribute(model, make_batch, idx, F, seed=0, max_n=40_000):
    """What the model actually uses -- measured, not asserted.

    Permutation and occlusion are applied to the INPUT, so the numbers are
    comparable across models and directly comparable to the GBM's gain
    ranking. Everything is relative to the intact model on the same rows.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    idx = subsample(idx, max_n, seed=seed)
    y = F.y.to_numpy()[idx]
    hs = F.half_spread.to_numpy()[idx] * TICK
    exc = F.excursion.to_numpy()[idx] * TICK

    def run(mut=None):
        model.eval()
        ps = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                img, sc, _, _ = make_batch(idx[s:s + 1024])
                if mut is not None:
                    img, sc = mut(img.clone(), sc.clone(), rng)
                ps.append(torch.sigmoid(model(img, sc)).numpy())
        return np.concatenate(ps)

    base_p = run()
    base_auc = roc_auc_score(y, base_p) if len(np.unique(y)) > 1 else np.nan
    # P&L at the model's median-split gate, a fixed reference for all mutations
    gate = np.quantile(base_p, 0.5)
    base_pnl = float((hs - exc)[base_p < gate].mean())

    rows = []

    def record(kind, name, p):
        a = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        q = p < gate
        pnl = float((hs - exc)[q].mean()) if q.any() else np.nan
        rows.append(dict(kind=kind, name=name, auc=a, auc_drop=base_auc - a,
                         pnl=pnl, pnl_drop=base_pnl - pnl,
                         quote_frac=float(q.mean())))

    nch = None
    img0, sc0, _, _ = make_batch(idx[:2])
    nch = img0.shape[1]

    # --- input channels ---------------------------------------------------- #
    names = (["bid_depth", "ask_depth"] if nch == 2 else
             ["bid_depth", "ask_depth", "d_bid_depth", "d_ask_depth"])
    for c, nm in enumerate(names):
        def mut(img, sc, rng, c=c):
            img[:, c] = img[torch.from_numpy(rng.permutation(len(img))), c]
            return img, sc
        record("channel", nm, run(mut))
    for c, nm in enumerate(["scalar_dmid", "scalar_rel_path", "scalar_spread"]):
        def mut(img, sc, rng, c=c):
            sc[:, :, c] = sc[torch.from_numpy(rng.permutation(len(sc))), :, c]
            return img, sc
        record("channel", nm, run(mut))

    # --- where in the book: three disjoint bands, one zeroed at a time ----- #
    bands = {
        "touch +/-1 tick": list(range(HALF - 1, HALF + 2)),
        "near 2-5 ticks": (list(range(HALF - 5, HALF - 1))
                           + list(range(HALF + 2, HALF + 6))),
        "deep 6-20 ticks": (list(range(0, HALF - 5))
                            + list(range(HALF + 6, NB))),
    }
    for nm, cols in bands.items():
        def mut(img, sc, rng, cols=cols):
            img[:, :, :, cols] = 0
            return img, sc
        record("price_band", f"zero {nm}", run(mut))

    # --- how far back ------------------------------------------------------ #
    L = img0.shape[2]
    for nm, sl in (("last 1s (5 fine frames)", slice(L - 5, L)),
                   ("fine window (4.8s)", slice(L - 24, L)),
                   ("coarse window (38.4s)", slice(0, L - 24))):
        def mut(img, sc, rng, sl=sl):
            img[:, :, sl] = 0
            sc[:, sl] = 0
            return img, sc
        record("time_band", f"zero {nm}", run(mut))

    R = pd.DataFrame(rows)
    R["base_auc"] = base_auc
    R["base_pnl"] = base_pnl
    return R


# --------------------------------------------------------------------------- #
def run_variant(variant, T, F, tr, va, te, OFF, STR, out, epochs, batch=384,
                lr=1e-3, seed=0, weights=None, suffix=""):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 4))

    W = np.ones(len(F), np.float32)
    if weights is not None:
        W[tr] = weights.astype(np.float32)
    make_batch, L = make_batcher(T, F, OFF, STR, variant, W)
    in_ch = 2 if variant == "raw" else 4
    model = build_model(in_ch, L)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"  cnn_depth_{variant}: {nparam:,} params, "
        f"input ({in_ch}, {L}, {NB}) = (channels, frames, ticks)")

    y_all = F.y.to_numpy(np.float32)
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
        order = rng.permutation(len(tr))
        for s in range(0, len(order), batch):
            img, sc, yb, wb = make_batch(tr[order[s:s + batch]])
            opt.zero_grad()
            loss = (lossf(model(img, sc), yb) * wb).mean()
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
            for s in range(0, len(idx), 1024):
                img, sc, _, _ = make_batch(idx[s:s + 1024])
                ps.append(torch.sigmoid(model(img, sc)).numpy())
        return np.concatenate(ps)

    name = f"cnn_depth_{variant}{suffix}"
    evaluate(name, predict(va), predict(te), F, va, te, out,
             extra=dict(nparam=nparam))
    torch.save(dict(state_dict=model.state_dict(), nparam=nparam,
                    loss_history=hist,
                    config=dict(variant=variant, in_ch=in_ch, L=L, NB=NB,
                                epochs=epochs, lr=lr, batch=batch, seed=seed)),
               os.path.join(JD, f"{name}.pt"))

    log(f"  attribution for {name}...")
    A = attribute(model, make_batch, te, F, seed=seed)
    A.insert(0, "model", name)
    ap = os.path.join(JD, f"attribution_{name}.csv")
    A.to_csv(ap, index=False)
    print(f"\n=== what {name} actually uses (test rows, "
          f"base AUC {A.base_auc.iloc[0]:.4f}) ===")
    print(A[["kind", "name", "auc", "auc_drop", "pnl_drop"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    return A


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-types", nargs="*",
                    default=["moneyline", "total", "spread"])
    ap.add_argument("--variants", nargs="*", default=["raw", "flow"],
                    choices=["raw", "flow"])
    ap.add_argument("--fine", type=int, default=24)
    ap.add_argument("--coarse", type=int, default=24)
    ap.add_argument("--coarse-stride", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=300_000)
    ap.add_argument("--max-val", type=int, default=150_000)
    ap.add_argument("--max-test", type=int, default=150_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--target", choices=["jump", "econ"], default="jump")
    ap.add_argument("--pnl-weight", action="store_true")
    ap.add_argument("--skip-deep", action="store_true")
    ap.add_argument("--max-book-age-s", type=float, default=5.0,
                    help="a row may be a prediction point only if its carried "
                         "book is at most this stale; pass -1 to disable")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    pd.set_option("display.width", 240)
    OFF, STR, _, lookback = frame_offsets(a.fine, a.coarse, a.coarse_stride)
    F, T, tr, va, te = load(a.market_types, lookback=lookback,
                            target=a.target, log=log,
                            max_book_age_s=(None if a.max_book_age_s < 0
                                            else a.max_book_age_s))

    log("computing explicit collapse features from the tensor...")
    F = add_collapse_features(F, T)
    have = [c for c in COLLAPSE if c in F.columns]
    log(f"  {len(have)} collapse features: {', '.join(have)}")

    tr = subsample(tr, a.max_train)
    va = subsample(va, a.max_val)
    te = subsample(te, a.max_test)
    log(f"using train {len(tr):,} val {len(va):,} test {len(te):,}")

    suffix = a.tag or ("" if (a.target == "jump" and not a.pnl_weight)
                       else f"_{a.target}{'_w' if a.pnl_weight else ''}")
    w = pnl_weights(F, tr) if a.pnl_weight else None
    out = []

    # --- reference leg: GBM told the answer explicitly -------------------- #
    import lightgbm as lgb
    log("gbm on explicit collapse features:")
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                             num_leaves=63, min_child_samples=200,
                             subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=5.0,
                             verbose=-1, n_jobs=10, random_state=0)
    gbm.fit(F.loc[tr, have].to_numpy(np.float32), F.y.to_numpy()[tr],
            sample_weight=w)
    evaluate(f"gbm_collapse{suffix}",
             gbm.predict_proba(F.loc[va, have].to_numpy(np.float32))[:, 1],
             gbm.predict_proba(F.loc[te, have].to_numpy(np.float32))[:, 1],
             F, va, te, out)
    gi = pd.DataFrame({"feature": have,
                       "gain": gbm.booster_.feature_importance("gain")})
    gi["gain"] /= max(gi.gain.sum(), 1e-12)
    gi = gi.sort_values("gain", ascending=False)
    gi.to_csv(os.path.join(JD, f"collapse_importance{suffix}.csv"), index=False)
    print("\n=== collapse features by GBM gain (the answer, handed over) ===")
    print(gi.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))

    # --- the ablation pair ------------------------------------------------ #
    if not a.skip_deep:
        for v in a.variants:
            log(f"depth-image CNN, variant '{v}':")
            run_variant(v, T, F, tr, va, te, OFF, STR, out, epochs=a.epochs,
                        weights=w, suffix=suffix)

    R = show(out, title="COLLAPSE -> JUMP")
    if {"cnn_depth_raw" + suffix, "cnn_depth_flow" + suffix} <= set(R.model):
        raw = R[R.model == "cnn_depth_raw" + suffix].iloc[0]
        flow = R[R.model == "cnn_depth_flow" + suffix].iloc[0]
        print(f"\n--- did the CNN derive withdrawal on its own? ---")
        print(f"  raw  (must construct the difference): AUC {raw.auc:.4f}  "
              f"pnl/opp {raw.pnl_per_opp:+.5f} "
              f"[{raw.pnl_opp_lo:+.5f}, {raw.pnl_opp_hi:+.5f}]")
        print(f"  flow (difference handed over)       : AUC {flow.auc:.4f}  "
              f"pnl/opp {flow.pnl_per_opp:+.5f} "
              f"[{flow.pnl_opp_lo:+.5f}, {flow.pnl_opp_hi:+.5f}]")
        print(f"  delta from being told               : AUC "
              f"{flow.auc - raw.auc:+.4f}  "
              f"pnl/opp {flow.pnl_per_opp - raw.pnl_per_opp:+.5f}")
        # The CIs are the point. A delta inside them says nothing.
        overlap = (raw.pnl_opp_lo <= flow.pnl_opp_hi
                   and flow.pnl_opp_lo <= raw.pnl_opp_hi)
        if overlap:
            print("  The two CIs OVERLAP, so being handed the difference did "
                  "not measurably help: on this data the CNN derives it "
                  "itself, and the answer to 'will it figure it out?' is yes "
                  "for this particular feature.")
        else:
            print("  The CIs are DISJOINT: being handed the difference helped "
                  "measurably, so the CNN did not derive a feature that is "
                  "one subtraction from its input.")

    with open(os.path.join(JD, f"collapse_config{suffix or '_base'}.json"), "w") as fh:
        json.dump(dict(vars(a), lookback=lookback, NB=NB,
                       n_train=len(tr), n_val=len(va), n_test=len(te),
                       collapse_features=have,
                       ts=time.strftime("%Y-%m-%dT%H:%M:%S")), fh, indent=2)


if __name__ == "__main__":
    main()
