"""Versioned model bundles: weights plus everything needed to run them again.

A bare `state_dict` is not a saved model. To reproduce a prediction you also
need the architecture arguments, the exact data spec the inputs were built
from, the preprocessing constants, and the decision threshold -- and if any of
those is only implicit in whatever the script happened to be doing that day,
the weights are decoration.

This repo has already lost work to that twice:

  * The first CNN-Transformer result (AUC 0.874) was never checkpointed at
    all, its model class lived inside a closure, and the CSV holding its
    metrics was overwritten by a later --skip-deep run. The number survives
    only as prose in reports/FINDINGS.md and cannot be reproduced.
  * `cnn_direction.pt` was silently overwritten when the break-even (4-tick)
    run reused the name, so the earlier 2-tick directional model is gone.

So a bundle here is a directory, versions auto-increment, and nothing is ever
overwritten:

    data/models/<name>_v<N>/
        weights.pt        state_dict only
        manifest.json     arch + args, data spec, preprocessing, threshold,
                          metrics, provenance, and a sha256 per source file
                          so the code version is pinned

`load(name, version)` returns a ready model and its manifest.
`verify(name)` re-scores the saved test index and checks the predictions match
the ones recorded at training time -- the only proof that a bundle actually
works rather than merely existing.

    python model_registry.py --bundle-all
    python model_registry.py --list
    python model_registry.py --verify cnn_direction
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.join(BASE, "data", "jump")
MODELS = os.path.join(BASE, "data", "models")

# Which builder reconstructs each architecture, and the source files whose
# content determines its behaviour. Hashing these pins the code version.
ARCH = {
    "cnn_transformer": dict(
        builder="jump_model.build_multiscale",
        sources=["jump_model.py", "jump_split.py", "jump_data.py"],
        inputs="mid-centered LOB level tensor (4 x 10) + mid-path scalars",
        target="P(path-max |dmid| >= 2 ticks within 5s)"),
    "cnn_transformer_nomid": dict(
        builder="jump_model.build_multiscale",
        sources=["jump_model.py", "jump_split.py", "jump_data.py"],
        inputs="same, mid-path scalars ZEROED (ablation)",
        target="P(path-max |dmid| >= 2 ticks within 5s)"),
    "cnn_depth_raw": dict(
        builder="collapse_cnn.build_model",
        sources=["collapse_cnn.py", "jump_split.py", "jump_data.py"],
        inputs="depth image (2 x 48 x 41) on a common price axis + scalars",
        target="P(path-max |dmid| >= 2 ticks within 5s)"),
    "cnn_depth_flow": dict(
        builder="collapse_cnn.build_model",
        sources=["collapse_cnn.py", "jump_split.py", "jump_data.py"],
        inputs="depth image + precomputed per-price deltas (4 x 48 x 41)",
        target="P(path-max |dmid| >= 2 ticks within 5s)"),
    "cnn_direction": dict(
        builder="taker_signal.build_dircnn",
        sources=["taker_signal.py", "collapse_cnn.py", "jump_split.py",
                 "jump_data.py", "polymarket_fees.py"],
        inputs="depth image (2 x 48 x 41) + scalars",
        target="3-class sign of dmid at 5s, threshold k ticks"),
}

# Preprocessing that is fixed in code rather than learned. Recorded because
# changing any of it silently invalidates the weights.
PREPROCESS = dict(
    tick=0.01, grid_ms=200, k_levels=10, depth_bins=41, depth_half_ticks=20,
    scalar_clip_ticks=30, scalar_scale=5.0,
    lob_prices="ticks relative to that row's own mid",
    lob_sizes="log1p(size * price), i.e. log dollars",
    depth_image="sizes binned on a price axis anchored at the mid at t",
    note="the mid path is supplied separately because mid-centering removes "
         "it from the tensor entirely -- see jump_data.py docstring")


def sha(path, n=16):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def metrics_for(name):
    """Pull whatever the comparison tables recorded for this model."""
    out = {}
    for csv, tag in ((("model_comparison.csv"), "magnitude"),
                     (("taker_comparison.csv"), "taker")):
        p = os.path.join(JD, csv)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        r = d[d.model == name]
        if len(r):
            out[tag] = {k: (None if pd.isna(v) else v)
                        for k, v in r.iloc[0].to_dict().items()}
    return out


def data_spec(name):
    """The split and filters the inputs were built from."""
    spec = dict(
        feat_files=[os.path.basename(f) for f in
                    sorted(glob.glob(os.path.join(JD, "feat_*.parquet")))],
        split="temporal 60/20/20 by ts, 120s gap discarded at each interior "
              "boundary; threshold chosen on val, test scored once",
        lookback_rows=200, receptive_field="coarse 24x8 + fine 24 = 43.2s",
        market_types=["moneyline", "total", "spread"],
        max_book_age_s=5.0)
    for cfg, keys in (("collapse_config_base.json", None),
                      ("taker_config.json", None),
                      ("run_config_base.json", None)):
        p = os.path.join(JD, cfg)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                spec[cfg] = json.load(fh)
    return spec


def provenance():
    """Where the underlying data came from, and whether it was sound."""
    prov = dict(recorded_by="live_recorder.py", built_by="jump_data.py")
    hp = os.path.join(JD, "recording_health.json")
    if os.path.exists(hp):
        with open(hp, encoding="utf-8") as fh:
            reps = json.load(fh)
        prov["recordings"] = [
            dict(path=r["path"], verdict=r["verdict"],
                 usable_for=r.get("usable_for"),
                 trade_prints=r.get("trade_prints"),
                 lag_sustained_spread_ms=r.get("lag_sustained_spread_ms"),
                 fails=r.get("fails"), warns=r.get("warns")) for r in reps]
    return prov


def next_version(name):
    existing = sorted(glob.glob(os.path.join(MODELS, f"{name}_v*")))
    n = 0
    for e in existing:
        try:
            n = max(n, int(os.path.basename(e).rsplit("_v", 1)[1]))
        except (IndexError, ValueError):
            pass
    return n + 1


def bundle(name, ckpt=None, notes=None):
    import torch
    ckpt = ckpt or os.path.join(JD, f"{name}.pt")
    if not os.path.exists(ckpt):
        return None
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = ARCH.get(name, {})
    v = next_version(name)
    outdir = os.path.join(MODELS, f"{name}_v{v}")
    os.makedirs(outdir, exist_ok=True)

    torch.save(blob["state_dict"], os.path.join(outdir, "weights.pt"))
    man = dict(
        name=name, version=v,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_checkpoint=os.path.relpath(ckpt, BASE),
        source_checkpoint_mtime=time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(ckpt))),
        nparam=blob.get("nparam"),
        train_config=blob.get("config", {}),
        loss_history=blob.get("loss_history"),
        architecture=arch,
        preprocessing=PREPROCESS,
        data_spec=data_spec(name),
        metrics=metrics_for(name),
        provenance=provenance(),
        code_sha256={s: sha(os.path.join(BASE, s))
                     for s in arch.get("sources", [])},
        notes=notes or "",
    )
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, default=str)
    # keep the predictions and index beside the weights so verify() has a
    # reference and downstream slicing needs no retraining
    for pat in (f"preds_{name}.npy", f"testidx_{name}.npy",
                f"takeredge_{name}.npy", f"takeridx_{name}.npy",
                f"attribution_{name}.csv"):
        src = os.path.join(JD, pat)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(outdir, pat))
    return outdir, man


def load(name, version=None):
    """Reconstruct a model from a bundle. Returns (model, manifest)."""
    import torch
    if version is None:
        version = next_version(name) - 1
    if version < 1:
        raise SystemExit(f"no bundle for {name}")
    outdir = os.path.join(MODELS, f"{name}_v{version}")
    with open(os.path.join(outdir, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    cfg = man["train_config"]
    b = man["architecture"]["builder"]
    L = cfg.get("L", 48)
    if b == "taker_signal.build_dircnn":
        from taker_signal import build_dircnn
        model = build_dircnn(L, in_ch=2, d=cfg.get("d_model", 64))
    elif b == "collapse_cnn.build_model":
        from collapse_cnn import build_model
        model = build_model(cfg.get("in_ch", 2), L, d=cfg.get("d_model", 64))
    elif b == "jump_model.build_multiscale":
        from jump_model import build_multiscale, frame_offsets
        _, _, scale, _ = frame_offsets(cfg.get("fine", 24),
                                       cfg.get("coarse", 24),
                                       cfg.get("stride", 8))
        model = build_multiscale(L, scale, d=cfg.get("d_model", 64))
    else:
        raise SystemExit(f"unknown builder {b!r} in {outdir}")
    model.load_state_dict(torch.load(os.path.join(outdir, "weights.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    return model, man


def verify(names=None, tol=1e-5):
    """Reload each bundle and check it reproduces its recorded predictions.

    This is the only thing that distinguishes a saved model from a saved
    file. A bundle that cannot be re-scored is exactly as useful as the
    CNN-Transformer result that was lost.
    """
    import torch
    from collapse_cnn import make_batcher
    from jump_model import frame_offsets
    from jump_split import TICK, load as load_data
    from taker_signal import batcher as dir_batcher, label3, prepare

    names = names or list(ARCH)
    OFF, STR, SCALE, lookback = frame_offsets(24, 24, 8)
    print("loading data once for all checks...")
    F, T, tr, va, te = load_data(["moneyline", "total", "spread"],
                                 lookback=lookback, max_book_age_s=5.0)
    F, _ = prepare(F, 5.0, "sports", 2.0)
    MID = F.mid.to_numpy(np.float32)
    zeros = np.zeros(len(F), np.float32)

    def multiscale_batch(j, mid_path=True):
        X = T[j[:, None] + OFF[None, :]]
        M = MID[j[:, None] + OFF[None, :]]
        rel = (M - MID[j][:, None]) / TICK
        dmid = (M - MID[j[:, None] + OFF[None, :] - STR[None, :]]) / TICK
        spr = X[:, :, 2, 0] - X[:, :, 0, 0]
        S = np.stack([np.clip(dmid, -30, 30) / 5.0,
                      np.clip(rel, -30, 30) / 5.0,
                      np.clip(spr, 0, 30) / 5.0], axis=-1)
        if not mid_path:
            S = np.zeros_like(S)
        return (torch.from_numpy(np.ascontiguousarray(X)),
                torch.from_numpy(S.astype(np.float32)))

    print(f"\n{'bundle':>28} {'n':>8} {'max|diff|':>11} {'result':>8}")
    allok = True
    for name in names:
        v = next_version(name) - 1
        if v < 1:
            continue
        d = os.path.join(MODELS, f"{name}_v{v}")
        ref_p = next((p for p in (f"takeredge_{name}.npy", f"preds_{name}.npy")
                      if os.path.exists(os.path.join(d, p))), None)
        idx_p = next((p for p in (f"takeridx_{name}.npy", f"testidx_{name}.npy")
                      if os.path.exists(os.path.join(d, p))), None)
        if not (ref_p and idx_p):
            print(f"{name+'_v'+str(v):>28} {'-':>8} {'-':>11} {'no ref':>8}")
            continue
        ref = np.load(os.path.join(d, ref_p))
        idx = np.load(os.path.join(d, idx_p))
        model, man = load(name, v)
        cfg = man["train_config"]

        got = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                j = idx[s:s + 1024]
                if man["architecture"]["builder"] == "jump_model.build_multiscale":
                    xb, sb = multiscale_batch(j, cfg.get("mid_path", True))
                    got.append(torch.sigmoid(model(xb, sb)).numpy())
                elif name == "cnn_direction":
                    mb = dir_batcher(T, F, OFF, STR,
                                     label3(F.signed_ticks.to_numpy(np.float64),
                                            cfg.get("k", 4)))
                    img, sc, _ = mb(j)
                    q = torch.softmax(model(img, sc), dim=-1).numpy()
                    got.append(q[:, 2] - q[:, 0])
                else:
                    mk, _ = make_batcher(T, F, OFF, STR,
                                         cfg.get("variant", "raw"), zeros)
                    img, sc, _, _ = mk(j)
                    got.append(torch.sigmoid(model(img, sc)).numpy())
        got = np.concatenate(got)
        diff = float(np.abs(got - ref).max()) if len(got) == len(ref) else np.nan
        ok = np.isfinite(diff) and diff <= tol
        allok &= bool(ok)
        print(f"{name+'_v'+str(v):>28} {len(idx):>8,} {diff:>11.2e} "
              f"{'OK' if ok else 'MISMATCH':>8}")
    print(f"\n{'all bundles reproduce their predictions' if allok else 'SOME BUNDLES DO NOT REPRODUCE -- do not trust those weights'}")
    return allok


def show_list():
    if not os.path.isdir(MODELS):
        print("no bundles yet")
        return
    rows = []
    for d in sorted(glob.glob(os.path.join(MODELS, "*_v*"))):
        mp = os.path.join(d, "manifest.json")
        if not os.path.exists(mp):
            continue
        with open(mp, encoding="utf-8") as fh:
            m = json.load(fh)
        mt = m.get("metrics", {})
        key = mt.get("taker") or mt.get("magnitude") or {}
        rows.append(dict(bundle=os.path.basename(d), nparam=m.get("nparam"),
                         created=m.get("created"),
                         auc=key.get("auc") or key.get("sign_auc"),
                         headline=(key.get("pnl_per_opp")
                                   if "pnl_per_opp" in key else None)))
    print(pd.DataFrame(rows).to_string(index=False) if rows else "no bundles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-all", action="store_true")
    ap.add_argument("--bundle", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", nargs="*", default=None,
                    help="reload bundles and check they reproduce their "
                         "recorded predictions; no names = all")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    if a.list:
        show_list()
        return
    if a.verify is not None:
        raise SystemExit(0 if verify(a.verify or None) else 1)
    names = list(ARCH) if a.bundle_all else (a.bundle or [])
    if not names:
        raise SystemExit("pass --bundle-all, --bundle NAME..., or --list")
    for n in names:
        r = bundle(n, notes=a.notes)
        if r is None:
            print(f"  [skip] {n}: no checkpoint at data/jump/{n}.pt")
            continue
        outdir, man = r
        print(f"  [ok]   {os.path.relpath(outdir, BASE)}  "
              f"{man['nparam']:,} params")
    print(f"\nbundles in {os.path.relpath(MODELS, BASE)}:")
    show_list()


if __name__ == "__main__":
    main()
