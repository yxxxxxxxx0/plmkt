"""Phase 5 runner: train and evaluate the sequence models, plus ablations.

Sections 8-13 of the brief. The comparison that matters is CNN vs CNN+LSTM vs
CNN+Transformer against the handcrafted baselines from baselines.py, all on
the same split, the same label, and the same held-out markets.

Sequence construction. Frames are taken at MULTI-SCALE offsets rather than
uniformly: fine spacing near the prediction instant where the book is most
informative, coarser going back to -60s. That covers the full window the brief
asks for without paying for 300 frames of 200ms grid, which the IO cannot
sustain on this machine.

Every frame is pulled at a NEGATIVE offset from the prediction row, and a
sample is dropped unless every frame is in the same series and at exactly the
expected time offset. Nothing at or after t enters the input.

Normalisation statistics come from TRAIN rows only. Class weighting is applied
to the training loss only; validation and test keep their natural prevalence.

    python train.py --smoke                      # tiny, checks the plumbing
    python train.py --J 0.02 --H 30
    python train.py --J 0.02 --H 30 --ablations
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
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import models as MD          # noqa: E402
import splits as SP          # noqa: E402
from evaluate import calibration, evaluate, pick_threshold  # noqa: E402

JD = os.path.join(ROOT, "data", "jump")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "jump_prediction")
GRID_MS = 200
TICK = 0.01


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def frame_offsets(fine_n=12, fine_step_s=0.6, coarse_n=18, back_s=60.0):
    """Negative grid-row offsets, fine near t and coarse going back."""
    f = -np.arange(fine_n) * int(fine_step_s * 1000 / GRID_MS)
    start = f.min() - int(fine_step_s * 1000 / GRID_MS)
    span = int(back_s * 1000 / GRID_MS) + start
    c = start - (np.arange(1, coarse_n + 1) * (abs(span) // max(coarse_n, 1)))
    offs = np.sort(np.unique(np.concatenate([f, c])))
    return offs.astype(np.int64)            # ascending, ends at 0


class Sessions:
    """Memmapped per-session arrays, opened once and shared by all batches."""

    def __init__(self, names):
        self.d = {}
        for n in names:
            F = pq.read_table(os.path.join(JD, f"feat_{n}_trimmed.parquet"),
                              columns=["mid", "spread_ticks", "ts", "sid",
                                       "book_age_ms"]).to_pandas()
            T = np.load(os.path.join(JD, f"lob_{n}_trimmed.npy"), mmap_mode="r")
            self.d[n] = dict(T=T, mid=F.mid.to_numpy(np.float32),
                             sp=F.spread_ticks.to_numpy(np.float32),
                             ts=F.ts.to_numpy(np.int64),
                             sid=F.sid.to_numpy(),
                             age=F.book_age_ms.to_numpy(np.float32))

    def valid_rows(self, name, rows, offs):
        s = self.d[name]
        n = len(s["ts"])
        ok = np.ones(len(rows), bool)
        for o in (offs.min(), offs.max()):
            j = rows + o
            ok &= (j >= 0) & (j < n)
        j0 = np.clip(rows + offs.min(), 0, n - 1)
        ok &= (s["sid"][j0] == s["sid"][np.clip(rows, 0, n - 1)])
        ok &= (s["ts"][np.clip(rows, 0, n - 1)] - s["ts"][j0]
               == -offs.min() * GRID_MS)
        return ok

    def batch(self, name, rows, offs):
        """(B, L, 6, K) frames, (B, L, 3) scalars, (B, L) seconds-before."""
        s = self.d[name]
        J = rows[:, None] + offs[None, :]
        B, L = J.shape
        X = np.asarray(s["T"][J.reshape(-1)]).reshape(B, L, 4, -1)
        bd, bl, ad, al = X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3]
        bm = (bl > 0).astype(np.float32)
        am = (al > 0).astype(np.float32)
        img = np.stack([bd * bm, bl, bm, ad * am, al, am], axis=2)
        mid = s["mid"][J]
        sc = np.stack([
            (mid - s["mid"][rows][:, None]) / TICK,     # mid path, in ticks
            s["sp"][J],                                  # spread
            np.log1p(s["age"][J] / 1000.0),              # staleness = delta_t
        ], axis=-1)
        t = (offs.astype(np.float32) * GRID_MS / 1000.0)[None, :].repeat(B, 0)
        return (img.astype(np.float32), sc.astype(np.float32), t)


def load_points(sessions=None):
    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    if sessions:
        ps = [p for p in ps if any(s in os.path.basename(p) for s in sessions)]
    cols = ["row", "session", "market", "market_type", "sid", "ts",
            "spread_ticks", "mid"]
    lab = None
    out = []
    for p in ps:
        sch = pq.ParquetFile(p).schema_arrow.names
        lab = [c for c in sch if c.startswith(("jump_", "valid_"))]
        out.append(pd.read_parquet(p, columns=cols + lab))
    return pd.concat(out, ignore_index=True)


def fit_norm(S, names, rows_by_session, offs, n=20000, rng=None):
    """Channel-wise mean/std from TRAIN rows only."""
    acc_i, acc_s = [], []
    for nm in names:
        r = rows_by_session.get(nm)
        if r is None or len(r) == 0:
            continue
        take = r if len(r) <= n // max(len(names), 1) else rng.choice(
            r, n // max(len(names), 1), replace=False)
        img, sc, _ = S.batch(nm, take, offs)
        acc_i.append(img.reshape(-1, img.shape[2], img.shape[3]))
        acc_s.append(sc.reshape(-1, sc.shape[2]))
    I = np.concatenate(acc_i, 0)
    C = np.concatenate(acc_s, 0)
    mi, si = I.mean((0, 2)), I.std((0, 2))
    ms, ss = C.mean(0), C.std(0)
    si[si < 1e-6] = 1.0
    ss[ss < 1e-6] = 1.0
    # masks are already 0/1 and meaningful; leave them alone
    mi[2] = 0.0; si[2] = 1.0
    mi[5] = 0.0; si[5] = 1.0
    return (mi.astype(np.float32), si.astype(np.float32),
            ms.astype(np.float32), ss.astype(np.float32))


def run_model(kind, S, D, split, ycol, offs, a, norm, rng, use_mask=True,
              use_dt=True, tag=""):
    import torch
    import torch.nn as nn
    torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    mi, si, ms, ss = norm

    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    thin = [s for s in ("train", "val", "test") if len(idx[s]) < 50]
    if thin:
        raise SystemExit(
            f"split(s) {thin} have fewer than 50 rows. Splitting is by whole "
            f"market, so a run needs enough markets for 70/15/15 to be "
            f"meaningful -- pass more --sessions (6 sessions = 75 markets).")
    for s, cap in (("train", a.max_train), ("val", a.max_val), ("test", a.max_test)):
        if cap and len(idx[s]) > cap:
            idx[s] = np.sort(rng.choice(idx[s], cap, replace=False))
    y_all = D[ycol].to_numpy(np.float32)
    sess = D.session.to_numpy()
    rowv = D.row.to_numpy()

    model = MD.build(kind, k_levels=S.d[list(S.d)[0]]["T"].shape[2],
                     d=a.d_model, use_mask=use_mask)
    pos = float(y_all[idx["train"]].mean())
    # class weighting from TRAIN prevalence only
    w = torch.tensor([(1 - pos) / max(pos, 1e-6)], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    log(f"  {kind}{tag}: {MD.n_params(model):,} params, {len(offs)} frames, "
        f"train {len(idx['train']):,} (pos {pos:.3f})")

    def make(ii):
        img_l, sc_l, t_l = [], [], []
        order = np.argsort(sess[ii], kind="stable")
        ii = ii[order]
        for nm in np.unique(sess[ii]):
            m = sess[ii] == nm
            img, sc, t = S.batch(nm, rowv[ii[m]], offs)
            img_l.append(img); sc_l.append(sc); t_l.append(t)
        img = np.concatenate(img_l); sc = np.concatenate(sc_l)
        t = np.concatenate(t_l)
        img = (img - mi[None, None, :, None]) / si[None, None, :, None]
        sc = (sc - ms) / ss
        if not use_dt:
            sc = sc.copy(); sc[:, :, 2] = 0.0
        return (torch.from_numpy(img), torch.from_numpy(sc),
                torch.from_numpy(t), torch.from_numpy(y_all[ii]))

    def predict(ii, bs=512):
        model.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(ii), bs):
                img, sc, t, _ = make(ii[k:k + bs])
                z = (model(img, sc, t) if kind == "cnn_transformer"
                     else model(img, sc))
                out.append(torch.sigmoid(z).numpy())
        return np.concatenate(out)

    from sklearn.metrics import average_precision_score
    best, best_state, bad = -1.0, None, 0
    tr = idx["train"]
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        t0 = time.time()
        for k in range(0, len(perm), a.batch):
            ii = tr[perm[k:k + a.batch]]
            if len(ii) < 8:
                continue
            img, sc, t, y = make(ii)
            z = (model(img, sc, t) if kind == "cnn_transformer"
                 else model(img, sc))
            loss = lossf(z, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        pv = predict(idx["val"])
        ap = average_precision_score(y_all[idx["val"]], pv)
        log(f"    epoch {ep}/{a.epochs} loss {tot/max(nb,1):.4f} "
            f"val PR-AUC {ap:.4f} ({time.time()-t0:.0f}s)")
        if ap > best + 1e-5:
            best, bad = ap, 0
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                log(f"    early stop; restoring best val PR-AUC {best:.4f}")
                break
    if best_state:
        model.load_state_dict(best_state)

    pv = predict(idx["val"])
    thr = pick_threshold(y_all[idx["val"]], pv)
    pt = predict(idx["test"])
    return pt, y_all[idx["test"]], thr, idx, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--J", type=float, default=0.02)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--models", nargs="*",
                    default=["cnn", "cnn_lstm", "cnn_transformer"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=120_000)
    ap.add_argument("--max-val", type=int, default=40_000)
    ap.add_argument("--max-test", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tight-only", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--label", default="raw",
                    choices=["raw", "clean", "durable", "both"],
                    help="'raw' is the original max-|mid| label. The others use "
                         "the artefact-free decomposition from "
                         "audit_durability.py (cache/moves_H*.parquet): 'clean' "
                         "requires the displaced price to be seen while the "
                         "book is still tight, 'durable' requires it to "
                         "persist, 'both' requires both. See run_clean.py.")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.max_train, a.max_val, a.max_test = 1, 4000, 2000, 2000
    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    # every artefact of this run carries the regime and label, so a corrected
    # run never silently overwrites the original one
    run_tag = (("_tight" if a.tight_only else "")
               + ("" if a.label == "raw" else f"_{a.label}")
               + ("_smoke" if a.smoke else ""))

    D = load_points(a.sessions)
    ycol, vcol = f"jump_{a.J:g}_H{a.H}", f"valid_label_H{a.H}"
    D = D[D[vcol].to_numpy(bool)].reset_index(drop=True)
    if a.label != "raw":
        # Swap in the artefact-free label. The raw max-|mid| label counts a mid
        # excursion however briefly it existed and however broken the book was
        # at the time; only ~25% of tight-book raw jumps are ever observed while
        # the book is still tight (audit_durability.py).
        mv = os.path.join(CACHE, f"moves_H{a.H}.parquet")
        if not os.path.exists(mv):
            raise SystemExit(f"run audit_durability.py --H {a.H} first")
        M = pd.read_parquet(mv)
        n0 = len(D)
        D = D.merge(M[["sid", "ts", "clean_move", "durable_move"]],
                    on=["sid", "ts"], how="inner")
        clean = D.clean_move >= a.J
        durable = D.durable_move >= a.J
        D[ycol] = (clean if a.label == "clean" else
                   durable if a.label == "durable" else clean & durable)
        D = D.drop(columns=["clean_move", "durable_move"])
        log(f"label '{a.label}': {n0:,} -> {len(D):,} rows after move join")
    if a.tight_only:
        D = D[D.spread_ticks <= 2.0].reset_index(drop=True)
    log(f"{len(D):,} points, prevalence {D[ycol].mean():.4f}")

    offs = frame_offsets()
    S = Sessions(sorted(D.session.unique()))
    keep = np.zeros(len(D), bool)
    for nm, g in D.groupby("session", observed=True):
        keep[g.index.to_numpy()] = S.valid_rows(nm, g.row.to_numpy(), offs)
    D = D[keep].reset_index(drop=True)
    log(f"{len(D):,} points with a complete contiguous {abs(offs.min())*0.2:.0f}s "
        f"window ({len(offs)} frames)")

    split, Sm = SP.assign(D)
    rows_by = {nm: g.row.to_numpy()
               for nm, g in D[split == "train"].groupby("session", observed=True)}
    norm = fit_norm(S, sorted(rows_by), rows_by, offs, rng=rng)
    log("normalisation fitted on TRAIN rows only")

    runs = [(m, True, True, "") for m in a.models]
    if a.ablations:
        runs += [("cnn_transformer", False, True, "_nomask"),
                 ("cnn_transformer", True, False, "_nodt"),
                 ("cnn_lstm", True, False, "_nodt")]

    rows = []
    for kind, use_mask, use_dt, tag in runs:
        pt, yt, thr, idx, model = run_model(kind, S, D, split, ycol, offs, a,
                                            norm, rng, use_mask, use_dt, tag)
        hours = len(yt) / 3600.0
        r = evaluate(kind + tag, yt, pt, thr, hours=hours,
                     extra=dict(J=a.J, H=a.H, n_params=MD.n_params(model),
                                frames=len(offs), tight_only=bool(a.tight_only)))
        rows.append(r)
        log(f"  {kind+tag}: TEST PR-AUC {r['pr_auc']:.4f} "
            f"(lift {r['pr_auc_lift']:.2f}x) ROC {r['roc_auc']:.4f} "
            f"P@1% {r['prec_top1pct']:.3f}")
        np.save(os.path.join(RES, f"preds_{kind}{tag}{run_tag}_J{a.J:g}_H{a.H}.npy"),
                pt)
        np.save(os.path.join(RES, f"testidx_{kind}{tag}{run_tag}_J{a.J:g}_H{a.H}.npy"),
                idx["test"])
        calibration(yt, pt).to_csv(
            os.path.join(RES,
                         f"calibration_{kind}{tag}{run_tag}_J{a.J:g}_H{a.H}.csv"),
            index=False)

    R = pd.DataFrame(rows)
    p = os.path.join(RES, f"deep_metrics_J{a.J:g}_H{a.H}{run_tag}.csv")
    R.to_csv(p, index=False)
    log(f"wrote {p}")
    print("\n" + R[["model", "prevalence", "pr_auc", "pr_auc_lift", "roc_auc",
                    "precision", "recall", "f1", "prec_top1pct", "brier_skill",
                    "n_params"]].to_string(index=False,
                                           float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
