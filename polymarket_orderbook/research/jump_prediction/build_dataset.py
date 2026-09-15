"""Phase 2: turn the audited 200ms grid into a modelling dataset.

Produces, per session, a cache holding one row per PREDICTION POINT:

  state features      instantaneous book geometry (features.state_features)
  trajectory features trailing changes over 5/10/30s (features.trajectory_*)
  labels              jump_{J} and dir_{J} for every J, at every horizon H
  row                 index into that session's trimmed LOB tensor, so the
                      sequence models can pull history without this cache
                      having to store it

Design decisions that matter for correctness:

* Prediction points are taken every `stride` grid rows (default 5 = 1 Hz)
  rather than every row. Adjacent 200ms rows are near-duplicates and inflate
  every count without adding information; the label still uses the full-
  resolution path, so nothing about the target is coarsened.
* Rows are kept only if the book is fresh, the book is not crossed, and the
  FULL forward label window is contiguous in time. The grid deliberately
  leaves gaps where a market went quiet past the fill cap, and a shift across
  one would compare prices minutes apart.
* Only in-game rows survive (trim_sessions.py already applied kickoff..final
  out), so post-resolution stale quotes cannot enter.
* Nothing here reads forward except the label, which is the point of the
  label. Normalisation is NOT applied here -- it is fitted on train only,
  later, in train.py.

    python build_dataset.py --sessions 2026-09-10        # smoke
    python build_dataset.py                              # all
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
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import features as FE  # noqa: E402

JD = os.path.join(ROOT, "data", "jump")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "jump_prediction")
GRID_MS = 200
HORIZONS = [10, 30, 60]
THRESHOLDS = [0.01, 0.02, 0.03, 0.05]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sessions_available():
    out = []
    for p in sorted(glob.glob(os.path.join(JD, "feat_books_*_trimmed.parquet"))):
        out.append(os.path.basename(p)[len("feat_"):-len("_trimmed.parquet")])
    return out


def build_one(name, stride=5, max_book_age_s=5.0, lags_s=(5, 10, 30),
              history_s=60):
    fp = os.path.join(JD, f"feat_{name}_trimmed.parquet")
    tp = os.path.join(JD, f"lob_{name}_trimmed.npy")
    tbl = pq.read_table(fp, columns=["mid", "spread_ticks", "ts", "sid",
                                     "book_age_ms", "series"])
    ser = tbl.column("series").combine_chunks().dictionary_encode().to_pandas()
    F0 = tbl.drop_columns(["series"]).to_pandas()
    del tbl
    T = np.load(tp, mmap_mode="r")
    n = len(F0)
    assert T.shape[0] == n, f"{name}: {n} rows vs tensor {T.shape[0]}"
    log(f"  {name}: {n:,} in-game grid rows, tensor {T.shape}")

    sid = F0.sid.to_numpy()
    ts = F0.ts.to_numpy(np.int64)
    mid = F0.mid.to_numpy(np.float64)

    # ---- labels at every horizon, on the FULL-resolution grid ------------- #
    lab = {}
    for H in HORIZONS:
        L = FE.jump_labels(mid, sid, ts, H, THRESHOLDS, GRID_MS)
        for c in L.columns:
            lab[f"{c}_H{H}"] = L[c].to_numpy()
    log(f"  {name}: labels built for H={HORIZONS}")

    # ---- prediction points ------------------------------------------------ #
    hist_rows = int(history_s * 1000 / GRID_MS)
    idx = np.arange(n)
    # contiguous history: row i-hist must be the same series and exactly
    # hist_rows*200ms earlier, so a 60s window is genuinely 60s of book
    back = idx - hist_rows
    ok_hist = (back >= 0)
    b = np.clip(back, 0, n - 1)
    ok_hist &= (sid[b] == sid) & (ts - ts[b] == hist_rows * GRID_MS)

    fresh = F0.book_age_ms.to_numpy() <= max_book_age_s * 1000
    any_label = np.zeros(n, bool)
    for H in HORIZONS:
        any_label |= lab[f"valid_label_H{H}"]
    keep = ok_hist & fresh & any_label & (idx % stride == 0)
    sel = np.where(keep)[0]
    log(f"  {name}: {len(sel):,} prediction points "
        f"(stride {stride}, {len(sel)/max(n,1):.1%} of rows; "
        f"history ok {ok_hist.mean():.1%}, fresh {fresh.mean():.1%})")
    if len(sel) == 0:
        return None

    # ---- features: state now, trajectory over trailing windows ------------ #
    # State is computed on the selected rows only. Trajectory needs the rows
    # in between, so it is computed on the full grid and then subset -- which
    # is also why it is computed with a groupby on sid rather than a plain
    # shift.
    d_all = FE.decode(T, mid)
    S_all = FE.state_features(d_all, spread_ticks=F0.spread_ticks.to_numpy())
    Tr_all = FE.trajectory_features(S_all, sid, lags_s=lags_s,
                                    grid_ms=GRID_MS,
                                    book_age_ms=F0.book_age_ms.to_numpy(),
                                    mid=mid)
    out = pd.concat([S_all.iloc[sel].reset_index(drop=True),
                     Tr_all.iloc[sel].reset_index(drop=True)], axis=1)
    del d_all, S_all, Tr_all

    out["row"] = sel.astype(np.int64)          # index into the trimmed tensor
    out["session"] = name
    out["sid"] = sid[sel]
    out["ts"] = ts[sel]
    out["book_age_ms"] = F0.book_age_ms.to_numpy()[sel]
    lut = ser.cat.categories if hasattr(ser, "cat") else pd.Index(ser.unique())
    codes = ser.cat.codes.to_numpy() if hasattr(ser, "cat") else None
    series_of_row = np.asarray(lut)[codes[sel]] if codes is not None \
        else ser.to_numpy()[sel]
    out["series"] = series_of_row
    out["market"] = pd.Series(series_of_row).str.split("|").str[0].to_numpy()
    out["market_type"] = pd.Series(series_of_row).str.split("|").str[1].to_numpy()
    for k, v in lab.items():
        out[k] = v[sel]

    # float32 everywhere except the integer bookkeeping columns
    for c, dt in out.dtypes.items():
        if dt == np.float64:
            out[c] = out[c].astype(np.float32)
    return out


def prevalence_table(D):
    rows = []
    for H in HORIZONS:
        v = D[f"valid_label_H{H}"].to_numpy(bool)
        for J in THRESHOLDS:
            y = D[f"jump_{J:g}_H{H}"].to_numpy(bool) & v
            d = D[f"dir_{J:g}_H{H}"].to_numpy()[v]
            rows.append(dict(
                H_s=H, J=J, n_valid=int(v.sum()), n_pos=int(y.sum()),
                prevalence=float(y.sum() / max(v.sum(), 1)),
                pos_up=int((d > 0).sum()), pos_down=int((d < 0).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=5,
                    help="grid rows between prediction points; 5 = 1 Hz")
    ap.add_argument("--history-s", type=float, default=60.0)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    names = sessions_available()
    if a.sessions:
        names = [n for n in names if any(s in n for s in a.sessions)]
    if not names:
        raise SystemExit("no trimmed sessions found; run trim_sessions.py first")
    log(f"building {len(names)} session(s): {names}")

    parts = []
    for nm in names:
        out = build_one(nm, stride=a.stride, history_s=a.history_s)
        if out is None:
            continue
        p = os.path.join(CACHE, f"points_{nm}.parquet")
        out.to_parquet(p, index=False)
        log(f"  wrote {p} ({os.path.getsize(p)/1e6:.0f} MB, {len(out):,} rows)")
        parts.append(out[["session", "market", "market_type", "ts"] +
                         [c for c in out.columns if c.startswith(("jump_",
                                                                  "dir_",
                                                                  "valid_"))]])
    if not parts:
        raise SystemExit("nothing built")
    D = pd.concat(parts, ignore_index=True)

    P = prevalence_table(D)
    pp = os.path.join(RES, "jump_prevalence.csv")
    P.to_csv(pp, index=False)
    log(f"wrote {pp}")
    print("\nclass prevalence by (J, H):")
    print(P.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    thin = P[P.n_pos < 500]
    if len(thin):
        print("\nSettings with fewer than 500 positives -- reported, not modelled:")
        print(thin[["H_s", "J", "n_pos", "prevalence"]].to_string(index=False))


if __name__ == "__main__":
    main()
