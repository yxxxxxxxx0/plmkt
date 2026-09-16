"""Stage 0/3: one 1-minute panel carrying price AND the 10-level book.

Source is the project's existing 200ms in-game grid, `data/jump/feat_*_trimmed
.parquet` plus the matching `lob_*_trimmed.npy` tensor. That pair is used rather
than the raw `top_of_book_*.csv` for one decisive reason: the CSV feed carries
only level 1 (best bid/ask), while this study needs the 10 depth levels the
paper's feature set is built on. The grid also covers six sessions instead of
four. Its `mid` is (best_bid + best_ask)/2 by construction (jump_data.py:406),
which is the price definition the brief asks for.

Nothing here writes to the grid or to the raw recordings; both are read-only.

Sampling rule (strictly backward-looking)
-----------------------------------------
Minute M takes the LAST valid 200ms row whose exchange timestamp falls at or
before the END of minute M, per series. No observation after the minute close
is used, so every row is causal. Minutes with no valid row are left MISSING and
the series is broken there -- no interpolation, no forward-fill across a gap.

Series identity is the `asset_id` embedded in the grid's `series` key
(slug|market_type|line|asset_id). asset_id is the tradable token; the first
three fields alone collide across the two legs of a spread, which is the defect
recorded in FINDINGS_SERIES_KEY.md.

    python build_panel.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "jump_prediction"))
import features as FE  # noqa: E402  (reuse the project's LOB decoder)

JD = os.path.join(ROOT, "data", "jump")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen")
MIN_MS = 60_000
HKT_MS = 8 * 3600 * 1000
K = 10


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def build_session(fp, tp):
    F = pq.read_table(fp, columns=["ts", "sid", "series", "mid", "spread_ticks",
                                   "valid", "book_age_ms"]).to_pandas()
    F["row"] = np.arange(len(F), dtype=np.int64)
    F = F[F.valid.to_numpy(bool)]
    F = F[np.isfinite(F.mid.to_numpy()) & (F.mid > 0) & (F.mid < 1)]
    F["minute"] = F.ts.to_numpy(np.int64) // MIN_MS

    # last valid 200ms row at or before each minute close
    F = F.sort_values(["sid", "ts"], kind="stable")
    g = F.groupby(["sid", "minute"], sort=False, observed=True)
    bars = g.agg(row=("row", "last"), ts=("ts", "last"), mid=("mid", "last"),
                 spread_ticks=("spread_ticks", "last"),
                 book_age_ms=("book_age_ms", "last"),
                 n_grid=("row", "size"), series=("series", "last")).reset_index()

    # 10-level book at exactly those rows
    T = np.load(tp, mmap_mode="r")
    idx = bars.row.to_numpy()
    d = FE.decode(np.asarray(T[idx]), bars.mid.to_numpy(np.float64))
    S = pd.DataFrame(FE.state_features(d, spread_ticks=bars.spread_ticks.to_numpy()))
    # state_features re-emits mid and spread_ticks; the bar already carries them
    S = S.drop(columns=[c for c in ("mid", "spread_ticks") if c in S.columns])
    out = pd.concat([bars.reset_index(drop=True), S.reset_index(drop=True)],
                    axis=1)
    assert not out.columns.duplicated().any(),         "duplicate feature columns: %s" % out.columns[out.columns.duplicated()].tolist()

    # raw per-level book, as the paper's input demands
    for i in range(K):
        out["bid_dist_%d" % (i + 1)] = d["bid_dist"][:, i]
        out["ask_dist_%d" % (i + 1)] = d["ask_dist"][:, i]
        out["bid_usd_%d" % (i + 1)] = np.log1p(d["bid_usd"][:, i])
        out["ask_usd_%d" % (i + 1)] = np.log1p(d["ask_usd"][:, i])
        out["bid_mask_%d" % (i + 1)] = d["bid_mask"][:, i]
        out["ask_mask_%d" % (i + 1)] = d["ask_mask"][:, i]
    # price gaps between adjacent levels (0 where either level is absent)
    for i in range(K - 1):
        bg = d["bid_dist"][:, i + 1] - d["bid_dist"][:, i]
        ag = d["ask_dist"][:, i + 1] - d["ask_dist"][:, i]
        m = d["bid_mask"][:, i + 1] * d["bid_mask"][:, i]
        a = d["ask_mask"][:, i + 1] * d["ask_mask"][:, i]
        out["bid_gap_%d" % (i + 1)] = bg * m
        out["ask_gap_%d" % (i + 1)] = ag * a

    out["asset_id"] = out.series.astype(str).str.rsplit("|", n=1).str[-1]
    out["session"] = os.path.basename(fp).replace("feat_", "").replace(
        "_trimmed.parquet", "")
    return out


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    fps = sorted(glob.glob(os.path.join(JD, "feat_books_*_trimmed.parquet")))
    frames, stats = [], []
    for fp in fps:
        tp = fp.replace("feat_", "lob_").replace(".parquet", ".npy")
        if not os.path.exists(tp):
            log("SKIP %s (no lob tensor)" % os.path.basename(fp))
            continue
        log("building %s" % os.path.basename(fp))
        B = build_session(fp, tp)
        frames.append(B)
        stats.append(dict(session=B.session.iloc[0], minute_bars=len(B),
                          series=int(B.asset_id.nunique()),
                          median_minutes=float(B.groupby("asset_id").size().median())))
        log("  %s bars, %d series" % ("{:,}".format(len(B)), B.asset_id.nunique()))
        del B
    P = pd.concat(frames, ignore_index=True)
    P["ts_hkt"] = pd.to_datetime(P.minute * MIN_MS + HKT_MS, unit="ms")
    P["date"] = P.ts_hkt.dt.date.astype(str)
    P["tod_min"] = P.ts_hkt.dt.hour * 60 + P.ts_hkt.dt.minute
    P["dow"] = P.ts_hkt.dt.dayofweek

    f = os.path.join(CACHE, "minute_panel.parquet")
    P.to_parquet(f, index=False)
    S = pd.DataFrame(stats)
    S.to_csv(os.path.join(RES, "panel_build_stats.csv"), index=False)
    print()
    print(S.to_string(index=False))
    print("\ntotal {:,} minute bars | {:,} series | {} sessions | dates {}..{}"
          .format(len(P), P.asset_id.nunique(), P.session.nunique(),
                  P.date.min(), P.date.max()))
    print("feature columns: %d" % (len(P.columns)))
    print("wrote", f)


if __name__ == "__main__":
    sys.exit(main())
