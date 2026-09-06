"""
The decisive test: can a model predict the EXECUTABLE taker return, without
being told in advance that a jump is coming?

ml_jump.py found direction is weakly predictable CONDITIONAL on a >= 5c move
occurring (AUC 0.55, +2.80c mean signed move in the top 5%). That conditioning
is not available live -- you cannot filter on an event that has not happened
yet. So this asks the unconditional question, and asks it against the target
that pays rather than the mid:

    fwd_long_10s  = bid(t+10s) - ask(t)     buy now as taker, sell in 10s
    fwd_short_10s = bid(t)     - ask(t+10s) sell now as taker, buy in 10s

Both already contain the spread. A model whose top predictions have POSITIVE
mean fwd_long has found a taker strategy; anything else has found a mid-price
signal too small to trade, which is what every previous attempt here produced.

Two separate models are fitted, one per direction, because the long and short
opportunities need not be symmetric -- and because a single signed model can
hide a losing side inside a winning average.

Split is BY GAME. Features include the Hawkes and Lee-Mykland statistics, so
the jump econometrics get their fair chance inside a nonlinear model rather
than only as standalone signals.

    python ml_directional.py --slate 2026-09-04 --market-type moneyline
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from jump_models import hawkes_intensity, kalman_fair, lee_mykland
from ml_jump import BOOKF, EXTRA, FEATS

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")
HORIZ = "10s"
STRIDE = 7


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build(path, market_type):
    tgt = [f"fwd_long_{HORIZ}", f"fwd_short_{HORIZ}"]
    cols = BOOKF + tgt + ["t", "mid_c", "bb_c", "ba_c", "asset_id", "market_type"]
    have = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in cols if c in have]
    tab = pq.read_table(path, columns=cols)
    d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
    ok = np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"]) & (d["ba_c"] > d["bb_c"])
    if market_type:
        ok &= (d["market_type"] == market_type)
    X, L, S = [], [], []
    for aid in np.unique(d["asset_id"][ok]):
        sel = ok & (d["asset_id"] == aid)
        if sel.sum() < 20000:
            continue
        t, mid = d["t"][sel], d["mid_c"][sel]
        Lm, r, sig = lee_mykland(mid)
        nn = max(mid.size, 2)
        Cn = (2 * np.log(nn)) ** 0.5 / 0.7979 - (
            np.log(np.pi) + np.log(np.log(nn))) / (2 * 0.7979 * (2 * np.log(nn)) ** 0.5)
        Sn = 1.0 / (0.7979 * (2 * np.log(nn)) ** 0.5)
        jump = np.abs(Lm) > (Cn + Sn * (-np.log(-np.log(0.99))))
        fair, _ = kalman_fair(mid)
        ex = {"hawkes_0.5": hawkes_intensity(t, jump, 0.5),
              "hawkes_2": hawkes_intensity(t, jump, 2.0),
              "hawkes_10": hawkes_intensity(t, jump, 10.0),
              "lm_abs": np.abs(Lm), "lm_signed": Lm, "kal_dev": fair - mid}
        fl = np.asarray(d[f"fwd_long_{HORIZ}"], np.float64)[sel]
        fs = np.asarray(d[f"fwd_short_{HORIZ}"], np.float64)[sel]
        rows = np.arange(0, t.size, STRIDE)
        rows = rows[np.isfinite(fl[rows]) & np.isfinite(fs[rows])]
        if rows.size < 500:
            continue
        X.append(np.column_stack(
            [np.asarray(d[c], np.float64)[sel][rows] if c in d
             else np.full(rows.size, np.nan) for c in BOOKF]
            + [ex[c][rows] for c in EXTRA]))
        L.append(fl[rows])
        S.append(fs[rows])
    if not X:
        return None
    return np.vstack(X), np.concatenate(L), np.concatenate(S)


def evaluate(name, Xtr, ytr_bin, Xte, yte_raw, out):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0)
    clf.fit(Xtr, ytr_bin)
    p = clf.predict_proba(Xte)[:, 1]
    yb = (yte_raw > 0).astype(np.int8)
    auc = roc_auc_score(yb, p) if 0 < yb.mean() < 1 else float("nan")
    print(f"\n  {name}:  test AUC {auc:.4f}   P(profit) base {yb.mean()*100:.2f}%   "
          f"mean over all rows {yte_raw.mean():+.4f}c")
    print(f"    {'top k%':>8} {'rows':>9} {'P(profit)':>11} {'mean exec c/share':>19} "
          f"{'verdict':>9}")
    rows = []
    for k in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0):
        n = max(int(len(p) * k / 100), 40)
        s = np.argsort(-p)[:n]
        m = float(yte_raw[s].mean())
        rows.append({"k": k, "n": n, "p_profit": float(yb[s].mean()), "exec_c": m})
        print(f"    {k:>7.2f}% {n:>9,} {yb[s].mean()*100:>10.2f}% {m:>+18.4f}c "
              f"{'PROFIT' if m > 0 else 'loses':>9}")
    out[name] = {"auc": float(auc), "rows": rows,
                 "mean_all": float(yte_raw.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", default="2026-09-04")
    ap.add_argument("--market-type", default="moneyline")
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.panel_dir, f"*-{args.slate}.parquet")))
    if len(files) < 6:
        raise SystemExit("need more games")
    ntr = int(len(files) * 0.65)
    log(f"train {ntr} games | test {len(files)-ntr} games (BY GAME)")

    def gather(fs):
        X, L, S = [], [], []
        for f in fs:
            r = build(f, args.market_type)
            if r is None:
                continue
            X.append(r[0]); L.append(r[1]); S.append(r[2])
        return np.vstack(X), np.concatenate(L), np.concatenate(S)

    Xtr, Ltr, Str = gather(files[:ntr])
    Xte, Lte, Ste = gather(files[ntr:])
    log(f"train {Xtr.shape[0]:,} rows | test {Xte.shape[0]:,} rows")

    print()
    print("=" * 96)
    print(f"UNCONDITIONAL EXECUTABLE PREDICTION  ({args.market_type}, "
          f"horizon {HORIZ}, taker both legs)")
    print("target already includes the spread; positive mean = a real taker edge")
    print("=" * 96)
    out = {}
    evaluate("LONG  (buy ask now, sell bid in 10s)", Xtr, (Ltr > 0).astype(np.int8),
             Xte, Lte, out)
    evaluate("SHORT (sell bid now, buy ask in 10s)", Xtr, (Str > 0).astype(np.int8),
             Xte, Ste, out)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
