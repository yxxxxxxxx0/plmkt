"""
Can a model predict the EXECUTABLE taker return well enough to trade it?

Every simple signal in backtest.py loses roughly the spread, at every latency.
The obvious next question is whether combining them helps. This tests that
properly rather than reporting an AUC and stopping.

Design
------
target      fwd_long_1s > 0, i.e. "buy the ask now, sell the bid in 1s, and
            come out ahead". This is the tradeable event, not a mid move. It
            is true only ~1.2% of the time, which is the whole problem in one
            number.
train       2026-08-27 + 2026-08-28   (22 games, chronologically first)
test        2026-09-04                (15 games, six days later, untouched)
            2026-08-29 is excluded entirely: its recorder stopped before most
            games started, leaving 5/15 games with a median 41% of the game.
model       HistGradientBoosting, which handles the NaNs (thin-book slope) and
            the scale differences without preprocessing that could leak.

The evaluation that matters is the last one: take the top-k% of predictions
and report the MEAN fwd_long of those rows. That is cents per share actually
earned by acting on the model. AUC can be excellent while that number stays
firmly negative, because ranking rows correctly is not the same as finding
rows where the move exceeds the spread.

    python ml_taker.py --market-type moneyline
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

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")
TRAIN = ("2026-08-27", "2026-08-28")
TEST = ("2026-09-04",)

FEATS = ["spread_c", "imb_touch", "imb_1c", "imb_2c", "imb_5c",
         "micro_less_mid_c", "slope_bid_c", "slope_ask_c",
         "bb_sz", "ba_sz", "n_bid", "n_ask",
         "dep_bid_1c", "dep_ask_1c", "dep_bid_2c", "dep_ask_2c",
         "dep_bid_5c", "dep_ask_5c",
         "rem_bid_1s", "rem_ask_1s", "add_bid_1s", "add_ask_1s",
         "rem_bid_5s", "rem_ask_5s",
         "tr_buy_1s", "tr_sell_1s", "tr_n_1s",
         "mom_0.1s_c", "mom_1s_c", "mom_5s_c",
         "absmove_1s_c", "absmove_5s_c", "dt_prev", "t_since_start"]
TARGET = "fwd_long_1s"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(slates, market_type, max_rows, seed=0):
    X, y, raw = [], [], []
    rng = np.random.default_rng(seed)
    files = []
    for s in slates:
        files += sorted(glob.glob(os.path.join(PANEL, f"*-{s}.parquet")))
    for path in files:
        cols = FEATS + [TARGET, "market_type", "bb_c", "ba_c"]
        have = set(pq.ParquetFile(path).schema.names)
        use = [c for c in cols if c in have]
        tab = pq.read_table(path, columns=use)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in use}
        m = np.isfinite(d[TARGET])
        # crossed books excluded -- they fabricate positive fwd_long
        m &= np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"]) & (d["ba_c"] > d["bb_c"])
        if market_type:
            m &= (d["market_type"] == market_type)
        if m.sum() < 100:
            continue
        idx = np.flatnonzero(m)
        if idx.size > max_rows // max(len(files), 1):
            idx = rng.choice(idx, max_rows // max(len(files), 1), replace=False)
            idx.sort()
        X.append(np.column_stack([np.asarray(d[f], dtype=np.float32)[idx]
                                  if f in d else np.full(idx.size, np.nan, np.float32)
                                  for f in FEATS]))
        raw.append(np.asarray(d[TARGET], dtype=np.float64)[idx])
        y.append((raw[-1] > 0).astype(np.int8))
    if not X:
        raise SystemExit("no rows loaded")
    return np.vstack(X), np.concatenate(y), np.concatenate(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-type", default="moneyline")
    ap.add_argument("--max-rows", type=int, default=3_000_000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    log("loading train...")
    Xtr, ytr, rtr = load(TRAIN, args.market_type, args.max_rows)
    log(f"  train {Xtr.shape[0]:,} rows, positive rate {ytr.mean()*100:.3f}%")
    log("loading test...")
    Xte, yte, rte = load(TEST, args.market_type, args.max_rows)
    log(f"  test  {Xte.shape[0]:,} rows, positive rate {yte.mean()*100:.3f}%")

    log("fitting...")
    clf = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0)
    clf.fit(Xtr, ytr)
    ptr = clf.predict_proba(Xtr)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]

    print()
    print("=" * 92)
    print(f"ML ON THE EXECUTABLE TAKER TARGET  ({args.market_type}, "
          f"target = fwd_long_1s > 0)")
    print(f"train {TRAIN} {Xtr.shape[0]:,} rows | test {TEST} {Xte.shape[0]:,} rows")
    print("=" * 92)
    print(f"  train AUC {roc_auc_score(ytr, ptr):.4f}")
    print(f"  TEST  AUC {roc_auc_score(yte, pte):.4f}")
    print(f"  base rate: train {ytr.mean()*100:.3f}%  test {yte.mean()*100:.3f}%")
    print()
    print("  acting on the model -- take the top-k% most confident rows and BUY:")
    print(f"  {'top k%':>8} {'rows':>10} {'hit rate':>10} {'lift':>7} "
          f"{'mean fwd_long':>15} {'verdict':>10}")
    print("-" * 74)
    rows = []
    for k in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0):
        n = max(int(len(pte) * k / 100), 50)
        sel = np.argsort(-pte)[:n]
        hit = float(yte[sel].mean())
        edge = float(rte[sel].mean())
        rows.append({"k": k, "n": n, "hit": hit, "edge_c": edge})
        print(f"  {k:>7.2f}% {n:>10,} {hit*100:>9.2f}% "
              f"{hit/max(yte.mean(),1e-9):>6.1f}x {edge:>+14.4f}c "
              f"{'PROFIT' if edge > 0 else 'loses':>10}")
    print()
    print(f"  for reference, mean fwd_long over ALL test rows: {rte.mean():+.4f}c")
    print(f"  (that is the spread you pay; the model must beat it to be worth anything)")

    imp = None
    try:
        from sklearn.inspection import permutation_importance
        sub = np.random.default_rng(0).choice(len(yte), min(60000, len(yte)), replace=False)
        r = permutation_importance(clf, Xte[sub], yte[sub], n_repeats=3,
                                   random_state=0, scoring="roc_auc")
        imp = sorted(zip(FEATS, r.importances_mean), key=lambda x: -x[1])[:10]
        print()
        print("  top features by permutation importance (test AUC drop):")
        for f, v in imp:
            print(f"    {f:>18} {v:+.5f}")
    except Exception as e:
        log(f"  (importance skipped: {type(e).__name__})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"market_type": args.market_type,
                       "train_auc": float(roc_auc_score(ytr, ptr)),
                       "test_auc": float(roc_auc_score(yte, pte)),
                       "base_rate_test": float(yte.mean()),
                       "mean_fwd_long_all": float(rte.mean()),
                       "topk": rows,
                       "importance": [[f, float(v)] for f, v in (imp or [])]}, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
