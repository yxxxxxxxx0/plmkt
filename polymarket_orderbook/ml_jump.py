"""
Can machine learning improve jump prediction -- and would it matter?

Three questions, deliberately separated, because they have different answers
and conflating them is how a jump strategy gets built that cannot work.

Q1. MAGNITUDE. Can a model beat the Hawkes kernel at forecasting that a large
    move is coming? Hawkes alone reaches 28.1% precision on ">= 5c within
    10s" against a 4.4% base rate. A GBM sees the whole book state plus the
    Hawkes and Lee-Mykland statistics, so it should do better.

Q2. DIRECTION -- the one that decides everything. CONDITIONAL on a large move
    actually occurring in the next 10s, can anything predict its SIGN? This
    is the sharpest possible test of whether jump capture is achievable: the
    sample is restricted to rows where a jump really did happen, so the model
    is handed the easiest version of the problem and asked only which way.
    If AUC is ~0.50 here, no amount of magnitude accuracy can rescue a taker
    strategy, because expected PnL is (probability-weighted move) minus the
    spread and the first term is zero by construction.

Q3. ADVERSE SELECTION. At the moment of a fill, how toxic will it be? This is
    the target that actually pays, because it feeds the sizing rule for the
    market-making strategy. Improving it does not need direction at all.

Split is BY GAME, never by row: adjacent rows are milliseconds apart and
near-identical, so a random split would report near-perfect accuracy that
means nothing.

    python ml_jump.py --market-type moneyline
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from jump_models import hawkes_intensity, lee_mykland

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")

BOOKF = ["spread_c", "imb_touch", "imb_1c", "imb_2c", "imb_5c",
         "micro_less_mid_c", "slope_bid_c", "slope_ask_c",
         "bb_sz", "ba_sz", "n_bid", "n_ask",
         "dep_bid_1c", "dep_ask_1c", "dep_bid_2c", "dep_ask_2c",
         "dep_bid_5c", "dep_ask_5c",
         "rem_bid_1s", "rem_ask_1s", "add_bid_1s", "add_ask_1s",
         "rem_bid_5s", "rem_ask_5s", "tr_buy_1s", "tr_sell_1s", "tr_n_1s",
         "mom_0.1s_c", "mom_1s_c", "mom_5s_c", "absmove_1s_c", "absmove_5s_c",
         "dt_prev", "t_since_start"]
EXTRA = ["hawkes_0.5", "hawkes_2", "hawkes_10", "lm_abs", "lm_signed", "kal_dev"]
FEATS = BOOKF + EXTRA
THETA = 5.0
H = 10.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_game(path, market_type, stride=7):
    """Features + jump targets for one game. `stride` subsamples rows: adjacent
    rows are ~7ms apart and almost identical, so keeping all of them inflates
    the sample without adding information."""
    cols = BOOKF + ["t", "mid_c", "bb_c", "ba_c", "asset_id", "market_type"]
    have = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in cols if c in have]
    tab = pq.read_table(path, columns=cols)
    d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
    ok = (np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"]) & (d["ba_c"] > d["bb_c"]))
    if market_type:
        ok &= (d["market_type"] == market_type)
    X, Ymag, Ydir, Mv = [], [], [], []
    for aid in np.unique(d["asset_id"][ok]):
        sel = ok & (d["asset_id"] == aid)
        if sel.sum() < 20000:
            continue
        t = d["t"][sel]
        mid = d["mid_c"][sel]
        L, r, sig = lee_mykland(mid)
        nn = max(mid.size, 2)
        Cn = (2 * np.log(nn)) ** 0.5 / 0.7979 - (
            np.log(np.pi) + np.log(np.log(nn))) / (2 * 0.7979 * (2 * np.log(nn)) ** 0.5)
        Sn = 1.0 / (0.7979 * (2 * np.log(nn)) ** 0.5)
        jump = np.abs(L) > (Cn + Sn * (-np.log(-np.log(0.99))))
        ex = {"hawkes_0.5": hawkes_intensity(t, jump, 0.5),
              "hawkes_2": hawkes_intensity(t, jump, 2.0),
              "hawkes_10": hawkes_intensity(t, jump, 10.0),
              "lm_abs": np.abs(L), "lm_signed": L}
        # Kalman deviation, filter only (see jump_models.kalman_fair)
        from jump_models import kalman_fair
        fair, _ = kalman_fair(mid)
        ex["kal_dev"] = fair - mid

        # targets: does a >=THETA move occur within H, and which way
        j = np.searchsorted(t, t + H, side="right")
        up = np.full(t.size, np.nan)
        dn = np.full(t.size, np.nan)
        end = np.full(t.size, np.nan)
        for i in range(0, t.size, stride):
            k = j[i]
            if k <= i + 1:
                continue
            seg = mid[i + 1:k]
            up[i] = seg.max() - mid[i]
            dn[i] = seg.min() - mid[i]
            end[i] = mid[k - 1] - mid[i]
        rows = np.arange(0, t.size, stride)
        rows = rows[np.isfinite(up[rows])]
        if rows.size < 500:
            continue
        big = (up[rows] >= THETA) | (dn[rows] <= -THETA)
        # direction target: sign of the move at the END of the window, so it is
        # what a position actually realises rather than the best moment inside
        updir = (end[rows] > 0).astype(np.int8)
        feat = np.column_stack(
            [np.asarray(d[c], np.float64)[sel][rows] if c in d
             else np.full(rows.size, np.nan) for c in BOOKF]
            + [ex[c][rows] for c in EXTRA])
        X.append(feat)
        Ymag.append(big.astype(np.int8))
        Ydir.append(updir)
        Mv.append(end[rows])
    if not X:
        return None
    return (np.vstack(X), np.concatenate(Ymag), np.concatenate(Ydir),
            np.concatenate(Mv))


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
    tr_files, te_files = files[:ntr], files[ntr:]
    log(f"train {len(tr_files)} games | test {len(te_files)} games (split BY GAME)")

    def gather(fs):
        Xs, Ym, Yd, Mv = [], [], [], []
        for f in fs:
            r = build_game(f, args.market_type)
            if r is None:
                continue
            Xs.append(r[0]); Ym.append(r[1]); Yd.append(r[2]); Mv.append(r[3])
            log(f"  {os.path.basename(f)[:-8]}: {r[0].shape[0]:,} rows")
        return (np.vstack(Xs), np.concatenate(Ym), np.concatenate(Yd),
                np.concatenate(Mv))

    Xtr, Mtr, Dtr, Vtr = gather(tr_files)
    Xte, Mte, Dte, Vte = gather(te_files)
    log(f"train {Xtr.shape[0]:,} rows | test {Xte.shape[0]:,} rows")

    out = {}
    print()
    print("=" * 92)
    print(f"Q1  MAGNITUDE: can ML beat Hawkes at 'a >= {THETA:g}c move within {H:g}s'?")
    print("=" * 92)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0)
    clf.fit(Xtr, Mtr)
    pm = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(Mte, pm)
    base = Mte.mean()
    print(f"  test AUC {auc:.4f}   base rate {base*100:.2f}%")
    print(f"  {'top k%':>8} {'rows':>9} {'precision':>11} {'lift':>7}")
    hk = Xte[:, FEATS.index("hawkes_0.5")]
    for k in (0.5, 1.0, 5.0, 10.0):
        n = max(int(len(pm) * k / 100), 50)
        s_ml = np.argsort(-pm)[:n]
        s_hk = np.argsort(-np.nan_to_num(hk, nan=-1e9))[:n]
        print(f"  {k:>7.1f}%  {n:>8,}  ML {Mte[s_ml].mean()*100:>7.2f}%  "
              f"{Mte[s_ml].mean()/base:>5.1f}x   |  Hawkes {Mte[s_hk].mean()*100:>6.2f}%  "
              f"{Mte[s_hk].mean()/base:>5.1f}x")
    out["magnitude"] = {"auc": float(auc), "base": float(base)}

    print()
    print("=" * 92)
    print("Q2  DIRECTION, conditional on a jump actually happening")
    print("    (sample restricted to rows where a >= "
          f"{THETA:g}c move DID occur -- the easiest possible version)")
    print("=" * 92)
    mtr, mte = Mtr == 1, Mte == 1
    print(f"  train {mtr.sum():,} jump rows | test {mte.sum():,} jump rows")
    if mtr.sum() > 2000 and mte.sum() > 500:
        clf2 = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=0)
        clf2.fit(Xtr[mtr], Dtr[mtr])
        pd_ = clf2.predict_proba(Xte[mte])[:, 1]
        auc2 = roc_auc_score(Dte[mte], pd_)
        print(f"  test AUC {auc2:.4f}   (0.50 = coin flip)")
        print(f"  base P(up) {Dte[mte].mean()*100:.2f}%")
        print(f"  {'top k%':>8} {'rows':>8} {'P(up)':>9} {'mean signed move':>18}")
        for k in (5.0, 10.0, 25.0, 50.0):
            n = max(int(mte.sum() * k / 100), 30)
            s = np.argsort(-pd_)[:n]
            print(f"  {k:>7.1f}% {n:>8,} {Dte[mte][s].mean()*100:>8.2f}% "
                  f"{Vte[mte][s].mean():>+17.4f}c")
        out["direction"] = {"auc": float(auc2),
                            "n_test": int(mte.sum()),
                            "base_up": float(Dte[mte].mean())}
    else:
        print("  not enough jump rows")

    print()
    print("  Interpretation: an AUC near 0.50 with mean signed move near 0c means")
    print("  the sign of a jump is unpredictable even when you KNOW one is coming.")
    print("  In that case no improvement in magnitude accuracy makes a taker")
    print("  strategy viable, because direction is the missing term, not precision.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
