"""PCA on the order-book shape: do its factors predict SIGNED price moves?

The GBM in ml_taker.py already consumes all of these columns and reaches
AUC 0.865 out of sample, so PCA cannot add information -- it is a linear
compression of the same inputs. It is run anyway because a factor structure
is interpretable in a way a boosted tree is not: if PC1 is "book pressure"
and it carries directional signal, that is a far more robust thing to trade
than a 250-tree ensemble.

Loadings are fitted on TRAIN slates and applied unchanged to TEST. The score
is the correlation with the SIGNED forward mid move, because every other
model in this repo has turned out to predict volatility rather than
direction, and direction is what a taker needs.
"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))

TRAIN = ("2026-08-27", "2026-08-28")
TEST = ("2026-09-04",)
BOOK = ["bb_sz", "ba_sz", "dep_bid_1c", "dep_ask_1c", "dep_bid_2c",
        "dep_ask_2c", "dep_bid_5c", "dep_ask_5c", "n_bid", "n_ask",
        "slope_bid_c", "slope_ask_c", "spread_c"]
TGT = ["fwd_mid_1s", "fwd_mid_5s", "fwd_long_1s"]


def load(slates, market_type="moneyline", cap=400_000):
    X, Y = [], {k: [] for k in TGT}
    files = []
    for s in slates:
        files += sorted(glob.glob(os.path.join("data", "panel", f"*-{s}.parquet")))
    per = max(cap // max(len(files), 1), 1000)
    rng = np.random.default_rng(0)
    for path in files:
        tab = pq.read_table(path, columns=BOOK + TGT + ["market_type", "bb_c", "ba_c"])
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in tab.column_names}
        m = (d["market_type"] == market_type)
        m &= np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"]) & (d["ba_c"] > d["bb_c"])
        for k in TGT:
            m &= np.isfinite(d[k])
        for c in BOOK:
            m &= np.isfinite(d[c])
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        if idx.size > per:
            idx = rng.choice(idx, per, replace=False)
        X.append(np.column_stack([np.asarray(d[c], np.float64)[idx] for c in BOOK]))
        for k in TGT:
            Y[k].append(np.asarray(d[k], np.float64)[idx])
    return np.vstack(X), {k: np.concatenate(v) for k, v in Y.items()}


# log-scale the size columns: depths span several orders of magnitude and a
# raw PCA would simply recover "which token has the biggest book".
def prep(X):
    Z = X.copy()
    for j, c in enumerate(BOOK):
        if c.startswith(("bb_sz", "ba_sz", "dep_")):
            Z[:, j] = np.log1p(np.maximum(Z[:, j], 0))
    return Z


Xtr, Ytr = load(TRAIN)
Xte, Yte = load(TEST)
Ztr, Zte = prep(Xtr), prep(Xte)
mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
Ztr = (Ztr - mu) / sd
Zte = (Zte - mu) / sd

U, S, Vt = np.linalg.svd(Ztr - Ztr.mean(0), full_matrices=False)
var = S ** 2 / np.sum(S ** 2)
print(f"train {Ztr.shape[0]:,} rows  test {Zte.shape[0]:,} rows  ({len(BOOK)} book columns)")
print("explained variance:", "  ".join(f"PC{i+1} {v*100:.1f}%" for i, v in enumerate(var[:5])))
print()
for i in range(4):
    load_i = sorted(zip(BOOK, Vt[i]), key=lambda x: -abs(x[1]))[:5]
    print(f"  PC{i+1} loadings: " + ", ".join(f"{n} {w:+.2f}" for n, w in load_i))
print()
Ftr = Ztr @ Vt.T
Fte = Zte @ Vt.T
print(f"{'factor':>8} " + " ".join(f"{k:>16}" for k in TGT))
print("-" * 62)
for i in range(5):
    cs = []
    for k in TGT:
        y = Yte[k]
        x = Fte[:, i]
        cs.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan)
    print(f"  PC{i+1:>5} " + " ".join(f"{c:>+16.5f}" for c in cs))
print()
print("(correlation with SIGNED forward move; direction is what a taker needs)")
