"""Stage 10: uncertainty on the test PR-AUCs, by cluster bootstrap over GAMES.

The test split is a single recording session, so a day-level bootstrap has one
cluster and is undefined -- that is what produced NaN confidence intervals on
the first pass. The correct resampling unit here is the GAME: contracts on the
same game move together (one run repriced the moneyline, the run line and the
total simultaneously), so treating the ~14 test games as exchangeable clusters
captures the dominant correlation while contracts or rows would not.

This reads the saved test predictions; nothing is retrained.

    python bootstrap_ci.py --tag L120_H1_train
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen")

LABEL = {"0_prevalence": "0 prevalence", "1_time_of_day": "1 time-of-day only",
         "2_logistic": "2 logistic (snapshot)", "mlp_snapshot": "3 MLP snapshot",
         "mlp_history": "4 MLP history", "cnn": "5 CNN", "lstm": "6 LSTM",
         "cnn_lstm": "7 CNN-LSTM", "cnn_lstm_attention": "8 CNN-LSTM-Attention"}


def boot(y, p, groups, n, rng):
    uniq = np.unique(groups)
    where = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([where[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(average_precision_score(y[idx], p[idx]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="L120_H1_train")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()

    z = np.load(os.path.join(RES, "preds", "test_predictions_%s.npz" % a.tag))
    y = z["y"].astype(int)
    M = pd.read_parquet(os.path.join(RES, "preds", "test_meta_%s.parquet" % a.tag))
    P = pd.read_parquet(os.path.join(CACHE, "panel_labelled.parquet"))
    smap = P.drop_duplicates("asset_id").set_index("asset_id").series
    # game slug is the first field of the series key
    games = M.asset_id.map(smap).astype(str).str.split("|").str[0].to_numpy()
    print("test: %d rows, %d games, %d positives (prevalence %.3f%%)"
          % (len(y), len(np.unique(games)), int(y.sum()), 100 * y.mean()))

    rng = np.random.default_rng(0)
    dists, rows = {}, []
    for k in z.files:
        if k == "y":
            continue
        d = boot(y, z[k], games, a.n, rng)
        dists[k] = d
        rows.append(dict(model=LABEL.get(k, k), pr_auc=average_precision_score(y, z[k]),
                         ci_lo=float(np.percentile(d, 2.5)),
                         ci_hi=float(np.percentile(d, 97.5)),
                         boot_median=float(np.median(d))))
    R = pd.DataFrame(rows).sort_values("model")
    R.to_csv(os.path.join(RES, "pr_auc_bootstrap_ci.csv"), index=False)
    print("\n=== PR-AUC with 95% cluster-bootstrap CI (resampling games) ===")
    print(R.to_string(index=False, float_format=lambda v: "%.4f" % v))

    # paired comparisons: P(model A > model B) across the same bootstrap draws
    print("\n=== paired comparisons (same resamples), P(A beats B) ===")
    pairs = [("cnn_lstm_attention", "cnn_lstm"),
             ("cnn_lstm_attention", "lstm"),
             ("cnn_lstm_attention", "cnn"),
             ("lstm", "cnn"),
             ("cnn", "mlp_snapshot"),
             ("mlp_history", "mlp_snapshot"),
             ("mlp_snapshot", "2_logistic"),
             ("2_logistic", "1_time_of_day")]
    out = []
    for A, B in pairs:
        if A not in dists or B not in dists:
            continue
        n = min(len(dists[A]), len(dists[B]))
        diff = dists[A][:n] - dists[B][:n]
        rec = dict(A=LABEL.get(A, A), B=LABEL.get(B, B),
                   mean_diff=float(diff.mean()),
                   ci_lo=float(np.percentile(diff, 2.5)),
                   ci_hi=float(np.percentile(diff, 97.5)),
                   p_A_beats_B=float((diff > 0).mean()))
        out.append(rec)
        print("  %-24s vs %-24s  diff %+.4f  [%+.4f, %+.4f]  P=%.3f"
              % (rec["A"], rec["B"], rec["mean_diff"], rec["ci_lo"],
                 rec["ci_hi"], rec["p_A_beats_B"]))
    pd.DataFrame(out).to_csv(os.path.join(RES, "pr_auc_paired_tests.csv"), index=False)
    print("\nwrote pr_auc_bootstrap_ci.csv and pr_auc_paired_tests.csv")


if __name__ == "__main__":
    sys.exit(main())
