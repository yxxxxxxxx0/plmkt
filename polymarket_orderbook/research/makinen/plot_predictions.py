"""Stage 11: what the model was doing around each kind of outcome.

For representative true positives, false positives, false negatives and true
negatives, plot on one clock:

  * the 1-minute mid price of that contract
  * the Lee-Mykland jumps actually detected on it
  * the model's predicted probability, and the frozen decision threshold

The question the true-positive panels are meant to answer is whether the
probability RISES BEFORE the jump or only at it. Window is 60 minutes before
the prediction instant and 15 after, as asked.

    python plot_predictions.py --model cnn_lstm_attention
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen")
OUT = os.path.join(RES, "prediction_examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="L120_H1_train")
    ap.add_argument("--model", default="cnn_lstm_attention")
    ap.add_argument("--n-each", type=int, default=2)
    ap.add_argument("--pre", type=int, default=60)
    ap.add_argument("--post", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    z = np.load(os.path.join(RES, "preds", "test_predictions_%s.npz" % a.tag))
    if a.model not in z:
        raise SystemExit("no predictions for %s (have %s)" % (a.model, list(z)))
    y, p = z["y"].astype(int), z[a.model]
    M = pd.read_parquet(os.path.join(RES, "preds", "test_meta_%s.parquet" % a.tag))
    M = M.reset_index(drop=True)
    M["y"], M["p"] = y, p
    R = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    label = {"mlp_snapshot": "3 MLP snapshot", "mlp_history": "4 MLP history",
             "cnn": "5 CNN", "lstm": "6 LSTM", "cnn_lstm": "7 CNN-LSTM",
             "cnn_lstm_attention": "8 CNN-LSTM-Attention"}[a.model]
    thr = float(R[R.model == label].threshold.iloc[0])
    M["pred"] = (M.p >= thr).astype(int)

    P = pd.read_parquet(os.path.join(CACHE, "panel_labelled.parquet"))

    groups = {
        "true_positive": M[(M.y == 1) & (M.pred == 1)].nlargest(a.n_each, "p"),
        "false_negative": M[(M.y == 1) & (M.pred == 0)].nsmallest(a.n_each, "p"),
        "false_positive": M[(M.y == 0) & (M.pred == 1)].nlargest(a.n_each, "p"),
        "true_negative": M[(M.y == 0) & (M.pred == 0)].nsmallest(a.n_each, "p"),
    }
    for kind, sub in groups.items():
        for i, (_, ev) in enumerate(sub.iterrows()):
            g = P[P.asset_id == ev.asset_id].sort_values("minute")
            w = g[(g.minute >= ev.minute - a.pre) & (g.minute <= ev.minute + a.post)]
            if len(w) < 10:
                continue
            pp = M[(M.asset_id == ev.asset_id)
                   & (M.minute >= ev.minute - a.pre)
                   & (M.minute <= ev.minute + a.post)].sort_values("minute")
            fig, ax = plt.subplots(figsize=(11, 4.6))
            ax.plot(w.ts_hkt, w.mid, lw=1.3, color="#1f77b4", label="mid price")
            j = w[w.jump == 1]
            ax.scatter(j.ts_hkt, j.mid, marker="*", s=200, color="#d62728",
                       edgecolor="black", zorder=4, label="Lee-Mykland jump")
            ax.axvline(ev.ts_hkt, color="#555555", ls="--", lw=1.0,
                       label="prediction instant t")
            ax.set_ylabel("mid price"); ax.set_xlabel("time (HKT)")
            ax2 = ax.twinx()
            ax2.plot(pp.ts_hkt, pp.p, lw=1.4, color="#ff7f0e",
                     label="P(jump in next minute)")
            ax2.axhline(thr, color="#ff7f0e", ls=":", lw=1.1,
                        label="frozen threshold %.3f" % thr)
            ax2.set_ylabel("predicted probability", color="#ff7f0e")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="y", colors="#ff7f0e")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
            ax.set_title("%s  |  %s\n%s  |  label=%d  p=%.3f  (%s)"
                         % (kind.replace("_", " ").upper(),
                            str(ev.session), ev.ts_hkt.strftime("%Y-%m-%d %H:%M HKT"),
                            int(ev.y), float(ev.p), label), fontsize=9)
            ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
            fig.savefig(os.path.join(OUT, "%s_%d.png" % (kind, i + 1)), dpi=130)
            plt.close(fig)
        print("  %-16s %d example(s)" % (kind, len(sub)))

    # attention: which features the paper's model leaned on
    fa = os.path.join(RES, "attention_weights.csv")
    if os.path.exists(fa) and a.model == "cnn_lstm_attention":
        A = pd.read_csv(fa).head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(A.feature[::-1], A.attention_weight[::-1], color="#1f77b4")
        ax.axvline(1.0, color="k", ls="--", lw=1, label="uniform weight = 1")
        ax.set_xlabel("mean feature-attention weight (test set)")
        ax.set_title("CNN-LSTM-Attention: top 20 weighted features")
        ax.legend(); ax.grid(alpha=.25, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "attention_top_features.png"), dpi=130)
        plt.close(fig)
        print("\n  top 10 attention-weighted features:")
        print(A.head(10).to_string(index=False))
    print("\nwrote examples to", OUT)


if __name__ == "__main__":
    sys.exit(main())
