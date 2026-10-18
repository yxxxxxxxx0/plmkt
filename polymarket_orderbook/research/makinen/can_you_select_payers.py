"""
Can a selector actually reach the 77.3% precision the marked set demands?

Justin's question: if I beat 77.3% precision at a 1.1% base rate, I capture the
935 paying jumps and make money. The arithmetic is right. This asks whether the
precision is reachable at all, using only information available at the instant
the jump starts.

Setup. Every durable jump from oracle_jump_scan.py becomes one row. The label is
whether that jump nets positive after the spread at both ends and the fee on
both legs. The features are the book and price state at the jump's onset --
nothing from the future, nothing from the move itself.

Split is by whole game and chronological: the last two sessions are held out, so
no game and no day appears in both sides.

The scoring that matters is not precision in the abstract, it is the realised
P&L of the jumps the model actually picks, at each operating point, under two
exit rules:

  * exit at the peak     -- the rule the label was built on, and an oracle exit
  * exit 3s after peak   -- what a system reacting to its own signal would get

Outputs, under results/makinen/oracle_jumps/
  selector_operating_points.csv   precision and realised P&L at each threshold
  selector_importance.csv         what the model leans on
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

FEATURES = [
    "spread_ticks", "imb1", "imb3", "imb5", "imb10",
    "log_bid_usd", "log_ask_usd", "conc_bid", "conc_ask",
    "reach_bid", "reach_ask", "microprice_dev",
    "rv_5", "rv_25", "rv_150", "ofi_5", "dmid_5", "ofi_25", "dmid_25",
    "stale", "book_age_ms", "mid",
]
def _test_sessions(n_held_out=2):
    """The chronologically LAST `n_held_out` sessions, discovered not listed.

    This was the hardcoded pair ["books_2026-09-12", "books_2026-09-13"], and
    it silently stopped being a chronological split the moment later sessions
    were built: 09-12 and 09-13 stayed the test set while 09-17 through 09-20
    -- recorded AFTER them -- joined TRAINING. The "no game straddles the
    split" assertion still passed, because no game does, so nothing announced
    it. A model fitted on the future of its own test period is not a held-out
    result, and every score it produces is suspect upwards.

    Session names are `books_YYYY-MM-DD`, so lexical order is chronological.
    """
    suffix = "_trimmed.parquet"
    built = sorted(q.name[len("feat_"):-len(suffix)]
                   for q in (ROOT / "data" / "jump").glob("feat_books_*" + suffix))
    return built[-n_held_out:]


TEST_SESSIONS = _test_sessions()


def build(outdir: Path) -> pd.DataFrame:
    ev = pd.read_parquet(outdir / "events.parquet")
    ev = ev[ev.is_jump & np.isfinite(ev.pnl_peak) & np.isfinite(ev.pnl_hold)].copy()

    parts = []
    for sess, g in ev.groupby("session"):
        src = ROOT / "data" / "jump" / f"feat_{sess}_trimmed.parquet"
        feat = pd.read_parquet(src, columns=FEATURES + ["series", "ts"])
        merged = g.merge(feat, on=["series", "ts"], how="left", suffixes=("", "_f"))
        parts.append(merged)
        print(f"  {sess}: {len(g):,} jumps, "
              f"{merged[FEATURES].isna().any(axis=1).sum():,} with missing features")
    df = pd.concat(parts, ignore_index=True)
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    args = ap.parse_args()
    outdir = Path(args.outdir)

    import lightgbm as lgb

    df = build(outdir)
    df["y"] = (df.pnl_peak > 0).astype(int)

    te = df.session.isin(TEST_SESSIONS)
    tr = ~te
    # No game may straddle the split.
    assert not set(df.slug[tr]) & set(df.slug[te]), "game leaked across the split"

    Xtr, ytr = df.loc[tr, FEATURES], df.loc[tr, "y"]
    Xte, yte = df.loc[te, FEATURES], df.loc[te, "y"]
    print()
    print(f"train {len(Xtr):,} jumps, {ytr.sum():,} payers ({ytr.mean():.2%})")
    print(f"test  {len(Xte):,} jumps, {yte.sum():,} payers ({yte.mean():.2%})")

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, class_weight="balanced",
        random_state=0, verbosity=-1,
    )
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    from sklearn.metrics import average_precision_score, roc_auc_score
    base = yte.mean()
    print()
    print(f"ROC-AUC {roc_auc_score(yte, p):.4f}   "
          f"PR-AUC {average_precision_score(yte, p):.4f} "
          f"(floor {base:.4f}, lift {average_precision_score(yte, p)/base:.2f}x)")

    sub = df.loc[te].copy()
    sub["p"] = p
    order = sub.sort_values("p", ascending=False)

    rows = []
    for frac in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]:
        k = max(1, int(round(frac * len(order))))
        sel = order.head(k)
        rows.append({
            "top_frac": frac,
            "n_selected": k,
            "n_payers": int(sel.y.sum()),
            "precision": float(sel.y.mean()),
            "recall": float(sel.y.sum() / max(1, yte.sum())),
            "ev_exit_at_peak": float(sel.pnl_peak.mean()),
            "ev_exit_3s_after": float(sel.pnl_hold.mean()),
            "total_ticks_at_peak": float(sel.pnl_peak.sum()),
        })
    ops = pd.DataFrame(rows)
    ops.to_csv(outdir / "selector_operating_points.csv", index=False)
    print()
    print("OPERATING POINTS on the held-out sessions")
    print(ops.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    imp = pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(outdir / "selector_importance.csv", index=False)
    print()
    print("top features by gain")
    print(imp.head(8).to_string(index=False))

    best = ops.loc[ops.precision.idxmax()]
    print()
    print(f"best precision anywhere on the curve: {best.precision:.1%} "
          f"at the top {best.top_frac:.1%} ({int(best.n_selected)} trades)")
    print(f"  -- the marked set needs 77.3%")
    print(f"  -- realised P&L there: {best.ev_exit_at_peak:+.2f} t/trade at an "
          f"oracle exit, {best.ev_exit_3s_after:+.2f} t/trade exiting 3s late")


if __name__ == "__main__":
    main()
