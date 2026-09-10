"""Small-sample, chronological screen of the recorded liquidity-collapse signals.

This is deliberately a *signal* test, not a PnL backtest.  The event archive
contains upward jumps and flat controls, but not the full population of
downward moves or executable fill information.  Treating its AUC as a trading
return would therefore be invalid.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "collapse" / "events.csv"


FEATURE_SETS = {
    "book imbalance": ["I5_at_0"],
    "directional withdrawals": ["withdrawal_diff", "I5_at_0"],
    "symmetric withdrawal activity": ["withdrawal_sum", "CA1_max_pre"],
    "all recorded signals": [
        "withdrawal_diff", "withdrawal_sum", "CA1_max_pre", "I5_at_0",
        "log_DA5", "log_DB5",
    ],
}


def prepare() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d["date"] = pd.to_datetime(d["slug"].str[-10:])
    d["y"] = (d["kind"] == "jump").astype(int)
    d["withdrawal_diff"] = d["CA5_max_pre"] - d["CB5_max_pre"]
    d["withdrawal_sum"] = d["CA5_max_pre"] + d["CB5_max_pre"]
    d["log_DA5"] = np.log1p(d["DA5_at_0"].clip(lower=0))
    d["log_DB5"] = np.log1p(d["DB5_at_0"].clip(lower=0))
    return d.replace([np.inf, -np.inf], np.nan).dropna()


def screen(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = sorted(d["date"].unique())
    for test_date in dates[1:]:
        train = d[d["date"] < test_date]
        test = d[d["date"] == test_date]
        for name, cols in FEATURE_SETS.items():
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.25, max_iter=2000)
            )
            model.fit(train[cols], train["y"])
            p = model.predict_proba(test[cols])[:, 1]
            rows.append({
                "test_date": pd.Timestamp(test_date).date().isoformat(),
                "model": name,
                "n": len(test),
                "base_rate": test["y"].mean(),
                "auc": roc_auc_score(test["y"], p),
                "brier": brier_score_loss(test["y"], p),
                "accuracy_05": accuracy_score(test["y"], p >= 0.5),
            })
    return pd.DataFrame(rows)


def main() -> None:
    d = prepare()
    out = screen(d)
    print(f"{len(d):,} selected events, {d.slug.nunique()} games, "
          f"{d.date.nunique()} dates")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMean across chronological test dates:")
    print(out.groupby("model")[["auc", "brier", "accuracy_05"]].mean()
          .sort_values("auc", ascending=False)
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nWARNING: upward-jump versus sampled-flat classification only; "
          "this dataset cannot establish directionally executable PnL.")


if __name__ == "__main__":
    main()
