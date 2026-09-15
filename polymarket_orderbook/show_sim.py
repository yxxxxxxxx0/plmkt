"""Print the walk-the-clock P&L summaries in one table.

Reads every data/jump/sim_summary_*.json so results from different sessions
sit side by side, which is the comparison that matters: the same model on
sessions it never trained on.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

JD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jump")


def main():
    pd.set_option("display.width", 220)
    rows = []
    for f in sorted(glob.glob(os.path.join(JD, "sim_summary_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        sess = os.path.basename(f).replace("sim_summary_", "").replace(".json", "")
        for r in d.get("results", []):
            rows.append(dict(session=sess, tau=d.get("tau"), **r))
    if not rows:
        print("no simulations yet")
        return
    R = pd.DataFrame(rows)
    cols = ["session", "entry_delay_s", "n_trades", "trades_per_hour",
            "hit_rate", "mean_pnl", "total_pnl", "total_return",
            "max_drawdown", "fees_paid", "n_series"]
    cols = [c for c in cols if c in R.columns]
    print(R[cols].to_string(index=False))


if __name__ == "__main__":
    main()
