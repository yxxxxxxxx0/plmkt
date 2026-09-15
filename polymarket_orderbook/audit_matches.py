"""Which matches survive the filters, across every built session.

Run before training so the training set is a deliberate choice rather than
whatever happened to be on disk.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from match_filter import load_windows, match_report

JD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jump")


def main():
    pd.set_option("display.width", 200)
    W = load_windows()
    allrep = []
    for f in sorted(glob.glob(os.path.join(JD, "feat_*.parquet"))):
        F = pd.read_parquet(f, columns=["series", "ts"])
        r = match_report(F, W)
        r.insert(0, "session", os.path.basename(f)
                 .replace("feat_books_", "").replace(".parquet", ""))
        allrep.append(r)
    if not allrep:
        raise SystemExit("no datasets built")
    R = pd.concat(allrep, ignore_index=True)
    print(R.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print(f"\n{'='*96}")
    kept = R[R.keep]
    print(f"KEEP  {len(kept):>3} matches, {kept.n.sum():>12,} rows")
    print(f"DROP  {(~R.keep).sum():>3} matches, {R[~R.keep].n.sum():>12,} rows")
    print(f"{'='*96}")
    for _, r in R[~R.keep].iterrows():
        print(f"  DROP {r.session:>12} {r.slug:<28} {r.reason}")
    print(f"\nper session:")
    g = R.groupby("session").agg(matches=("slug", "size"),
                                 kept=("keep", "sum"),
                                 rows=("n", "sum"))
    g["kept_rows"] = R[R.keep].groupby("session").n.sum()
    g["ingame_mean"] = R[R.keep].groupby("session").ingame_frac.mean()
    print(g.to_string(float_format=lambda v: f"{v:,.3f}"))
    R.to_csv(os.path.join(JD, "match_audit.csv"), index=False)
    print(f"\nwrote data/jump/match_audit.csv")


if __name__ == "__main__":
    main()
