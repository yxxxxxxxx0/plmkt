"""End-to-end: build features, run the walk-forward backtest, write reports."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import collect, features, odds
from .backtest import summarize, walk_forward
from .config import OUT, PROC, TEST_SEASONS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def build(use_odds: bool = True) -> pd.DataFrame:
    games = collect.load("games")
    team_logs = collect.load("team_logs")
    pitcher_logs = collect.load("pitcher_logs")
    print(f"games={len(games)} team_logs={len(team_logs)} pitcher_logs={len(pitcher_logs)}")

    df = features.build_dataset(games, team_logs, pitcher_logs)
    print(f"dataset={len(df)} cols={df.shape[1]}")

    if use_odds and (PROC / "odds.parquet").exists():
        od = pd.read_parquet(PROC / "odds.parquet")
        # MLB team id -> franchise name, taken from the schedule itself
        names = {}
        raw = collect._get("/teams", dict(sportId=1, season=2024), "teams_2024")
        for yr in sorted(games.season.unique()):
            j = collect._get("/teams", dict(sportId=1, season=int(yr)), f"teams_{yr}")
            for t in j["teams"]:
                names[t["id"]] = t["name"]
        df = odds.attach_odds(df, od, names)
        hit = df.mkt_p_home.notna()
        print(f"odds matched on {hit.sum()}/{len(df)} games "
              f"({hit[df.season >= min(TEST_SEASONS)].mean():.1%} of test seasons)")
    df.to_parquet(PROC / "dataset.parquet", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--seasons", type=int, nargs="*", default=TEST_SEASONS)
    args = ap.parse_args()

    if args.rebuild or not (PROC / "dataset.parquet").exists():
        df = build()
    else:
        df = pd.read_parquet(PROC / "dataset.parquet")

    # Only games where both starters were listed -- that is the realistic
    # betting universe and it avoids a large block of missing pitcher features.
    df = df[df.has_both_sp].copy()
    feat = features.feature_columns(df)
    print(f"\n{len(feat)} features, {len(df)} usable games\n")

    print("Walk-forward backtest (train = all prior seasons, test = held-out season)")
    res, preds = walk_forward(df, feat, args.seasons, edge_thresh=args.edge)

    res.to_csv(OUT / "results_by_season.csv", index=False)
    preds.to_parquet(OUT / "predictions.parquet", index=False)

    summ = summarize(res)
    summ.to_csv(OUT / "summary.csv", index=False)

    print("\n=== POOLED RESULTS ACROSS TEST SEASONS ===")
    cols = [c for c in ["model", "n", "winrate", "mkt_winrate", "logloss", "brier",
                        "auc", "cal_slope", "bets", "bet_winrate", "flat_roi"]
            if c in summ.columns]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n=== WINRATE BY SEASON ===")
    piv = res.pivot_table(index="model", columns="season", values="winrate")
    piv["mean"] = piv.mean(axis=1)
    print(piv.sort_values("mean", ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== LOG LOSS BY SEASON (lower is better) ===")
    piv2 = res.pivot_table(index="model", columns="season", values="logloss")
    piv2["mean"] = piv2.mean(axis=1)
    print(piv2.sort_values("mean").to_string(float_format=lambda v: f"{v:.4f}"))

    if "bet_flat_roi" in res.columns:
        print("\n=== BETTING SIM vs CLOSING MONEYLINE ===")
        b = res[res.bet_bets.fillna(0) > 0]
        if len(b):
            piv3 = b.pivot_table(index="model", columns="season",
                                 values="bet_flat_roi")
            print(piv3.to_string(float_format=lambda v: f"{v:+.3f}"))

    print(f"\nwrote {OUT/'summary.csv'}, {OUT/'results_by_season.csv'}, "
          f"{OUT/'predictions.parquet'}")


if __name__ == "__main__":
    main()
