"""Backtest immediate match-market jumps against related championship markets.

This is deliberately conservative:
* only manually validated team/competition links are used;
* signals come from the main match winner market, not props or map/game markets;
* an entry occurs after the market's historical taker delay;
* buys enter at the related future's ask and exit at its bid;
* stale/missing quotes are rejected.

The downloaded dataset records one outcome token per condition.  For a binary
match, metadata's YES token is the first named team and NO is the second team.
Probabilities and bid/ask quotes are complemented when the recorded token is
the other team.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path("polymarket_sports")
META_PATH = ROOT / "metadata" / "sports_markets.parquet"
REPORT_DIR = ROOT / "reports"

TAKER_DELAY_SECONDS = 3.0
SIGNAL_WINDOW_SECONDS = 3.0
MAX_QUOTE_WAIT_SECONDS = 10.0
COOLDOWN_SECONDS = 60.0
THRESHOLDS = (0.03, 0.05, 0.10)
HORIZONS = (10, 30, 60, 300)


@dataclass(frozen=True)
class Pair:
    label: str
    day: str
    team: str
    match_condition: str
    future_condition: str
    team_is_first: bool
    relationship: str


PAIRS = (
    Pair(
        "MOUZ vs FUT Esports -> FUT ESL Pro League",
        "2026-03-13",
        "FUT Esports",
        "0x3400943a4af3236bd99361f2905026de8c6c7f4b79f5419e95631ff7ff5b3c0a",
        "0xe4096ca705411ed8b808cf75940e1d7c3ab1f4f21ca02584e67e75e83b3f29a4",
        False,
        "same tournament",
    ),
    Pair(
        "NAVI vs FUT Esports -> FUT ESL Pro League",
        "2026-03-14",
        "FUT Esports",
        "0x056e5327893e1488134b69ef7bff7c600219227939211974645e1f6ac1da0735",
        "0xe4096ca705411ed8b808cf75940e1d7c3ab1f4f21ca02584e67e75e83b3f29a4",
        False,
        "same tournament",
    ),
    Pair(
        "FUT Esports vs Astralis -> FUT ESL Pro League",
        "2026-03-15",
        "FUT Esports",
        "0xf5edbfaf29c7e9a158eeaa71f7748b3311a1d0bc9c500a9231d1094d376e93af",
        "0xe4096ca705411ed8b808cf75940e1d7c3ab1f4f21ca02584e67e75e83b3f29a4",
        True,
        "same tournament",
    ),
    Pair(
        "Gen.G vs LYON -> Gen.G LCK playoffs",
        "2026-03-19",
        "Gen.G",
        "0x0a0daf16a2e58158d63d46c018b7161eb06b36d49f3f15188d58aa5b4e1c3c4f",
        "0x36668e37b463d0762bf6e4c053a44682f89fec1198ec07c279812067bb732604",
        True,
        "cross-tournament strength signal",
    ),
    Pair(
        "Gen.G vs G2 -> Gen.G LCK playoffs",
        "2026-03-21",
        "Gen.G",
        "0x79c98c99467097f0a00bf7ad55e4fed572cfe759e5200ef6c6564af307c18407",
        "0x36668e37b463d0762bf6e4c053a44682f89fec1198ec07c279812067bb732604",
        True,
        "cross-tournament strength signal",
    ),
)


def metadata() -> pd.DataFrame:
    return pq.read_table(META_PATH).to_pandas()


def load_conditions(day: str, condition_ids: list[str]) -> pd.DataFrame:
    path = ROOT / "orderbook" / f"orderbook_{day}.parquet"
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=[
            "timestamp_received",
            "market_id",
            "token_id",
            "best_bid",
            "best_ask",
            "mid_price",
        ],
        filter=ds.field("market_id").isin(condition_ids),
    )
    frame = table.to_pandas()
    frame = frame.sort_values("timestamp_received", kind="stable")
    return frame.drop_duplicates(
        ["timestamp_received", "market_id", "token_id", "best_bid", "best_ask", "mid_price"]
    )


def normalized_book(
    rows: pd.DataFrame, condition_id: str, yes_token: str, want_yes: bool
) -> pd.DataFrame:
    x = rows.loc[rows.market_id == condition_id].copy()
    if x.empty:
        return x
    observed_is_yes = x.token_id.astype(str).eq(str(yes_token))
    # YES book from recorded token. Complemented quotes reverse sides:
    # YES bid = 1 - NO ask; YES ask = 1 - NO bid.
    yes_mid = np.where(observed_is_yes, x.mid_price, 1.0 - x.mid_price)
    yes_bid = np.where(observed_is_yes, x.best_bid, 1.0 - x.best_ask)
    yes_ask = np.where(observed_is_yes, x.best_ask, 1.0 - x.best_bid)
    if want_yes:
        x["p"], x["bid"], x["ask"] = yes_mid, yes_bid, yes_ask
    else:
        x["p"], x["bid"], x["ask"] = 1.0 - yes_mid, 1.0 - yes_ask, 1.0 - yes_bid
    x = x[["timestamp_received", "p", "bid", "ask"]]
    x = x.dropna().sort_values("timestamp_received", kind="stable")
    return x.drop_duplicates("timestamp_received", keep="last")


def unit_scale(ts: np.ndarray) -> float:
    median = float(np.nanmedian(ts))
    if median > 1e17:
        return 1e9
    if median > 1e14:
        return 1e6
    if median > 1e11:
        return 1e3
    return 1.0


def first_at_or_after(times: np.ndarray, target: float) -> int | None:
    idx = int(np.searchsorted(times, target, side="left"))
    return idx if idx < len(times) else None


def signals(match: pd.DataFrame, threshold: float, scale: float) -> list[tuple[int, float]]:
    times = match.timestamp_received.to_numpy(dtype=np.float64) / scale
    prices = match.p.to_numpy(dtype=np.float64)
    earlier = np.searchsorted(times, times - SIGNAL_WINDOW_SECONDS, side="right") - 1
    candidates: list[tuple[int, float]] = []
    last_time = -np.inf
    for idx, prior in enumerate(earlier):
        if prior < 0:
            continue
        jump = prices[idx] - prices[prior]
        if jump >= threshold and times[idx] - last_time >= COOLDOWN_SECONDS:
            candidates.append((idx, float(jump)))
            last_time = times[idx]
    return candidates


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    meta = metadata().set_index("condition_id", drop=False)
    results: list[dict] = []
    coverage: list[dict] = []
    signal_audit: list[dict] = []

    for pair in PAIRS:
        rows = load_conditions(pair.day, [pair.match_condition, pair.future_condition])
        match_meta = meta.loc[pair.match_condition]
        future_meta = meta.loc[pair.future_condition]
        match = normalized_book(
            rows,
            pair.match_condition,
            str(match_meta.clob_token_id_yes),
            pair.team_is_first,
        )
        future = normalized_book(
            rows,
            pair.future_condition,
            str(future_meta.clob_token_id_yes),
            True,
        )
        if match.empty or future.empty:
            coverage.append({"pair": pair.label, "status": "missing rows", "match_rows": len(match), "future_rows": len(future)})
            continue

        scale = unit_scale(match.timestamp_received.to_numpy(dtype=np.float64))
        mt = match.timestamp_received.to_numpy(dtype=np.float64) / scale
        ft = future.timestamp_received.to_numpy(dtype=np.float64) / scale
        coverage.append(
            {
                "pair": pair.label,
                "status": "tested",
                "relationship": pair.relationship,
                "match_rows": len(match),
                "future_rows": len(future),
                "match_start_utc": pd.to_datetime(mt[0], unit="s", utc=True),
                "match_end_utc": pd.to_datetime(mt[-1], unit="s", utc=True),
                "future_start_utc": pd.to_datetime(ft[0], unit="s", utc=True),
                "future_end_utc": pd.to_datetime(ft[-1], unit="s", utc=True),
            }
        )

        for threshold in THRESHOLDS:
            candidate_signals = signals(match, threshold, scale)
            executable_entries = 0
            for signal_idx, jump in candidate_signals:
                signal_time = mt[signal_idx]
                entry_target = signal_time + TAKER_DELAY_SECONDS
                entry_idx = first_at_or_after(ft, entry_target)
                if entry_idx is None or ft[entry_idx] - entry_target > MAX_QUOTE_WAIT_SECONDS:
                    continue
                entry_ask = float(future.ask.iloc[entry_idx])
                if not (0.0 < entry_ask < 1.0):
                    continue
                executable_entries += 1
                base = {
                    "pair": pair.label,
                    "team": pair.team,
                    "relationship": pair.relationship,
                    "threshold": threshold,
                    "signal_time_utc": pd.to_datetime(signal_time, unit="s", utc=True),
                    "match_jump": jump,
                    "match_probability": float(match.p.iloc[signal_idx]),
                    "entry_time_utc": pd.to_datetime(ft[entry_idx], unit="s", utc=True),
                    "entry_quote_wait_s": float(ft[entry_idx] - entry_target),
                    "entry_ask": entry_ask,
                }
                for horizon in HORIZONS:
                    exit_target = ft[entry_idx] + horizon
                    exit_idx = first_at_or_after(ft, exit_target)
                    if exit_idx is None or ft[exit_idx] - exit_target > MAX_QUOTE_WAIT_SECONDS:
                        continue
                    exit_bid = float(future.bid.iloc[exit_idx])
                    if not (0.0 < exit_bid < 1.0):
                        continue
                    pnl = exit_bid - entry_ask
                    results.append(
                        {
                            **base,
                            "horizon_s": horizon,
                            "exit_time_utc": pd.to_datetime(ft[exit_idx], unit="s", utc=True),
                            "exit_bid": exit_bid,
                            "pnl_per_share": pnl,
                            "pnl_5_shares": 5.0 * pnl,
                            "return_on_cost": pnl / entry_ask,
                        }
                    )
            signal_audit.append(
                {
                    "pair": pair.label,
                    "relationship": pair.relationship,
                    "threshold": threshold,
                    "candidate_signals": len(candidate_signals),
                    "executable_entries": executable_entries,
                }
            )

    detail = pd.DataFrame(results)
    coverage_df = pd.DataFrame(coverage)
    signal_audit_df = pd.DataFrame(signal_audit)
    detail.to_csv(REPORT_DIR / "cross_market_lead_lag_trades.csv", index=False)
    coverage_df.to_csv(REPORT_DIR / "cross_market_lead_lag_coverage.csv", index=False)
    signal_audit_df.to_csv(REPORT_DIR / "cross_market_lead_lag_signals.csv", index=False)
    if detail.empty:
        summary = pd.DataFrame()
        print("No executable simulated trades passed the quote freshness checks.")
    else:
        summary = (
            detail.groupby(["relationship", "threshold", "horizon_s"], dropna=False)
            .agg(
                trades=("pnl_per_share", "size"),
                wins=("pnl_per_share", lambda x: int((x > 0).sum())),
                win_rate=("pnl_per_share", lambda x: float((x > 0).mean())),
                mean_pnl_per_share=("pnl_per_share", "mean"),
                median_pnl_per_share=("pnl_per_share", "median"),
                total_pnl_5_shares=("pnl_5_shares", "sum"),
                mean_return_on_cost=("return_on_cost", "mean"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))
    summary.to_csv(REPORT_DIR / "cross_market_lead_lag_summary.csv", index=False)
    print("\nCoverage:\n", coverage_df.to_string(index=False))
    print("\nSignals:\n", signal_audit_df.to_string(index=False))
    print(f"\nWrote reports under {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
