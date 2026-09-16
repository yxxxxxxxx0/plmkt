"""Read-only current-book audit for a small ($100) maker account."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import requests


INPUT = Path("polymarket_sports/reports/current_taker_delay_audit.csv")
OUTPUT = Path("polymarket_sports/reports/current_maker_feasibility.csv")
BANKROLL = 100.0


def book_stats(book: dict, outcome: str) -> dict:
    bids = sorted(((float(x["price"]), float(x["size"])) for x in book.get("bids", [])), reverse=True)
    asks = sorted(((float(x["price"]), float(x["size"])) for x in book.get("asks", [])))
    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None
    spread = ask - bid if bid is not None and ask is not None else None
    midpoint = (bid + ask) / 2 if bid is not None and ask is not None else None
    bid_dollars = bid * bids[0][1] if bids else 0.0
    ask_dollars = ask * asks[0][1] if asks else 0.0
    return {
        "outcome": outcome, "best_bid": bid, "best_ask": ask, "midpoint": midpoint,
        "spread": spread, "spread_cents": spread * 100 if spread is not None else None,
        "best_bid_size": bids[0][1] if bids else 0.0, "best_ask_size": asks[0][1] if asks else 0.0,
        "best_bid_dollars": bid_dollars, "best_ask_dollars": ask_dollars,
        "levels_bid": len(bids), "levels_ask": len(asks),
        "within_bankroll_ask_shares": BANKROLL / ask if ask else 0.0,
    }


session = requests.Session()
source = list(csv.DictReader(INPUT.open(encoding="utf-8")))
out = []
for row in source:
    if row["status"] != "checked":
        continue
    market = session.get("https://clob.polymarket.com/markets/" + row["condition_id"], timeout=30).json()
    for token in market.get("tokens", []):
        book = session.get(
            "https://clob.polymarket.com/book", params={"token_id": token["token_id"]}, timeout=30
        ).json()
        stats = book_stats(book, token.get("outcome", ""))
        out.append({
            "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_category": row["dataset_category"], "question": row["question"],
            "condition_id": row["condition_id"], "seconds_delay": row["seconds_delay"],
            "minimum_order_size": market.get("minimum_order_size"),
            "minimum_tick_size": market.get("minimum_tick_size"),
            "maker_base_fee": market.get("maker_base_fee"),
            "taker_base_fee": market.get("taker_base_fee"),
            "reward_min_size": (market.get("rewards") or {}).get("min_size"),
            "reward_max_spread": (market.get("rewards") or {}).get("max_spread"),
            "token_id": token["token_id"], **stats,
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(out[0]))
    writer.writeheader(); writer.writerows(out)

print(f"Wrote {len(out)} token rows to {OUTPUT}")
for row in out:
    print(
        f"{row['dataset_category']:<24} {row['outcome']:<5} delay={row['seconds_delay']}s "
        f"bid={row['best_bid']} ask={row['best_ask']} spread={row['spread_cents']}c "
        f"top_sizes={row['best_bid_size']}/{row['best_ask_size']} min={row['minimum_order_size']}"
    )
