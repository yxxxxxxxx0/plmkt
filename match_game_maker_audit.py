"""Read-only live book audit for the strict current-match report."""
import csv
from pathlib import Path
import requests

src = Path("polymarket_sports/reports/current_match_game_delay_audit.csv")
dst = Path("polymarket_sports/reports/current_match_game_maker_audit.csv")
session = requests.Session(); rows = []
for r in csv.DictReader(src.open(encoding="utf-8")):
    if r["status"] != "checked":
        continue
    m = session.get("https://clob.polymarket.com/markets/" + r["condition_id"], timeout=30).json()
    for tok in m.get("tokens", []):
        b = session.get("https://clob.polymarket.com/book", params={"token_id": tok["token_id"]}, timeout=30).json()
        bids = sorted([(float(x["price"]), float(x["size"])) for x in b.get("bids", [])], reverse=True)
        asks = sorted([(float(x["price"]), float(x["size"])) for x in b.get("asks", [])])
        bid, ask = (bids[0] if bids else (None, 0)), (asks[0] if asks else (None, 0))
        spread = (ask[0] - bid[0]) * 100 if bid[0] is not None and ask[0] is not None else None
        rows.append({"dataset_category": r["dataset_category"], "event_title": r["event_title"],
                     "question": r["question"], "seconds_delay": r["seconds_delay"],
                     "outcome": tok.get("outcome"), "minimum_order_size": m.get("minimum_order_size"),
                     "reward_min_size": (m.get("rewards") or {}).get("min_size"),
                     "reward_max_spread": (m.get("rewards") or {}).get("max_spread"),
                     "best_bid": bid[0], "best_ask": ask[0], "spread_cents": spread,
                     "best_bid_size": bid[1], "best_ask_size": ask[1]})
with dst.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"Wrote {len(rows)} token rows to {dst}")
for r in rows:
    print(f"{r['dataset_category']:<20} {r['outcome']:<12} delay={r['seconds_delay']}s spread={r['spread_cents']}c sizes={r['best_bid_size']}/{r['best_ask_size']} reward_min={r['reward_min_size']}")
