"""Scan active Polymarket books for small-bankroll maker regimes.

Read-only: this program never authenticates and never places or cancels orders.
It evaluates paired resting BUY quotes on the two outcomes of a binary market.
If both quotes fill, one complete set pays $1; if only one fills, the maker has
directional inventory and adverse-selection risk.

The score is intentionally a screening score, not an expected-PnL estimate.
Run multiple samples so spread persistence and short-horizon markout risk enter
the decision.  A CSV is written for later paper-fill validation.
"""

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests


CLOB_SAMPLING = "https://clob.polymarket.com/sampling-markets"
CLOB_BOOKS = "https://clob.polymarket.com/books"
GEO = "https://polymarket.com/api/geoblock"
BASE = Path(__file__).resolve().parent


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def discover(session, limit):
    rows, cursor = [], None
    while limit <= 0 or len(rows) < limit:
        params = {}
        if cursor:
            params["next_cursor"] = cursor
        response = session.get(CLOB_SAMPLING, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("data", [])
        if not page:
            break
        for market in page:
            tokens = market.get("tokens") or []
            if (market.get("active", True) and market.get("enable_order_book")
                    and market.get("accepting_orders", True) and not market.get("closed")
                    and len(tokens) == 2):
                market["_tokens"] = [str(t["token_id"]) for t in tokens]
                market["_outcomes"] = [t["outcome"] for t in tokens]
                # Normalize CLOB sampling fields used by the scoring code.
                market["id"] = market.get("condition_id", "")
                market["slug"] = market.get("market_slug", "")
                market["orderMinSize"] = market.get("minimum_order_size")
                market["orderPriceMinTickSize"] = market.get("minimum_tick_size")
                market["endDate"] = market.get("end_date_iso")
                market["feesEnabled"] = bool(market.get("taker_base_fee"))
                market["sportsMarketType"] = ",".join(market.get("tags") or [])
                rows.append(market)
                if limit > 0 and len(rows) >= limit:
                    break
        next_cursor = payload.get("next_cursor")
        # CLOB uses base64("-1") == "LTE=" as its terminal cursor sentinel.
        if not next_cursor or next_cursor == "LTE=" or next_cursor == cursor:
            break
        cursor = next_cursor
    return rows


def fetch_books(session, token_ids):
    result = {}
    for start in range(0, len(token_ids), 200):
        payload = [{"token_id": token} for token in token_ids[start:start + 200]]
        response = session.post(CLOB_BOOKS, json=payload, timeout=40)
        response.raise_for_status()
        for book in response.json():
            result[str(book["asset_id"])] = book
    return result


def touch(book):
    bids, asks = book.get("bids") or [], book.get("asks") or []
    if not bids or not asks:
        return None
    bid_row = max(bids, key=lambda row: number(row.get("price"), -1))
    ask_row = min(asks, key=lambda row: number(row.get("price"), 2))
    bid, ask = number(bid_row["price"]), number(ask_row["price"])
    if not 0 < bid < ask < 1:
        return None
    return {
        "bid": bid, "ask": ask,
        "bid_size": number(bid_row.get("size")),
        "ask_size": number(ask_row.get("size")),
        "mid": (bid + ask) / 2,
        "spread": ask - bid,
    }


def hours_to_end(market):
    raw = market.get("endDate") or market.get("endDateIso")
    if not raw:
        return math.inf
    try:
        end = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return (end - datetime.now(timezone.utc)).total_seconds() / 3600
    except ValueError:
        return math.inf


def snapshot(markets, books, bankroll, allocation, max_spread, min_mid):
    now = time.time()
    rows = []
    per_market_budget = bankroll * allocation
    for market in markets:
        a = touch(books.get(market["_tokens"][0], {}))
        b = touch(books.get(market["_tokens"][1], {}))
        if not a or not b:
            continue
        min_shares = number(market.get("orderMinSize"), 5.0)
        tick = number(market.get("orderPriceMinTickSize"), 0.01)

        # Paired BUY quotes at the current best bids. If both fill, this is the
        # complete-set margin. Capital is reserved for both outstanding legs.
        pair_margin = 1.0 - a["bid"] - b["bid"]
        pair_capital = min_shares * (a["bid"] + b["bid"])
        queue_ahead = a["bid_size"] + b["bid_size"]
        min_spread = min(a["spread"], b["spread"])
        mid_consistency = abs(a["mid"] + b["mid"] - 1.0)
        liquid_book = (
            max(a["spread"], b["spread"]) <= max_spread + 1e-12
            and min_mid <= a["mid"] <= 1 - min_mid
            and min_mid <= b["mid"] <= 1 - min_mid
            and min(a["bid_size"], b["bid_size"]) >= min_shares
        )
        rows.append({
            "ts": now,
            "id": str(market.get("id", "")),
            "slug": market.get("slug", ""),
            "question": market.get("question", ""),
            "outcome_a": market["_outcomes"][0],
            "outcome_b": market["_outcomes"][1],
            "bid_a": a["bid"], "ask_a": a["ask"], "mid_a": a["mid"],
            "bid_b": b["bid"], "ask_b": b["ask"], "mid_b": b["mid"],
            "spread_a": a["spread"], "spread_b": b["spread"],
            "pair_margin": pair_margin,
            "pair_capital": pair_capital,
            "queue_ahead": queue_ahead,
            "min_spread": min_spread,
            "mid_consistency": mid_consistency,
            "min_shares": min_shares,
            "tick": tick,
            "volume24h": number(market.get("volume24hrClob") or market.get("volume24hr")),
            "liquidity": number(market.get("liquidityClob") or market.get("liquidity")),
            "hours_to_end": hours_to_end(market),
            "fees_enabled": bool(market.get("feesEnabled")),
            "sports_type": market.get("sportsMarketType") or "",
            "capital_ok": pair_capital <= per_market_budget,
            "two_ticks": min_spread >= 2 * tick - 1e-12,
            "liquid_book": liquid_book,
        })
    return rows


def summarize(history, min_samples, enter_score, exit_score):
    output = []
    for market_id, samples in history.items():
        if len(samples) < min_samples:
            continue
        last = samples[-1].copy()
        mids = [row["mid_a"] for row in samples]
        margins = [row["pair_margin"] for row in samples]
        valid = [row["capital_ok"] and row["two_ticks"] and row["liquid_book"]
                 and row["hours_to_end"] > 0
                 for row in samples]
        changes = [abs(mids[i] - mids[i - 1]) for i in range(1, len(mids))]
        touch_changes = sum(
            any(samples[i][key] != samples[i - 1][key]
                for key in ("bid_a", "ask_a", "bid_b", "ask_b"))
            for i in range(1, len(samples))
        )
        volatility = sum(changes) / max(1, len(changes))
        persistence = sum(valid) / len(valid)
        margin = sum(margins) / len(margins)
        # Higher pair margin and activity help. Volatility relative to the
        # margin, a huge touch queue, and long capital lock-up hurt.
        activity = math.log1p(last["volume24h"])
        queue_penalty = math.log1p(last["queue_ahead"])
        lock_days = max(0.0, min(last["hours_to_end"] / 24, 365.0))
        score = (
            45.0 * margin
            + 1.8 * persistence
            + 0.09 * activity
            - 18.0 * volatility
            - 0.055 * queue_penalty
            - 0.002 * lock_days
            - 8.0 * last["mid_consistency"]
        )
        hard_ok = (
            last["capital_ok"] and last["two_ticks"] and last["liquid_book"]
            and persistence >= 0.75
            and margin > 0 and last["hours_to_end"] > 0
            and volatility <= max(last["tick"], margin / 2)
            and touch_changes > 0
        )
        last.update({
            "samples": len(samples), "persistence": persistence,
            "mean_abs_mid_move": volatility, "mean_pair_margin": margin,
            "touch_changes": touch_changes,
            "score": score, "enter": hard_ok and score >= enter_score,
            "remain": hard_ok and score >= exit_score,
        })
        output.append(last)
    return sorted(output, key=lambda row: row["score"], reverse=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bankroll", type=float, default=50.0)
    parser.add_argument("--allocation", type=float, default=0.10,
                        help="maximum fraction reserved for one paired quote")
    parser.add_argument("--max-markets", type=int, default=0,
                        help="0 scans every active binary market")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--monitor-markets", type=int, default=300,
                        help="after the universe snapshot, repeatedly sample only this shortlist")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--enter-score", type=float, default=2.3)
    parser.add_argument("--exit-score", type=float, default=1.9)
    parser.add_argument("--max-spread", type=float, default=0.05,
                        help="reject dead/wide books; applied to both outcome books")
    parser.add_argument("--min-mid", type=float, default=0.05,
                        help="reject extreme-probability outcomes")
    parser.add_argument("--output", type=Path,
                        default=BASE / "data" / "maker_regimes.csv")
    args = parser.parse_args()
    if not 0 < args.allocation <= 0.5:
        raise SystemExit("--allocation must be in (0, 0.5]")

    session = requests.Session()
    try:
        geo = session.get(GEO, timeout=15).json()
        print(f"geoblock: blocked={geo.get('blocked')} country={geo.get('country')} "
              f"region={geo.get('region')}")
    except Exception as exc:
        print(f"geoblock check unavailable: {type(exc).__name__}")

    markets = discover(session, args.max_markets)
    print(f"active binary markets discovered: {len(markets)}")
    history = defaultdict(list)
    for index in range(args.samples):
        tokens = [token for market in markets for token in market["_tokens"]]
        books = fetch_books(session, tokens)
        rows = snapshot(markets, books, args.bankroll, args.allocation,
                        args.max_spread, args.min_mid)
        for row in rows:
            history[row["id"]].append(row)
        print(f"sample {index + 1}/{args.samples}: {len(rows)} complete books")
        if index == 0 and args.samples > 1:
            eligible = [row for row in rows if row["capital_ok"] and row["two_ticks"]
                        and row["liquid_book"]
                        and row["pair_margin"] > 0 and row["hours_to_end"] > 0]
            eligible.sort(
                key=lambda row: (45 * row["pair_margin"]
                                 - 0.055 * math.log1p(row["queue_ahead"])
                                 - 0.002 * min(row["hours_to_end"] / 24, 365)),
                reverse=True,
            )
            keep = {row["id"] for row in eligible[:args.monitor_markets]}
            markets = [market for market in markets if str(market.get("id")) in keep]
            print(f"static eligible: {len(eligible)}; monitoring shortlist: {len(markets)}")
        if index + 1 < args.samples:
            time.sleep(args.interval)

    ranked = summarize(history, args.samples, args.enter_score, args.exit_score)
    write_csv(args.output, ranked)
    enter = [row for row in ranked if row["enter"]]
    print(f"ranked markets: {len(ranked)}; ENTER candidates: {len(enter)}")
    print(f"output: {args.output}")
    shown = enter[:args.top] if enter else ranked[:args.top]
    heading = "ENTER candidates" if enter else "Top rejected candidates"
    print(f"\n{heading} (screening score, not expected profit):")
    for row in shown:
        state = "ENTER" if row["enter"] else ("HOLD" if row["remain"] else "SKIP")
        print(
            f"{state:5} score={row['score']:5.2f} margin={row['mean_pair_margin']:.3f} "
            f"capital=${row['pair_capital']:.2f} persist={row['persistence']:.0%} "
            f"move={row['mean_abs_mid_move']:.4f} vol24h=${row['volume24h']:.0f} "
            f"q={row['question'][:65]}"
        )
    print("\nNo orders were placed. ENTER means eligible for paper-fill tracking only.")


if __name__ == "__main__":
    main()
