"""Long-running, read-only paired-maker paper simulation.

Conservative mechanics:
* markets must report seconds_delay == 0;
* quotes improve the best bid by one tick but remain post-only;
* a maker fill is counted only after a changed last-trade price trades at or
  below our BUY quote (or the subsequent ask crosses it);
* both outcome fills are immediately combined for their fixed $1 payout;
* an unmatched leg is liquidated at the displayed bid on regime exit/timeout,
  with a conservative 7% fee-curve rate: C*r*p*(1-p);
* no real credentials are loaded and no orders are sent.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from maker_regime_scanner import discover, fetch_books, hours_to_end, number, touch


BASE = Path(__file__).resolve().parent


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def get_rows(markets, books, max_pair_capital, max_spread, min_mid):
    rows = []
    for market in markets:
        # Explicit equality: missing/None delay is not accepted as zero.
        if number(market.get("seconds_delay"), -1) != 0:
            continue
        ta = touch(books.get(market["_tokens"][0], {}))
        tb = touch(books.get(market["_tokens"][1], {}))
        if not ta or not tb:
            continue
        tick = number(market.get("orderPriceMinTickSize"), 0.01)
        shares = number(market.get("orderMinSize"), 5)
        qa = min(ta["bid"] + tick, ta["ask"] - tick)
        qb = min(tb["bid"] + tick, tb["ask"] - tick)
        margin = 1 - qa - qb
        capital = shares * (qa + qb)
        spread_ok = (2 * tick <= ta["spread"] <= max_spread
                     and 2 * tick <= tb["spread"] <= max_spread)
        mid_ok = (min_mid <= ta["mid"] <= 1 - min_mid
                  and min_mid <= tb["mid"] <= 1 - min_mid)
        consistent = abs(ta["mid"] + tb["mid"] - 1) <= 2 * tick
        if not (margin > 0 and capital <= max_pair_capital and spread_ok
                and mid_ok and consistent and hours_to_end(market) > 0):
            continue
        rows.append({
            "id": str(market["id"]), "market": market, "ta": ta, "tb": tb,
            "qa": qa, "qb": qb, "shares": shares, "margin": margin,
            "capital": capital,
            # Prefer margin, shorter lock-up, and smaller queue at the old touch.
            "score": 50 * margin
                     - 0.05 * math.log1p(ta["bid_size"] + tb["bid_size"])
                     - 0.002 * min(hours_to_end(market) / 24, 365),
        })
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def liquidate(order, row, fee_rate):
    side = "a" if order["fill_a"] else "b"
    bid = row["ta"]["bid"] if side == "a" else row["tb"]["bid"]
    cost = order["shares"] * (order["qa"] if side == "a" else order["qb"])
    gross = order["shares"] * bid
    fee = order["shares"] * fee_rate * bid * (1 - bid)
    return gross - fee - cost, gross - fee, fee, side, bid


def write_status(path, state, orders, extra=None):
    payload = {
        "updated_at": iso_now(), "starting_capital": state["starting"],
        "cash": state["cash"], "realized_pnl": state["cash"] - state["starting"],
        "open_orders": len(orders), "completed_pairs": state["pairs"],
        "one_leg_exits": state["one_leg_exits"], "maker_fills": state["maker_fills"],
        "fees_paid": state["fees"], "zero_delay_only": True,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", required=True, help="ISO timestamp with offset")
    ap.add_argument("--capital", type=float, default=50)
    ap.add_argument("--max-pair-capital", type=float, default=5)
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--poll", type=float, default=5)
    ap.add_argument("--refresh", type=float, default=1800)
    ap.add_argument("--one-leg-timeout", type=float, default=1800)
    ap.add_argument("--max-spread", type=float, default=0.05)
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--taker-fee-rate", type=float, default=0.07)
    ap.add_argument("--shortlist", type=int, default=300)
    ap.add_argument("--dir", type=Path, default=BASE / "data" / "paper_maker")
    args = ap.parse_args()
    end = datetime.fromisoformat(args.until).astimezone(timezone.utc).timestamp()
    if end <= time.time():
        raise SystemExit("--until must be in the future")

    args.dir.mkdir(parents=True, exist_ok=True)
    events_path = args.dir / "events.csv"
    status_path = args.dir / "status.json"
    config_path = args.dir / "config.json"
    config_path.write_text(json.dumps(vars(args) | {"dir": str(args.dir)},
                                      indent=2, default=str), encoding="utf-8")

    import requests
    session = requests.Session()
    state = {"starting": args.capital, "cash": args.capital, "pairs": 0,
             "one_leg_exits": 0, "maker_fills": 0, "fees": 0.0}
    orders, previous_trade = {}, {}
    markets, shortlist, rows = [], [], {}
    next_refresh = 0.0

    while time.time() < end:
        try:
            if time.time() >= next_refresh or not shortlist:
                universe = discover(session, 0)
                zero_delay = [m for m in universe
                              if number(m.get("seconds_delay"), -1) == 0]
                all_books = fetch_books(
                    session, [t for m in zero_delay for t in m["_tokens"]]
                )
                ranked = get_rows(zero_delay, all_books, args.max_pair_capital,
                                  args.max_spread, args.min_mid)
                shortlist = [row["market"] for row in ranked[:args.shortlist]]
                markets = {str(m["id"]): m for m in shortlist}
                next_refresh = time.time() + args.refresh
                append_csv(events_path, {
                    "time": iso_now(), "event": "UNIVERSE", "market": "",
                    "question": "", "side": "", "price": "", "shares": "",
                    "pnl": "", "fee": "", "cash": state["cash"],
                    "detail": f"all={len(universe)} zero_delay={len(zero_delay)} "
                              f"eligible={len(ranked)} shortlist={len(shortlist)}",
                })

            books = fetch_books(session, [t for m in shortlist for t in m["_tokens"]])
            ranked = get_rows(shortlist, books, args.max_pair_capital,
                              args.max_spread, args.min_mid)
            rows = {row["id"]: row for row in ranked}

            # Update fills and exits.
            for market_id, order in list(orders.items()):
                row = rows.get(market_id)
                if row:
                    for side in ("a", "b"):
                        if order[f"fill_{side}"]:
                            continue
                        book = books[order[f"token_{side}"]]
                        last = number(book.get("last_trade_price"), -1)
                        changed_trade = last >= 0 and last != previous_trade.get(order[f"token_{side}"])
                        crossed = row[f"t{side}"]["ask"] <= order[f"q{side}"]
                        if crossed or (changed_trade and last <= order[f"q{side}"]):
                            cost = order["shares"] * order[f"q{side}"]
                            state["cash"] -= cost
                            order[f"fill_{side}"] = True
                            order[f"fill_time_{side}"] = time.time()
                            state["maker_fills"] += 1
                            append_csv(events_path, {
                                "time": iso_now(), "event": "MAKER_FILL",
                                "market": market_id, "question": order["question"],
                                "side": side, "price": order[f"q{side}"],
                                "shares": order["shares"], "pnl": "", "fee": 0,
                                "cash": state["cash"], "detail": "maker fee=0",
                            })
                    if order["fill_a"] and order["fill_b"]:
                        payout = order["shares"]
                        pnl = payout - order["shares"] * (order["qa"] + order["qb"])
                        state["cash"] += payout
                        state["pairs"] += 1
                        append_csv(events_path, {
                            "time": iso_now(), "event": "COMPLETE_SET",
                            "market": market_id, "question": order["question"],
                            "side": "both", "price": "", "shares": order["shares"],
                            "pnl": pnl, "fee": 0, "cash": state["cash"],
                            "detail": "both maker legs filled",
                        })
                        del orders[market_id]
                        continue

                age = time.time() - order["created"]
                one_leg = order["fill_a"] != order["fill_b"]
                should_exit = row is None or (one_leg and age >= args.one_leg_timeout)
                if should_exit:
                    if one_leg and row:
                        pnl, proceeds, fee, side, price = liquidate(
                            order, row, args.taker_fee_rate
                        )
                        state["cash"] += proceeds
                        state["fees"] += fee
                        state["one_leg_exits"] += 1
                        append_csv(events_path, {
                            "time": iso_now(), "event": "TAKER_EXIT",
                            "market": market_id, "question": order["question"],
                            "side": side, "price": price, "shares": order["shares"],
                            "pnl": pnl, "fee": fee, "cash": state["cash"],
                            "detail": "regime exit/one-leg timeout",
                        })
                    elif one_leg:
                        # No executable book: retain at zero in the ledger. This
                        # is deliberately punitive rather than inventing a fill.
                        state["one_leg_exits"] += 1
                    del orders[market_id]

            # Reserve capital for outstanding paired quotes and open best rows.
            reserved = sum(o["capital"] for o in orders.values()
                           if not (o["fill_a"] or o["fill_b"]))
            available = state["cash"] - reserved
            for row in ranked:
                if len(orders) >= args.max_open or row["id"] in orders:
                    continue
                if row["capital"] > available:
                    continue
                m = row["market"]
                orders[row["id"]] = {
                    "created": time.time(), "question": m.get("question", ""),
                    "shares": row["shares"], "qa": row["qa"], "qb": row["qb"],
                    "capital": row["capital"], "fill_a": False, "fill_b": False,
                    "token_a": m["_tokens"][0], "token_b": m["_tokens"][1],
                }
                available -= row["capital"]
                append_csv(events_path, {
                    "time": iso_now(), "event": "QUOTE_PAIR", "market": row["id"],
                    "question": m.get("question", ""), "side": "both",
                    "price": f"{row['qa']:.4f}+{row['qb']:.4f}",
                    "shares": row["shares"], "pnl": "", "fee": 0,
                    "cash": state["cash"], "detail": f"margin={row['margin']:.4f}",
                })

            for token, book in books.items():
                previous_trade[token] = number(book.get("last_trade_price"), -1)
            write_status(status_path, state, orders, {
                "ends_at": datetime.fromtimestamp(end, timezone.utc).isoformat(),
                "shortlist": len(shortlist), "eligible_now": len(ranked),
            })
            time.sleep(min(args.poll, max(0, end - time.time())))
        except Exception as exc:
            append_csv(events_path, {
                "time": iso_now(), "event": "ERROR", "market": "", "question": "",
                "side": "", "price": "", "shares": "", "pnl": "", "fee": "",
                "cash": state["cash"], "detail": f"{type(exc).__name__}: {exc}",
            })
            write_status(status_path, state, orders, {"last_error": str(exc)})
            time.sleep(15)

    # End-of-run: cancel unfilled quotes and liquidate unmatched inventory at
    # the last executable bid with the conservative taker fee.
    unmatched = 0
    for market_id, order in list(orders.items()):
        one_leg = order["fill_a"] != order["fill_b"]
        row = rows.get(market_id)
        if one_leg and row:
            pnl, proceeds, fee, side, price = liquidate(order, row, args.taker_fee_rate)
            state["cash"] += proceeds
            state["fees"] += fee
            state["one_leg_exits"] += 1
            append_csv(events_path, {
                "time": iso_now(), "event": "FINAL_TAKER_EXIT",
                "market": market_id, "question": order["question"], "side": side,
                "price": price, "shares": order["shares"], "pnl": pnl, "fee": fee,
                "cash": state["cash"], "detail": "scheduled final liquidation",
            })
        elif one_leg:
            unmatched += 1
        del orders[market_id]
    write_status(status_path, state, orders, {"finished": True,
                                             "unmatched_inventory": unmatched})
    append_csv(events_path, {
        "time": iso_now(), "event": "FINISH", "market": "", "question": "",
        "side": "", "price": "", "shares": "", "pnl": state["cash"] - state["starting"],
        "fee": state["fees"], "cash": state["cash"],
        "detail": f"unmatched_inventory={unmatched}",
    })


if __name__ == "__main__":
    main()
