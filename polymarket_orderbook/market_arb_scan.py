"""Read-only scan for executable binary complete-set mispricing.

For every active two-outcome CLOB market, compare the best asks for both
outcomes.  Buying one share of each pays $1 in every state, so ask_yes +
ask_no < 1 is a gross arbitrage before fees.  This scanner never places an
order and deliberately reports gross opportunities separately from fees.
"""

import argparse
import json
from decimal import Decimal

import requests


GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB_BOOKS = "https://clob.polymarket.com/books"


def active_markets(session, max_markets):
    out, offset = [], 0
    while len(out) < max_markets:
        # Gamma currently caps this endpoint at 100 rows even if a larger
        # limit is requested; paginate explicitly so the scan is not silently
        # restricted to its first page.
        n = min(100, max_markets - len(out))
        r = session.get(GAMMA, params={"active": "true", "closed": "false",
                                      "limit": n, "offset": offset}, timeout=30)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        for m in page:
            try:
                tokens = json.loads(m.get("clobTokenIds") or "[]")
            except json.JSONDecodeError:
                continue
            if m.get("enableOrderBook") and len(tokens) == 2 and m.get("acceptingOrders", True):
                out.append((m, tokens))
        offset += len(page)
        if len(page) < n:
            break
    return out[:max_markets]


def books(session, token_ids):
    result = {}
    for i in range(0, len(token_ids), 200):
        body = [{"token_id": t} for t in token_ids[i:i + 200]]
        r = session.post(CLOB_BOOKS, json=body, timeout=30)
        r.raise_for_status()
        for b in r.json():
            result[str(b["asset_id"])] = b
    return result


def best_ask(book):
    asks = book.get("asks") or []
    return min((Decimal(str(x["price"])) for x in asks), default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-markets", type=int, default=1000)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    s = requests.Session()
    markets = active_markets(s, args.max_markets)
    bookmap = books(s, [t for _, ts in markets for t in ts])
    rows = []
    for m, ts in markets:
        a0 = best_ask(bookmap.get(ts[0], {}))
        a1 = best_ask(bookmap.get(ts[1], {}))
        if a0 is None or a1 is None:
            continue
        gross = Decimal("1") - a0 - a1
        rows.append((gross, a0, a1, m.get("feesEnabled", False),
                     m.get("orderMinSize"), m.get("question", ""), m.get("slug", "")))
    rows.sort(reverse=True, key=lambda x: x[0])

    print(f"active binary markets fetched: {len(markets)}")
    print(f"markets with both asks:       {len(rows)}")
    print(f"gross ask-sum arbitrages:     {sum(r[0] > 0 for r in rows)}")
    print("\nBest raw opportunities (negative means paying more than the $1 payout):")
    for gross, a0, a1, fees, mos, question, slug in rows[:args.top]:
        cost_for_min = (a0 + a1) * Decimal(str(mos or 1))
        print(f"{gross:+.4f}  asks={a0}+{a1}  min_cost=${cost_for_min:.2f}  "
              f"fees={fees!s:<5}  {question[:74]}  [{slug}]")


if __name__ == "__main__":
    main()
