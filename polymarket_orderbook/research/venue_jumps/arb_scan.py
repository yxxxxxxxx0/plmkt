"""Snapshot arbitrage scan of open Polymarket events (politics, weather, futures, ...).

1. Sum of outcomes, multi-outcome (neg-risk) events, every outcome tradable:
     buy all YES  profit = 1 - sum(ask) - fees     only safe if exactly one outcome MUST win
     buy all NO   profit = sum(bid) - 1 - fees     n NOs cost sum(1-bid) and pay >= n-1
2. Nested futures: a team's champion price can't exceed its conference/league price:
     profit = bid(champion) - ask(conference) - fees   (buy conference YES, buy champion NO)
Prices are Gamma's bestBid/bestAsk, a summary that can lag; confirm any hit on the CLOB book
(`curl https://clob.polymarket.com/book?token_id=...`) and read the resolution rules before trading.
Taker fee 0.05*p(1-p) per leg (sports rate; most other categories are 0.04 or less).

    python research/venue_jumps/arb_scan.py
"""
import json
import subprocess

GAMMA = "https://gamma-api.polymarket.com/events?closed=false&active=true&limit=100&offset={}"
NESTED = [("Pro Football: 2027 Champion", ["Pro Football: 2027 NFC Champion", "Pro Football: 2027 AFC Champion"]),
          ("NBA: 2027 Champion", ["NBA: 2027 Western Conference Champion", "NBA: 2027 Eastern Conference Champion"])]
fee = lambda p: 0.05 * p * (1 - p)


def events():
    out = {}
    for off in range(0, 5000, 100):     # Gamma caps plain offsets; deeper needs /events/keyset
        d = json.loads(subprocess.run(["curl", "-s", GAMMA.format(off)], capture_output=True, text=True, encoding="utf-8").stdout or "[]")
        if not isinstance(d, list) or not d:
            break
        out.update({e["id"]: e for e in d})
    return list(out.values())


def quotes(e):
    q = {}
    for m in e.get("markets", []):
        if m.get("closed") or not m.get("acceptingOrders"):
            continue
        try:
            q[m["groupItemTitle"].strip()] = (float(m.get("bestBid") or 0), float(m["bestAsk"]))
        except (KeyError, TypeError, ValueError):
            pass
    return q


def main():
    E = events()
    print(f"{len(E)} open events")
    rows = []
    for e in E:
        if not e.get("negRisk"):
            continue
        q = quotes(e)
        if len(q) < 3 or len(q) != sum(not m.get("closed") for m in e["markets"]):
            continue
        A, B = [a for _, a in q.values()], [b for b, _ in q.values()]
        yes, no = 1 - sum(A) - sum(map(fee, A)), sum(B) - 1 - sum(fee(1 - b) for b in B)
        rows.append((max(yes, no), "YES" if yes > no else "NO", len(q), e["title"][:60], (e.get("endDate") or "")[:10], e["slug"]))
    rows.sort(reverse=True)
    print(f"\n1. sum of outcomes: {len(rows)} multi-outcome events, {sum(r[0] > 0 for r in rows)} with a positive edge")
    for r in rows[:10]:
        print(f"   {100 * r[0]:+6.2f}c  buy all {r[1]:3s}  n={r[2]:3d}  ends {r[4]}  {r[3]}  ({r[5]})")
    by_title = {e["title"].strip(): e for e in E}
    print("\n2. nested futures (champion <= conference)")
    for champ, confs in NESTED:
        C = quotes(by_title.get(champ, {}))
        pairs = [(sb - ca - fee(ca) - fee(1 - sb), team) for conf in confs
                 for team, (cb, ca) in quotes(by_title.get(conf, {})).items() if team in C for sb in [C[team][0]]]
        if pairs:
            best = max(pairs)
            print(f"   {champ}: {len(pairs)} teams, closest {best[1]} {100 * best[0]:+.2f}c")


if __name__ == "__main__":
    main()
