"""Paper-trading arbitrage scanner for Polymarket (politics, weather, futures, ...).

Each run: pull every open event from Gamma, shortlist anything whose summary prices come within
2c of an arbitrage, then confirm on the real CLOB order books and walk the depth to find how many
full sets you could actually buy and the locked-in profit. Every confirmed gap is appended to
data/arb/gaps.jsonl; --report summarises the log (a gap seen on several scans is one opportunity).

  buy all YES   one YES per outcome; pays $1 if exactly one outcome wins.
                Only safe when the outcomes are exhaustive: READ THE RULES before trading.
  buy all NO    one NO per outcome; pays n-1 if one wins (n if none does), so it is always safe.
  nested        team champion NO + same team conference YES; pays >= $1 whatever happens.

Fee: taker 0.05 * p(1-p) per share on every leg (the sports rate; most other categories are
lower, so this is conservative). No orders are sent.

    python research/venue_jumps/arb_scan.py             one scan (run every few minutes)
    python research/venue_jumps/arb_scan.py --report    summary of everything logged so far
"""
import argparse
import json
import os
import subprocess
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(BASE, "data", "arb", "gaps.jsonl")
GAMMA = "https://gamma-api.polymarket.com/events?closed=false&active=true&limit=100&offset={}"
NESTED = [("Pro Football: 2027 Champion", ["Pro Football: 2027 NFC Champion", "Pro Football: 2027 AFC Champion"]),
          ("NBA: 2027 Champion", ["NBA: 2027 Western Conference Champion", "NBA: 2027 Eastern Conference Champion"])]
SHORTLIST = -0.02          # Gamma edge above this gets checked on the real books
BUDGET = 100.0             # cap the cost of one paper trade at the bankroll
fee = lambda p: 0.05 * p * (1 - p)


def curl(url, body=None):
    cmd = ["curl", "-s", "--max-time", "30", url]
    if body is not None:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(body)]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
    try:
        return json.loads(out or "null")
    except json.JSONDecodeError:
        return None


def events():
    out = {}
    for off in range(0, 5000, 100):     # Gamma caps plain offsets; deeper needs /events/keyset
        d = curl(GAMMA.format(off))
        if not isinstance(d, list) or not d:
            break
        out.update({e["id"]: e for e in d})
    return list(out.values())


def books(tokens):
    """token_id -> asks sorted cheapest first [(price, size)], via the batch endpoint."""
    out = {}
    tokens = list(tokens)
    for i in range(0, len(tokens), 100):
        for b in curl("https://clob.polymarket.com/books", [{"token_id": t} for t in tokens[i:i + 100]]) or []:
            out[b["asset_id"]] = sorted((float(x["price"]), float(x["size"])) for x in b.get("asks", []))
    return out


def fill(legs, payout):
    """Buy one share of every leg per set, walking each ask ladder. Returns (sets, cost, profit)."""
    legs = [list(l) for l in legs]
    if not legs or any(not l for l in legs):
        return 0.0, 0.0, 0.0
    sets = cost = profit = 0.0
    while all(legs):
        prices = [l[0][0] for l in legs]
        unit = sum(p + fee(p) for p in prices)
        if payout - unit <= 0:
            break
        q = min(l[0][1] for l in legs)
        q = min(q, (BUDGET - cost) / unit)
        if q <= 0:
            break
        sets += q; cost += q * unit; profit += q * (payout - unit)
        for l in legs:
            l[0] = (l[0][0], l[0][1] - q)
            if l[0][1] <= 1e-9:
                l.pop(0)
    return sets, cost, profit


def tokens_of(m):
    t = json.loads(m.get("clobTokenIds") or "[]")
    return (t[0], t[1]) if len(t) == 2 else (None, None)


def scan():
    E = events()
    now = time.time()
    cand = []                                   # (kind, event, [(name, yes_tok, no_tok)])
    for e in E:
        ms = [m for m in e.get("markets", []) if not m.get("closed")]
        if not e.get("negRisk") or len(ms) < 3 or not all(m.get("acceptingOrders") for m in ms):
            continue
        try:
            A = [float(m["bestAsk"]) for m in ms]; B = [float(m.get("bestBid") or 0) for m in ms]
        except (KeyError, TypeError, ValueError):
            continue
        legs = [(m.get("groupItemTitle", "").strip(),) + tokens_of(m) for m in ms]
        if 1 - sum(A) - sum(map(fee, A)) > SHORTLIST:
            cand.append(("buy all YES", e, legs))
        if sum(B) - 1 - sum(fee(1 - b) for b in B) > SHORTLIST:
            cand.append(("buy all NO", e, legs))
    by_title = {e["title"].strip(): e for e in E}
    nested = []
    for champ, confs in NESTED:
        C = {m.get("groupItemTitle", "").strip(): m for m in by_title.get(champ, {}).get("markets", []) if not m.get("closed")}
        for conf in confs:
            for m in by_title.get(conf, {}).get("markets", []):
                t = m.get("groupItemTitle", "").strip()
                if t in C and not m.get("closed"):
                    nested.append((champ, conf, t, tokens_of(C[t])[1], tokens_of(m)[0]))   # champion NO, conference YES
    need = {tok for _, _, legs in cand for _, y, n in legs for tok in (y, n) if tok}
    need |= {tok for *_, a, b in nested for tok in (a, b) if tok}
    asks = books(need)
    hits = []
    for kind, e, legs in cand:
        idx = 1 if kind == "buy all YES" else 2
        ladders = [asks.get(l[idx], []) for l in legs]
        payout = 1.0 if kind == "buy all YES" else len(legs) - 1.0
        sets, cost, profit = fill(ladders, payout)
        if profit > 0:
            hits.append(dict(t=now, kind=kind, slug=e["slug"], title=e["title"], n=len(legs), end=(e.get("endDate") or "")[:10],
                             sets=round(sets, 2), cost=round(cost, 2), profit=round(profit, 4)))
    for champ, conf, team, a, b in nested:
        sets, cost, profit = fill([asks.get(a, []), asks.get(b, [])], 1.0)
        if profit > 0:
            hits.append(dict(t=now, kind="nested", slug=f"{champ} / {conf}", title=team, n=2, end="",
                             sets=round(sets, 2), cost=round(cost, 2), profit=round(profit, 4)))
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
        f.write(json.dumps(dict(t=now, kind="scan", events=len(E), shortlisted=len(cand) + len(nested), hits=len(hits))) + "\n")
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {len(E)} events, {len(cand)} shortlisted + {len(nested)} nested pairs, "
          f"{len(hits)} confirmed on the book")
    for h in hits:
        print(f"  {h['kind']:11s} profit ${h['profit']:.2f} on ${h['cost']:.2f} ({h['sets']} sets), ends {h['end']}  {h['title'][:60]}")


def report():
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8")]
    scans = [r for r in rows if r["kind"] == "scan"]
    hits = [r for r in rows if r["kind"] != "scan"]
    if not scans:
        print("nothing logged yet"); return
    span = (scans[-1]["t"] - scans[0]["t"]) / 3600
    print(f"{len(scans)} scans over {span:.1f} h, {len(hits)} confirmed gap sightings")
    opp = {}
    for h in hits:
        k = (h["kind"], h["slug"], h["title"])
        o = opp.setdefault(k, dict(first=h["t"], last=h["t"], seen=0, best=h, end=h["end"]))
        o["last"] = h["t"]; o["seen"] += 1
        if h["profit"] > o["best"]["profit"]:
            o["best"] = h
    print(f"{len(opp)} distinct opportunities (same event and trade = one)\n")
    tot = 0.0
    for (kind, slug, title), o in sorted(opp.items(), key=lambda x: -x[1]["best"]["profit"]):
        b = o["best"]; tot += b["profit"]
        print(f"  ${b['profit']:7.2f} on ${b['cost']:6.2f}  seen {o['seen']:3d}x over {(o['last'] - o['first']) / 60:6.0f} min  "
              f"ends {o['end'] or '?':10s}  {kind:11s} {title[:50]}")
    print(f"\nprofit if each opportunity were taken once at its best size (capped at ${BUDGET:.0f}): ${tot:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    report() if ap.parse_args().report else scan()
