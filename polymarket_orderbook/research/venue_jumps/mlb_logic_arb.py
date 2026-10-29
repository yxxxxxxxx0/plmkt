"""Logical arbitrage inside one MLB game on Polymarket (no taker delay on MLB).

Relationships that must hold, from the recorded top of book (ms timestamps):
  totals    Over L1 >= Over L2 for L1 < L2         arb: bid(Over L2) > ask(Over L1)
  run line  Team -k  <= Team wins (moneyline)       arb: bid(Team -k) > ask(Team ML)
            Team -k1 >= Team -k2 for k1 < k2        arb: bid(Team -k2) > ask(Team -k1)
  exclusive A -k + B -k <= 1                        arb: bid(A -k) + bid(B -k) > 1  (buy both NOs)
Each arb buys the cheap side at its ask and the other at (1 - bid); payoff >= $1 in every outcome.
Edge = gap - taker fee on both legs (0.05 * p(1-p)). A window is open while the edge stays > 0
with both quotes updated within the last 2 s. Size = smaller top-of-book size of the two legs.

    python research/venue_jumps/mlb_logic_arb.py data/live/top_of_book_2026-10-27.csv
"""
import collections
import re
import sys

import pandas as pd

fee = lambda p: 0.05 * p * (1 - p)
FRESH = 2000


def main(path):
    cols = ["ts_recv_ms", "event_slug", "market_type", "market_question", "line", "outcome",
            "best_bid", "best_bid_size", "best_ask", "best_ask_size"]
    d = pd.read_csv(path, usecols=cols, dtype={"line": str})
    d = d[d.event_slug.str.startswith("mlb") & d.market_type.isin(["moneyline", "total", "spread"]) & (d.outcome == "YES")]
    d = d.sort_values("ts_recv_ms")
    print(f"{len(d):,} YES-side quote updates, {d.event_slug.nunique()} games")

    # key each YES contract by (kind, team, line); moneyline YES = first-named team
    def key(r):
        if r.market_type == "moneyline":
            return ("ml", r.market_question.split(" vs. ")[0].strip(), 0.0)
        if r.market_type == "total":
            return ("over", "", float(r.line))
        m = re.match(r"Spread: (.+) \((-?[\d.]+)\)", r.market_question)
        return ("spread", m.group(1).strip(), abs(float(m.group(2)))) if m else None

    quote = collections.defaultdict(dict)          # event -> key -> (t, bid, bsz, ask, asz)
    teams = {}                                     # event -> (teamA, teamB) from the moneyline question
    open_w, W = {}, []
    seen = collections.defaultdict(list)          # rel -> every edge evaluated (sanity check)

    def pairs(ev, q):
        """Yield (name, cheap_key, rich_key_or_None, gap, size) for every relationship in this game."""
        overs = sorted(k for k in q if k[0] == "over")
        for i, a in enumerate(overs):
            for b in overs[i + 1:]:
                yield ("totals", a, b)
        A, B = teams.get(ev, (None, None))
        for team in (A, B):
            spreads = sorted((k for k in q if k[0] == "spread" and k[1] == team), key=lambda k: k[2])
            ml = ("ml", A, 0.0)
            for s in spreads:
                if team == A and ml in q:
                    yield ("runline<=ML", ml, s)
            for i, a in enumerate(spreads):
                for b in spreads[i + 1:]:
                    yield ("runline k1>=k2", a, b)
        for k in {k[2] for k in q if k[0] == "spread"}:
            a, b = ("spread", A, k), ("spread", B, k)
            if a in q and b in q:
                yield ("A-k + B-k <= 1", a, b)

    def edge(name, a, b, q, now):
        qa, qb = q[a], q[b]
        if now - qa[0] > FRESH or now - qb[0] > FRESH:
            return None
        if name == "A-k + B-k <= 1":               # buy NO on both at 1 - bid each; pays >= $1
            pa, pb = 1 - qa[1], 1 - qb[1]
            return 1 - pa - pb - fee(pa) - fee(pb), min(qa[2], qb[2])
        pin, pout = qa[3], qb[1]                    # buy cheap (ask), buy NO on rich (1 - bid)
        return pout - pin - fee(pin) - fee(1 - pout), min(qa[4], qb[2])

    for r in d.itertuples(index=False):
        k = key(r)
        if k is None or pd.isna(r.best_bid) or pd.isna(r.best_ask):
            continue
        ev, now = r.event_slug, int(r.ts_recv_ms)
        if k[0] == "ml" and ev not in teams:
            a, b = (s.strip() for s in r.market_question.split(" vs. "))
            teams[ev] = (a, b)
        quote[ev][k] = (now, float(r.best_bid), float(r.best_bid_size or 0), float(r.best_ask), float(r.best_ask_size or 0))
        q = quote[ev]
        for name, a, b in pairs(ev, q):
            if k not in (a, b):
                continue
            e = edge(name, a, b, q, now)
            wk = (ev, name, a, b)
            if e:
                seen[name].append(e[0])
            if e and e[0] > 0:
                w = open_w.setdefault(wk, dict(game=ev, rel=name, a=a, b=b, start=now, edge=0.0, size=0.0))
                w["edge"], w["size"], w["end"] = max(w["edge"], e[0]), max(w["size"], e[1]), now
            elif wk in open_w:
                w = open_w.pop(wk); w["end"] = now; W.append(w)
    W.extend(open_w.values())
    for name, v in seen.items():
        v = pd.Series(v)
        print(f"  {name:16s} evaluations {len(v):9,d}  best edge {100*v.max():+.2f}c  share within 1c of arb {(v > -0.01).mean():.2%}")
    W = pd.DataFrame(W)
    if W.empty:
        print("no windows"); return
    W["ms"] = W.end - W.start
    W["usd"] = W.edge * W["size"].clip(upper=200)
    print(f"\n{len(W)} arbitrage windows")
    bins = pd.cut(W.ms, [-1, 100, 500, 1000, 5000, 1e12], labels=["<0.1 s", "0.1-0.5 s", "0.5-1 s", "1-5 s", ">5 s"])
    print(W.groupby([W.rel, bins], observed=True).agg(n=("ms", "size"), median_edge_c=("edge", lambda x: round(100 * x.median(), 2)),
                                                      median_size=("size", "median"), usd_if_taken=("usd", "sum")).to_string())
    L = W[W.ms >= 500]
    print(f"\nwindows open >= 0.5 s: {len(L)}, profit if every one taken (size capped at 200 shares): ${L.usd.sum():.2f}")
    print(L.sort_values("usd", ascending=False).head(8)[["game", "rel", "a", "b", "ms", "edge", "size", "usd"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1])
