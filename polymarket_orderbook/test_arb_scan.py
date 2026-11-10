"""python test_arb_scan.py -- sizing, fills, inventory and quoting logic."""
from types import SimpleNamespace

from arb_scan import Account, Sim, SimOrder, fill, ladder, on_book, on_trade, sets_held, size_arb

# 3 outcomes, asks sum 0.90 for 10 sets, then 0.99 for more: no fee
books = [{"bids": {0.25: 50}, "asks": {0.30: 10, 0.33: 100}} for _ in range(3)]
lads = [ladder(b, "BUY_YES", {}) for b in books]
sets, lim, cost = size_arb(lads, [0, 0, 0], 1.0, 0.005, 1000)
assert abs(sets - 110) < 1e-6 and lim == [0.33] * 3, (sets, lim)
assert abs(size_arb(lads, [0] * 3, 1.0, 0.05, 1000)[0] - 10) < 1e-6   # edge stops it
assert abs(size_arb(lads, [0] * 3, 1.0, 0.05, 4.5)[0] - 5) < 1e-6     # cash caps it
assert size_arb(lads, [0.05] * 3, 1.0, 0.005, 1000)[0] == 10          # fees kill 0.99

# NO side: bids 0.40 x3 -> NO at 0.60, 3 NO pay 2
nb = [{"bids": {0.40: 20}, "asks": {0.45: 20}} for _ in range(3)]
s, lim, _ = size_arb([ladder(b, "BUY_NO", {}) for b in nb], [0] * 3, 2.0, 0.005, 1000)
assert abs(s - 20) < 1e-6 and abs(lim[0] - 0.60) < 1e-9
assert fill([(0.30, 10), (0.33, 100)], 0.30, 50, 0)[0] == 10          # limit respected
assert ladder(books[0], "BUY_YES", {0.30: 10}) == [(0.33, 100)]       # footprint

# queue (same cases as sigbot tests/test_pmmm.py)
o = SimOrder("bid", 0.30, 100, ahead=150, live_at=0)
assert on_trade(o, 0.30, 100, 1) == 0 and o.ahead == 50
assert on_trade(o, 0.30, 80, 1) == 30 and o.size == 70
assert on_trade(o, 0.29, 20, 1) == 20
assert on_book(o, 0.0, 0.30, 1) == 50 and o.size == 0
assert on_trade(SimOrder("ask", 0.40, 10, 0, live_at=5), 0.41, 10, 1) == 0

# account: close-before-open, NO as negative YES
acc = Account(50.0)
acc.fill("t", "bid", 0.30, 100)
acc.fill("t", "ask", 0.32, 150)
assert abs(acc.cash - 18.0) < 1e-9 and acc.net["t"] == -50
assert acc.cost("t", "bid", 0.3, 80) == 30 * 0.3
assert sets_held({"a": 7, "b": 5, "c": 9}, "abc") == (5, 0)
assert sets_held({"a": -2, "b": -3}, "ab") == (0, 2)

# maker skew: long outcome A beyond the sets -> its bid leans down, an ask appears
a = SimpleNamespace(bankroll=100, latency_ms=0, min_edge=0.005, mm_events=1,
                    mm_max_legs=8, mm_size=5, half_spread=0.01, max_inv=50,
                    skew_at_cap=0.02, pull_jump=0.5, pull_cooldown_s=30)
legs = [{"asset": x, "name": x, "min": 5, "tick": 0.01, "fee": 0} for x in "AB"]
ev = {"slug": "e", "end": None, "days": 1.0, "legs": legs}
sim = Sim([ev], a)
sim.busy.add("e")
for t in "AB":
    sim.books[t] = {"bids": {0.48: 100}, "asks": {0.52: 100}}
sim.mm, sim.mm_state["e"] = [ev], {"hist": [], "paused_until": 0}
q0 = sim._event_quotes(ev, 0)
assert q0[("A", "bid")] == (0.49, 5) and ("A", "ask") not in q0
sim.acct.net["A"] = 50                                     # at cap, no B
q1 = sim._event_quotes(ev, 0)
assert ("A", "bid") not in q1 and q1[("A", "ask")] == (0.49, 5), q1   # 0.5-0.02+0.01
assert q1[("B", "bid")] == (0.49, 5)
# cash scaling: $2 free can't cover two 5-share bids at 0.49 -> both dropped below min
sim.acct.cash, sim.acct.net = 2.0, {}
sim.requote(0)
assert not sim.orders
print("ok")
