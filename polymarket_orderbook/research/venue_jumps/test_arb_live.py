"""python research/venue_jumps/test_arb_live.py  (from polymarket_orderbook/)"""
import json
from arb_live import fill, build_groups, fee

# 3 outcomes at 0.30 each: 0.90 + fees < 1 -> profitable, limited by the thinnest leg (5 shares)
s, c, p, g = fill([[(0.30, 10)], [(0.30, 5)], [(0.30, 10)]], 1.0)
assert abs(s - 5) < 1e-9 and abs(g - 5 * 0.10) < 1e-9 and abs(p - 5 * (0.10 - 3 * fee(0.30))) < 1e-9
# asks summing over $1: nothing
assert fill([[(0.5, 10)], [(0.51, 10)]], 1.0)[0] == 0
# walks the ladder: second level makes it unprofitable, so only the first level fills
s, *_ = fill([[(0.40, 2), (0.60, 50)], [(0.40, 50)]], 1.0)
assert abs(s - 2) < 1e-9

m = lambda i, delay=None: dict(closed=False, acceptingOrders=True, clobTokenIds=json.dumps([f"y{i}", f"n{i}"]),
                              groupItemTitle=f"o{i}", **({"secondsDelay": delay} if delay is not None else {}))
E = [dict(id="1", negRisk=True, title="free", markets=[m(1), m(2, 0)]),
     dict(id="2", negRisk=True, title="delayed", markets=[m(3), m(4, 1)])]
G, nf, na = build_groups(E)
assert set(G) == {"1:YES", "1:NO"} and G["1:NO"]["payout"] == 1.0 and nf == 3 and na == 4
print("ok")
