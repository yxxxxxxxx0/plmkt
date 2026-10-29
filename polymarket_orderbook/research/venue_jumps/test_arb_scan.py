"""python research/venue_jumps/test_arb_scan.py -- checks the depth walk in arb_scan.fill."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arb_scan import fill, fee

# two legs, 10 shares each at 0.40 and 0.50: profit per set = 1 - 0.9 - fees
s, c, p = fill([[(0.40, 10)], [(0.50, 20)]], 1.0)
assert s == 10 and abs(p - 10 * (0.1 - fee(0.4) - fee(0.5))) < 1e-9
# second level makes the set unprofitable -> stop at the first level
s, c, p = fill([[(0.40, 5), (0.60, 50)], [(0.50, 50)]], 1.0)
assert s == 5
# no edge -> nothing
assert fill([[(0.5, 10)], [(0.5, 10)]], 1.0)[0] == 0
# empty ladder -> nothing
assert fill([[], [(0.1, 5)]], 1.0) == (0.0, 0.0, 0.0)
print("ok")
