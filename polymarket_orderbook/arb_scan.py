"""Live Polymarket neg-risk arbitrage: taker scanner + inventory-managed maker, paper only.

The only structural arbitrage on Polymarket is inside a neg-risk event (exactly
one of N outcomes resolves YES). A plain binary has none: the CLOB mirrors one
book, so NO ask is exactly 1 - YES bid.

  TAKER  BUY_YES  sum(YES ask) + fees < 1       one YES per outcome pays $1
         BUY_NO   sum(NO ask)  + fees < N - 1   one NO per outcome pays N-1
  MAKER  quote every outcome of the --mm-events soonest-resolving events and
         collect the spread as sets complete

Inventory management is ported from sig-predictions-cup sigbot/pmmm.py
(Account, SimOrder, on_trade, on_book, the skew/cap/cash rules in requote):

  * Account   net YES per token (+YES / -NO); a fill closes before it opens,
              and cost() charges cash only for the opening part
  * fair      outcome mids rescaled to sum to 1 (pmmm makes D+R sum to 1)
  * skew      an outcome held beyond the completed sets ("naked") leans its
              quotes down by skew_at_cap * naked / max_inv; at the cap it stops
              bidding, and an ask works the naked shares off
  * cash      every opening quote shrinks pro rata until all of them fit in
              free cash assuming all fill; below orderMinSize it is not placed
  * pull      a fair move >= pull_jump within 30 s pulls the event's quotes for
              pull_cooldown_s
  * queue     a quote joins the back of its level; prints at its price eat the
              queue ahead first; a print or the opposite book through it fills it

Realism on the taker side: orders land --latency-ms after the signal against
the book as it is then; fills walk depth under a limit; fee = rate*p*(1-p) per
market feeType; minimum order size; liquidity we took is not re-offered. Legs
that fill unevenly leave naked shares: in a maker event they stay as inventory
for the skew to work off, elsewhere they are dumped into the book one more
latency later. Complete sets are held to resolution, so their capital is locked.

Nothing is sent to the exchange. Writes data/arb/trades.csv (taker attempts)
and data/arb/fills.csv (every simulated fill).

    python arb_scan.py
    python arb_scan.py --mm-events 0 --latency-ms 400      # taker only
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field

import requests
import websockets

from live_recorder import apply_change, apply_snapshot, new_book
from polymarket_fees import TAKER_FEE_RATE

GAMMA = "https://gamma-api.polymarket.com/events"
WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "arb")
STATE = os.path.join(OUT, "state.json")
EPS = 1e-9


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------- inventory (from sigbot/pmmm.py)
@dataclass
class SimOrder:
    side: str          # bid | ask (YES-space)
    price: float
    size: float
    ahead: float       # displayed size at our level when we joined
    live_at: float     # placement time + latency


def on_trade(o: SimOrder, price: float, size: float, now: float) -> float:
    """Shares of our order filled by one print (mutates queue/size)."""
    if now < o.live_at or o.size <= 0:
        return 0.0
    through = price < o.price if o.side == "bid" else price > o.price
    if through:
        f = min(o.size, size)
    elif abs(price - o.price) < EPS:
        f = min(o.size, max(0.0, size - o.ahead))
        o.ahead = max(0.0, o.ahead - size)
    else:
        return 0.0
    o.size -= f
    return f


def on_book(o: SimOrder, level_size: float, best_opp, now: float) -> float:
    """Book update at our level / opposite best. Returns shares filled by a cross."""
    # ponytail: clamping ahead to the level can double-count a print whose book
    # update arrived first -- optimistic by at most one print. Upgrade: match
    # prints to level decrements by exchange timestamp.
    o.ahead = min(o.ahead, level_size)
    if now >= o.live_at and best_opp is not None and o.size > 0:
        if (o.side == "bid" and best_opp <= o.price) or (o.side == "ask" and best_opp >= o.price):
            f, o.size = o.size, 0.0
            return f
    return 0.0


@dataclass
class Account:
    cash: float
    net: dict = field(default_factory=dict)       # token -> net YES (+YES / -NO)

    def fill(self, token, side, price, qty):
        """Buying YES while short closes NO pairs (+1 each); selling YES while long
        sells inventory, else opens NO."""
        n = self.net.get(token, 0.0)
        if side == "bid":
            close = min(qty, max(0.0, -n))
            self.cash += close * (1 - price) - (qty - close) * price
            self.net[token] = n + qty
        else:
            close = min(qty, max(0.0, n))
            self.cash += close * price - (qty - close) * (1 - price)
            self.net[token] = n - qty

    def cost(self, token, side, price, qty):
        """Cash this order would consume if filled (0 for the part that closes)."""
        n = self.net.get(token, 0.0)
        if side == "bid":
            return max(0.0, qty - max(0.0, -n)) * price
        return max(0.0, qty - max(0.0, n)) * (1 - price)


def tick_floor(p, tick):
    return math.floor(round(p / tick, 6)) * tick


def tick_ceil(p, tick):
    return math.ceil(round(p / tick, 6)) * tick


# ---------------------------------------------------------------- discovery
def discover(tags):
    """Exhaustive neg-risk events whose every unresolved outcome is tradable."""
    events = {}
    for tag in tags:
        offset = 0
        while True:
            page = requests.get(GAMMA, params={"tag_slug": tag, "closed": "false",
                                               "limit": 100, "offset": offset},
                                timeout=20).json()
            for e in page:
                ev = _event(e)
                if ev:
                    events[ev["slug"]] = ev
            if len(page) < 100:
                break
            offset += 100
    return list(events.values())


def _event(e):
    # augmented events can add outcomes later, so the set is not exhaustive
    if not e.get("negRisk") or e.get("negRiskAugmented"):
        return None
    legs = []
    for m in e.get("markets", []):
        if m.get("closed"):
            if float(json.loads(m.get("outcomePrices") or '["1"]')[0]) != 0:
                return None             # resolved YES, or closed but unresolved
            continue                    # resolved NO: drops out of the set
        if not (m.get("active") and m.get("acceptingOrders")):
            return None                 # an outcome we can't trade
        fee_cat = (m.get("feeType") or "").replace("_fees", "")
        legs.append({
            "asset": json.loads(m["clobTokenIds"])[0],
            "name": m.get("groupItemTitle") or m.get("question"),
            "min": float(m.get("orderMinSize") or 5),
            "tick": float(m.get("orderPriceMinTickSize") or 0.01),
            "fee": TAKER_FEE_RATE.get(fee_cat, 0.05) if m.get("feesEnabled") else 0.0,
        })
    if len(legs) < 2:
        return None
    return {"slug": e["slug"], "end": e.get("endDate"), "legs": legs,
            "days": days_to(e.get("endDate"))}


def days_to(end_iso):
    try:
        t = time.mktime(time.strptime(end_iso[:19], "%Y-%m-%dT%H:%M:%S"))
        return max((t - time.time()) / 86400, 1 / 24)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ pricing
def fee(rate, p):
    return rate * p * (1 - p)


def ladder(book, side, taken):
    """Levels as (price_we_pay, size) best first, net of what we already took.

    BUY_YES walks YES asks; BUY_NO walks YES bids, priced as NO = 1 - bid.
    """
    if side == "BUY_YES":
        lv = sorted(book["asks"].items())
        return [(p, s - taken.get(p, 0)) for p, s in lv if s - taken.get(p, 0) > EPS]
    lv = sorted(book["bids"].items(), reverse=True)
    return [(1 - p, s - taken.get(p, 0)) for p, s in lv if s - taken.get(p, 0) > EPS]


def size_arb(ladders, fees, payoff, min_edge, cash):
    """Greedy depth walk: keep buying sets while the marginal set clears.

    Returns (sets, limit price per leg, cost incl. fees).
    """
    idx = [0] * len(ladders)
    left = [ld[0][1] if ld else 0 for ld in ladders]
    sets, cost = 0.0, 0.0
    limits = [None] * len(ladders)
    while all(i < len(ld) for i, ld in zip(idx, ladders)):
        px = [ld[i][0] for i, ld in zip(idx, ladders)]
        unit = sum(px) + sum(fee(f, p) for f, p in zip(fees, px))
        if unit > payoff - min_edge:
            break
        step = min(min(left), (cash - cost) / unit)
        if step <= EPS:
            break
        sets += step
        cost += step * unit
        limits = px
        for k in range(len(ladders)):
            left[k] -= step
            if left[k] <= EPS:
                idx[k] += 1
                if idx[k] < len(ladders[k]):
                    left[k] = ladders[k][idx[k]][1]
    return sets, limits, cost


def fill(ladder_now, limit, qty, rate):
    """Marketable-limit fill. Returns (qty, notional, fee) in the price we pay."""
    got, notional, fees = 0.0, 0.0, 0.0
    for p, s in ladder_now:
        if p > limit + EPS or got >= qty - EPS:
            break
        q = min(s, qty - got)
        got += q
        notional += q * p
        fees += q * fee(rate, p)
    return got, notional, fees


def exit_ladder(book, kind):
    """Levels as (price_we_receive, size) best first for selling a held set.

    YES sets sell into YES bids; NO sets sell by buying YES at the ask, NO = 1 - ask.
    """
    if kind == "yes":
        return sorted(book["bids"].items(), reverse=True)
    return [(1 - p, s) for p, s in sorted(book["asks"].items())]


def size_exit(ladders, fees, floor, held):
    """Greedy depth walk: keep selling sets while the marginal set brings in >= floor.

    Returns (sets, limit price per leg, proceeds net of fees).
    """
    idx = [0] * len(ladders)
    left = [ld[0][1] if ld else 0 for ld in ladders]
    sets, proceeds = 0.0, 0.0
    limits = [None] * len(ladders)
    while all(i < len(ld) for i, ld in zip(idx, ladders)) and held - sets > EPS:
        px = [ld[i][0] for i, ld in zip(idx, ladders)]
        unit = sum(px) - sum(fee(f, p) for f, p in zip(fees, px))
        if unit < floor:
            break
        step = min(min(left), held - sets)
        sets += step
        proceeds += step * unit
        limits = px
        for k in range(len(ladders)):
            left[k] -= step
            if left[k] <= EPS:
                idx[k] += 1
                if idx[k] < len(ladders[k]):
                    left[k] = ladders[k][idx[k]][1]
    return sets, limits, proceeds


def exit_fill(ladder_now, limit, qty, rate):
    """Marketable-limit sell. Returns (qty, proceeds, fee) in the price we receive."""
    got, notional, fees = 0.0, 0.0, 0.0
    for p, s in ladder_now:
        if p < limit - EPS or got >= qty - EPS:
            break
        q = min(s, qty - got)
        got += q
        notional += q * p
        fees += q * fee(rate, p)
    return got, notional, fees


def sets_held(net, tokens):
    """(complete YES sets, complete NO sets) held across an event's outcomes."""
    return (min(max(net.get(t, 0.0), 0.0) for t in tokens),
            min(max(-net.get(t, 0.0), 0.0) for t in tokens))


# ---------------------------------------------------------------- simulator
class Sim:
    def __init__(self, events, a):
        self.a = a
        self.events = events
        self.acct = Account(a.bankroll)
        self.basis = {}                              # slug -> kind -> [sets, cost incl. fees]
        self.load()                                  # before the books: adds held events
        self.books ={l["asset"]: new_book() for e in events for l in e["legs"]}
        self.by_asset = {l["asset"]: e for e in events for l in e["legs"]}
        self.leg = {l["asset"]: l for e in events for l in e["legs"]}
        self.taken = {k: {} for k in self.books}     # our taker footprint per level
        self.busy = set()                            # events with a taker order out
        self.reserved = 0.0                        # cash promised to taker orders in flight
        self.fees = 0.0
        self.orders = {}                             # (token, side) -> SimOrder
        self.mm = []                                 # events being quoted
        self.mm_state = {}                           # slug -> {hist, paused_until}
        self.exits = {}                              # token -> resting SimOrder selling a held set's leg
        self.naked_since = {}                        # slug -> when a resting exit first broke a set
        self.n_sig = self.n_fill = 0
        os.makedirs(OUT, exist_ok=True)
        self.w = self._csv("trades.csv", ["ts", "event", "kind", "legs", "planned_sets",
                                          "planned_edge", "sets", "cost", "fees",
                                          "days_to_end", "ann_return"])
        self.wf = self._csv("fills.csv", ["ts", "event", "outcome", "token", "side",
                                          "price", "qty", "fee", "source", "net_after"])

    def _csv(self, name, header):
        path = os.path.join(OUT, name)
        new = not os.path.exists(path)
        fh = open(path, "a", newline="")
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        return w, fh

    def free_cash(self):
        return self.acct.cash - self.reserved

    # ---- state: positions survive the hourly restart so held sets can still be sold
    def load(self):
        if not os.path.exists(STATE):
            return
        st = json.load(open(STATE))
        self.acct = Account(st["cash"], st["net"])
        self.basis = st["basis"]
        have = {e["slug"] for e in self.events}
        for ev in st["held"]:                    # keep watching events discovery dropped
            if ev["slug"] not in have:
                ev["days"] = days_to(ev["end"])
                self.events.append(ev)
        log(f"resumed: cash ${self.acct.cash:.2f}, sets held in {len(self.basis)} events")

    def save(self):
        held = [e for e in self.events
                if any(abs(self.acct.net.get(l["asset"], 0.0)) > EPS for l in e["legs"])]
        st = dict(cash=self.acct.cash, net={k: v for k, v in self.acct.net.items() if abs(v) > EPS},
                  basis=self.basis, held=[{k: e[k] for k in ("slug", "end", "legs")} for e in held])
        json.dump(st, open(STATE + ".tmp", "w"), indent=1)
        os.replace(STATE + ".tmp", STATE)

    # ---- inventory bookkeeping: every fill goes through here
    def book_fill(self, token, side, price, qty, fee_usd, source):
        self.acct.fill(token, side, price, qty)
        self.acct.cash -= fee_usd
        self.fees += fee_usd
        self.n_fill += 1
        ev = self.by_asset[token]
        w, fh = self.wf
        w.writerow([int(time.time() * 1000), ev["slug"], self.leg[token]["name"], token,
                    side, round(price, 4), round(qty, 4), round(fee_usd, 5), source,
                    round(self.acct.net[token], 4)])
        fh.flush()

    # ---- feed hooks
    def on_book(self, asset, price=None):
        if price is None:
            self.taken[asset] = {}
        else:
            self.taken[asset].pop(price, None)       # level updated: footprint gone
        bk = self.books[asset]
        bb = max(bk["bids"], default=None)
        ba = min(bk["asks"], default=None)
        now = time.time()
        for side, lv, opp in (("bid", bk["bids"], ba), ("ask", bk["asks"], bb)):
            o = self.orders.get((asset, side))
            if o and (f := on_book(o, lv.get(o.price, 0.0), opp, now)):
                self._maker_fill(asset, side, o, f)
        o = self.exits.get(asset)
        if o and (f := on_book(o, (bk["bids"] if o.side == "bid" else bk["asks"]).get(o.price, 0.0),
                               ba if o.side == "bid" else bb, now)):
            self._exit_fill(asset, o, f)
        ev = self.by_asset.get(asset)
        if ev and ev["slug"] not in self.busy:
            self.check(ev)

    def on_trade(self, asset, price, size):
        now = time.time()
        for side in ("bid", "ask"):
            o = self.orders.get((asset, side))
            if o and (f := on_trade(o, price, size, now)):
                self._maker_fill(asset, side, o, f)
        o = self.exits.get(asset)
        if o and (f := on_trade(o, price, size, now)):
            self._exit_fill(asset, o, f)

    # ---- passive exits: rest one order per leg of a held set at fair + the profit we want,
    # never crossing (sigbot's passive sliced exits), so the set is sold before resolution
    # without waiting for the bids to add up
    def _exit_fill(self, asset, o, qty):
        self.book_fill(asset, o.side, o.price, qty, 0.0, "exit_rest")   # makers pay no fee
        if o.size <= EPS:
            self.exits.pop(asset, None)
        ev = self.by_asset[asset]
        self.reconcile(ev)
        log(f"exit_rest {ev['slug']}: {self.leg[asset]['name']} {o.side} {qty:.1f} @ {o.price:.3f}")

    def reconcile(self, ev):
        """Shrink the basis to the complete sets still held; a broken set's legs become naked."""
        tokens = [l["asset"] for l in ev["legs"]]
        b = self.basis.get(ev["slug"], {})
        for kind, have in zip(("yes", "no"), sets_held(self.acct.net, tokens)):
            if kind in b and have < b[kind][0] - EPS:
                b[kind][1] *= have / b[kind][0]
                b[kind][0] = have
                if have <= EPS:
                    del b[kind]
        if not b:
            self.basis.pop(ev["slug"], None)
        ys, ns = sets_held(self.acct.net, tokens)
        naked = any(abs(self.acct.net.get(t, 0.0) - ys + ns) > EPS for t in tokens)
        if naked:
            self.naked_since.setdefault(ev["slug"], time.time())
        else:
            self.naked_since.pop(ev["slug"], None)
        self.save()

    def place_exits(self, now):
        a = self.a
        want = {}
        for ev in self.events:
            if ev in self.mm or ev["slug"] in self.busy:
                continue                             # the maker's skew works these off
            tokens = [l["asset"] for l in ev["legs"]]
            books = [self.books[t] for t in tokens]
            since = self.naked_since.get(ev["slug"])
            if since and now - since > a.naked_s:    # a broken set waited too long: sell the rest
                self.naked_since.pop(ev["slug"], None)
                self.busy.add(ev["slug"])
                asyncio.get_running_loop().create_task(self._dump(ev))
                continue
            b = self.basis.get(ev["slug"])
            if not b and not since:
                continue
            if not all(x["bids"] or x["asks"] for x in books):
                continue
            # a leg with no bid is dead (asks at 0.001): nobody buys it, so it is worth ~0 and not offered;
            # a leg with no ask is the near-certain winner
            mids = [(max(x["bids"]) + min(x["asks"])) / 2 if x["bids"] and x["asks"]
                    else (max(x["bids"]) if x["bids"] else 0.0) for x in books]
            if sum(mids) <= EPS:
                continue
            fair = [m / sum(mids) for m in mids]
            n = len(tokens)
            # Ask at fair (the legs' fairs sum to a set's $1 payout), plus an equal share of whatever the
            # floor needs above that. Legs left naked by a broken set keep resting at fair until naked_s.
            extra = {"yes": 0.0, "no": 0.0}
            for kind, (sets, paid) in (b or {}).items():
                floor = paid / sets + a.tp_min
                live = sum(1 for x in books if x["bids"] and x["asks"])   # the legs the extra is spread over
                extra[kind] = max(0.0, floor - (1 if kind == "yes" else n - 1)) / max(live, 1)
            for l, t, bk, f in zip(ev["legs"], tokens, books, fair):
                held = self.acct.net.get(t, 0.0)
                if not (bk["bids"] and bk["asks"]):
                    continue                            # dead, or a winner nobody sells: no resting order
                if held > EPS:                          # sell YES
                    p = max(tick_ceil(f + extra["yes"], l["tick"]), round(max(bk["bids"]) + l["tick"], 6))
                    want[t] = ("ask", p, held)
                elif held < -EPS:                       # sell NO: bid on YES at 1 - NO target
                    p = min(tick_floor(f - extra["no"], l["tick"]), round(min(bk["asks"]) - l["tick"], 6))
                    want[t] = ("bid", p, -held)
        for t in list(self.exits):
            if t not in want:
                del self.exits[t]
        for t, (side, p, qty) in want.items():
            if not 0 < p < 1 or qty < self.leg[t]["min"]:
                self.exits.pop(t, None)
                continue
            o = self.exits.get(t)
            if o and o.side == side and abs(o.price - p) < EPS and o.size <= qty + EPS:
                continue                              # keep queue position
            lv = self.books[t]["bids" if side == "bid" else "asks"]
            self.exits[t] = SimOrder(side, p, qty, lv.get(p, 0.0), now + a.latency_ms / 1000)

    async def _dump(self, ev):
        for l in ev["legs"]:
            self.exits.pop(l["asset"], None)
        await self._flatten(ev)
        self.reconcile(ev)
        log(f"exit_rest {ev['slug']}: broken set held > {self.a.naked_s / 60:.0f} min, sold the rest")
        self.busy.discard(ev["slug"])

    def _maker_fill(self, asset, side, o, qty):
        self.book_fill(asset, side, o.price, qty, 0.0, "maker")   # makers pay no fee
        if o.size <= EPS:
            self.orders.pop((asset, side), None)

    # ---- taker signal
    def check(self, ev):
        legs = ev["legs"]
        if any(not self.books[l["asset"]]["bids"] and not self.books[l["asset"]]["asks"]
               for l in legs):
            return                                   # book not seeded yet
        fees = [l["fee"] for l in legs]
        tokens = [l["asset"] for l in legs]
        for kind, held in zip(("yes", "no"), sets_held(self.acct.net, tokens)):
            n, paid = self.basis.get(ev["slug"], {}).get(kind, (0.0, 0.0))
            held = min(held, n)                      # only sets we know the cost of
            if held < max(l["min"] for l in legs) - EPS:
                continue
            lads = [exit_ladder(self.books[t], kind) for t in tokens]
            floor = paid / n + self.a.tp_min         # break-even per set plus the profit we want
            sets, limits, proceeds = size_exit(lads, fees, floor, held)
            if sets >= max(l["min"] for l in legs) - EPS:
                self.n_sig += 1
                self.busy.add(ev["slug"])
                asyncio.get_running_loop().create_task(
                    self.take_profit(ev, kind, sets, limits, proceeds))
                return
        # sizing, as sigbot's risk caps: at most max_event of the bankroll per event, and never
        # below the cash reserve, so one slow-resolving event can't lock up the whole account
        # an outcome already priced near certain: the "gap" is dead buckets at 0.001 while the winner's ask
        # gets lifted first, so only the worthless legs fill (2026-11-12, same-day temperature events)
        if any(self.books[l["asset"]]["bids"] and max(self.books[l["asset"]]["bids"]) >= self.a.mm_max_fav
               for l in legs):
            return
        held = sum(b[1] for b in self.basis.get(ev["slug"], {}).values())
        budget =min(self.free_cash() - self.a.reserve * self.a.bankroll,
                     self.a.max_event * self.a.bankroll - held)
        for side, payoff in (("BUY_YES", 1.0), ("BUY_NO", len(legs) - 1.0)):
            lads = [ladder(self.books[l["asset"]], side, self.taken[l["asset"]])
                    for l in legs]
            sets, limits, cost = size_arb(lads, fees, payoff, self.a.min_edge, max(budget, 0.0))
            # money locked until the end date has to earn its keep (score-marks' TRADE rule)
            ann = (sets * payoff - cost) / cost * 365 / ev["days"] if cost > EPS else 0.0
            if sets >= max(l["min"] for l in legs) - EPS and ann >= self.a.min_ann:
                self.n_sig += 1
                self.busy.add(ev["slug"])
                self.reserved += cost
                asyncio.get_running_loop().create_task(
                    self.take(ev, side, payoff, sets, limits, cost))
                return

    async def take(self, ev, side, payoff, sets, limits, plan_cost):
        legs = ev["legs"]
        tokens = [l["asset"] for l in legs]
        before = sets_held(self.acct.net, tokens)
        await asyncio.sleep(self.a.latency_ms / 1000)          # one batch order
        self.reserved -= plan_cost
        cost = fees = 0.0
        for l, lim in zip(legs, limits):
            lad = ladder(self.books[l["asset"]], side, self.taken[l["asset"]])
            q, notional, f = fill(lad, lim, sets, l["fee"])
            if q <= EPS:
                continue
            self._footprint(l["asset"], side, lad, q)
            avg = notional / q
            # BUY_NO is a YES sell in Account terms, at YES price 1 - NO price
            self.book_fill(l["asset"], "bid" if side == "BUY_YES" else "ask",
                           avg if side == "BUY_YES" else 1 - avg, q, f, "taker")
            cost += notional
            fees += f
        if ev not in self.mm:
            await self._flatten(ev)          # no skew working this event: dump naked legs
        after = sets_held(self.acct.net, tokens)
        got = (after[0] - before[0]) if side == "BUY_YES" else (after[1] - before[1])
        edge = got * payoff - cost - fees
        ann = edge / (cost + fees) * 365 / ev["days"] if cost + fees > EPS else 0.0
        if got > EPS:
            # ponytail: legs that filled unevenly and were flattened leave their gain or
            # loss in cash, not in the basis
            b = self.basis.setdefault(ev["slug"], {}).setdefault(
                "yes" if side == "BUY_YES" else "no", [0.0, 0.0])
            b[0] += got
            b[1] += cost + fees
            self.save()
        w, fh = self.w
        w.writerow([int(time.time() * 1000), ev["slug"], "taker_" + side, len(legs),
                    round(sets, 2), round(sets * payoff - plan_cost, 4), round(got, 2),
                    round(cost, 4), round(fees, 4), round(ev["days"], 1), round(ann, 3)])
        fh.flush()
        log(f"taker_{side} {ev['slug']}: planned {sets:.1f} sets edge "
            f"${sets * payoff - plan_cost:.3f} -> completed {got:.1f}, set edge ${edge:+.3f}")
        self.busy.discard(ev["slug"])

    async def take_profit(self, ev, kind, sets, limits, plan_proceeds):
        """Sell held sets before resolution once the books pay more than they cost."""
        legs = ev["legs"]
        tokens = [l["asset"] for l in legs]
        for t in tokens:
            self.exits.pop(t, None)                            # the bids pay now: no need to rest
        before = sets_held(self.acct.net, tokens)
        await asyncio.sleep(self.a.latency_ms / 1000)          # one batch order
        proceeds = fees = 0.0
        for l, lim in zip(legs, limits):
            lad = exit_ladder(self.books[l["asset"]], kind)
            q, notional, f = exit_fill(lad, lim, sets, l["fee"])
            if q <= EPS:
                continue
            # selling a YES set sells YES; selling a NO set buys YES back at 1 - NO price
            avg = notional / q
            self.book_fill(l["asset"], "ask" if kind == "yes" else "bid",
                           avg if kind == "yes" else 1 - avg, q, f, "take_profit")
            proceeds += notional
            fees += f
        if ev not in self.mm:
            await self._flatten(ev)          # legs that did not fill: sell what is left over
        after = sets_held(self.acct.net, tokens)
        sold = (before[0] - after[0]) if kind == "yes" else (before[1] - after[1])
        b = self.basis[ev["slug"]][kind]
        cost = b[1] * sold / b[0] if b[0] > EPS else 0.0
        b[0] -= sold
        b[1] -= cost
        if b[0] <= EPS:
            del self.basis[ev["slug"]][kind]
            if not self.basis[ev["slug"]]:
                del self.basis[ev["slug"]]
        self.save()
        profit = proceeds - fees - cost
        w, fh = self.w
        w.writerow([int(time.time() * 1000), ev["slug"], "exit_" + kind, len(legs),
                    round(sets, 2), round(plan_proceeds - cost, 4), round(sold, 2),
                    round(-proceeds, 4), round(fees, 4), round(ev["days"], 1), ""])
        fh.flush()
        log(f"take_profit {kind} {ev['slug']}: sold {sold:.1f} of {sets:.1f} sets, "
            f"freed ${proceeds - fees:.2f}, profit ${profit:+.3f}")
        self.busy.discard(ev["slug"])

    async def _flatten(self, ev):
        """Sell every share beyond the complete sets into the book, paying the fee."""
        tokens = [l["asset"] for l in ev["legs"]]
        ys, ns = sets_held(self.acct.net, tokens)
        naked = {t: self.acct.net.get(t, 0.0) - ys + ns for t in tokens}
        if all(abs(n) <= EPS for n in naked.values()):
            return
        await asyncio.sleep(self.a.latency_ms / 1000)
        for t, n in naked.items():
            bk, rate = self.books[t], self.leg[t]["fee"]
            lv = (sorted(bk["bids"].items(), reverse=True) if n > 0
                  else sorted(bk["asks"].items()))
            left = abs(n)
            for p, s in lv:
                if left <= EPS:
                    break
                q = min(s, left)
                self.book_fill(t, "ask" if n > 0 else "bid", p, q, q * fee(rate, p), "flatten")
                left -= q
            # ponytail: shares the book can't absorb stay as inventory (marked at mid)

    def _footprint(self, asset, side, lad, qty):
        t = self.taken[asset]
        for p, s in lad:
            if qty <= EPS:
                break
            q = min(s, qty)
            key = p if side == "BUY_YES" else round(1 - p, 6)
            t[key] = t.get(key, 0) + q
            qty -= q

    # ---- maker: pmmm.requote, generalised from D/R to N outcomes
    def pick_mm(self):
        def decided(e):     # an outcome priced near certain: the result is (all but) known
            return any(max(self.books[l["asset"]]["bids"]) >= self.a.mm_max_fav for l in e["legs"])

        ok = [e for e in self.events
              if len(e["legs"]) <= self.a.mm_max_legs and e["days"] == e["days"]
              and e["days"] >= self.a.mm_min_days
              and all(self.books[l["asset"]]["bids"] and self.books[l["asset"]]["asks"]
                      for l in e["legs"]) and not decided(e)]
        self.mm = sorted(ok, key=lambda e: e["days"])[:self.a.mm_events]
        for e in self.mm:
            self.mm_state.setdefault(e["slug"], {"hist": [], "paused_until": 0.0})
        log("maker events: " + ", ".join(f"{e['slug']} ({len(e['legs'])} legs, "
                                          f"{e['days']:.1f}d)" for e in self.mm))

    def requote(self, now):
        a = self.a
        want = {}
        for ev in self.mm:
            want.update(self._event_quotes(ev, now))
        # cash: shrink every opening quote pro rata until all fit (worst case: all fill)
        need = sum(self.acct.cost(t, s, px, sz) for (t, s), (px, sz) in want.items())
        scale = min(1.0, max(self.free_cash(), 0.0) / need) if need > 0 else 1.0
        for k, (px, sz) in list(want.items()):
            unit = px if k[1] == "bid" else 1 - px
            free = sz - self.acct.cost(k[0], k[1], px, sz) / unit
            sz = math.floor(free + (sz - free) * scale)
            want[k] = (px, sz if sz >= self.leg[k[0]]["min"] else 0)
        for k in list(self.orders):
            if k not in want or want[k][1] <= 0:
                del self.orders[k]
        for (tok, side), (px, sz) in want.items():
            if sz <= 0:
                continue
            o = self.orders.get((tok, side))
            if o and abs(o.price - px) < EPS and o.size >= sz / 2:
                o.size = min(o.size, sz)              # keep queue position
                continue                              # (else re-post: back of queue)
            lv = self.books[tok]["bids" if side == "bid" else "asks"]
            self.orders[(tok, side)] = SimOrder(side, px, sz, lv.get(px, 0.0),
                                                now + a.latency_ms / 1000)

    def _event_quotes(self, ev, now):
        a = self.a
        tokens = [l["asset"] for l in ev["legs"]]
        books = [self.books[t] for t in tokens]
        if not all(b["bids"] and b["asks"] for b in books):
            return {}
        mids = [(max(b["bids"]) + min(b["asks"])) / 2 for b in books]
        fair = [m / sum(mids) for m in mids]                     # outcomes sum to 1
        st = self.mm_state[ev["slug"]]
        st["hist"] = [(t, f) for t, f in st["hist"] if now - t <= 30] + [(now, fair)]
        if any(abs(x - y) >= a.pull_jump for x, y in zip(fair, st["hist"][0][1])):
            st["paused_until"], st["hist"] = now + a.pull_cooldown_s, []
        if now < st["paused_until"]:
            return {}
        ys, _ = sets_held(self.acct.net, tokens)
        out = {}
        for l, t, b, f in zip(ev["legs"], tokens, books, fair):
            naked = self.acct.net.get(t, 0.0) - ys              # held beyond complete sets
            fv = f - a.skew_at_cap * naked / a.max_inv
            bb, ba, tick = max(b["bids"]), min(b["asks"]), l["tick"]
            size = max(l["min"], a.mm_size)
            bid = min(tick_floor(fv - a.half_spread, tick), round(ba - tick, 6))
            if naked < a.max_inv and 0 < bid < 1:
                out[(t, "bid")] = (round(bid, 4), size)
            if naked > EPS:                                     # only ever sell inventory
                ask = max(tick_ceil(fv + a.half_spread, tick), round(bb + tick, 6))
                if 0 < ask < 1:
                    out[(t, "ask")] = (round(ask, 4), min(size, math.floor(naked)))
        return out

    # ---- reporting
    def mark(self):
        """(complete-set payout, naked inventory at mid, mark-to-market P&L)."""
        locked = naked_v = 0.0
        for ev in self.events:
            tokens = [l["asset"] for l in ev["legs"]]
            if not any(abs(self.acct.net.get(t, 0.0)) > EPS for t in tokens):
                continue
            ys, ns = sets_held(self.acct.net, tokens)
            locked += ys + ns * (len(tokens) - 1)
            for t in tokens:
                n = self.acct.net.get(t, 0.0) - ys + ns
                b = self.books[t]
                if b["bids"] and b["asks"]:
                    m = (max(b["bids"]) + min(b["asks"])) / 2
                    naked_v += n * m if n >= 0 else -n * (1 - m)
        return locked, naked_v, self.acct.cash + locked + naked_v - self.a.bankroll


# --------------------------------------------------------------------- feed
async def feed(sim, assets):
    while True:
        try:
            async with websockets.connect(WS, ping_interval=20, ping_timeout=60,
                                          max_size=None) as ws:
                await ws.send(json.dumps({"assets_ids": assets, "type": "market"}))
                async for raw in ws:
                    msgs = json.loads(raw)
                    for m in msgs if isinstance(msgs, list) else [msgs]:
                        handle(sim, m)
        except Exception as e:                               # noqa: BLE001
            log(f"feed reconnect ({type(e).__name__}: {e})")
            await asyncio.sleep(2)


def handle(sim, m):
    et = m.get("event_type")
    if et == "book" and m.get("asset_id") in sim.books:
        apply_snapshot(sim.books[m["asset_id"]], m)
        sim.on_book(m["asset_id"])
    elif et == "price_change":
        for ch in m.get("price_changes") or m.get("changes") or []:
            a = ch.get("asset_id") or m.get("asset_id")
            if a in sim.books:
                p = float(ch["price"])
                apply_change(sim.books[a], p, ch["side"], float(ch["size"]))
                sim.on_book(a, p)
    elif et == "last_trade_price" and m.get("asset_id") in sim.books:
        sim.on_trade(m["asset_id"], float(m["price"]), float(m.get("size") or 0))
    elif et == "tick_size_change" and m.get("asset_id") in sim.leg:
        sim.leg[m["asset_id"]]["tick"] = float(m["new_tick_size"])


async def maker_loop(sim):
    if sim.a.mm_events <= 0:
        return
    await asyncio.sleep(15)                  # let every book seed first
    sim.pick_mm()
    while True:
        sim.requote(time.time())
        await asyncio.sleep(1)


async def exit_loop(sim):
    await asyncio.sleep(15)                  # let every book seed first
    for ev in sim.events:                    # broken sets from before a restart
        if any(abs(sim.acct.net.get(l["asset"], 0.0)) > EPS for l in ev["legs"]):
            sim.reconcile(ev)
    while True:
        sim.place_exits(time.time())
        await asyncio.sleep(2)


async def status(sim):
    while True:
        await asyncio.sleep(60)
        locked, naked, pnl = sim.mark()
        sim.save()
        log(f"[status] taker signals {sim.n_sig}  fills {sim.n_fill}  quotes "
            f"{len(sim.orders)}  exits {len(sim.exits)}  cash ${sim.acct.cash:.2f}  sets ${locked:.2f}  "
            f"naked ${naked:.2f}  fees ${sim.fees:.3f}  mtm pnl ${pnl:+.3f}")


async def run(a):
    events = discover(a.tags)
    sim = Sim(events, a)
    assets = list(sim.books)
    log(f"{len(events)} neg-risk events, {len(assets)} outcomes "
        f"(tags {','.join(a.tags)})")
    tasks = [feed(sim, assets[i:i + 200]) for i in range(0, len(assets), 200)]
    tasks += [status(sim), maker_loop(sim), exit_loop(sim)]
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=a.duration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["politics", "weather", "futures"])
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--latency-ms", type=float, default=250)
    ap.add_argument("--min-edge", type=float, default=0.005,
                    help="$ per set required after fees (taker)")
    ap.add_argument("--naked-s", type=float, default=1800,
                    help="a set broken by a resting exit is sold at the bids after this long")
    ap.add_argument("--mm-min-days", type=float, default=1.0, help="maker skips events ending sooner")
    ap.add_argument("--mm-max-fav", type=float, default=0.95,
                    help="maker skips events where an outcome's bid is this high (decided)")
    ap.add_argument("--max-event",type=float, default=0.25, help="max share of bankroll in one event")
    ap.add_argument("--reserve", type=float, default=0.3, help="share of bankroll kept as free cash")
    ap.add_argument("--min-ann", type=float, default=0.10, help="min locked-in return per year to take")
    ap.add_argument("--tp-min",type=float, default=0.005,
                    help="$ per set above cost (fees included) to sell a held set early")
    # maker / inventory (pmmm Cfg names)
    ap.add_argument("--mm-events", type=int, default=3)
    ap.add_argument("--mm-max-legs", type=int, default=8)
    ap.add_argument("--mm-size", type=float, default=5)
    ap.add_argument("--half-spread", type=float, default=0.01)
    ap.add_argument("--max-inv", type=float, default=50, help="naked shares per outcome")
    ap.add_argument("--skew-at-cap", type=float, default=0.02)
    ap.add_argument("--pull-jump", type=float, default=0.02)
    ap.add_argument("--pull-cooldown-s", type=float, default=30)
    ap.add_argument("--duration", type=float, default=3600,
                    help="seconds; then exit so a supervisor restarts with fresh discovery")
    a = ap.parse_args()
    try:
        asyncio.run(run(a))
    except asyncio.TimeoutError:
        log("duration reached; exiting for rediscovery")


if __name__ == "__main__":
    main()
