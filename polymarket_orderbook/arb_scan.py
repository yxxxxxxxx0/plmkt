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


def sets_held(net, tokens):
    """(complete YES sets, complete NO sets) held across an event's outcomes."""
    return (min(max(net.get(t, 0.0), 0.0) for t in tokens),
            min(max(-net.get(t, 0.0), 0.0) for t in tokens))


# ---------------------------------------------------------------- simulator
class Sim:
    def __init__(self, events, a):
        self.a = a
        self.events = events
        self.books = {l["asset"]: new_book() for e in events for l in e["legs"]}
        self.by_asset = {l["asset"]: e for e in events for l in e["legs"]}
        self.leg = {l["asset"]: l for e in events for l in e["legs"]}
        self.taken = {k: {} for k in self.books}     # our taker footprint per level
        self.busy = set()                            # events with a taker order out
        self.acct = Account(a.bankroll)
        self.reserved = 0.0                          # cash promised to taker orders in flight
        self.fees = 0.0
        self.orders = {}                             # (token, side) -> SimOrder
        self.mm = []                                 # events being quoted
        self.mm_state = {}                           # slug -> {hist, paused_until}
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
        ev = self.by_asset.get(asset)
        if ev and ev["slug"] not in self.busy:
            self.check(ev)

    def on_trade(self, asset, price, size):
        now = time.time()
        for side in ("bid", "ask"):
            o = self.orders.get((asset, side))
            if o and (f := on_trade(o, price, size, now)):
                self._maker_fill(asset, side, o, f)

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
        for side, payoff in (("BUY_YES", 1.0), ("BUY_NO", len(legs) - 1.0)):
            lads = [ladder(self.books[l["asset"]], side, self.taken[l["asset"]])
                    for l in legs]
            sets, limits, cost = size_arb(lads, fees, payoff, self.a.min_edge,
                                          max(self.free_cash(), 0.0))
            if sets >= max(l["min"] for l in legs) - EPS:
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
        w, fh = self.w
        w.writerow([int(time.time() * 1000), ev["slug"], "taker_" + side, len(legs),
                    round(sets, 2), round(sets * payoff - plan_cost, 4), round(got, 2),
                    round(cost, 4), round(fees, 4), round(ev["days"], 1), round(ann, 3)])
        fh.flush()
        log(f"taker_{side} {ev['slug']}: planned {sets:.1f} sets edge "
            f"${sets * payoff - plan_cost:.3f} -> completed {got:.1f}, set edge ${edge:+.3f}")
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
        ok = [e for e in self.events
              if len(e["legs"]) <= self.a.mm_max_legs and e["days"] == e["days"]
              and all(self.books[l["asset"]]["bids"] and self.books[l["asset"]]["asks"]
                      for l in e["legs"])]
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


async def status(sim):
    while True:
        await asyncio.sleep(60)
        locked, naked, pnl = sim.mark()
        log(f"[status] taker signals {sim.n_sig}  fills {sim.n_fill}  quotes "
            f"{len(sim.orders)}  cash ${sim.acct.cash:.2f}  sets ${locked:.2f}  "
            f"naked ${naked:.2f}  fees ${sim.fees:.3f}  mtm pnl ${pnl:+.3f}")


async def run(a):
    events = discover(a.tags)
    sim = Sim(events, a)
    assets = list(sim.books)
    log(f"{len(events)} neg-risk events, {len(assets)} outcomes "
        f"(tags {','.join(a.tags)})")
    tasks = [feed(sim, assets[i:i + 200]) for i in range(0, len(assets), 200)]
    tasks += [status(sim), maker_loop(sim)]
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=a.duration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["politics", "weather", "futures"])
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--latency-ms", type=float, default=250)
    ap.add_argument("--min-edge", type=float, default=0.005,
                    help="$ per set required after fees (taker)")
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
