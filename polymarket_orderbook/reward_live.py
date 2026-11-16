"""Live liquidity-rewards quoter: reward_quoter.py's selection and quoting rules, with real orders.

Without --live nothing is sent: the orders it would post are logged (a wiring check; nothing fills).

What changes against the paper quoter:
  orders   every quote is a post-only GTC order. A YES-space bid buys YES at p; an ask buys NO at 1 - p. When
           the other token is held, the quote sells that instead (the same trade), so inventory is closed
           and cash does not stay locked in YES + NO pairs. Pairs under 5 shares cannot be sold this way:
           merge them on polymarket.com.
  fills    come from the exchange, never from the queue model: the user feed's order events, and a REST pass
           every RECONCILE_S that also cancels any order of ours this process does not know.
  book     our own resting size is taken out before the mid, the spread test and the reward share are
           computed, so the quotes never anchor on themselves.
  safety   orders are cancelled on start, on exit, when the market feed drops or goes silent, and by the
           exchange itself if the heartbeat stops (a crash). Positions are checked against token balances;
           the exchange wins. Quoting halts when the ledger is down --max-loss.

The wallet must hold nothing else in the quoted markets: positions there are taken to be this bot's.
A resolved market is booked at its payout but not redeemed: do that on polymarket.com.

Credentials, from the environment (never from a file in the repo):
  POLY_PRIVATE_KEY   the signing key
  POLY_FUNDER        the address holding the funds (the Polymarket proxy wallet), if not the key's own
  POLY_SIG_TYPE      0 = plain wallet (default), 1 = email / Magic login, 2 = browser-wallet proxy

    python reward_live.py                              dry run, 1 market, $50
    python reward_live.py --live                       real orders
    python reward_live.py --live --markets 3 --bankroll 150 --max-loss 15
    python reward_live.py --cancel-all                 cancel this state file's markets' orders and exit

State and logs are in data/rewards_live (same files as the paper quoter; `reward_quoter.py --report` reads
them after set_out, or run `python reward_live.py --report`).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import time

import websockets

import reward_quoter as rq
from arb_scan import SimOrder
from reward_quoter import Quoter, log, select

USER_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
MIN_ORDER = 5           # shares, the exchange's minimum order
RETRY_S = 10            # after a rejected order, leave that side alone this long
RECONCILE_S = 30
SETTLE_S = 120          # a token balance can lag a fill this long before a mismatch is believed
YOUNG_S = 5             # an order this new may not be in the open-orders listing yet


@dataclasses.dataclass(eq=False)
class Leg:
    """One real order behind a YES-space quote."""
    id: str
    t: str              # YES token of the market
    side: str           # bid | ask, YES space
    token: str
    buy: bool
    price: float        # on `token`
    size: float
    how: str
    t0: float
    matched: float = 0.0
    open: bool = True


class Exchange:
    """The few CLOB calls the quoter needs."""
    def __init__(self):
        from py_clob_client.client import ClobClient
        self.c = ClobClient(rq.CLOB, chain_id=137, key=os.environ["POLY_PRIVATE_KEY"],
                            signature_type=int(os.environ.get("POLY_SIG_TYPE", "0")),
                            funder=os.environ.get("POLY_FUNDER") or None)
        self.creds = self.c.create_or_derive_api_creds()
        self.c.set_api_creds(self.creds)

    def post(self, token, buy, price, size, taker=False):
        """Order id, or None if the exchange refused it. Makers are post-only; a taker order is fill-and-kill."""
        from py_clob_client.clob_types import OrderArgs, OrderType
        o = self.c.create_order(OrderArgs(token_id=token, price=price, size=size, side="BUY" if buy else "SELL"))
        r = self.c.post_order(o, OrderType.FAK if taker else OrderType.GTC, post_only=not taker)
        if not r.get("success") or not r.get("orderID"):
            log(f"order refused: {r.get('errorMsg') or r}")
            return None
        return r["orderID"]

    def cancel(self, ids):
        self.c.cancel_orders(ids)

    def cancel_tokens(self, tokens):
        for t in tokens:
            self.c.cancel_market_orders(asset_id=t)

    def open_orders(self):
        return self.c.get_orders()

    def order(self, oid):
        return self.c.get_order(oid)

    def balance(self, token=None):
        """Shares of a token, or dollars of collateral."""
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        r = self.c.get_balance_allowance(BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL if token else AssetType.COLLATERAL, token_id=token))
        return float(r["balance"]) / 1e6

    def heartbeat(self, hb):
        return self.c.post_heartbeat(hb)


class DryExchange:
    """Logs what would be sent. Nothing fills."""
    creds = None

    def __init__(self):
        self.o, self.n = {}, 0

    def post(self, token, buy, price, size, taker=False):
        self.n += 1
        oid = f"dry{self.n}"
        self.o[oid] = dict(id=oid, asset_id=token, size_matched="0")
        log(f"DRY {'FAK' if taker else 'post-only'} {'BUY ' if buy else 'SELL'} {size:g} @ {price:g}  token ..{token[-6:]}")
        return oid

    def cancel(self, ids):
        for i in ids:
            self.o.pop(i, None)
        log(f"DRY cancel {len(ids)}")

    def cancel_tokens(self, tokens):
        self.o = {i: o for i, o in self.o.items() if o["asset_id"] not in tokens}

    def open_orders(self):
        return list(self.o.values())

    def order(self, oid):
        return self.o.get(oid) or dict(id=oid, size_matched="0")

    def balance(self, token=None):
        return 0.0 if token else float("inf")

    def heartbeat(self, hb):
        return {}


class LiveQuoter(Quoter):
    def __init__(self, a, st, ex):
        super().__init__(a, st)
        self.ex = ex
        self.legs = {}          # (token, side) -> [Leg] behind the resting quote
        self.by_id = {}         # order id -> Leg, kept after a cancel: its fill can still arrive
        self.gone = []          # cancelled legs whose final matched size has not been read back yet
        self.retry = {}         # (token, side) -> no new order before this time
        self.last_fill = {}     # token -> time
        self.hold = {}          # YES or NO token -> shares held (both at once = a pair, cash locked until sold)
        self.halted = None      # why quoting stopped for good

    # ---- the paper quoter's simulated fills are off: the exchange reports the real ones -----------------
    def on_trade(self, t, price, size):
        pass

    def on_book(self, t):
        pass

    def others_book(self, t):
        b = {"bids": dict(self.book[t]["bids"]), "asks": dict(self.book[t]["asks"])}
        for s, o in self.orders[t].items():
            lv = b["bids" if s == "bid" else "asks"]
            for k in [k for k in lv if abs(k - o.price) < 1e-9]:
                lv[k] -= o.size
                if lv[k] < 1e-6:
                    del lv[k]
        return b

    # ---- orders -------------------------------------------------------------------------------------------
    def send(self, t, side, token, buy, price, size, taker=False, how="live"):
        try:
            oid = self.ex.post(token, buy, round(price, 4), size, taker)
        except Exception as e:                               # noqa: BLE001
            log(f"order failed ({type(e).__name__}: {e})")
            oid = None
        if oid is None:
            return None
        leg = self.by_id[oid] = Leg(oid, t, side, token, buy, round(price, 4), size, how, time.time())
        return leg

    def place(self, t, side, p, size, ahead, now):
        if self.halted or now < self.retry.get((t, side), 0):
            return
        m = self.mk[t]
        own, other = (m["yes"], m["no"]) if side == "bid" else (m["no"], m["yes"])   # bid buys YES, ask buys NO
        # selling the other token is the same trade as buying this one, and frees cash where buying locks it:
        # sell whatever is held of it first (inventory being closed, or one half of a YES + NO pair)
        close = math.floor(min(size, max(0.0, self.hold.get(other, 0.0))) * 100) / 100
        close = close if close >= MIN_ORDER else 0.0
        opn = round(size - close, 2)
        opn = opn if opn >= MIN_ORDER else 0.0
        px = p if side == "bid" else 1 - p                                            # price on `own`
        legs = []
        if close:
            legs.append(self.send(t, side, other, False, 1 - px, close))             # sell what is held
        if opn:
            legs.append(self.send(t, side, own, True, px, opn))
        legs = [l for l in legs if l]
        if not legs:
            self.retry[(t, side)] = now + RETRY_S
            return
        self.legs[(t, side)] = legs
        self.orders[t][side] = SimOrder(side, p, sum(l.size for l in legs), ahead, now)

    def drop(self, t, side=None):
        for s in [side] if side else ["bid", "ask"]:
            self.orders[t].pop(s, None)
            legs = [l for l in self.legs.pop((t, s), []) if l.open]
            if not legs:
                continue
            self.gone += legs
            try:
                self.ex.cancel([l.id for l in legs])
            except Exception as e:                           # noqa: BLE001  (reconcile cancels it as a stray)
                log(f"cancel failed ({type(e).__name__}: {e})")

    def matched(self, oid, total):
        """An order's cumulative matched size, from the user feed or from REST: book what is new."""
        l = self.by_id.get(oid)
        if l is None or total - l.matched < 1e-6:
            return
        d, l.matched = total - l.matched, total
        o = self.orders[l.t].get(l.side)
        if o and l in self.legs.get((l.t, l.side), ()):
            o.size = max(0.0, o.size - d)
        self.last_fill[l.t] = time.time()
        self.hold[l.token] = self.hold.get(l.token, 0.0) + (d if l.buy else -d)
        yes_price = l.price if l.token == self.mk[l.t]["yes"] else round(1 - l.price, 6)
        self.fill(l.t, l.side, yes_price, d, l.how)
        # the other side's quote was built for the old position: left alone it would buy the opposite token
        # and lock the cash in a YES + NO pair, where it should now sell what this fill bought. Requote rebuilds it.
        self.drop(l.t, "ask" if l.side == "bid" else "bid")
        log(f"FILL {l.side} {d:g} @ {yes_price:g} ({l.how})  {self.mk[l.t]['q'][:40]}  net {self.acct.net.get(l.t, 0):+.2f}")

    def time_stop(self, t, net):
        """Sell what is held into the best level, fill-and-kill. The taker fee is not in the ledger."""
        b, m = self.book[t], self.mk[t]
        self.drop(t)
        q = math.floor(abs(net) * 100) / 100
        leg = None
        if q >= MIN_ORDER and not self.halted:
            if net > 0 and b["bids"]:
                leg = self.send(t, "ask", m["yes"], False, max(b["bids"]), q, taker=True, how="time_stop")
            elif net < 0 and b["asks"]:
                leg = self.send(t, "bid", m["no"], False, 1 - min(b["asks"]), q, taker=True, how="time_stop")
        if leg:
            self.gone.append(leg)                            # its matched size is read back at the next reconcile
        if t in self.entry:
            self.entry[t]["opened"] = time.time() - rq.MAX_HOLD + 60     # what is left: retry in 1 min

    def settle(self, t, payout):
        super().settle(t, payout)
        log("  booked at the payout; redeem it on polymarket.com")

    def lost_feed(self, toks):
        super().lost_feed(toks)
        for t in toks:                                       # blind: nothing of ours may rest
            self.drop(t)

    # ---- truth from the exchange --------------------------------------------------------------------------
    def all_tokens(self):
        return [x for m in self.mk.values() for x in (m["yes"], m["no"])]

    def cancel_everything(self):
        for t in self.mk:
            self.orders[t].clear()
        self.gone += [l for ls in self.legs.values() for l in ls if l.open]
        self.legs.clear()
        try:
            self.ex.cancel_tokens(self.all_tokens())
            log("all orders in the quoted markets cancelled")
        except Exception as e:                               # noqa: BLE001
            log(f"CANCEL-ALL FAILED ({type(e).__name__}: {e}): check polymarket.com for resting orders")

    def reconcile(self, force=False):
        now, toks = time.time(), set(self.all_tokens())
        active = {l.id: l for ls in self.legs.values() for l in ls if l.open}
        seen, stray = set(), []
        for o in self.ex.open_orders():
            if o.get("asset_id") not in toks:
                continue
            seen.add(o["id"])
            self.matched(o["id"], float(o.get("size_matched") or 0))
            if o["id"] not in active:
                stray.append(o["id"])
        if stray:
            log(f"cancelling {len(stray)} order(s) this process does not own")
            self.ex.cancel(stray)
        closed = [l for i, l in active.items() if i not in seen]     # filled, or cancelled there
        gone, self.gone = self.gone, []
        for l in closed + gone:
            o = self.ex.order(l.id) or {}
            self.matched(l.id, float(o.get("size_matched") or 0))
            l.open = l in closed and now - l.t0 <= YOUNG_S           # too new to be listed yet: look again
        for (t, s), ls in list(self.legs.items()):
            if not any(l.open for l in ls):
                del self.legs[(t, s)]
                self.orders[t].pop(s, None)
        for t, m in self.mk.items():
            if t in self.done:
                continue
            if not (force or now - self.last_fill.get(t, 0) > SETTLE_S):     # balances may still lag the fill
                continue
            self.hold[m["yes"]], self.hold[m["no"]] = self.ex.balance(m["yes"]), self.ex.balance(m["no"])
            net, pos = self.acct.net.get(t, 0.0), self.hold[m["yes"]] - self.hold[m["no"]]
            if abs(pos - net) > 0.5:
                log(f"POSITION MISMATCH {m['q'][:40]}: ledger {net:+.2f}, exchange {pos:+.2f}; using the exchange's "
                    f"(the cash ledger is now off by the missed trades)")
                self.acct.net[t] = pos
                if abs(pos) < 1 or pos * net <= 0:
                    self.entry.pop(t, None)
        eq = self.acct.cash - self.a.bankroll
        for t, n in self.acct.net.items():
            mid = self.mid(t) if t in self.mk and t not in self.done else None
            if mid is not None:
                eq += n * mid if n > 0 else -n * (1 - mid)
            elif t in self.mk and t not in self.done:
                return                                       # a book is missing: no verdict this pass
        if eq < -self.a.max_loss and not self.halted:
            self.halted = f"ledger down ${-eq:.2f}, limit ${self.a.max_loss:.2f}"
            log(f"HALT: {self.halted}. Orders cancelled; positions are left as they are.")
            self.cancel_everything()

    async def reconcile_loop(self):
        bad = 0
        while True:
            await asyncio.sleep(RECONCILE_S)
            try:
                self.reconcile()
                bad = 0
            except Exception as e:                           # noqa: BLE001
                bad += 1
                log(f"reconcile failed {bad}x ({type(e).__name__}: {e})")
                if bad >= 5:                                 # flying blind on fills: stop (run() cancels on the way out)
                    raise

    async def user_feed(self):
        c = self.ex.creds
        while c is not None:
            try:
                async with websockets.connect(USER_WS, ping_interval=10, ping_timeout=20, max_size=None) as ws:
                    await ws.send(json.dumps({"auth": {"apiKey": c.api_key, "secret": c.api_secret,
                                                       "passphrase": c.api_passphrase},
                                              "markets": [m["cond"] for m in self.mk.values()], "type": "user"}))
                    async for raw in ws:
                        try:
                            P = json.loads(raw)
                        except ValueError:
                            continue
                        for m in P if isinstance(P, list) else [P]:
                            if isinstance(m, dict) and m.get("event_type") == "order":
                                self.matched(m.get("id"), float(m.get("size_matched") or 0))
            except Exception as e:                           # noqa: BLE001  (reconcile covers the gap)
                log(f"user feed reconnect ({type(e).__name__}: {e})")
                await asyncio.sleep(3)

    async def heartbeat_loop(self):
        """While these arrive the exchange keeps our orders; when they stop (a crash, a dead link) it cancels them."""
        hb, warned = None, False
        while not self.a.no_heartbeat:
            try:
                hb = (self.ex.heartbeat(hb) or {}).get("heartbeat_id", hb)
                warned = False
            except Exception as e:                           # noqa: BLE001
                hb = None
                if not warned:
                    log(f"heartbeat failed ({type(e).__name__}: {e})")
                    warned = True
            await asyncio.sleep(5)

    async def run(self):
        toks = list(self.mk)
        try:
            self.ex.cancel_tokens(self.all_tokens())             # nothing left over from an earlier run
            self.reconcile(force=True)                       # positions as the exchange has them
            log(f"wallet ${self.ex.balance():.2f}, bankroll ${self.a.bankroll:.2f}, ledger cash ${self.acct.cash:.2f}")
            await asyncio.gather(self.feed(toks), self.quote_loop(), self.sample_loop(), self.user_feed(),
                                 self.reconcile_loop(), self.heartbeat_loop())
        finally:
            self.cancel_everything()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="send real orders (default: log them only)")
    ap.add_argument("--fresh", action="store_true", help="pick new markets; do this only when flat")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--cancel-all", action="store_true")
    ap.add_argument("--markets", type=int, default=1)
    ap.add_argument("--bankroll", type=float, default=50.0)
    ap.add_argument("--max-min-size", type=float, default=50)
    ap.add_argument("--max-loss", type=float, default=10.0, help="halt when the ledger is down this many dollars")
    ap.add_argument("--no-heartbeat", action="store_true")
    a = ap.parse_args()
    a.survey = a.all = False
    a.shards, a.shard = 1, 0
    rq.set_out(os.path.join(os.path.dirname(rq.OUT), "rewards_live" if a.live or a.cancel_all or a.report else "rewards_live_dry"))
    rq.LATENCY = 0.0
    if a.report:
        rq.report()
        raise SystemExit
    if a.cancel_all:
        LiveQuoter(a, json.load(open(rq.STATE)), Exchange()).cancel_everything()
        raise SystemExit
    if a.fresh or not os.path.exists(rq.STATE):
        os.makedirs(rq.OUT, exist_ok=True)
        json.dump(dict(started=time.time(), bankroll=a.bankroll, cash=a.bankroll, net={}, reward=0.0, markets=select(a)),
                  open(rq.STATE, "w"), indent=1)
    st = json.load(open(rq.STATE))
    a.bankroll = st["bankroll"]
    log(f"{'LIVE' if a.live else 'DRY RUN'}: quoting {len(st['markets'])} markets, ledger cash ${st['cash']:.2f}, "
        f"max loss ${a.max_loss:.2f}")
    asyncio.run(LiveQuoter(a, st, Exchange() if a.live else DryExchange()).run())
