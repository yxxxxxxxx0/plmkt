"""python test_reward_live.py -- LiveQuoter against a fake exchange: random fills, lost fill messages, failed
cancels and refused orders. After every reconcile the ledger must equal the exchange."""
import json
import os
import random
import tempfile
import types

import reward_live as rl
import reward_quoter as rq


class Clock:
    t = 1_800_000_000.0
    time = classmethod(lambda cls: cls.t)
    strftime = staticmethod(lambda f, *a: "")


class Fake:
    """Holds orders and balances the way the CLOB does; fills what the test tells it to."""
    creds = None

    def __init__(self, r, cash):
        self.r, self.o, self.n, self.bal, self.cash = r, {}, 0, {}, cash
        self.oversold = self.cancel_fail = self.refused = self.broke = 0

    def post(self, token, buy, price, size, taker=False):
        assert 0 < price < 1 and size >= rl.MIN_ORDER and abs(size * 100 - round(size * 100)) < 1e-6, (price, size)
        if self.r.random() < 0.05:
            self.refused += 1
            return None
        if buy:                                              # a buy needs the cash, less what other buys hold
            locked = sum((o["size"] - o["m"]) * o["price"] for o in self.o.values() if o["live"] and o["buy"])
            if size * price > self.cash - locked + 1e-6:
                self.broke += 1
                return None
        else:                                                # a sell needs the shares, less what other sells hold
            locked = sum(o["size"] - o["m"] for o in self.o.values() if o["live"] and not o["buy"] and o["asset_id"] == token)
            if size > self.bal.get(token, 0) - locked + 1e-6:
                self.oversold += 1
                return None
        self.n += 1
        oid = f"o{self.n}"
        self.o[oid] = dict(id=oid, asset_id=token, buy=buy, price=price, size=size, m=0.0, live=True)
        if taker:
            self.fill(oid, size * self.r.random())
            self.o[oid]["live"] = False
        return oid

    def fill(self, oid, q):
        o = self.o[oid]
        q = min(q, o["size"] - o["m"])
        o["m"] += q
        sgn = 1 if o["buy"] else -1
        self.bal[o["asset_id"]] = self.bal.get(o["asset_id"], 0) + sgn * q
        self.cash -= sgn * q * o["price"]
        if o["size"] - o["m"] < 1e-9:
            o["live"] = False

    def cancel(self, ids):
        if self.r.random() < 0.05:
            self.cancel_fail += 1
            raise RuntimeError("503")
        for i in ids:
            self.o[i]["live"] = False

    def cancel_tokens(self, tokens):
        for o in self.o.values():
            if o["asset_id"] in tokens:
                o["live"] = False

    def open_orders(self):
        return [dict(id=o["id"], asset_id=o["asset_id"], size_matched=str(o["m"])) for o in self.o.values() if o["live"]]

    def order(self, oid):
        return dict(id=oid, size_matched=str(self.o[oid]["m"]))

    def balance(self, token=None):
        return self.bal.get(token, 0.0) if token else self.cash

    def heartbeat(self, hb):
        return {}


def book(t, mid, tick, r, wide=False):
    h = tick * (r.randint(1, 2) if not wide else 30)
    lv = lambda s: [dict(price=f"{mid + s * (h + j * tick):.4f}", size=f"{r.uniform(50, 500):.2f}") for j in range(5)]
    return dict(event_type="book", asset_id=t, bids=lv(-1), asks=lv(1))


def run(seed, steps=40000):
    r = random.Random(seed)
    rq.time = rl.time = Clock
    rq.log = rl.log = lambda m: None
    rq.set_out(tempfile.mkdtemp())
    ms = [dict(cond=f"0x{i}", q=f"m{i}", yes=f"Y{i}", no=f"N{i}", rate=100, v=0.045, minsz=20, tick=tick, fee="tech_fees",
               size=sz) for i, (tick, sz) in enumerate([(0.01, 20), (0.001, 50), (0.01, 20)])]
    st = dict(started=Clock.t, bankroll=200.0, cash=200.0, net={}, reward=0.0, markets=ms)
    json.dump(st, open(rq.STATE, "w"))
    ex = Fake(r, 200.0)
    q = rl.LiveQuoter(types.SimpleNamespace(all=False, shards=1, bankroll=200.0, max_loss=1e9, no_heartbeat=True), st, ex)
    mid = {m["yes"]: r.uniform(0.3, 0.7) for m in ms}
    fills = checks = max_pairs = 0
    for i in range(steps):
        Clock.t += 1
        for m in ms:
            t = m["yes"]
            mid[t] = min(0.9, max(0.1, mid[t] + r.gauss(0, 0.0004)))
            c = round(mid[t] / m["tick"]) * m["tick"]
            if r.random() < 0.002:
                q.lost_feed([t])
            q.handle(book(t, c, m["tick"], r, wide=r.random() < 0.01))
            q.requote(t, Clock.t)
        if i % 3000 == 2999:                                 # age a position past the time stop
            for e in q.entry.values():
                e["opened"] -= rq.MAX_HOLD
        live = [o for o in ex.o.values() if o["live"]]
        if live and r.random() < 0.05:
            o = r.choice(live)
            ex.fill(o["id"], r.choice([1.0, 0.3]) * (o["size"] - o["m"]) if r.random() < 0.8 else 0.37)
            fills += 1
            if r.random() < 0.5:                             # half the fill messages are lost: REST must find them
                q.matched(o["id"], o["m"])
        if i % 30 == 29:
            try:
                q.reconcile()
                q.reconcile()                                # the second pass reads back what the first cancelled
            except RuntimeError:                             # a failed stray cancel: reconcile_loop retries
                continue
            checks += 1
            for m in ms:
                pos = ex.bal.get(m["yes"], 0) - ex.bal.get(m["no"], 0)
                assert abs(q.acct.net.get(m["yes"], 0) - pos) < 1e-6, (i, m["q"], q.acct.net.get(m["yes"]), pos)
                assert abs(pos) <= 2 * m["size"] + m["size"] + 1e-6, (i, pos)
            # the ledger nets YES against NO at $1 a pair; the exchange holds such pairs until they are merged
            pairs = sum(min(ex.bal.get(m["yes"], 0), ex.bal.get(m["no"], 0)) for m in ms)
            max_pairs = max(max_pairs, pairs)
            for m in ms:                                     # holdings token by token, not just the net
                assert all(abs(q.hold.get(m[k], 0) - ex.bal.get(m[k], 0)) < 1e-6 for k in ("yes", "no")), i
            assert abs(q.acct.cash - ex.cash - pairs) < 1e-6, (i, q.acct.cash, ex.cash, pairs)
            active = {l.id: l for ls in q.legs.values() for l in ls if l.open}
            live = {o["id"] for o in ex.o.values() if o["live"]}
            assert live <= set(active), i                    # nothing rests that we do not track
            assert all(Clock.t - active[x].t0 <= rl.YOUNG_S for x in set(active) - live), i
            for (t, s), ls in q.legs.items():
                assert abs(q.orders[t][s].size - sum(l.size - l.matched for l in ls)) < 1e-6
    q.cancel_everything()
    assert not any(o["live"] for o in ex.o.values())
    # sells the exchange refused for lack of shares: a failed cancel or an unseen fill, cleared by the next reconcile
    how = {}
    for l in q.by_id.values():
        how[l.how] = how.get(l.how, 0) + (l.matched > 0)
    print(f"seed {seed}: {len(ex.o)} orders, {fills} fills ({how}), {checks} reconciles matched; refused {ex.refused}, "
          f"failed cancels {ex.cancel_fail}, sells refused for shares {ex.oversold}, buys refused for cash {ex.broke}, max |net|/size "
          f"{max(abs(n) / q.mk[t]['size'] for t, n in q.acct.net.items()) if q.acct.net else 0:.2f} at the end, "
          f"most YES+NO pairs held {max_pairs:.1f}")


if __name__ == "__main__":
    for s in (1, 2, 3):
        run(s)
    print("ok")
