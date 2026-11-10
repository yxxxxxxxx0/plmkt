"""Paper quoter for Polymarket liquidity rewards. Nothing is sent to the exchange.

Question it answers: does a small two-sided quote earn more in rewards than it loses to the
traders who fill it? The 2026-11-10 snapshot screen said a ~$19 quote earns a median ~1%/day
of its capital in rewards; this measures the other half, the fills, live.

Selection (once, saved in the state file so a restart resumes the same markets):
  every market in /rewards/markets/current with min size <= --max-min-size, no taker delay,
  not sports, 9-120 days to its end date, traded in the last 24 h, mid 15-85c, another maker
  already scoring (an empty book is usually a new or decided market), book spread within 2 x the
  reward band (a wider book's mid is no fair value: on 2026-11-10 other makers re-centred 10-25c
  away from it and ran over every quote), then the --markets best
  by estimated reward per dollar that fit in the bankroll.

Quoting, in YES space (an ask on YES is a bid on NO): size = max(min size, 20) shares, half way
out in the reward band (score 0.25 of the best), re-priced when the target moves a tick, pulled
for 60 s when the mid moves 5c within 30 s. Inventory cap 2 x size per side, with quotes leaned
against the position (sigbot/pmmm.py's skew) so it is worked off by the opposite quote, not held. Fills use the queue
model from arb_scan.py (join the back of the level; prints at our price eat the queue first; a
print or the opposite book through our price fills us). Rewards are scored once a minute with the
docs' formula: S = ((v - s) / v)^2 per share, min of the two sides (single side / 3 when the mid
is 10-90c), pro rata against everyone else resting in the band.

Writes data/rewards/minutes.csv (one row per market per minute), data/rewards/fills.csv and
data/rewards/state.json.

    python reward_quoter.py                 run (resumes data/rewards/state.json if present)
    python reward_quoter.py --fresh         pick new markets (keeps the old logs)
    python reward_quoter.py --report        rewards vs fill P&L so far, with markouts
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import os
import random
import time

import requests
import websockets

from arb_scan import Account, SimOrder, on_book, on_trade, tick_ceil, tick_floor
from live_recorder import apply_change, apply_snapshot, new_book

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rewards")
STATE, MIN_CSV, FILL_CSV = (os.path.join(OUT, f) for f in ("state.json", "minutes.csv", "fills.csv"))
WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CLOB, GAMMA = "https://clob.polymarket.com", "https://gamma-api.polymarket.com"
LATENCY, PULL_MOVE, PULL_S = 0.25, 0.05, 60


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def get(url, **kw):
    for i in range(4):
        try:
            r = requests.get(url, params=kw, timeout=60)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 + 3 * i)
    return None


def score(levels, mid, v, minsz):
    """Reward points of resting (price, size) levels within v of the mid."""
    return sum(((v - abs(p - mid)) / v) ** 2 * q for p, q in levels if abs(p - mid) < v and q >= minsz)


def q_min(q_bid, q_ask, mid):
    if 0.10 <= mid <= 0.90:
        return max(min(q_bid, q_ask), max(q_bid, q_ask) / 3)
    return min(q_bid, q_ask)


def books(tokens):
    out = {}
    for i in range(0, len(tokens), 100):
        try:
            r = requests.post(f"{CLOB}/books", json=[{"token_id": t} for t in tokens[i:i + 100]], timeout=60).json()
            out.update({b["asset_id"]: b for b in r})
        except (requests.RequestException, ValueError, KeyError):
            pass
    return out


def select(a):
    R, cur = [], ""
    while cur != "LTE=":
        d = get(f"{CLOB}/rewards/markets/current", next_cursor=cur) or {}
        R += d.get("data") or []
        cur = d.get("next_cursor") or "LTE="
    R = {r["condition_id"]: r for r in R if 0 < r["rewards_min_size"] <= a.max_min_size and r["rewards_max_spread"] > 0
         and r["total_daily_rate"] > 0}
    log(f"{len(R)} reward markets with min size <= {a.max_min_size}")
    ids, M = list(R), []
    for i in range(0, len(ids), 50):
        M += get(f"{GAMMA}/markets", condition_ids=ids[i:i + 50], limit=50) or []
    now = dt.datetime.now(dt.timezone.utc)
    C = []
    for m in M:
        try:
            days = (dt.datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")) - now).days
            yes, no = json.loads(m["clobTokenIds"])
        except (KeyError, ValueError, TypeError):
            continue
        if (m.get("secondsDelay") or 0) or "sports" in (m.get("feeType") or "") or not m.get("acceptingOrders") \
                or not 9 <= days <= 120 or float(m.get("volume24hrClob") or 0) <= 0:
            continue
        r = R[m["conditionId"]]
        C.append(dict(cond=m["conditionId"], q=m["question"][:80], yes=yes, no=no, rate=r["total_daily_rate"],
                      v=r["rewards_max_spread"] / 100, minsz=r["rewards_min_size"], tick=float(m.get("orderPriceMinTickSize") or 0.01),
                      end=m["endDate"][:10], fee=m.get("feeType") or ""))
    log(f"{len(C)} pass the market filters; reading their books")
    B = books([c["yes"] for c in C])
    for c in C:
        b = B.get(c["yes"]) or {}
        bid = [(float(x["price"]), float(x["size"])) for x in b.get("bids", [])]
        ask = [(float(x["price"]), float(x["size"])) for x in b.get("asks", [])]
        c["est"] = 0.0
        if not bid or not ask:
            continue
        mid = (max(p for p, _ in bid) + min(p for p, _ in ask)) / 2
        others = q_min(score(bid, mid, c["v"], c["minsz"]), score(ask, mid, c["v"], c["minsz"]), mid)
        n = max(c["minsz"], 20)
        c.update(mid=mid, size=n, capital=n * (1 - c["v"]), others=others)   # both sides rest: n*(mid-v/2) + n*(1-mid-v/2)
        spread = min(p for p, _ in ask) - max(p for p, _ in bid)
        if 0.15 <= mid <= 0.85 and others > 0 and spread <= 2 * c["v"]:
            mine = 0.25 * n
            c["est"] = c["rate"] * mine / (mine + others) / c["capital"]
    C.sort(key=lambda c: -c["est"])
    pick, spent = [], 0.0
    for c in C:
        if c["est"] <= 0 or len(pick) == a.markets:
            break
        if spent + c["capital"] <= a.bankroll:
            pick.append(c); spent += c["capital"]
    for c in pick:
        log(f"  {100 * c['est']:5.2f}%/day est  ${c['rate']:>6.1f}/day pool  mid {c['mid']:.2f}  v {100 * c['v']:.1f}c  "
            f"{c['size']} sh  ends {c['end']}  {c['q']}")
    return pick


class Quoter:
    def __init__(self, a, st):
        self.a, self.mk = a, {m["yes"]: m for m in st["markets"]}
        self.acct = Account(cash=st["cash"], net=st["net"])
        self.reward = st["reward"]
        self.book = {t: new_book() for t in self.mk}
        self.orders = {t: {} for t in self.mk}        # token -> side -> SimOrder
        self.hist = {t: [] for t in self.mk}          # (time, mid) for the pull rule
        self.pulled = {t: 0.0 for t in self.mk}
        self.n_fill = 0
        os.makedirs(OUT, exist_ok=True)
        self.fw = self._csv(FILL_CSV, ["ts", "cond", "side", "price", "qty", "mid", "net_after", "how"])
        self.mw = self._csv(MIN_CSV, ["ts", "cond", "mid", "best_bid", "best_ask", "my_bid", "my_ask", "my_q", "others_q",
                                      "share", "reward_min", "net", "cash"])

    @staticmethod
    def _csv(path, header):
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a", newline="", encoding="utf-8")
        w = csv.writer(f)
        if new:
            w.writerow(header)
        return f, w

    def mid(self, t):
        b = self.book[t]
        return (max(b["bids"]) + min(b["asks"])) / 2 if b["bids"] and b["asks"] else None

    def fill(self, t, side, price, qty, how):
        if qty <= 0:
            return
        self.acct.fill(t, side, price, qty)
        self.n_fill += 1
        f, w = self.fw
        w.writerow([round(time.time(), 3), self.mk[t]["cond"], side, price, round(qty, 4), self.mid(t),
                    round(self.acct.net.get(t, 0), 4), how]); f.flush()

    def on_trade(self, t, price, size):
        for side, o in list(self.orders[t].items()):
            self.fill(t, side, o.price, on_trade(o, price, size, time.time()), "print")

    def on_book(self, t):
        b, now = self.book[t], time.time()
        for side, o in list(self.orders[t].items()):
            lvl = (b["bids"] if side == "bid" else b["asks"]).get(o.price, 0.0)
            opp = (min(b["asks"]) if b["asks"] else None) if side == "bid" else (max(b["bids"]) if b["bids"] else None)
            self.fill(t, side, o.price, on_book(o, lvl, opp, now), "cross")

    def requote(self, t, now):
        m, mid, b = self.mk[t], self.mid(t), self.book[t]
        if mid is None:
            self.orders[t] = {}
            return
        h = self.hist[t]
        h.append((now, mid))
        while h and h[0][0] < now - 30:
            h.pop(0)
        if max(x for _, x in h) - min(x for _, x in h) >= PULL_MOVE:
            self.pulled[t] = now + PULL_S
        if now < self.pulled[t] or min(b["asks"]) - max(b["bids"]) > 2 * m["v"]:   # no anchor for the mid
            self.orders[t] = {}
            return
        net, n = self.acct.net.get(t, 0.0), m["size"]
        # inventory skew, as in sigbot/pmmm.py: lean both quotes against the position so the other side
        # fills and flattens it instead of holding to resolution. At the cap the closing quote sits at the mid.
        c = mid - m["v"] / 2 * net / (2 * n)
        tgt = {"bid": tick_floor(c - m["v"] / 2, m["tick"]), "ask": tick_ceil(c + m["v"] / 2, m["tick"])}
        tgt["bid"] = min(tgt["bid"], round(min(b["asks"]) - m["tick"], 6))   # never cross
        tgt["ask"] = max(tgt["ask"], round(max(b["bids"]) + m["tick"], 6))
        for side in ("bid", "ask"):
            capped = net >= 2 * n if side == "bid" else net <= -2 * n
            p = tgt[side]
            if capped or not 0 < p < 1:
                self.orders[t].pop(side, None)
                continue
            o = self.orders[t].get(side)
            if o and abs(o.price - p) < 1e-9 and o.size > 0:
                continue
            if self.acct.cost(t, side, p, n) > self.free_cash(t, side):
                self.orders[t].pop(side, None)
                continue
            ahead = (b["bids"] if side == "bid" else b["asks"]).get(p, 0.0)
            self.orders[t][side] = SimOrder(side, p, n, ahead, now + LATENCY)

    def free_cash(self, t, side):
        """Cash not reserved by the other resting orders (all of them could fill)."""
        used = sum(self.acct.cost(u, s, o.price, o.size) for u, os_ in self.orders.items() for s, o in os_.items()
                   if (u, s) != (t, side))
        return self.acct.cash - used

    def sample(self):
        now = time.time()
        for t, m in self.mk.items():
            mid, b = self.mid(t), self.book[t]
            if mid is None:
                continue
            others = q_min(score(b["bids"].items(), mid, m["v"], m["minsz"]), score(b["asks"].items(), mid, m["v"], m["minsz"]), mid)
            live = {s: o for s, o in self.orders[t].items() if now >= o.live_at and o.size > 0}
            mine = q_min(*(score([(o.price, o.size)], mid, m["v"], m["minsz"]) if (o := live.get(s)) else 0.0
                           for s in ("bid", "ask")), mid)
            share = mine / (mine + others) if mine > 0 else 0.0
            r = m["rate"] / 1440 * share
            self.reward += r
            f, w = self.mw
            w.writerow([round(now), m["cond"], mid, max(b["bids"]), min(b["asks"]),
                        live["bid"].price if "bid" in live else "", live["ask"].price if "ask" in live else "",
                        round(mine, 2), round(others, 2), round(share, 4), round(r, 5),
                        round(self.acct.net.get(t, 0), 4), round(self.acct.cash, 4)])
            f.flush()
        st = json.load(open(STATE))
        st.update(cash=self.acct.cash, net=self.acct.net, reward=self.reward)
        json.dump(st, open(STATE, "w"), indent=1)

    def handle(self, msg):
        et, t = msg.get("event_type"), msg.get("asset_id")
        if et == "book" and t in self.book:
            apply_snapshot(self.book[t], msg); self.on_book(t)
        elif et == "price_change":
            for ch in msg.get("price_changes") or msg.get("changes") or []:
                u = ch.get("asset_id") or t
                if u in self.book:
                    apply_change(self.book[u], float(ch["price"]), ch["side"], float(ch["size"])); self.on_book(u)
        # ponytail: YES-token prints only; NO-token prints could double-count a minting match. Fills
        # from NO-side flow still arrive through on_book's cross check.
        elif et == "last_trade_price" and t in self.book:
            self.on_trade(t, float(msg["price"]), float(msg.get("size") or 0))
        elif et == "tick_size_change" and t in self.mk:
            self.mk[t]["tick"] = float(msg["new_tick_size"])

    async def feed(self):
        while True:
            try:
                async with websockets.connect(WS, ping_interval=20, ping_timeout=60, max_size=None) as ws:
                    await ws.send(json.dumps({"assets_ids": list(self.mk), "type": "market"}))
                    async for raw in ws:
                        P = json.loads(raw)
                        for m in P if isinstance(P, list) else [P]:
                            if isinstance(m, dict):
                                self.handle(m)
            except Exception as e:                           # noqa: BLE001
                for t in self.book:                          # stale until the resubscribe snapshot
                    self.book[t] = new_book(); self.orders[t] = {}
                log(f"feed reconnect ({type(e).__name__}: {e})")
                await asyncio.sleep(3)

    async def quote_loop(self):
        while True:
            now = time.time()
            for t in self.mk:
                self.requote(t, now)
            await asyncio.sleep(1)

    async def sample_loop(self):
        while True:
            await asyncio.sleep(60 - time.time() % 60 + random.uniform(0, 59))   # one random instant per minute
            self.sample()
            if int(time.time() // 60) % 15 == 0:
                log(f"reward ${self.reward:.2f}  fills {self.n_fill}  cash ${self.acct.cash:.2f}  "
                    f"positions {{{', '.join(f'{self.mk[t]['q'][:20]}: {n:+.0f}' for t, n in self.acct.net.items() if abs(n) > 1e-6)}}}")

    async def run(self):
        await asyncio.gather(self.feed(), self.quote_loop(), self.sample_loop())


def report():
    import pandas as pd
    st = json.load(open(STATE))
    M = pd.read_csv(MIN_CSV)
    F = pd.read_csv(FILL_CSV) if os.path.exists(FILL_CSV) and os.path.getsize(FILL_CSV) else pd.DataFrame()
    days = (M.ts.max() - M.ts.min()) / 86400
    print(f"{len(st['markets'])} markets, {days:.2f} days logged, bankroll ${st['bankroll']:.0f}")
    by = M.groupby("cond").agg(minutes=("ts", "size"), quoted=("my_q", lambda x: (x > 0).mean()), share=("share", "mean"),
                               reward=("reward_min", "sum"))
    q = {m["cond"]: m["q"][:50] for m in st["markets"]}
    by.index = [q.get(c, c[:12]) for c in by.index]
    print(by.round(3).to_string())
    daily = M.assign(day=pd.to_datetime(M.ts, unit="s").dt.date).groupby("day").reward_min.sum()
    paid = daily.where(daily >= 1, 0)                  # payouts under $1 a day are not paid (assumed account-wide)
    print(f"\nrewards accrued ${daily.sum():.2f}; paid after the $1/day minimum ${paid.sum():.2f}")
    last = M.sort_values("ts").groupby("cond").tail(1).set_index("cond")
    cond = {m["yes"]: m["cond"] for m in st["markets"]}
    for how, col in (("mid", "mid"), ("price you could sell or cover at", None)):
        px = lambda t, n: last.loc[cond[t], "mid"] if col else last.loc[cond[t], "best_bid" if n > 0 else "best_ask"]
        pos = sum(n * px(t, n) if n > 0 else -n * (1 - px(t, n)) for t, n in st["net"].items() if t in cond)   # short YES = long NO
        fill_pnl = st["cash"] + pos - st["bankroll"]
        print(f"fill P&L marked at the last {how}: ${fill_pnl:+.2f} ({len(F)} fills)  ->  net ${paid.sum() + fill_pnl:+.2f}")
    if len(F):
        rt = F.assign(sgn=F.side.map({"bid": -1, "ask": 1}))
        print(f"  fills: {(F.side == 'bid').sum()} buys, {(F.side == 'ask').sum()} sells; "
              f"shares still held {sum(abs(n) for n in st['net'].values()):.0f}")
        M2 = M.sort_values("ts").astype({"ts": float})
        for h in (60, 600, 3600):
            x = pd.merge_asof(F.assign(t2=F.ts + h).sort_values("t2"), M2[["ts", "cond", "mid"]].rename(columns={"ts": "t2", "mid": "mid_h"}),
                              on="t2", by="cond", direction="forward").dropna(subset=["mid_h"])
            sgn = x.side.map({"bid": 1, "ask": -1})
            mk = sgn * (x.mid_h - x.price)
            print(f"  markout +{h // 60:>3} min: {100 * (mk * x.qty).sum() / max(x.qty.sum(), 1):+.2f}c/share over {len(x)} fills "
                  f"(spread captured at fill {100 * (sgn * (x.mid - x.price) * x.qty).sum() / max(x.qty.sum(), 1):+.2f}c)")


def selftest():
    assert abs(score([(0.49, 100)], 0.50, 0.03, 20) - 100 * (2 / 3) ** 2) < 1e-9
    assert score([(0.49, 10)], 0.50, 0.03, 20) == 0                 # under the min size
    assert q_min(9, 0, 0.5) == 3 and q_min(9, 0, 0.95) == 0
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--markets", type=int, default=5)
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--max-min-size", type=float, default=50)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.report:
        report()
    else:
        if a.fresh or not os.path.exists(STATE):
            os.makedirs(OUT, exist_ok=True)
            json.dump(dict(started=time.time(), bankroll=a.bankroll, cash=a.bankroll, net={}, reward=0.0, markets=select(a)),
                      open(STATE, "w"), indent=1)
        st = json.load(open(STATE))
        log(f"quoting {len(st['markets'])} markets, cash ${st['cash']:.2f}, reward so far ${st['reward']:.2f}")
        asyncio.run(Quoter(a, st).run())
