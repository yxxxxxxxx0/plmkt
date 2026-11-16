"""python reward_recorder.py [--markets 150] -- record books and prints for liquidity-reward markets.

The sports recorder only covers games. This one follows the non-sports markets the reward quoter
trades (Powell-type: politics, economics, tech), so jumps there can be studied the same way. It
writes data/live_rewards/books_<utc date>.jsonl with the same row shape as live_recorder: "book"
and "price_change" rows carry the top LEVELS of each side, "trade" rows carry the print. YES
tokens only, reselected every RESELECT seconds."""
import argparse
import asyncio
import datetime as dt
import json
import os
import time

import websockets

from live_recorder import apply_change, apply_snapshot, new_book
from reward_quoter import CLOB, GAMMA, WS, get, log

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "live_rewards")
LEVELS = 20
RESELECT = 6 * 3600


def select(n):
    R, cur = [], ""
    while cur != "LTE=":
        d = get(f"{CLOB}/rewards/markets/current", next_cursor=cur) or {}
        R += d.get("data") or []
        cur = d.get("next_cursor") or "LTE="
    R = {r["condition_id"]: r for r in R if r["total_daily_rate"] > 0}
    ids, M = list(R), []
    for i in range(0, len(ids), 50):
        M += get(f"{GAMMA}/markets", condition_ids=ids[i:i + 50], limit=50) or []
    C = []
    for m in M:
        try:
            yes = json.loads(m["clobTokenIds"])[0]
            p = float(json.loads(m["outcomePrices"])[0])
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if "sports" in (m.get("feeType") or "") or not m.get("acceptingOrders") or not 0.05 <= p <= 0.95:
            continue
        C.append((R[m["conditionId"]]["total_daily_rate"], yes, m["conditionId"], m["question"][:80]))
    C.sort(reverse=True)
    log(f"recording {min(n, len(C))} of {len(C)} non-sports reward markets")
    mk = {yes: dict(cond=c, q=q) for _, yes, c, q in C[:n]}
    try:                                       # always include what the quoter is trading
        for m in json.load(open(os.path.join(OUT, "..", "rewards", "state.json")))["markets"]:
            mk.setdefault(m["yes"], dict(cond=m["cond"], q=m["q"]))
    except (OSError, ValueError, KeyError):
        pass
    return mk


class Recorder:
    def __init__(self, mk):
        self.mk, self.book, self.seq, self.f, self.day = mk, {t: new_book() for t in mk}, 0, None, None

    def write(self, row):
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        if day != self.day:
            if self.f:
                self.f.close()
            os.makedirs(OUT, exist_ok=True)
            self.f, self.day = open(os.path.join(OUT, f"books_{day}.jsonl"), "a"), day
        self.seq += 1
        self.f.write(json.dumps({"seq": self.seq, **row}) + "\n")

    def emit(self, et, t, ts, recv):
        b = self.book[t]
        m = self.mk[t]
        self.write({"ts": ts, "recv": recv, "et": et, "asset_id": t, "condition_id": m["cond"],
                    "market_question": m["q"],
                    "bids": sorted(b["bids"].items(), reverse=True)[:LEVELS],
                    "asks": sorted(b["asks"].items())[:LEVELS]})

    def handle(self, m, recv):
        et, t = m.get("event_type"), m.get("asset_id")
        ts = int(m.get("timestamp") or 0)
        if et == "book" and t in self.book:
            apply_snapshot(self.book[t], m)
            self.emit("book", t, ts, recv)
        elif et == "price_change":
            seen = set()
            for ch in m.get("price_changes") or m.get("changes") or []:
                u = ch.get("asset_id") or t
                if u in self.book:
                    apply_change(self.book[u], float(ch["price"]), ch["side"], float(ch["size"]))
                    seen.add(u)
            for u in seen:
                self.emit("price_change", u, ts, recv)
        elif et == "last_trade_price" and t in self.book:
            self.write({"ts": ts, "recv": recv, "et": "trade", "asset_id": t, "condition_id": self.mk[t]["cond"],
                        "price": float(m["price"]), "size": float(m.get("size") or 0), "side": m.get("side")})

    async def run(self, until):
        while time.time() < until:
            try:
                async with websockets.connect(WS, ping_interval=20, ping_timeout=60, max_size=None) as ws:
                    await ws.send(json.dumps({"assets_ids": list(self.mk), "type": "market"}))
                    while time.time() < until:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=max(1, until - time.time()))
                        except asyncio.TimeoutError:
                            break
                        recv = int(time.time() * 1000)
                        P = json.loads(raw)
                        for m in P if isinstance(P, list) else [P]:
                            if isinstance(m, dict):
                                self.handle(m, recv)
                        self.f and self.f.flush()
            except Exception as e:                           # noqa: BLE001
                for t in self.book:                          # stale until the resubscribe snapshot
                    self.book[t] = new_book()
                log(f"feed reconnect ({type(e).__name__}: {e})")
                await asyncio.sleep(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markets", type=int, default=150)
    a = p.parse_args()
    while True:
        mk = select(a.markets)
        r = Recorder(mk)
        asyncio.run(r.run(time.time() + RESELECT))
        r.f and r.f.close()


if __name__ == "__main__":
    main()
