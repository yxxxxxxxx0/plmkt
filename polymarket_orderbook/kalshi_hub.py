"""
One Kalshi poller shared by every kalshi_recorder.py on this machine.

Why: Kalshi rate-limits by IP. With one recorder per match group, five
groups at once drove each down to ~0.7 good polls/s and thousands of 429s
(2026-10-18). But the batch endpoint returns up to ~60 books per request
for the same cost as one, so a single poller covering every live ticker
gives every game the full rate again.

How:
  * Listens on a unix socket (data/kalshi_hub.sock). A recorder connects
    and sends one line: {"tickers": [...], "session": TAG}.
  * Polls /markets/orderbooks for the union of all subscribers' tickers
    (phase-offset pollers, adaptive interval, 429 back-off -- the same
    scheme kalshi_recorder.py used on its own) and the global trade feed.
  * Every reply is forwarded, as it arrives, to each subscriber whose
    tickers it contains, filtered to those tickers:
        {"type": "reply", "poll_id", "t_send", "t_recv", "status", "orderbooks": [...]}
        {"type": "trade", "t_seen", "trade": {...raw Kalshi trade...}}
    The recorder diffs and writes exactly as before, so its files keep
    their format.
  * poll_ids start at the hub's start time in ms, so they never repeat
    across hub restarts.
  * Started on demand by the first recorder that cannot connect; a lock
    file keeps it to one instance. Exits after --idle-exit seconds with
    no subscribers.

Usage (normally not run by hand):
    python kalshi_hub.py
"""
import argparse
import asyncio
import fcntl
import os
import statistics
import time
from collections import deque

import aiohttp

from kalshi_recorder import API, HEADERS, _iso_to_s, _loads
from live_recorder import log, now_ms

try:
    import orjson

    def _dumpb(o):
        return orjson.dumps(o) + b"\n"
except ImportError:
    import json

    def _dumpb(o):
        return (json.dumps(o, separators=(",", ":")) + "\n").encode()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOCK = os.path.join(BASE_DIR, "data", "kalshi_hub.sock")
LOCK = os.path.join(BASE_DIR, "data", "kalshi_hub.lock")
MAX_BUFFER = 64 << 20          # drop a subscriber that stops reading


class Sub:
    def __init__(self, writer, tickers, name):
        self.w = writer
        self.tickers = set(tickers)
        self.name = name
        self.dead = False

    def send(self, data):
        if self.dead:
            return
        if self.w.is_closing() or self.w.transport.get_write_buffer_size() > MAX_BUFFER:
            self.dead = True
            log(f"[hub] dropping subscriber {self.name} (closed or not reading)")
            self.w.close()
            return
        self.w.write(data)


class Hub:
    def __init__(self, args):
        self.args = args
        self.subs = set()
        self.poll_id = now_ms()
        self.inflight = 0
        self.rtts = deque(maxlen=200)
        self.penalty = 1.0
        self.last_429 = 0.0
        self.last_sub = time.monotonic()
        self.n_polls = self.n_ok = self.n_429 = self.n_err = 0
        self.sess = None
        self.stop = asyncio.Event()

    def union(self):
        return sorted(set().union(*(s.tickers for s in self.subs if not s.dead))) if self.subs else []

    def chunks(self):
        u, bs = self.union(), self.args.batch_size
        return [u[i:i + bs] for i in range(0, len(u), bs)]

    # -- subscribers ------------------------------------------------------------
    async def handle(self, reader, writer):
        try:
            hello = _loads(await reader.readline())
        except Exception:
            writer.close()
            return
        sub = Sub(writer, hello.get("tickers") or [], hello.get("session") or "?")
        self.subs.add(sub)
        log(f"[hub] + {sub.name} ({len(sub.tickers)} tickers); "
            f"{len(self.subs)} subscriber(s), {len(self.union())} tickers")
        try:
            await reader.read()                  # returns at EOF: subscriber gone
        except Exception:
            pass
        finally:
            sub.dead = True
            self.subs.discard(sub)
            self.last_sub = time.monotonic()
            writer.close()
            log(f"[hub] - {sub.name}; {len(self.subs)} subscriber(s), {len(self.union())} tickers")

    def fan_out(self, msg_for):
        """msg_for(sub) -> dict or None"""
        for s in list(self.subs):
            m = msg_for(s)
            if m is not None:
                s.send(_dumpb(m))

    # -- book polling -----------------------------------------------------------
    def interval(self, n_chunks):
        rtt = statistics.median(self.rtts) if self.rtts else 1.0
        base = max(rtt / self.args.pollers, self.args.min_interval, n_chunks / self.args.max_rps)
        return base * self.penalty

    async def poll_chunk(self, poll_id, tickers):
        t_send = now_ms()
        t0 = time.monotonic()
        status, data, err = None, None, None
        try:
            async with self.sess.get(API + "/markets/orderbooks",
                                     params=[("tickers", t) for t in tickers]) as r:
                body = await r.read()
                status = r.status
                data = _loads(body) if status == 200 else None
        except Exception as e:
            status, data = None, None
            err = f"{type(e).__name__}: {e}"[:200]
        t_recv = now_ms()
        if status == 200 and data is not None:
            self.n_ok += 1
            self.rtts.append(time.monotonic() - t0)
        elif status == 429:
            self.n_429 += 1
            self.last_429 = time.monotonic()
            old = self.penalty
            self.penalty = min(self.penalty * 2, 32.0)
            if self.penalty != old:
                log(f"[hub] 429 rate-limited; interval x{self.penalty:g}")
        else:
            self.n_err += 1
        obs = (data or {}).get("orderbooks", []) if status == 200 else []
        chunk = set(tickers)

        def msg(s):
            mine = s.tickers & chunk
            if not mine:
                return None
            m = {"type": "reply", "poll_id": poll_id, "t_send": t_send, "t_recv": t_recv,
                 "status": status, "n": len(mine),
                 "orderbooks": [ob for ob in obs if ob.get("ticker") in s.tickers]}
            if err:
                m["err"] = err
            return m
        self.fan_out(msg)

    async def poll_once(self, poll_id, chunks):
        self.inflight += 1
        try:
            await asyncio.gather(*(self.poll_chunk(poll_id, c) for c in chunks))
        finally:
            self.inflight -= 1

    async def book_loop(self):
        tasks = set()
        nxt = time.monotonic()
        while not self.stop.is_set():
            if self.penalty > 1 and time.monotonic() - self.last_429 > 30:
                self.penalty = max(1.0, self.penalty / 2)
                self.last_429 = time.monotonic()
                log(f"[hub] easing rate-limit back-off to x{self.penalty:g}")
            chunks = self.chunks()
            if chunks and self.inflight < self.args.pollers:
                self.poll_id += 1
                self.n_polls += 1
                t = asyncio.create_task(self.poll_once(self.poll_id, chunks))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
            nxt = max(nxt + self.interval(max(len(chunks), 1)), time.monotonic())
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))

    # -- trades -----------------------------------------------------------------
    async def trade_loop(self):
        seen = {}
        watermark = time.time() - 10
        while not self.stop.is_set():
            if not self.subs:
                watermark = time.time() - 10
                await asyncio.sleep(self.args.trade_interval)
                continue
            t_seen = now_ms()
            cursor, pages, newest = None, 0, watermark
            try:
                while pages < self.args.trade_max_pages:
                    params = {"limit": 1000, "min_ts": int(watermark - 5)}
                    if cursor:
                        params["cursor"] = cursor
                    async with self.sess.get(API + "/markets/trades", params=params) as r:
                        body = await r.read()
                        status = r.status
                    pages += 1
                    if status == 429:
                        self.n_429 += 1
                        self.last_429 = time.monotonic()
                        break
                    if status != 200:
                        break
                    data = _loads(body)
                    trades = data.get("trades") or []
                    for tr in trades:
                        tid = tr.get("trade_id")
                        if tid in seen:
                            continue
                        ts = _iso_to_s(tr.get("created_time"))
                        seen[tid] = ts
                        newest = max(newest, ts)
                        tk = tr.get("ticker")
                        self.fan_out(lambda s: {"type": "trade", "t_seen": t_seen, "trade": tr}
                                     if tk in s.tickers else None)
                    cursor = data.get("cursor")
                    if not trades or not cursor:
                        break
                if pages >= self.args.trade_max_pages and cursor:
                    log(f"[hub] trade feed: hit {pages} pages in one cycle; some trades may be missing")
            except Exception as e:
                log(f"[hub] trade poll failed ({type(e).__name__}: {e})")
            watermark = newest
            cut = watermark - 120
            for tid in [k for k, v in seen.items() if v < cut]:
                del seen[tid]
            await asyncio.sleep(self.args.trade_interval)

    # -- housekeeping -------------------------------------------------------------
    async def stats_loop(self):
        t0 = time.monotonic()
        while not self.stop.is_set():
            await asyncio.sleep(self.args.stats_interval)
            el = time.monotonic() - t0
            rt = sorted(self.rtts)
            p50 = rt[len(rt) // 2] * 1000 if rt else float("nan")
            log(f"[hub] {el:6.0f}s  subs {len(self.subs)}  tickers {len(self.union())}  "
                f"polls {self.n_polls:,} ok {self.n_ok:,} 429 {self.n_429} err {self.n_err}  "
                f"rtt p50 {p50:.0f} ms  interval {self.interval(max(len(self.chunks()), 1))*1000:.0f} ms")
            if not self.subs and time.monotonic() - self.last_sub > self.args.idle_exit:
                log(f"[hub] no subscribers for {self.args.idle_exit:.0f}s; exiting")
                self.stop.set()

    async def run(self):
        conn = aiohttp.TCPConnector(limit=self.args.pollers * 4 + 8, ttl_dns_cache=300, keepalive_timeout=60)
        timeout = aiohttp.ClientTimeout(total=self.args.request_timeout)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout, headers=HEADERS) as sess:
            self.sess = sess
            if os.path.exists(SOCK):
                os.unlink(SOCK)
            server = await asyncio.start_unix_server(self.handle, path=SOCK, limit=1 << 20)
            log(f"[hub] listening on {SOCK}")
            tasks = [asyncio.create_task(c) for c in (self.book_loop(), self.trade_loop(), self.stats_loop())]
            await self.stop.wait()
            server.close()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if os.path.exists(SOCK):
                os.unlink(SOCK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pollers", type=int, default=4)
    ap.add_argument("--max-rps", type=float, default=12.0,
                    help="book requests per second for ALL games together")
    ap.add_argument("--min-interval", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--request-timeout", type=float, default=5.0)
    ap.add_argument("--trade-interval", type=float, default=1.0)
    ap.add_argument("--trade-max-pages", type=int, default=10)
    ap.add_argument("--stats-interval", type=float, default=60.0)
    ap.add_argument("--idle-exit", type=float, default=900.0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another kalshi_hub is running; exiting")
        return
    asyncio.run(Hub(args).run())


if __name__ == "__main__":
    main()
