"""
Kalshi order book recorder -- REST polling, no API key.

Kalshi's websocket rejects unauthenticated connections (HTTP 401), but its
public REST endpoints serve the same order books to anyone. This recorder
turns those into a time series that can be lined up against the Polymarket
recording from live_recorder.py.

How the polling works:

  * ONE batch request (GET /markets/orderbooks?tickers=A&tickers=B...)
    returns every tracked book. 72 EPL markets came back in ~1 s from Hong
    Kong, so the request rate does not grow with the number of matches.
  * Several requests are kept in flight at once, launched at evenly spaced
    offsets ("phase-offset pollers"). With a round trip R and P pollers a
    fresh snapshot lands every ~R/P. The interval adapts to the measured
    round trip and is capped by --max-rps.
  * A 429 doubles the interval; it decays back after a quiet spell. Every
    change is logged, so a rate-limited stretch is visible in the data.
  * Responses are NOT cached by CloudFront (60 rapid requests were all
    "Miss"), so every poll is a fresh read.

Timing caveats, which the analysis has to respect:

  * Book responses carry no exchange timestamp. Each snapshot gets t_send
    and t_recv (local ms); the true state is somewhere in between.
  * A later request can answer first (~17% of replies). Books are diffed
    in ARRIVAL order, which was measured to track the server's state far
    better than send order (see BookState); such replies are flagged
    "overtook" so the analysis can check both orderings.
  * Trades DO carry Kalshi's own created_time (microseconds), and are polled
    from the global trade feed, so they anchor the Kalshi timeline.

Outputs (append-only, under data/live/kalshi/, one set per --session):
    kbooks_TAG.jsonl   one line per CHANGED book (first sighting = "seed")
    ktrades_TAG.jsonl  one line per trade on a tracked ticker
    kpolls_TAG.jsonl   one line per book request + status/rate-limit events
    ktop_TAG.csv       top of book, one row whenever it changes

Usage:
    python kalshi_recorder.py --events KXMLBGAME-26OCT172000TBCLE --duration 60
    python kalshi_recorder.py --events KXEPLGAME-26OCT18BOUSUN ... \\
        --session 2026-10-18_2100 --stop-when-closed 10
    python kalshi_recorder.py --events ... --game-pks 849817 --stop-when-final 20
"""

import argparse
import asyncio
import csv
import json
import os
import queue
import statistics
import threading
import time
from collections import deque

import aiohttp

from live_recorder import final_watch, log, now_ms

try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "live", "kalshi")
API = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"User-Agent": "Mozilla/5.0 (orderbook-research)", "Accept": "application/json"}

# A market in any of these states will never trade again.
FINISHED = {"closed", "determined", "settled", "finalized"}

TOP_HEADER = ["poll_id", "t_send", "t_recv", "event", "ticker", "outcome",
              "best_bid", "best_bid_size", "best_ask", "best_ask_size", "midpoint", "spread"]


# ---------------------------------------------------------------------------
# Disk writer: same idea as live_recorder.Writer -- a daemon thread, so no
# write or fsync ever runs on the event loop -- but with four outputs.
# ---------------------------------------------------------------------------

class Writer(threading.Thread):
    def __init__(self, session=None, fsync_interval=30.0, batch=2000):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.fsync_interval = fsync_interval
        self.batch = batch
        self._stop_evt = threading.Event()
        self.written = 0
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = f"_{session}" if session else ""
        self.paths = {
            "book": os.path.join(OUT_DIR, f"kbooks{tag}.jsonl"),
            "trade": os.path.join(OUT_DIR, f"ktrades{tag}.jsonl"),
            "poll": os.path.join(OUT_DIR, f"kpolls{tag}.jsonl"),
            "top": os.path.join(OUT_DIR, f"ktop{tag}.csv"),
        }

    def put(self, kind, payload):
        self.q.put_nowait((kind, payload))

    def run(self):
        new_top = not (os.path.exists(self.paths["top"]) and os.path.getsize(self.paths["top"]) > 0)
        fh = {k: open(p, "a", encoding="utf-8", newline="" if k == "top" else None)
              for k, p in self.paths.items()}
        tw = csv.writer(fh["top"])
        if new_top:
            tw.writerow(TOP_HEADER)
        last_sync = time.monotonic()
        try:
            while not (self._stop_evt.is_set() and self.q.empty()):
                drained = 0
                try:
                    kind, payload = self.q.get(timeout=0.5)
                except queue.Empty:
                    kind = None
                while kind is not None:
                    if kind == "top":
                        tw.writerow(payload)
                    else:
                        fh[kind].write(payload)
                        fh[kind].write("\n")
                    self.written += 1
                    drained += 1
                    if drained >= self.batch:
                        break
                    try:
                        kind, payload = self.q.get_nowait()
                    except queue.Empty:
                        break
                if drained:
                    for f in fh.values():
                        f.flush()
                now = time.monotonic()
                if now - last_sync >= self.fsync_interval:
                    for f in fh.values():
                        os.fsync(f.fileno())
                    last_sync = now
        finally:
            for f in fh.values():
                f.flush()
                os.fsync(f.fileno())
                f.close()

    def stop(self):
        self._stop_evt.set()


def dumps(obj):
    return json.dumps(obj, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Book conversion
# ---------------------------------------------------------------------------

def parse_book(ob):
    """Kalshi returns two lists of RESTING BIDS: yes_dollars and no_dollars,
    each [[price, size], ...] ascending. A NO bid at p is a YES ask at 1-p,
    so the book is expressed in YES terms, the same way the Polymarket
    recorder stores a token's book: bids high->low, asks low->high."""
    bids = sorted(((float(p), float(s)) for p, s in ob.get("yes_dollars") or [] if float(s) > 0),
                  key=lambda x: -x[0])
    asks = sorted(((round(1 - float(p), 4), float(s)) for p, s in ob.get("no_dollars") or [] if float(s) > 0),
                  key=lambda x: x[0])
    return tuple(bids), tuple(asks)


class BookState:
    """Per-ticker last-applied state, applied in ARRIVAL order.

    Arrival order, not send order, is what tracks the server's state. Measured
    on two live matches (2026-10-18, 1,251 responses, 17% overtaken by a
    later-sent request): ordering by send time produced 168 one-poll
    A->B->A flickers, ordering by arrival only 22. A request that is slow to
    come back was evidently served late too, so its book is the NEWER one --
    discarding it as "stale" threw away the freshest data."""

    def __init__(self):
        self.book = {}        # ticker -> (bids, asks)
        self.top = {}         # ticker -> (bb, bbs, ba, bas)

    def apply(self, ticker, bids, asks):
        """Returns 'seed', 'change' or None (unchanged)."""
        prev = self.book.get(ticker)
        if prev == (bids, asks):
            return None
        self.book[ticker] = (bids, asks)
        return "seed" if prev is None else "change"


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class KalshiRecorder:
    def __init__(self, args, markets, writer, stop_event):
        self.args = args
        self.markets = markets              # ticker -> {"event", "outcome"}
        self.tickers = sorted(markets)
        bs = args.batch_size
        self.chunks = [self.tickers[i:i + bs] for i in range(0, len(self.tickers), bs)]
        self.writer = writer
        self.stop_event = stop_event
        self.state = BookState()
        self.seq = 0
        self.poll_id = 0
        self.inflight = 0
        self.rtts = deque(maxlen=200)       # seconds, successful book requests
        self.recv_times = deque(maxlen=400) # t_recv of successful responses, ms
        self.penalty = 1.0                  # 429 back-off multiplier
        self.last_429 = 0.0
        self.n_polls = self.n_ok = self.n_429 = self.n_err = self.n_overtook = 0
        self.max_answered = 0
        self.n_changes = self.n_trades = 0
        self.status = {}                    # ticker -> market status
        self.sess = None

    def next_seq(self):
        self.seq += 1
        return self.seq

    # -- HTTP ---------------------------------------------------------------
    async def get(self, path, params):
        async with self.sess.get(API + path, params=params) as r:
            body = await r.read()
            return r.status, (_loads(body) if r.status == 200 else None)

    # -- book polling ---------------------------------------------------------
    def interval(self):
        """Seconds between launches. Aim for one snapshot every RTT/pollers,
        never above --max-rps, and slower while rate-limited."""
        rtt = statistics.median(self.rtts) if self.rtts else 1.0
        base = max(rtt / self.args.pollers, self.args.min_interval,
                   len(self.chunks) / self.args.max_rps)
        return base * self.penalty

    async def poll_chunk(self, poll_id, tickers):
        t_send = now_ms()
        t0 = time.monotonic()
        rec = {"type": "poll", "poll_id": poll_id, "t_send": t_send}
        try:
            status, data = await self.get("/markets/orderbooks", [("tickers", t) for t in tickers])
        except Exception as e:
            self.n_err += 1
            rec.update(t_recv=now_ms(), status=None, err=f"{type(e).__name__}: {e}"[:200])
            self.writer.put("poll", dumps(rec))
            return
        t_recv = now_ms()
        rec.update(t_recv=t_recv, status=status)
        if status == 429:
            self.n_429 += 1
            self.last_429 = time.monotonic()
            old = self.penalty
            self.penalty = min(self.penalty * 2, 32.0)
            if self.penalty != old:
                log(f"[kalshi] 429 rate-limited; interval x{self.penalty:g} "
                    f"(now {self.interval()*1000:.0f} ms)")
            self.writer.put("poll", dumps(rec))
            return
        if status != 200 or data is None:
            self.n_err += 1
            self.writer.put("poll", dumps(rec))
            return

        self.rtts.append(time.monotonic() - t0)
        self.apply_reply(poll_id, t_send, t_recv, data.get("orderbooks", []), len(tickers))

    def apply_reply(self, poll_id, t_send, t_recv, orderbooks, n):
        """A successful reply, in arrival order -- from our own poll or the hub."""
        self.n_ok += 1
        self.recv_times.append(t_recv)
        # overtook: a later-sent poll already answered. Kept and applied (see
        # BookState), but flagged so the analysis can test either ordering.
        overtook = poll_id < self.max_answered
        self.max_answered = max(self.max_answered, poll_id)
        if overtook:
            self.n_overtook += 1
        changed = 0
        for ob in orderbooks:
            ticker = ob.get("ticker")
            meta = self.markets.get(ticker)
            if meta is None:
                continue
            bids, asks = parse_book(ob.get("orderbook_fp") or {})
            kind = self.state.apply(ticker, bids, asks)
            if kind is None:
                continue
            changed += 1
            self.emit(ticker, meta, poll_id, t_send, t_recv, kind, bids, asks, overtook)
        self.n_changes += changed
        self.writer.put("poll", dumps({"type": "poll", "poll_id": poll_id, "t_send": t_send,
                                       "t_recv": t_recv, "status": 200, "n": n,
                                       "changed": changed, "overtook": overtook}))

    def emit(self, ticker, meta, poll_id, t_send, t_recv, kind, bids, asks, overtook):
        self.writer.put("book", dumps({
            "seq": self.next_seq(), "poll_id": poll_id,
            "t_send": t_send, "t_recv": t_recv, "et": kind, "overtook": overtook,
            "event": meta["event"], "ticker": ticker, "outcome": meta["outcome"],
            "bids": bids, "asks": asks,
        }))
        bb, bbs = bids[0] if bids else (None, None)
        ba, bas = asks[0] if asks else (None, None)
        top = (bb, bbs, ba, bas)
        if self.state.top.get(ticker) != top:
            self.state.top[ticker] = top
            both = bb is not None and ba is not None
            self.writer.put("top", [
                poll_id, t_send, t_recv, meta["event"], ticker, meta["outcome"],
                bb, bbs, ba, bas,
                round((bb + ba) / 2, 5) if both else None,
                round(ba - bb, 5) if both else None,
            ])

    async def poll_once(self, poll_id):
        self.inflight += 1
        try:
            await asyncio.gather(*(self.poll_chunk(poll_id, c) for c in self.chunks))
        finally:
            self.inflight -= 1

    async def book_loop(self):
        """Launch a poll every interval(); never more than --pollers at once,
        so a latency spike slows the cadence instead of piling up requests."""
        tasks = set()
        nxt = time.monotonic()
        while not self.stop_event.is_set():
            # 429 penalty decays after 30 s without one
            if self.penalty > 1 and time.monotonic() - self.last_429 > 30:
                self.penalty = max(1.0, self.penalty / 2)
                self.last_429 = time.monotonic()
                log(f"[kalshi] easing rate-limit back-off to x{self.penalty:g}")
            if self.inflight < self.args.pollers:
                self.poll_id += 1
                self.n_polls += 1
                t = asyncio.create_task(self.poll_once(self.poll_id))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
            nxt = max(nxt + self.interval(), time.monotonic())
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=nxt - time.monotonic())
            except asyncio.TimeoutError:
                pass
        if tasks:
            await asyncio.wait(tasks, timeout=10)

    # -- trades ---------------------------------------------------------------
    async def trade_loop(self):
        """Poll Kalshi's global trade feed (filtering by ticker is one request
        per market; the global feed is ~60 trades/s, i.e. one or two pages a
        second for everything). min_ts trails the newest trade seen so a
        slow cycle cannot leave a gap; trade_id dedupes the overlap."""
        seen = {}                     # trade_id -> created ts (s)
        watermark = time.time() - 10
        while not self.stop_event.is_set():
            t_seen = now_ms()
            cursor, pages, newest = None, 0, watermark
            try:
                while pages < self.args.trade_max_pages:
                    params = {"limit": 1000, "min_ts": int(watermark - 5)}
                    if cursor:
                        params["cursor"] = cursor
                    status, data = await self.get("/markets/trades", params)
                    pages += 1
                    if status == 429:
                        self.n_429 += 1
                        self.last_429 = time.monotonic()
                        break
                    if status != 200 or not data:
                        break
                    trades = data.get("trades") or []
                    for tr in trades:
                        tid = tr.get("trade_id")
                        if tid in seen:
                            continue
                        ts = _iso_to_s(tr.get("created_time"))
                        seen[tid] = ts
                        newest = max(newest, ts)
                        self.record_trade(tr, ts, t_seen)
                    cursor = data.get("cursor")
                    if not trades or not cursor:
                        break
                if pages >= self.args.trade_max_pages and cursor:
                    log(f"[kalshi] trade feed: hit {pages} pages in one cycle; "
                        f"some trades may be missing")
            except Exception as e:
                log(f"[kalshi] trade poll failed ({type(e).__name__}: {e})")
            watermark = newest
            cut = watermark - 120
            for tid in [k for k, v in seen.items() if v < cut]:
                del seen[tid]
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.args.trade_interval)
            except asyncio.TimeoutError:
                pass

    def record_trade(self, tr, ts, t_seen):
        meta = self.markets.get(tr.get("ticker"))
        if meta is None:
            return
        self.n_trades += 1
        self.writer.put("trade", dumps({
            "trade_id": tr.get("trade_id"), "ts_exchange": tr.get("created_time"),
            "ts_ms": int(ts * 1000), "t_seen": t_seen,
            "event": meta["event"], "ticker": tr["ticker"],
            "outcome": meta["outcome"],
            "yes_price": float(tr.get("yes_price_dollars") or 0),
            "size": float(tr.get("count_fp") or 0),
            "taker_side": tr.get("taker_side"),
        }))

    # -- shared hub (kalshi_hub.py) -----------------------------------------------
    async def hub_loop(self):
        """Receive books and trades from the shared hub instead of polling.
        Starts the hub if none is running; reconnects if it goes away (the
        lost stretch shows as a gap in kpolls)."""
        from kalshi_hub import SOCK
        spawned = 0.0
        while not self.stop_event.is_set():
            try:
                reader, w = await asyncio.open_unix_connection(SOCK, limit=1 << 26)
            except OSError:
                if time.monotonic() - spawned > 20:
                    spawned = time.monotonic()
                    spawn_hub()
                    log("[kalshi] hub not running; started it")
                await asyncio.sleep(0.5)
                continue
            w.write((dumps({"tickers": self.tickers, "session": self.args.session}) + "\n").encode())
            await w.drain()
            log(f"[kalshi] subscribed to hub ({len(self.tickers)} tickers)")
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    m = _loads(line)
                    if m["type"] == "reply":
                        self.n_polls += 1
                        if m["status"] == 200:
                            self.rtts.append((m["t_recv"] - m["t_send"]) / 1000)
                            self.apply_reply(m["poll_id"], m["t_send"], m["t_recv"], m["orderbooks"], m["n"])
                        else:
                            if m["status"] == 429:
                                self.n_429 += 1
                            else:
                                self.n_err += 1
                            rec = {"type": "poll", "poll_id": m["poll_id"], "t_send": m["t_send"],
                                   "t_recv": m["t_recv"], "status": m["status"]}
                            if m.get("err"):
                                rec["err"] = m["err"]
                            self.writer.put("poll", dumps(rec))
                    elif m["type"] == "trade":
                        tr = m["trade"]
                        self.record_trade(tr, _iso_to_s(tr.get("created_time")), m["t_seen"])
            except (ConnectionError, asyncio.IncompleteReadError, ValueError) as e:
                log(f"[kalshi] hub connection error: {e}")
            finally:
                w.close()
            if not self.stop_event.is_set():
                log("[kalshi] hub connection lost; reconnecting")
                await asyncio.sleep(0.5)

    # -- market status ----------------------------------------------------------
    async def status_loop(self):
        """Once a minute, read each event's market status. Logs every change
        (e.g. active -> closed at the final whistle) and, with
        --stop-when-closed, stops MIN minutes after every market finished."""
        events = sorted({m["event"] for m in self.markets.values()})
        done_since = None
        while not self.stop_event.is_set():
            for ev in events:
                try:
                    status, data = await self.get("/markets", {"event_ticker": ev, "limit": 200})
                except Exception as e:
                    log(f"[kalshi] status poll failed for {ev}: {e}")
                    continue
                if status != 200 or not data:
                    continue
                for m in data.get("markets", []):
                    tk, st = m.get("ticker"), m.get("status")
                    if tk in self.markets and self.status.get(tk) != st:
                        if tk in self.status:
                            log(f"[kalshi] {tk}: {self.status[tk]} -> {st}")
                        self.status[tk] = st
                        self.writer.put("poll", dumps({"type": "status", "t": now_ms(),
                                                       "ticker": tk, "status": st,
                                                       "result": m.get("result")}))
            if self.args.stop_when_closed and self.status and len(self.status) == len(self.markets):
                if all(s in FINISHED for s in self.status.values()):
                    if done_since is None:
                        done_since = time.monotonic()
                        log(f"[kalshi] every market finished; stopping in "
                            f"{self.args.stop_when_closed:.0f} min")
                    elif time.monotonic() - done_since >= self.args.stop_when_closed * 60:
                        log("[kalshi] close grace elapsed; stopping recorder")
                        self.stop_event.set()
                        return
                else:
                    done_since = None
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=60)
            except asyncio.TimeoutError:
                pass

    # -- stats ------------------------------------------------------------------
    async def stats_loop(self, t0):
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.args.stats_interval)
                return
            except asyncio.TimeoutError:
                pass
            el = time.monotonic() - t0
            rt = sorted(self.rtts)
            p50 = rt[len(rt) // 2] * 1000 if rt else float("nan")
            p95 = rt[int(len(rt) * 0.95)] * 1000 if rt else float("nan")
            rv = list(self.recv_times)
            gaps = sorted(b - a for a, b in zip(rv, rv[1:]) if b >= a)
            gap = gaps[len(gaps) // 2] if gaps else float("nan")
            log(f"[stats] {el:6.0f}s  polls {self.n_polls:>7,} ({self.n_polls/max(el,1):4.1f}/s)  "
                f"rtt p50 {p50:4.0f} p95 {p95:4.0f} ms  gap~{gap:4.0f} ms  "
                f"429s {self.n_429}  err {self.n_err}  overtook {self.n_overtook}  "
                f"changes {self.n_changes:>7,}  trades {self.n_trades:>6,}  "
                f"queued {self.writer.q.qsize():>5,}")

    # -- run --------------------------------------------------------------------
    async def run(self, extra_tasks=()):
        conn = aiohttp.TCPConnector(limit=self.args.pollers * len(self.chunks) + 8,
                                    ttl_dns_cache=300, keepalive_timeout=60)
        timeout = aiohttp.ClientTimeout(total=self.args.request_timeout)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout, headers=HEADERS) as sess:
            self.sess = sess
            t0 = time.monotonic()
            tasks = [asyncio.create_task(c) for c in (self.status_loop(), self.stats_loop(t0))]
            if self.args.direct:
                tasks.append(asyncio.create_task(self.book_loop()))
                if not self.args.no_trades:
                    tasks.append(asyncio.create_task(self.trade_loop()))
                hub = None
            else:
                hub = asyncio.create_task(self.hub_loop())
            tasks += [asyncio.create_task(c) for c in extra_tasks]
            await self.stop_event.wait()
            if hub:
                hub.cancel()
                await asyncio.gather(hub, return_exceptions=True)
            await asyncio.gather(*tasks, return_exceptions=True)
            return time.monotonic() - t0


def spawn_hub():
    """Start kalshi_hub.py detached (its lock keeps it to one instance)."""
    import subprocess
    import sys
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    out = open(os.path.join(BASE_DIR, "logs", "kalshi_hub.log"), "a")
    subprocess.Popen([sys.executable, "-u", os.path.join(BASE_DIR, "kalshi_hub.py")],
                     cwd=BASE_DIR, stdout=out, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)


def _iso_to_s(s):
    from datetime import datetime
    if not s:
        return time.time()
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]:0<6}"
    return datetime.fromisoformat(s + "+00:00").timestamp()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(event_tickers):
    """event ticker -> its markets, read live from GET /markets."""
    import requests
    markets = {}
    for ev in event_tickers:
        try:
            r = requests.get(f"{API}/markets", params={"event_ticker": ev, "limit": 200},
                             headers=HEADERS, timeout=20)
            r.raise_for_status()
            ms = r.json().get("markets", [])
        except Exception as e:
            log(f"[discovery] {ev}: fetch failed: {e}")
            continue
        if not ms:
            log(f"[discovery] {ev}: no markets (wrong ticker?)")
        for m in ms:
            if m.get("status") in FINISHED:
                continue
            markets[m["ticker"]] = {"event": ev, "outcome": m.get("yes_sub_title") or m["ticker"]}
    return markets


async def main_async(args):
    markets = discover(args.events)
    if not markets:
        raise SystemExit("No open Kalshi markets found; nothing to record.")
    writer = Writer(session=args.session, fsync_interval=args.fsync_interval)
    writer.start()
    stop_event = asyncio.Event()
    rec = KalshiRecorder(args, markets, writer, stop_event)
    log(f"Recording {len(markets)} Kalshi markets across {len(args.events)} event(s), "
        + (f"{len(rec.chunks)} batch request(s) per poll, {args.pollers} pollers" if args.direct
           else "via the shared kalshi_hub.py"))
    for p in writer.paths.values():
        log(f"  -> {p}")

    extra = []
    if args.stop_when_final and args.game_pks:
        extra.append(final_watch(stop_event, args.game_pks, args.stop_when_final))

    async def guard():
        if args.duration:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.duration)
            except asyncio.TimeoutError:
                log(f"[done] --duration {args.duration}s reached; stopping")
                stop_event.set()
    extra.append(guard())

    el = await rec.run(extra)
    writer.stop()
    writer.join(timeout=30)
    print()
    log("=== summary ===")
    log(f"  {rec.n_polls:,} polls in {el:.0f}s ({rec.n_polls/max(el,1):.1f}/s), "
        f"{rec.n_ok:,} ok, {rec.n_429} rate-limited, {rec.n_err} errors, "
        f"{rec.n_overtook} arrived after a later-sent poll")
    log(f"  {rec.n_changes:,} book changes, {rec.n_trades:,} trades, "
        f"{writer.written:,} records written, {writer.q.qsize()} still queued")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True, help="Kalshi event tickers")
    ap.add_argument("--session", default=None, metavar="TAG",
                    help="write kbooks_TAG.jsonl etc.; one tag per session")
    ap.add_argument("--direct", action="store_true",
                    help="poll Kalshi from this process instead of the shared kalshi_hub.py "
                         "(only safe when it is the only recorder on this IP)")
    ap.add_argument("--pollers", type=int, default=4,
                    help="book requests kept in flight; snapshot gap ~ RTT / pollers")
    ap.add_argument("--max-rps", type=float, default=25.0,
                    help="hard cap on book requests per second")
    ap.add_argument("--min-interval", type=float, default=0.05,
                    help="never launch polls closer than this (s); below ~50 ms "
                         "round-trip jitter swamps the extra resolution")
    ap.add_argument("--batch-size", type=int, default=60,
                    help="tickers per batch request (72 verified working)")
    ap.add_argument("--request-timeout", type=float, default=5.0)
    ap.add_argument("--no-trades", action="store_true")
    ap.add_argument("--trade-interval", type=float, default=1.0)
    ap.add_argument("--trade-max-pages", type=int, default=10)
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--stop-when-closed", type=float, default=0, metavar="MIN",
                    help="stop MIN minutes after Kalshi reports every market closed")
    ap.add_argument("--game-pks", type=int, nargs="*", default=None,
                    help="MLB gamePks for --stop-when-final")
    ap.add_argument("--stop-when-final", type=float, default=0, metavar="MIN",
                    help="stop MIN minutes after the MLB API reports every gamePk Final")
    ap.add_argument("--stats-interval", type=float, default=30.0)
    ap.add_argument("--fsync-interval", type=float, default=30.0)
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nstopped by user.")


if __name__ == "__main__":
    main()
