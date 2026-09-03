"""
Live Polymarket order book recorder -- one independent websocket per match.

Replaces multi_recorder.py, which lost ~94% of the feed. Three defects were
measured in that recording (see README "Why the first recording failed"):

  1. It called flush()+fsync() once PER ROW on the asyncio event loop. Its
     busiest second produced 70,926 depth rows, and at ~0.36 ms per fsync
     that needs ~25 s of disk time to write 1 s of data. The loop froze past
     the 20 s keepalive deadline, so `websockets` tore both connections down
     with "1011 keepalive ping timeout" every few minutes -- and every
     price_change sent during the outage was gone for good.
  2. It stamped rows with wall-clock time at the moment it got around to
     writing them, ignoring the `timestamp` field the exchange puts on every
     message. So even surviving rows carry the time the backlog drained, not
     when the event happened.
  3. It spread each match's assets across shared connections (100 assets per
     socket, chunked by discovery order), so one drop damaged many games.

What this recorder does differently:

  * ONE websocket per match. 15 matches = 15 independent tasks; a drop or a
    reconnect affects that match only, and matches record concurrently.
  * The exchange's `timestamp` is the event time. Local receive time is kept
    alongside it, so clock skew and delivery lag stay measurable.
  * NO disk I/O on the event loop. Rows go to a plain queue.Queue and a
    daemon thread batches them to disk. fsync happens at most every
    --fsync-interval seconds, in that thread, never inline.
  * ONE record per book snapshot, holding the whole book, instead of one row
    per price level. That is ~90x fewer records, and it removes the
    snapshot-boundary ambiguity that made the old one-row-per-level CSV
    genuinely hard to parse (snapshots colliding inside a single millisecond
    had to be split by file order).
  * Unmatched message shapes are logged, not silently dropped.
  * Prints per-match health every --stats-interval seconds (messages,
    snapshots, reconnects, seconds since last message) so a stalled feed is
    obvious while it is still running.

Outputs (both append-only, safe to resume):
    data/live/books.jsonl            one JSON object per book snapshot
    data/live/top_of_book.csv        one row per snapshot, top of book only

Usage:
    python live_recorder.py
    python live_recorder.py --duration 120          # short smoke test
    python live_recorder.py --slugs mlb-tb-det-2026-08-26
    python live_recorder.py --live-discover 40      # ignore matches.py, grab
                                                    # the 40 most active open
                                                    # markets (for testing)
"""

import argparse
import asyncio
import csv
import json
import os
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests
import websockets

import discovery
import matches as matches_module

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "live")

CLOB_REST_BOOK_URL = "https://clob.polymarket.com/book"
CLOB_WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

TOP_HEADER = [
    "seq", "ts_exchange_ms", "ts_recv_ms", "event_slug", "market_type", "market_question",
    "line", "outcome", "condition_id", "asset_id",
    "best_bid", "best_bid_size", "best_ask", "best_ask_size", "midpoint", "spread",
]


def now_ms():
    return int(time.time() * 1000)


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def log(msg):
    print(f"[{iso(now_ms())}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Disk writer: a daemon thread, so no write ever runs on the event loop.
# ---------------------------------------------------------------------------

class Writer(threading.Thread):
    def __init__(self, fsync_interval=30.0, batch=2000, session=None):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.fsync_interval = fsync_interval
        self.batch = batch
        self._stop = threading.Event()
        self.written = 0
        self.dropped = 0
        os.makedirs(OUT_DIR, exist_ok=True)
        # Files are opened append-only, so without a per-session name a second
        # night's recording lands on top of the first one's multi-GB file --
        # slower to split, awkward to archive, and impossible to hand over as
        # "one slate". `session` keeps each run in its own pair of files.
        tag = f"_{session}" if session else ""
        self.books_path = os.path.join(OUT_DIR, f"books{tag}.jsonl")
        self.top_path = os.path.join(OUT_DIR, f"top_of_book{tag}.csv")

    def put(self, kind, payload):
        # never blocks the event loop; the queue is unbounded on purpose so a
        # burst is buffered in RAM rather than applying backpressure to the
        # websocket (which is what strangled the previous recorder).
        self.q.put_nowait((kind, payload))

    def run(self):
        new_top = not (os.path.exists(self.top_path) and os.path.getsize(self.top_path) > 0)
        bf = open(self.books_path, "a", encoding="utf-8")
        tf = open(self.top_path, "a", newline="", encoding="utf-8")
        tw = csv.writer(tf)
        if new_top:
            tw.writerow(TOP_HEADER)
        last_sync = time.monotonic()
        try:
            while not (self._stop.is_set() and self.q.empty()):
                drained = 0
                try:
                    kind, payload = self.q.get(timeout=0.5)
                except queue.Empty:
                    kind = None
                while kind is not None:
                    if kind == "book":
                        bf.write(payload)
                        bf.write("\n")
                    else:
                        tw.writerow(payload)
                    self.written += 1
                    drained += 1
                    if drained >= self.batch:
                        break
                    try:
                        kind, payload = self.q.get_nowait()
                    except queue.Empty:
                        break
                if drained:
                    bf.flush()
                    tf.flush()
                now = time.monotonic()
                if now - last_sync >= self.fsync_interval:
                    # durability checkpoint, off the event loop and rare
                    os.fsync(bf.fileno())
                    os.fsync(tf.fileno())
                    last_sync = now
        finally:
            bf.flush(); tf.flush()
            os.fsync(bf.fileno()); os.fsync(tf.fileno())
            bf.close(); tf.close()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Book state
# ---------------------------------------------------------------------------

def new_book():
    return {"bids": {}, "asks": {}}


def apply_snapshot(book, msg):
    book["bids"] = {float(l["price"]): float(l["size"]) for l in msg.get("bids", []) if float(l["size"]) > 0}
    book["asks"] = {float(l["price"]): float(l["size"]) for l in msg.get("asks", []) if float(l["size"]) > 0}


def apply_change(book, price, side, size):
    if side in ("BUY", "BID"):
        levels = book["bids"]
    elif side in ("SELL", "ASK"):
        levels = book["asks"]
    else:
        return False
    if size <= 0:
        levels.pop(price, None)
    else:
        levels[price] = size
    return True


def sorted_book(book):
    bids = sorted(book["bids"].items(), key=lambda kv: -kv[0])
    asks = sorted(book["asks"].items(), key=lambda kv: kv[0])
    return bids, asks


class Seq:
    """Process-wide monotonic counter stamped on every record.

    The exchange's `timestamp` is only millisecond-resolution and it stamps
    bursts of messages with the same value -- 56 separate messages sharing
    one millisecond was observed live. Recovering their order from file order
    alone is exactly what made the old one-row-per-level CSV painful to
    parse. An explicit seq makes replay order unambiguous forever, no matter
    how the records are later sorted, split or merged.
    """

    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return self.n


class MatchRecorder:
    """One match, one websocket, its own reconnect loop and its own state."""

    def __init__(self, slug, assets, writer, stop_event, seq):
        self.seq = seq
        self.slug = slug
        self.assets = assets          # asset_id -> meta
        self.writer = writer
        self.stop_event = stop_event
        self.books = {aid: new_book() for aid in assets}
        self.last_top = {}
        self.msgs = 0
        self.snapshots = 0
        self.reconnects = 0
        self.unknown = 0
        self.last_msg_ms = None
        self.lag_sum = 0.0
        self.lag_n = 0

    # -- emit ---------------------------------------------------------------
    def emit(self, asset_id, ts_ex, ts_recv, event_type):
        meta = self.assets[asset_id]
        bids, asks = sorted_book(self.books[asset_id])
        bb = bids[0][0] if bids else None
        ba = asks[0][0] if asks else None
        bbs = bids[0][1] if bids else None
        bas = asks[0][1] if asks else None

        self.writer.put("book", json.dumps({
            "seq": self.seq.next(),
            "ts": ts_ex, "recv": ts_recv, "et": event_type,
            "slug": self.slug, "asset_id": asset_id,
            "market_type": meta["market_type"], "outcome": meta["outcome"],
            "market_question": meta.get("market_question"),
            "outcome_name": meta.get("outcome_name"),
            "line": meta["line"], "condition_id": meta["condition_id"],
            "bids": bids, "asks": asks,
        }, separators=(",", ":")))
        self.snapshots += 1

        top = (bb, bbs, ba, bas)
        if self.last_top.get(asset_id) != top:
            self.last_top[asset_id] = top
            mid = (bb + ba) / 2 if (bb is not None and ba is not None) else None
            spr = (ba - bb) if (bb is not None and ba is not None) else None
            self.writer.put("top", [
                self.seq.n, ts_ex, ts_recv, self.slug, meta["market_type"], meta["market_question"],
                meta["line"], meta["outcome"], meta["condition_id"], asset_id,
                bb, bbs, ba, bas, mid, spr,
            ])

    # -- message handling ---------------------------------------------------
    def handle(self, msg, ts_recv):
        if not isinstance(msg, dict):
            return
        et = msg.get("event_type")
        ts_ex = msg.get("timestamp")
        try:
            ts_ex = int(ts_ex) if ts_ex is not None else ts_recv
        except (TypeError, ValueError):
            ts_ex = ts_recv

        self.last_msg_ms = ts_recv
        lag = ts_recv - ts_ex
        if -60000 < lag < 600000:
            self.lag_sum += lag
            self.lag_n += 1

        if et == "book":
            aid = msg.get("asset_id")
            if aid not in self.books:
                return
            apply_snapshot(self.books[aid], msg)
            self.emit(aid, ts_ex, ts_recv, "book")
            return

        if et == "price_change":
            # Observed live shape: no top-level asset_id; a `price_changes`
            # array whose entries each carry their own asset_id.
            changes = msg.get("price_changes") or msg.get("changes")
            if changes is None and "price" in msg:
                changes = [msg]
            if not changes:
                self.unknown += 1
                log(f"[{self.slug}] unrecognised price_change shape, keys={sorted(msg.keys())}")
                return
            touched = set()
            for ch in changes:
                aid = ch.get("asset_id") or msg.get("asset_id")
                if aid not in self.books:
                    continue
                try:
                    price = float(ch["price"]); size = float(ch["size"]); side = ch["side"]
                except (KeyError, TypeError, ValueError):
                    self.unknown += 1
                    continue
                if apply_change(self.books[aid], price, side, size):
                    touched.add(aid)
            for aid in touched:
                self.emit(aid, ts_ex, ts_recv, "price_change")
            return

        # tick_size_change / last_trade_price / anything else: not book state.

    # -- seeding ------------------------------------------------------------
    async def seed(self):
        async def one(aid):
            try:
                data = await asyncio.to_thread(
                    lambda: requests.get(CLOB_REST_BOOK_URL, params={"token_id": aid}, timeout=15).json())
            except Exception as e:
                log(f"[{self.slug}] seed failed for {aid[:12]}: {e}")
                return
            apply_snapshot(self.books[aid], data)
            ts = data.get("timestamp")
            try:
                ts = int(ts)
            except (TypeError, ValueError):
                ts = now_ms()
            self.emit(aid, ts, now_ms(), "seed")
        await asyncio.gather(*(one(a) for a in self.assets))

    # -- run ----------------------------------------------------------------
    async def run(self):
        await self.seed()
        backoff = 1
        ids = list(self.assets)
        while not self.stop_event.is_set():
            down_since = None
            try:
                async with websockets.connect(
                    CLOB_WS_MARKET_URL,
                    ping_interval=20,
                    ping_timeout=60,      # generous: a slow moment must not kill the feed
                    max_queue=None,       # buffer bursts instead of back-pressuring
                    max_size=None,
                    close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({"assets_ids": ids, "type": "market"}))
                    log(f"[{self.slug}] connected, {len(ids)} assets")
                    backoff = 1
                    while not self.stop_event.is_set():
                        raw = await ws.recv()
                        ts_recv = now_ms()
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            self.unknown += 1
                            continue
                        if isinstance(payload, dict):
                            payload = [payload]
                        self.msgs += len(payload)
                        for m in payload:
                            try:
                                self.handle(m, ts_recv)
                            except Exception as e:
                                log(f"[{self.slug}] handler error: {e}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self.stop_event.is_set():
                    break
                self.reconnects += 1
                down_since = time.monotonic()
                log(f"[{self.slug}] disconnected ({e}); retry in {backoff}s")
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, 20)
                if down_since:
                    log(f"[{self.slug}] outage ~{time.monotonic()-down_since:.1f}s "
                        f"-- price_changes during it are unrecoverable")


# ---------------------------------------------------------------------------
# Game-status watch
# ---------------------------------------------------------------------------

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def _final_states(game_pks):
    """gamePk -> abstractGameState ('Preview' / 'Live' / 'Final')."""
    r = requests.get(MLB_SCHEDULE_URL, timeout=25, params={
        "sportId": 1, "gamePks": ",".join(str(p) for p in game_pks)})
    r.raise_for_status()
    out = {}
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            out[g["gamePk"]] = (g.get("status") or {}).get("abstractGameState")
    return out


async def final_watch(stop_event, game_pks, grace_min, poll=60.0):
    """Stops recording once the MLB API says every tracked game has ended.

    Better than a fixed --duration guard, which has to be padded for the
    worst case (rain delay, extra innings) and therefore over-records a normal
    slate and still cuts a long one short. This asks the source of truth.

    `abstractGameState` is used rather than detailedState because it collapses
    'Final', 'Game Over' and 'Completed Early' into one value. A grace period
    runs after the last game goes Final so the post-game market tail is still
    captured, and the state is re-checked during it -- a game can leave Final
    if the API corrects itself, and then the countdown is abandoned.

    Any API failure is logged and retried; it never stops recording, because
    losing the feed to a transient HTTP error would be far worse than
    over-recording. --duration remains the hard backstop.
    """
    if not game_pks:
        return
    log(f"[final-watch] tracking {len(game_pks)} gamePk(s), "
        f"stop {grace_min:.0f} min after the last one is Final")
    final_since = None
    while not stop_event.is_set():
        try:
            states = await asyncio.to_thread(_final_states, game_pks)
            pending = [p for p in game_pks if states.get(p) != "Final"]
            if not pending:
                if final_since is None:
                    final_since = time.monotonic()
                    log(f"[final-watch] all {len(game_pks)} games Final; "
                        f"stopping in {grace_min:.0f} min")
                elif time.monotonic() - final_since >= grace_min * 60:
                    log("[final-watch] grace elapsed; stopping recorder")
                    stop_event.set()
                    return
            else:
                if final_since is not None:
                    log(f"[final-watch] {len(pending)} game(s) no longer Final; "
                        f"cancelling stop")
                final_since = None
        except Exception as e:
            log(f"[final-watch] poll failed ({type(e).__name__}: {e}); retrying")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def assets_from_matches(slugs=None, core_only=False):
    ms = matches_module.matches
    if slugs:
        ms = [m for m in ms if m["slug"] in slugs]
    targets, found, errs = discovery.discover_target_markets(ms, core_only=core_only)
    per_match = defaultdict(dict)
    for tm in targets:
        if tm["errors"]:
            continue
        outs = tm["outcomes"]
        for name, aid, label in ((outs[0], tm["yes_asset_id"], "YES"),
                                  (outs[1], tm["no_asset_id"], "NO")):
            per_match[tm["event_slug"]][aid] = {
                "market_type": tm["market_type"], "market_question": tm["market_question"],
                "line": tm["line"], "condition_id": tm["condition_id"],
                "outcome": label, "outcome_name": name,
            }
    for e in errs:
        log(f"[discovery] {e}")
    log(f"[discovery] {found} events, {len(per_match)} matches with assets")
    return per_match


def assets_from_event_slugs(slugs, core_only=False):
    """Record arbitrary Gamma event slugs, not just the MLB slate in
    matches.py. Every order-book market on the event is taken; MLB market
    types get discovery.py's friendly names, anything else keeps its raw
    `sportsMarketType` so it is still identifiable downstream."""
    per_match = defaultdict(dict)
    for slug in slugs:
        try:
            rows = requests.get(GAMMA_EVENTS_URL, params={"slug": slug}, timeout=30).json()
        except Exception as e:
            log(f"[discovery] {slug}: fetch failed: {e}")
            continue
        if not rows:
            log(f"[discovery] {slug}: not found")
            continue
        ev = rows[0]
        for m in ev.get("markets", []):
            if not m.get("enableOrderBook") or m.get("closed"):
                continue
            smt = m.get("sportsMarketType")
            table = discovery.CORE_SPORTS_MARKET_TYPES if core_only else discovery.ALL_SPORTS_MARKET_TYPES
            mt = table.get(smt)
            if mt is None:
                if core_only:
                    continue
                mt = smt or "other"
            try:
                ids = json.loads(m.get("clobTokenIds") or "[]")
                outs = json.loads(m.get("outcomes") or "[]")
            except json.JSONDecodeError:
                continue
            if len(ids) != 2 or len(outs) != 2:
                continue
            for aid, label, name in ((ids[0], "YES", outs[0]), (ids[1], "NO", outs[1])):
                per_match[slug][aid] = {
                    "market_type": mt,
                    "market_question": m.get("question"),
                    "line": None,
                    "condition_id": m.get("conditionId"),
                    "outcome": label,
                    "outcome_name": name,
                }
    log(f"[discovery] {sum(len(v) for v in per_match.values())} assets across "
        f"{len(per_match)} event(s)")
    return per_match


def assets_live_discover(limit):
    """Grab the most active currently-open markets. For smoke-testing the
    recorder when no tracked game is live."""
    ev = requests.get(GAMMA_EVENTS_URL,
                      params={"closed": "false", "limit": 80,
                              "order": "volume24hr", "ascending": "false"},
                      timeout=30).json()
    per_match = defaultdict(dict)
    n = 0
    for e in ev:
        slug = e.get("slug") or str(e.get("id"))
        for m in e.get("markets", []):
            if not m.get("enableOrderBook") or m.get("closed"):
                continue
            try:
                ids = json.loads(m.get("clobTokenIds") or "[]")
                outs = json.loads(m.get("outcomes") or "[]")
            except json.JSONDecodeError:
                continue
            if len(ids) != 2 or len(outs) != 2:
                continue
            for aid, label, name in ((ids[0], "YES", outs[0]), (ids[1], "NO", outs[1])):
                per_match[slug][aid] = {
                    "market_type": m.get("sportsMarketType") or "other",
                    "market_question": m.get("question"), "line": None,
                    "condition_id": m.get("conditionId"),
                    "outcome": label, "outcome_name": name,
                }
                n += 1
            if n >= limit:
                break
        if n >= limit:
            break
    log(f"[discovery] live mode: {n} assets across {len(per_match)} events")
    return per_match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def stats_loop(recs, writer, interval, stop_event, t0):
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        el = time.monotonic() - t0
        tot_m = sum(r.msgs for r in recs)
        tot_s = sum(r.snapshots for r in recs)
        rc = sum(r.reconnects for r in recs)
        log(f"[stats] {el:6.0f}s  msgs {tot_m:>8,} ({tot_m/max(el,1):>6.1f}/s)  "
            f"snapshots {tot_s:>8,}  reconnects {rc}  "
            f"queued {writer.q.qsize():>6,}  written {writer.written:>8,}")
        stale = []
        for r in recs:
            age = (now_ms() - r.last_msg_ms) / 1000 if r.last_msg_ms else None
            if age is None or age > 120:
                stale.append(f"{r.slug}({'never' if age is None else f'{age:.0f}s'})")
        if stale:
            log(f"[stats] quiet >120s: {', '.join(stale[:8])}")


async def main_async(args):
    if args.event_slugs:
        per_match = assets_from_event_slugs(args.event_slugs, core_only=args.core_only)
    elif args.live_discover:
        per_match = assets_live_discover(args.live_discover)
    else:
        per_match = assets_from_matches(args.slugs, core_only=args.core_only)

    if not per_match:
        raise SystemExit("No assets discovered; nothing to record.")

    writer = Writer(fsync_interval=args.fsync_interval, session=args.session)
    writer.start()

    stop_event = asyncio.Event()
    seq = Seq()
    recs = [MatchRecorder(slug, assets, writer, stop_event, seq)
            for slug, assets in sorted(per_match.items())]

    log(f"Recording {len(recs)} matches / {sum(len(r.assets) for r in recs)} assets, "
        f"one websocket each")
    log(f"  -> {writer.books_path}")
    log(f"  -> {writer.top_path}")

    t0 = time.monotonic()
    tasks = [asyncio.create_task(r.run()) for r in recs]
    tasks.append(asyncio.create_task(stats_loop(recs, writer, args.stats_interval, stop_event, t0)))

    if args.stop_when_final:
        pks = [m["gamePk"] for m in matches_module.matches
               if m.get("gamePk") and (not args.slugs or m["slug"] in args.slugs)]
        if pks:
            tasks.append(asyncio.create_task(
                final_watch(stop_event, pks, args.stop_when_final)))
        else:
            log("[final-watch] no gamePk in matches.py; "
                "regenerate it with plan_slate.py to enable. Falling back to --duration.")

    if args.duration:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=args.duration)
        except asyncio.TimeoutError:
            log(f"[done] --duration {args.duration}s reached; stopping")
            stop_event.set()
    else:
        await stop_event.wait()

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    writer.stop()
    writer.join(timeout=30)

    el = time.monotonic() - t0
    print()
    log("=== summary ===")
    for r in sorted(recs, key=lambda x: -x.snapshots):
        lag = (r.lag_sum / r.lag_n) if r.lag_n else float("nan")
        log(f"  {r.slug:<34} msgs {r.msgs:>7,}  snapshots {r.snapshots:>7,}  "
            f"reconnects {r.reconnects:>3}  mean delivery lag {lag:>7.0f} ms")
    tot_s = sum(r.snapshots for r in recs)
    log(f"  TOTAL {tot_s:,} snapshots in {el:.0f}s  ({tot_s/max(el,1):.1f}/s), "
        f"{sum(r.reconnects for r in recs)} reconnects, "
        f"{writer.written:,} records written, {writer.q.qsize()} still queued")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*", default=None, help="subset of matches.py slugs")
    ap.add_argument("--live-discover", type=int, default=0, metavar="N",
                     help="ignore matches.py; record the N most active open markets")
    ap.add_argument("--event-slugs", nargs="*", default=None,
                     help="record these Gamma event slugs instead of matches.py "
                          "(any sport or non-sport event, one websocket each)")
    ap.add_argument("--core-only", action="store_true",
                     help="record only moneyline/spread/total/first-inning-run. "
                          "Default records EVERY order-book market on the event, "
                          "including 1st-5-innings spreads/totals and extra innings.")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run until Ctrl+C)")
    ap.add_argument("--stats-interval", type=float, default=30.0)
    ap.add_argument("--stop-when-final", type=float, default=0, metavar="MIN",
                    help="stop MIN minutes after the MLB API reports every tracked "
                         "game Final (0 = disabled). Needs gamePk in matches.py, "
                         "which plan_slate.py writes. --duration still applies as a "
                         "hard backstop.")
    ap.add_argument("--session", default=None, metavar="TAG",
                    help="write to data/live/books_TAG.jsonl instead of books.jsonl. "
                         "Use one tag per slate (e.g. --session 2026-08-28); the files "
                         "are append-only, so without this a new run extends the "
                         "previous slate's file.")
    ap.add_argument("--fsync-interval", type=float, default=30.0,
                     help="seconds between durability checkpoints (done off the event loop)")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nstopped by user.")


if __name__ == "__main__":
    main()
