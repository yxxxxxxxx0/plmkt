"""
Backfill the public trade tape for every recorded match.

The recordings in data/live are order BOOK snapshots only -- the recorder
receives `last_trade_price` on the same websocket and drops it
(live_recorder.py, end of handle()). So nothing on disk says when a trade
actually happened, and without that you cannot tell whether a resting order
would have filled, which is the only question that matters for a maker.

This recovers it after the fact. data-api.polymarket.com/trades is public,
unauthenticated, and pages back through a closed market's entire life, so the
tape for games already recorded is still retrievable.

    python fetch_trades.py --all
    python fetch_trades.py --slate 2026-08-27
    python fetch_trades.py --slugs mlb-ari-sf-2026-08-27

Output: data/trades/<slug>.jsonl, one trade per line, raw as returned.
        data/trades/_index.json, per-market counts.

What this tape is, and is not
-----------------------------
It carries proxyWallet, side, size, price, outcome and asset -- richer than
the live websocket event, and wallet identity means a trader can be followed
across markets and games.

But its `timestamp` is SECONDS, while the book is microseconds. Aligning a
trade to the book state that preceded it is therefore ambiguous to within a
second, and up to 78 book snapshots can share a single millisecond here. Good
enough for rates, flow imbalance and wallet behaviour; NOT good enough to
attribute an individual fill to an individual quote. For that, capture
`last_trade_price` live -- it is millisecond-stamped.

Truncation is the failure mode that matters
-------------------------------------------
A short read looks exactly like a quiet market, and the whole repo's history
is of silent shrinkage going unnoticed. So:

  * paging stops only on a genuinely empty or short page, never on an error;
    any market that ends in an error is recorded as FAILED and its count is
    not written as if final
  * output goes to <slug>.jsonl.part and is renamed only after every market
    of that game succeeded, so a killed run leaves no file that a later step
    would mistake for complete
  * _index.json records the per-market trade count, so "did this game fetch
    fully" is answerable later without refetching
  * 429 is expected, not exceptional -- it is retried with backoff rather
    than being allowed to end a page loop early
"""
import argparse
import json
import os
import sys
import time

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "data", "discovery_cache.json")
OUT_DIR = os.path.join(BASE, "data", "trades")
API = "https://data-api.polymarket.com/trades"

PAGE = 1000
PAUSE = 0.15          # polite gap between pages
MAX_RETRY = 6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_page(session, cid, offset):
    """One page, retrying 429/5xx with exponential backoff.

    Returns the list, or raises. It must never return [] because of an error
    -- an empty list is the signal that paging is finished, so conflating the
    two would silently truncate a market.
    """
    delay = 0.5
    last = None
    for attempt in range(MAX_RETRY):
        try:
            r = session.get(API, params={"market": cid, "limit": PAGE,
                                         "offset": offset}, timeout=45)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"{cid[:12]} offset={offset}: {last}")
        except requests.RequestException as e:
            last = f"{type(e).__name__}"
        time.sleep(delay)
        delay = min(delay * 2, 20.0)
    raise RuntimeError(f"{cid[:12]} offset={offset}: gave up after "
                       f"{MAX_RETRY} attempts ({last})")


def fetch_market(session, cid):
    """Every trade for one condition_id, paged to exhaustion."""
    out, offset = [], 0
    while True:
        rows = get_page(session, cid, offset)
        if not rows:
            break
        out += rows
        offset += len(rows)
        if len(rows) < PAGE:
            break
        time.sleep(PAUSE)
    return out


def trade_key(t):
    """Identity for dedupe. Offset paging can repeat a row if the tape shifts
    under us; these markets are closed so it should not happen, but a silent
    duplicate would inflate every volume statistic built on this."""
    return (t.get("transactionHash"), t.get("asset"), t.get("timestamp"),
            t.get("size"), t.get("price"), t.get("side"),
            t.get("proxyWallet"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*", default=None)
    ap.add_argument("--slate", default=None, help="US date, e.g. 2026-08-27")
    ap.add_argument("--all", action="store_true", help="every slug in the cache")
    ap.add_argument("--force", action="store_true", help="refetch games already on disk")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    with open(CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    events = {s: e for s, e in cache.items()
              if isinstance(e, dict) and e.get("markets")}

    if args.slugs:
        slugs = [s for s in args.slugs]
    elif args.slate:
        slugs = sorted(s for s in events if s.endswith(args.slate))
    elif args.all:
        slugs = sorted(events)
    else:
        raise SystemExit("pass --slugs, --slate or --all")
    missing = [s for s in slugs if s not in events]
    if missing:
        raise SystemExit(f"not in discovery cache: {missing}")

    os.makedirs(args.out_dir, exist_ok=True)
    index_path = os.path.join(args.out_dir, "_index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

    session = requests.Session()
    session.headers["User-Agent"] = "polymarket-orderbook-research/1.0"
    log(f"{len(slugs)} game(s) to fetch")
    grand = 0
    t_all = time.time()

    for slug in slugs:
        final = os.path.join(args.out_dir, f"{slug}.jsonl")
        if os.path.exists(final) and not args.force:
            log(f"  {slug}: already on disk, skipping (--force to refetch)")
            continue

        markets = events[slug]["markets"]
        part = final + ".part"
        counts, failed, seen, n = {}, [], set(), 0
        t0 = time.time()
        with open(part, "w", encoding="utf-8") as fh:
            for m in markets:
                cid = m.get("conditionId")
                if not cid:
                    continue
                try:
                    rows = fetch_market(session, cid)
                except Exception as e:
                    failed.append(f"{m.get('sportsMarketType')}:{cid[:10]}: {e}")
                    continue
                kept = 0
                for t in rows:
                    k = trade_key(t)
                    if k in seen:
                        continue
                    seen.add(k)
                    fh.write(json.dumps(t, separators=(",", ":")) + "\n")
                    kept += 1
                counts[cid] = {"market_type": m.get("sportsMarketType"),
                               "question": (m.get("question") or "")[:80],
                               "trades": kept, "raw": len(rows)}
                n += kept
                time.sleep(PAUSE)

        if failed:
            # Leave the .part in place: it holds real data and refetching is
            # slow, but do NOT promote it, so nothing downstream can read a
            # partial game as if it were whole.
            log(f"  {slug}: {len(failed)} market(s) FAILED -- left as .part, not promoted")
            for f_ in failed[:3]:
                log(f"      {f_}")
            index[slug] = {"status": "incomplete", "trades": n,
                           "markets_ok": len(counts), "markets_failed": len(failed),
                           "errors": failed[:10], "per_market": counts}
            continue

        os.replace(part, final)
        dupes = sum(c["raw"] for c in counts.values()) - n
        index[slug] = {"status": "complete", "trades": n,
                       "markets": len(counts), "duplicates_dropped": dupes,
                       "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "per_market": counts}
        grand += n
        log(f"  {slug}: {n:,} trades over {len(counts)} markets "
            f"({os.path.getsize(final)/1e6:.1f} MB, {time.time()-t0:.0f}s"
            f"{f', {dupes} dupes dropped' if dupes else ''})")

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=1)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    bad = [s for s, v in index.items() if v.get("status") != "complete"]
    log(f"done: {grand:,} trades in {(time.time()-t_all)/60:.1f} min -> {args.out_dir}")
    if bad:
        log(f"INCOMPLETE games ({len(bad)}): {', '.join(bad[:8])}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
