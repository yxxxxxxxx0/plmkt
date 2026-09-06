"""
Build a DIVERSE (price, outcome) dataset from resolved sports EVENTS.

Replaces the /markets approach, which failed twice in instructive ways:
  * `tag_slug` is silently ignored on /markets -- every query returned the same
    rows regardless of the tag asked for.
  * Ordering by endDate returned whatever happened to close most recently,
    which was 769 of 775 non-sports markets dominated by a single family of
    crypto token-launch contracts. 58 of them shared one launch window and
    all resolved the same way. That is one correlated bet, not 58
    observations, and it manufactured a textbook-looking favourite-longshot
    bias that does not survive inspection.

/events DOES honour tag_slug and paginates thousands deep through a full
season of MLB game events. Each GAME is a genuinely independent trial, which
is the property a calibration study needs and the crypto sample entirely
lacked.

Each row records its event slug so within-event correlation can be measured
rather than assumed away: the moneyline, run line and totals for one game all
resolve off a single result, so they are not independent of each other even
though the games are independent of each other.

    python fetch_resolved_sports.py --tag baseball --pages 40
"""
import argparse
import json
import os
import statistics
import sys
import time

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "resolved", "sports.jsonl")
EVENTS = "https://gamma-api.polymarket.com/events"
TRADES = "https://data-api.polymarket.com/trades"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get(session, url, params, retries=5):
    delay = 0.4
    for _ in range(retries):
        try:
            r = session.get(url, params=params, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                return None
        except requests.RequestException:
            pass
        time.sleep(delay)
        delay = min(delay * 2, 12)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseball")
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--page-size", type=int, default=20)
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--min-volume", type=float, default=200.0)
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pause", type=float, default=0.10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["condition_id"])
                except Exception:
                    pass
        log(f"resuming: {len(done):,} markets on disk")

    s = requests.Session()
    s.headers["User-Agent"] = "polymarket-orderbook-research/1.0"
    kept = n_ev = 0

    with open(args.out, "a", encoding="utf-8") as out:
        for page in range(args.pages):
            off = args.start_offset + page * args.page_size
            evs = get(s, EVENTS, {"closed": "true", "tag_slug": args.tag,
                                  "limit": args.page_size, "offset": off,
                                  "order": "endDate", "ascending": "false"})
            if not evs:
                log(f"no events at offset {off}; stopping")
                break
            for ev in evs:
                n_ev += 1
                eslug = ev.get("slug") or ""
                for m in ev.get("markets", []):
                    cid = m.get("conditionId")
                    if not cid or cid in done:
                        continue
                    try:
                        outs = json.loads(m.get("outcomes") or "[]")
                        op = json.loads(m.get("outcomePrices") or "[]")
                        toks = json.loads(m.get("clobTokenIds") or "[]")
                    except json.JSONDecodeError:
                        continue
                    if len(op) != 2 or len(toks) != 2:
                        continue
                    try:
                        a, b = float(op[0]), float(op[1])
                    except ValueError:
                        continue
                    if not ((a > 0.99 and b < 0.01) or (a < 0.01 and b > 0.99)):
                        continue
                    if float(m.get("volumeNum") or 0) < args.min_volume:
                        continue
                    done.add(cid)

                    tr = get(s, TRADES, {"market": cid, "limit": 1000})
                    time.sleep(args.pause)
                    if not tr or len(tr) < args.min_trades:
                        continue
                    rows = sorted(((float(x["timestamp"]), float(x["price"]),
                                    x.get("asset")) for x in tr),
                                  key=lambda z: z[0])
                    n = len(rows)
                    # middle third BY TIME, not by trade index: volume spikes
                    # near resolution, so the middle third of the trade
                    # sequence is late in the market's life and already knows
                    # too much.
                    t0, t1 = rows[0][0], rows[-1][0]
                    lo, hi = t0 + (t1 - t0) / 3.0, t0 + 2 * (t1 - t0) / 3.0
                    seg = [r for r in rows if lo <= r[0] <= hi] or rows
                    px = [r[1] for r in seg if r[2] == toks[0]]
                    if len(px) < 4:
                        px = [1.0 - r[1] for r in seg if r[2] == toks[1]]
                    if len(px) < 4:
                        continue
                    out.write(json.dumps({
                        "condition_id": cid,
                        "event_slug": eslug,
                        "question": (m.get("question") or "")[:160],
                        "end_date": m.get("endDate"),
                        "volume": float(m.get("volumeNum") or 0),
                        "n_trades": n,
                        "outcome0": outs[0] if outs else None,
                        "resolved0": 1 if a > 0.5 else 0,
                        "price0_mid_life": statistics.median(px),
                        "sports_market_type": m.get("sportsMarketType"),
                    }, separators=(",", ":")) + "\n")
                    kept += 1
            out.flush()
            log(f"  offset {off:>5}  events {n_ev:>5}  kept {kept:>5}")

    log(f"done: {n_ev} events scanned, {kept} markets kept -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
