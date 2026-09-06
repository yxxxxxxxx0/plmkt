"""
Build a large (price, outcome) dataset from Polymarket's RESOLVED markets.

Why this exists
---------------
The calibration question -- does a contract priced at p resolve YES p of the
time? -- needs only two numbers per market: a price at some point before
resolution, and the outcome. It does NOT need order books, so it is not
limited to the handful of games this machine happened to record.

The 52 recorded games yield 259 settled contracts, which turned out to be far
too few: with one independent observation per contract, no price band reached
significance. This pulls thousands instead, from two public endpoints:

    gamma /markets?closed=true   -> the RESOLUTION (outcomePrices is ["1","0"]
                                    or ["0","1"] once a market has settled)
    data-api /trades?market=...  -> the trade tape, from which a pre-resolution
                                    price is taken

Design choices that keep the sample honest
------------------------------------------
  * Only two-outcome markets that resolved cleanly to 1/0. Anything still
    open, void, or ambiguous is skipped rather than guessed at.
  * The price used is the median trade price in the MIDDLE THIRD of the
    market's trading life. Using the last price before close would be
    circular -- by then the market usually knows the answer -- and using the
    first would sample an illiquid opening.
  * A minimum trade count, because a market that traded five times has a
    price that means very little.
  * One row per market (never per token): YES and NO are the same contract.

    python fetch_resolved.py --limit 3000 --min-volume 1000
    python fetch_resolved.py --tag baseball --limit 2000
"""
import argparse
import json
import os
import statistics
import sys
import time

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data", "resolved")
GAMMA = "https://gamma-api.polymarket.com/markets"
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
    ap.add_argument("--limit", type=int, default=3000, help="markets to scan")
    ap.add_argument("--min-volume", type=float, default=1000.0)
    ap.add_argument("--min-trades", type=int, default=25)
    ap.add_argument("--tag", default=None, help="gamma tag_slug filter, e.g. baseball")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "resolved.jsonl"))
    ap.add_argument("--pause", type=float, default=0.12)
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
        log(f"resuming: {len(done):,} markets already on disk")

    s = requests.Session()
    s.headers["User-Agent"] = "polymarket-orderbook-research/1.0"
    kept = scanned = 0
    offset = 0
    PAGE = 100

    with open(args.out, "a", encoding="utf-8") as out:
        while scanned < args.limit:
            p = {"closed": "true", "limit": PAGE, "offset": offset,
                 "order": "endDate", "ascending": "false"}
            if args.tag:
                p["tag_slug"] = args.tag
            rows = get(s, GAMMA, p)
            if not rows:
                log(f"no more markets at offset {offset}")
                break
            offset += len(rows)
            for m in rows:
                scanned += 1
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
                # resolved cleanly to one side
                try:
                    a, b = float(op[0]), float(op[1])
                except ValueError:
                    continue
                if not ((a > 0.99 and b < 0.01) or (a < 0.01 and b > 0.99)):
                    continue
                if float(m.get("volumeNum") or 0) < args.min_volume:
                    continue

                tr = get(s, TRADES, {"market": cid, "limit": 1000})
                time.sleep(args.pause)
                if not tr or len(tr) < args.min_trades:
                    continue
                # price on the FIRST outcome's side, middle third of life
                rows_t = sorted(((float(x["timestamp"]), float(x["price"]),
                                  x.get("asset")) for x in tr), key=lambda z: z[0])
                n = len(rows_t)
                mid_slice = rows_t[n // 3: 2 * n // 3] or rows_t
                tok0 = toks[0]
                px = [r[1] for r in mid_slice if r[2] == tok0]
                if len(px) < 5:
                    # fall back to the complement, mirrored
                    px = [1.0 - r[1] for r in mid_slice if r[2] == toks[1]]
                if len(px) < 5:
                    continue
                price0 = statistics.median(px)
                rec = {
                    "condition_id": cid,
                    "question": (m.get("question") or "")[:160],
                    "slug": m.get("slug"),
                    "end_date": m.get("endDate"),
                    "volume": float(m.get("volumeNum") or 0),
                    "n_trades": n,
                    "outcome0": outs[0] if outs else None,
                    "resolved0": 1 if a > 0.5 else 0,
                    "price0_mid_life": price0,
                    "sports_market_type": m.get("sportsMarketType"),
                }
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                kept += 1
                if kept % 25 == 0:
                    out.flush()
                    log(f"  scanned {scanned:,}  kept {kept:,}")
            if len(rows) < PAGE:
                break

    log(f"done: scanned {scanned:,}, kept {kept:,} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
