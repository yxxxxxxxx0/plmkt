"""
Split a live-captured trade tape into one file per game.

live_recorder writes every match's trades into a single
data/live/trades_<session>.jsonl. Everything downstream works per game, so
this fans it out the same way rebuild_multi.split_books_jsonl does for books.

Kept SEPARATE from data/trades/ (the data-api backfill) on purpose. The two
tapes are not interchangeable:

    data/trades/<slug>.jsonl        backfilled, SECOND-resolution timestamps,
                                    carries proxyWallet (trader identity)
    data/trades_live/<slug>.jsonl   captured live, MILLISECOND timestamps,
                                    carries fee_rate_bps, no wallet

A second holds a median of ~85 book updates on these markets, so only the live
tape can be aligned to the book per event. Overwriting one with the other
would silently destroy whichever property the next analysis needed.

    python split_live_trades.py --session 2026-09-04
    python split_live_trades.py --all
"""
import argparse
import glob
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(BASE, "data", "live")
OUT = os.path.join(BASE, "data", "trades_live")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def split_one(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files, counts, bad = {}, {}, 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    slug = json.loads(line)["slug"]
                except (json.JSONDecodeError, KeyError):
                    bad += 1
                    continue
                fh = files.get(slug)
                if fh is None:
                    fh = files[slug] = open(os.path.join(out_dir, f"{slug}.jsonl"),
                                            "w", encoding="utf-8")
                    counts[slug] = 0
                fh.write(line + "\n")
                counts[slug] += 1
    finally:
        for fh in files.values():
            fh.close()
    return counts, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="session tag, e.g. 2026-09-04")
    ap.add_argument("--all", action="store_true", help="every trades_*.jsonl")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    if args.session:
        paths = [os.path.join(LIVE, f"trades_{args.session}.jsonl")]
    elif args.all:
        paths = sorted(glob.glob(os.path.join(LIVE, "trades_*.jsonl")))
    else:
        raise SystemExit("pass --session or --all")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing: {missing}")

    for p in paths:
        counts, bad = split_one(p, args.out_dir)
        tot = sum(counts.values())
        log(f"{os.path.basename(p)}: {tot:,} trades -> {len(counts)} game(s)"
            + (f", {bad} unparseable" if bad else ""))
        for slug in sorted(counts):
            log(f"    {slug:<30} {counts[slug]:>7,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
