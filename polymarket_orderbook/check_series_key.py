"""Does one (slug, market_type, line) key ever cover more than one asset?

jump_data.extract builds its series key as (slug, market_type, line) and
forward-fills a single book per key. If two distinct asset_ids share a key,
their updates interleave into one series and the resulting "mid" alternates
between two unrelated price levels -- which would look exactly like the
violent 0.70 <-> 0.02 oscillation and constant +/-70 tick moves seen in the
plotted spread market, and would make those moves an artifact rather than
anything the market did.

Reads the raw recording directly, so it tests the source rather than the
derived dataset.
"""
from __future__ import annotations

import argparse
import collections
import os

import orjson

from jump_data import open_recording


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/live/books_2026-09-10.jsonl.xz")
    ap.add_argument("--max-lines", type=int, default=400_000)
    a = ap.parse_args()

    key_assets = collections.defaultdict(set)
    asset_price = {}
    n = 0
    with open_recording(a.raw) as fh:
        for line in fh:
            n += 1
            if n > a.max_lines:
                break
            try:
                r = orjson.loads(line)
            except Exception:
                continue
            if r.get("outcome") != "YES":
                continue
            mt = r.get("market_type")
            if mt not in ("moneyline", "spread", "total", "first_inning_run"):
                continue
            bids, asks = r.get("bids"), r.get("asks")
            if not bids or not asks:
                continue
            key = (r.get("slug"), mt, r.get("line"))
            aid = r.get("asset_id")
            key_assets[key].add(aid)
            if aid:
                asset_price[aid] = (bids[0][0] + asks[0][0]) / 2.0

    multi = {k: v for k, v in key_assets.items() if len(v) > 1}
    print(f"scanned {n:,} lines")
    print(f"{len(key_assets)} distinct (slug, market_type, line) keys")
    print(f"{len(multi)} keys map to MORE THAN ONE asset_id\n")

    if not multi:
        print("OK: every key is one asset. The oscillation is not this.")
        return

    by_mt = collections.Counter(k[1] for k in multi)
    print("affected keys by market type:", dict(by_mt))
    print("\nexamples (mids of the assets sharing one key):")
    for k, v in list(multi.items())[:6]:
        mids = sorted(round(asset_price[a2], 3) for a2 in v if a2 in asset_price)
        print(f"  {k}")
        print(f"    {len(v)} assets, mids {mids}")
    print("\nIf the mids under one key sit at very different levels, the "
          "series built from that key alternates between them and every "
          "'move' it shows is an artifact of the interleaving.")


if __name__ == "__main__":
    main()
