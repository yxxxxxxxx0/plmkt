"""Every stretch in which a recorded book was CROSSED, from the raw recordings.

A crossed book (best ask <= best bid) cannot exist on the exchange: the two
orders would trade. In the recording it means our copy of the book holds a
GHOST level -- an order the exchange has removed but whose removal message we
never applied -- and every price in that book is suspect until the next full
snapshot replaces it.

Found on mlb-ari-sf-2026-08-29, Over 7.5: the recorded ask sat at 0.52 for
three minutes while the recorded bid ran up to 0.66 through it and Over 8.5
and Over 6.5 repriced far above. It produced a 7.6-cent "arbitrage" that was
never there.

jump_data.py drops every crossed record (`asks[0][0] <= bids[0][0]`) and
forward-fills the last uncrossed book for up to 60s, so in the 200ms grid a
ghost looks like an ordinary, slightly stale book. The continuity scan cannot
see it either: records keep arriving throughout. The only place it is visible
is here, in the raw stream.

Output: data/crossed/<session>.parquet, one row per crossed span:
  asset_id, slug, market_type, line, start (exchange ms), end (first uncrossed
  record after it), n_records, last_uncrossed (exchange ms of the last good
  record before it -- the book the grid carried through the span).

    python crossed_scan.py                       # every session, in parallel
    python crossed_scan.py books_2026-08-29      # one
"""
from __future__ import annotations

import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import orjson
import pandas as pd

from jump_data import KEEP_TYPES, open_recording

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "crossed")


def scan(path: str) -> str:
    tag = os.path.basename(path).split(".jsonl")[0]
    state = {}          # asset -> dict(crossed, start, n, last_good, meta)
    spans = []
    with open_recording(path) as fh:
        for raw in fh:
            r = orjson.loads(raw)
            if r.get("outcome") != "YES" or r.get("market_type") not in KEEP_TYPES:
                continue
            bids, asks = r.get("bids"), r.get("asks")
            if not bids or not asks:
                continue
            a = r.get("asset_id")
            ts = r.get("ts")
            st = state.get(a)
            if st is None:
                st = state[a] = dict(crossed=False, start=None, n=0, last_good=None,
                                     meta=(r.get("slug"), r.get("market_type"), r.get("line")))
            crossed = asks[0][0] <= bids[0][0]
            if crossed and not st["crossed"]:
                st.update(crossed=True, start=ts, n=1)
            elif crossed:
                st["n"] += 1
            elif st["crossed"]:
                slug, mt, line = st["meta"]
                spans.append(dict(asset_id=a, slug=slug, market_type=mt, line=line,
                                  start=st["start"], end=ts, n_records=st["n"],
                                  last_uncrossed=st["last_good"], closed=True))
                st.update(crossed=False, start=None, n=0)
            if not crossed:
                st["last_good"] = ts
    for a, st in state.items():                 # still crossed at end of file
        if st["crossed"]:
            slug, mt, line = st["meta"]
            spans.append(dict(asset_id=a, slug=slug, market_type=mt, line=line,
                              start=st["start"], end=st["start"], n_records=st["n"],
                              last_uncrossed=st["last_good"], closed=False))
    os.makedirs(OUT, exist_ok=True)
    out = pd.DataFrame(spans)
    out.to_parquet(os.path.join(OUT, f"{tag}.parquet"), index=False)
    secs = ((out.end - out.start).sum() / 1000) if len(out) else 0
    return f"{tag}: {len(out)} crossed span(s), {secs / 60:.1f} min in total"


PAD_S = 10.0       # suspect from this long before the first crossed record ...
MERGE_S = 30.0     # ... and crossed spans closer than this are one episode
GHOST_S = 1.0      # an episode is a ghost only if the book stayed crossed this long


def suspect_windows(session: str, pad_s=PAD_S, merge_s=MERGE_S, ghost_s=GHOST_S) -> dict:
    """asset_id -> (k, 2) array of [start, end] exchange-ms windows to distrust.

    A ghost level only shows as a crossing when the bid happens to pass
    through it, so one ghost appears as many short crossed spans with
    ordinary-looking books between them (Over 7.5 on ari-sf 08-29: dozens of
    sub-second crossings over three minutes). Spans within merge_s of each
    other are merged into one episode and padded by pad_s either side, and the
    whole window is treated as suspect, not only the crossed instants.

    Only episodes crossed for at least ghost_s IN TOTAL count. Measured over
    all 19 recordings, 9,380 of 9,798 episodes are crossed for zero time --
    several updates delivered in one exchange millisecond, a bid landing a
    moment before the delete of the ask it traded against -- and they cluster
    in the busiest books at the busiest moments. Treating those as ghosts
    removed 676 moneyline jump marks, most of them genuine. Between the two
    populations the distribution is nearly empty (69 episodes under 1s), and
    the ari-sf ghost sits far above it: 164s crossed over 3,409 records.
    """
    p = os.path.join(OUT, f"{session}.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"no crossed scan for {session}: run crossed_scan.py first")
    d = pd.read_parquet(p)
    out = {}
    if not len(d):
        return out
    for a, g in d.sort_values("start").groupby("asset_id"):
        w = []                                   # [start, end, crossed ms]
        for s, e in zip(g.start.to_numpy(), g.end.to_numpy()):
            c = e - s
            s, e = s - pad_s * 1000, e + pad_s * 1000
            if w and s <= w[-1][1] + merge_s * 1000:
                w[-1][1] = max(w[-1][1], e)
                w[-1][2] += c
            else:
                w.append([s, e, c])
        keep = [x[:2] for x in w if x[2] >= ghost_s * 1000]
        if keep:
            out[a] = np.array(keep, dtype=np.int64)
    return out


def main() -> None:
    live = os.path.join(BASE, "data", "live")
    if len(sys.argv) > 1:
        paths = [p for t in sys.argv[1:] for p in glob.glob(os.path.join(live, t + ".jsonl*.xz"))]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(live, "books_2026-*.jsonl*.xz")))
    with Pool(min(len(paths), 10)) as pool:
        for msg in pool.imap_unordered(scan, paths):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
