"""In-game continuity: every silence between first pitch and the final out.

match_quality.py counts a game minute as covered when it holds ANY record, so
a few stray updates make a minute look live. It also takes outage length from
the recorder's `conn` records, and those measure only the retry sleep, not how
long the socket had already been dead: on books_2026-09-25 five games lost
2.3 minutes each at 03:19Z while the recorder logged ~10s of outage, and all
five were graded GOOD. Heartbeats do not settle it either -- the recorder
writes one whenever IT has heard nothing, so a dead socket produces them too.

The test here uses the market itself. An in-play MLB book changes many times a
minute across a game's ~34 assets, so a silence of more than GAP_S seconds on
the whole match stream inside the game window is data that was never captured.
Receive time is used, not exchange time: the question is when the socket
delivered. A late start or early stop is just a silence touching an edge.

Each silence is attributed, so causes can be told apart:
  disconnect       a disconnect/outage record falls inside it
  session_silent   share of it during which NO match in the session received
                   anything -- near 1.0 means the local network or recorder,
                   not one market

Checked on the 2026-09-25/26 holes: records whose exchange ts falls inside a
hole number in the tens, i.e. only book snapshots stamped with their last
change time. The updates were lost, not delivered late.

    python continuity_scan.py data/live/books_2026-09-26.jsonl.xz      # scan
    python continuity_scan.py --exclusions                            # classify

Scans write data/continuity/<session>.json. --exclusions reads them all and
writes data/excluded_matches.json, which match_filter.judge_match honours.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import lzma
import os
from array import array
from collections import defaultdict

import numpy as np
import orjson

BASE = os.path.dirname(os.path.abspath(__file__))
WIN = os.path.join(BASE, "data", "game_windows.json")
OUT_DIR = os.path.join(BASE, "data", "continuity")
EXCLUDED = os.path.join(BASE, "data", "excluded_matches.json")

# Silences shorter than this are ordinary: one game, a few seconds, no
# disconnect, while every other game keeps streaming.
GAP_S = 5
# A silence longer than this is a hole even without a disconnect record.
HOLE_S = 30
# Total loss a match may carry and still be kept. The same tolerance
# match_filter applies to a late start or an early stop, and the observed
# break: kept matches lose at most 4.5 min, the next worst loses 8.
MAX_LOST_S = 300


def to_ms(s):
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def scan(path, windows, gap_s=GAP_S):
    allr = defaultdict(lambda: array("q"))   # slug -> recv of data records
    ml = defaultdict(lambda: array("q"))     # slug -> recv of moneyline records
    conn = defaultdict(list)                 # slug -> (recv, event, outage_s)
    hb = defaultdict(list)
    every = array("q")                       # every data record, whole session
    op = lzma.open if path.endswith(".xz") else open
    with op(path, "rb") as fh:
        for raw in fh:
            try:
                r = orjson.loads(raw)
            except Exception:
                continue
            s, et = r.get("slug"), r.get("et")
            t = r.get("recv") or r.get("ts")
            if not s or t is None:
                continue
            if et == "conn":
                conn[s].append((t, r.get("event"), r.get("outage_s")))
                continue
            if et == "heartbeat":
                hb[s].append((t, r.get("quiet_s")))
                continue
            allr[s].append(t)
            every.append(t)
            if r.get("market_type") == "moneyline":
                ml[s].append(t)

    gs = np.sort(np.frombuffer(every, dtype=np.int64))
    del every

    def gaps(ts, a, b):
        x = np.sort(np.frombuffer(ts, dtype=np.int64))
        x = x[(x >= a) & (x <= b)]
        pts = np.concatenate([[a], x, [b]])
        idx = np.nonzero(np.diff(pts) > gap_s * 1000)[0]
        return [(int(pts[i]), int(pts[i + 1])) for i in idx], int(len(x))

    def session_silent(a, b):
        i = np.searchsorted(gs, a, "right")
        j = np.searchsorted(gs, b, "left")
        if i >= j:
            return 1.0
        pts = np.concatenate([[a], gs[i:j], [b]])
        return float(np.clip(np.diff(pts) - gap_s * 1000, 0, None).sum() / (b - a))

    res = []
    for s in sorted(allr):
        w = windows.get(s)
        row = dict(slug=s, window=bool(w))
        if not w:
            res.append(row)
            continue
        a, b = to_ms(w["start_utc"]), to_ms(w["end_utc"])

        def attr(x, y):
            d = [c for c in conn[s] if x - 5000 <= c[0] <= y + 5000
                 and c[1] != "connected"]
            return dict(start=x, end=y, secs=round((y - x) / 1000, 1),
                        disconnect=bool(d),
                        session_silent=round(session_silent(x, y), 2),
                        at_start=(x == a), at_end=(y == b))

        g_all, n_in = gaps(allr[s], a, b)
        g_ml, n_ml = gaps(ml[s], a, b)
        discs = [c for c in conn[s] if a - 600_000 <= c[0] <= b + 600_000
                 and c[1] == "disconnected"]
        in_hb = [h for h in hb[s] if a <= h[0] <= b]
        row.update(game_min=round((b - a) / 60000, 1), recs_in_game=n_in,
                   ml_recs_in_game=n_ml,
                   gaps=[attr(*g) for g in g_all],
                   ml_gaps=[attr(*g) for g in g_ml],
                   disconnects_in_game=len(discs),
                   heartbeats_in_game=len(in_hb),
                   max_hb_quiet=max([h[1] or 0 for h in in_hb], default=0))
        res.append(row)
    return res


def classify(row):
    """('continuous' | 'slight' | 'severe' | 'no_game', reason)."""
    if not row["window"]:
        return "no_game", "no game window"
    g = row["gaps"]
    start = sum(x["secs"] for x in g if x["at_start"])
    end = sum(x["secs"] for x in g if x["at_end"])
    holes = [x for x in g if x["secs"] > HOLE_S
             or (x["disconnect"] and not (x["at_start"] or x["at_end"]))]
    lost = sum(x["secs"] for x in holes)
    if start > MAX_LOST_S or end > MAX_LOST_S or lost > MAX_LOST_S:
        return "severe", (f"lost {lost / 60:.1f} of {row['game_min']:.0f} min "
                          f"(late start {start / 60:.0f}m, "
                          f"early stop {end / 60:.0f}m)")
    if lost > 0:
        return "slight", f"lost {lost:.0f}s in {len(holes)} hole(s)"
    return "continuous", ""


def write_exclusions():
    out, counts = {}, defaultdict(int)
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.json"))):
        session = os.path.basename(f)[:-5]
        for row in json.load(open(f)):
            c, why = classify(row)
            counts[c] += 1
            if c in ("severe", "no_game"):
                out[row["slug"]] = dict(session=session, verdict=c, reason=why)
    json.dump(dict(
        note=("Matches whose recording is too incomplete to use, from "
              "continuity_scan.py. Raw archives are untouched; "
              "match_filter.judge_match drops these slugs."),
        max_lost_s=MAX_LOST_S, hole_s=HOLE_S,
        matches=dict(sorted(out.items()))),
        open(EXCLUDED, "w"), indent=1)
    print(dict(counts), f"-> {len(out)} excluded, written to {EXCLUDED}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--exclusions", action="store_true")
    args = ap.parse_args()
    windows = json.load(open(WIN))
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in args.paths:
        session = os.path.basename(p).split(".")[0].replace("books_", "")
        res = scan(p, windows)
        json.dump(res, open(os.path.join(OUT_DIR, f"{session}.json"), "w"))
        print("scanned", p, len(res), "matches")
    if args.exclusions:
        write_exclusions()


if __name__ == "__main__":
    main()
