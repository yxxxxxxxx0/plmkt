"""
Rebuilds exact, tick-level order book histories for the 15-game
multi_recorder.py run -- every "tick" is a real recorded order-book change,
not a resampled/downsampled grid point.

Unlike rebuild.py (which replays a single-game orderbook_raw.jsonl event
log), this reads the multi-game recorder's depth CSV directly:
  - data/orderbooks_depth.csv    (full depth, one row per price level per
    real change -- ~19M rows / 6+ GB across all 15 games)

Each row is already a real, deduped change (multi_recorder only writes when
the book actually differs from the last recorded state), so there's no
diff-replay needed like the single-game jsonl format requires: grouping all
depth rows sharing a (timestamp, asset_id) already gives the full snapshot
at that instant.

No resampling and no window trimming: the output covers the full recorded
span for the game (every asset's first recorded snapshot through its last),
and every asset's timeline keeps its exact original change timestamps --
nothing is snapped to a grid or dropped. A shared, match-wide "tick" axis is
built as the sorted union of every real change timestamp across all of a
game's assets, so the viewer can still offer one slider per match while
every token's book at a given tick is looked up via forward-fill against
that token's own exact-timestamp runs.

Kickoff / actual-game-end times (from the MLB Stats API; see
data/game_windows.json) are carried through as reference markers only --
they no longer clip the data.

Usage:
    python rebuild_multi.py
    python rebuild_multi.py --games mlb-cle-laa-2026-08-26
"""

import argparse
import bisect
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEPTH_CSV_PATH = os.path.join(DATA_DIR, "orderbooks_depth.csv")
GAME_WINDOWS_PATH = os.path.join(DATA_DIR, "game_windows.json")
SPLIT_DIR = os.path.join(DATA_DIR, "depth_by_game")
OUTPUT_DIR = os.path.join(DATA_DIR, "uniform")

# live_recorder.py output: one JSON object per book snapshot
BOOKS_JSONL_PATH = os.path.join(DATA_DIR, "live", "books.jsonl")
JSONL_SPLIT_DIR = os.path.join(DATA_DIR, "live", "by_game")

DEPTH_HEADER = [
    "timestamp_utc", "timestamp_hkt", "event_slug", "match", "market_type",
    "market_question", "line", "outcome", "condition_id", "asset_id",
    "side", "level", "price", "size",
]


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_game_windows():
    """Optional reference markers (kickoff / actual game end from the MLB
    Stats API). Missing entirely is fine -- markers just won't be set."""
    if not os.path.exists(GAME_WINDOWS_PATH):
        return {}
    with open(GAME_WINDOWS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    windows = {}
    for slug, info in raw.items():
        windows[slug] = {
            "kickoff_ts": parse_ts(info["start_utc"]),
            "game_end_ts": parse_ts(info["end_utc"]),
            "status": info["status"],
        }
    return windows


# ---------------------------------------------------------------------------
# Step 1: split the big depth CSV into one file per game (single streaming
# pass; avoids loading 6+ GB into memory and avoids re-reading it per game).
# ---------------------------------------------------------------------------

def split_depth_by_game(slugs):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    existing = {
        slug for slug in slugs
        if os.path.exists(os.path.join(SPLIT_DIR, f"{slug}.csv"))
        and os.path.getsize(os.path.join(SPLIT_DIR, f"{slug}.csv")) > 0
    }
    if existing == set(slugs):
        print(f"[split] reusing existing per-game depth files in {SPLIT_DIR}")
        return

    print(f"[split] streaming {DEPTH_CSV_PATH} -> per-game files in {SPLIT_DIR} ...")
    writers = {}
    files = {}
    try:
        for slug in slugs:
            f = open(os.path.join(SPLIT_DIR, f"{slug}.csv"), "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(DEPTH_HEADER)
            files[slug] = f
            writers[slug] = w

        n = 0
        with open(DEPTH_CSV_PATH, "r", encoding="utf-8") as src:
            reader = csv.reader(src)
            header = next(reader)
            slug_idx = header.index("event_slug")
            for row in reader:
                slug = row[slug_idx]
                w = writers.get(slug)
                if w is not None:
                    w.writerow(row)
                n += 1
                if n % 2_000_000 == 0:
                    print(f"[split]   {n:,} rows scanned...")
        print(f"[split] done: {n:,} rows scanned")
    finally:
        for f in files.values():
            f.close()


# ---------------------------------------------------------------------------
# Step 2: for one game's split file, group rows into per-asset snapshot
# timelines. Each (timestamp, asset_id) group of rows IS the full snapshot
# at that instant -- no diff replay needed.
# ---------------------------------------------------------------------------

def build_timelines(slug):
    """Returns {asset_id: [(ts, snapshot_dict), ...]} sorted by ts, plus
    per-asset market metadata.

    Snapshot boundaries come from FILE ORDER, not from the timestamp.
    multi_recorder writes one snapshot as a contiguous block of rows --
    BID level 1..N then ASK level 1..M -- through a single queue and a
    single writer, so a snapshot's rows are always contiguous and its level
    numbers always ascend within a side.

    This matters because the recorder truncates its timestamps to
    milliseconds, and during a burst it can emit many genuinely different
    book states inside one millisecond (up to 13 observed). Grouping rows
    by (timestamp, asset_id) therefore merges several distinct snapshots
    into one corrupt book with duplicated, conflicting levels -- and loses
    every tick but one. On one game that collapsed 38,584 real snapshots
    down to 17,291. The state machine below splits them back apart:

        new snapshot  <=  a BID row arrives after ASK rows
                      or  a row's level does not advance within its side
                      or  the timestamp changed

    Snapshots that share a truncated millisecond keep their real recorded
    order and are nudged apart by 1 microsecond each (see below), so every
    tick survives as its own distinct point on the time axis.
    """
    path = os.path.join(SPLIT_DIR, f"{slug}.csv")
    meta = {}
    timelines = defaultdict(list)
    open_snap = {}  # asset_id -> in-progress snapshot

    def finish(asset_id, c):
        bids = sorted(c["bids"], key=lambda x: -x[0])
        asks = sorted(c["asks"], key=lambda x: x[0])
        timelines[asset_id].append((c["ts"], {
            "bids": bids,
            "asks": asks,
            "best_bid": bids[0][0] if bids else None,
            "best_ask": asks[0][0] if asks else None,
        }))

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asset_id = row["asset_id"]
            ts = parse_ts(row["timestamp_utc"])
            side = row["side"]
            level = int(row["level"])
            price = float(row["price"])
            size = float(row["size"])

            c = open_snap.get(asset_id)
            if c is None:
                starts_new = True
            elif c["ts"] != ts:
                starts_new = True
            elif side == "BID" and c["last_side"] == "ASK":
                starts_new = True
            elif side == c["last_side"] and level <= c["last_level"]:
                starts_new = True
            else:
                starts_new = False

            if starts_new:
                if c is not None:
                    finish(asset_id, c)
                c = {"ts": ts, "bids": [], "asks": [], "last_side": None, "last_level": 0}
                open_snap[asset_id] = c

            (c["bids"] if side == "BID" else c["asks"]).append((price, size))
            c["last_side"] = side
            c["last_level"] = level

            if asset_id not in meta:
                meta[asset_id] = {
                    "market_type": row["market_type"],
                    "market_question": row["market_question"],
                    "line": row["line"],
                    "outcome": row["outcome"],
                    "condition_id": row["condition_id"],
                }

    for asset_id, c in open_snap.items():
        finish(asset_id, c)

    # Separate snapshots that share a truncated-to-ms timestamp by nudging
    # each successive one 1us forward. The recorder truncated real sub-ms
    # ordering away; file order preserves that ordering, so this restores
    # distinct, strictly-increasing tick times without inventing or
    # dropping a single snapshot. Max drift is (collisions x 1us) -- with
    # 13 snapshots in the worst observed millisecond, under 13us.
    for asset_id, tl in timelines.items():
        tl.sort(key=lambda x: x[0])  # stable: preserves file order within a ts
        prev = None
        for i, (ts, snap) in enumerate(tl):
            if prev is not None and ts <= prev:
                ts = round(prev + 1e-6, 6)
            tl[i] = (ts, snap)
            prev = ts

    return timelines, meta


# ---------------------------------------------------------------------------
# live_recorder.py input (data/live/books.jsonl)
#
# Far simpler than the CSV path: every line is already one complete,
# self-contained book snapshot, so there is no one-row-per-level state machine
# and no snapshot-boundary ambiguity to recover. Order comes from the
# recorder's monotonic `seq`; event time is the exchange's own `timestamp`.
# ---------------------------------------------------------------------------

def split_books_jsonl(path=BOOKS_JSONL_PATH, out_dir=JSONL_SPLIT_DIR):
    """One streaming pass, splitting books.jsonl into one file per game."""
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    n = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    slug = json.loads(line)["slug"]
                except (json.JSONDecodeError, KeyError):
                    continue
                fh = files.get(slug)
                if fh is None:
                    fh = files[slug] = open(os.path.join(out_dir, f"{slug}.jsonl"),
                                             "w", encoding="utf-8")
                fh.write(line + "\n")
                n += 1
                if n % 500_000 == 0:
                    print(f"[split]   {n:,} records...")
    finally:
        for fh in files.values():
            fh.close()
    print(f"[split] {n:,} records -> {len(files)} game file(s) in {out_dir}")
    return sorted(files)


def build_timelines_jsonl(slug, split_dir=JSONL_SPLIT_DIR):
    """Returns (timelines, meta) in exactly the shape build_timelines gives,
    so every downstream step is shared between the two input formats."""
    path = os.path.join(split_dir, f"{slug}.jsonl")
    meta = {}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
            aid = r["asset_id"]
            if aid not in meta:
                meta[aid] = {
                    "market_type": r.get("market_type"),
                    "market_question": r.get("market_question") or "",
                    "line": r.get("line"),
                    "outcome": r.get("outcome"),
                    "condition_id": r.get("condition_id"),
                }

    rows.sort(key=lambda r: r.get("seq", 0))

    timelines = defaultdict(list)
    for r in rows:
        bids = sorted(((float(p), float(s)) for p, s in r.get("bids", [])), key=lambda x: -x[0])
        asks = sorted(((float(p), float(s)) for p, s in r.get("asks", [])), key=lambda x: x[0])
        timelines[r["asset_id"]].append((r["ts"] / 1000.0, {
            "bids": bids,
            "asks": asks,
            "best_bid": bids[0][0] if bids else None,
            "best_ask": asks[0][0] if asks else None,
        }))

    # Same 1us separation as the CSV path. The exchange clock is only
    # millisecond-resolution, so distinct snapshots can share a timestamp;
    # `seq` already fixed their order, this just makes the time axis strictly
    # increasing. Nothing is merged or dropped.
    for aid, tl in timelines.items():
        prev = None
        for i, (ts, snap) in enumerate(tl):
            if prev is not None and ts <= prev:
                ts = round(prev + 1e-6, 6)
            tl[i] = (ts, snap)
            prev = ts

    return timelines, meta


def build_runs_exact(tl, start_ts, end_ts):
    """Turns one asset's exact (ts, snapshot) timeline into runs covering
    [t_start, t_end) with NO grid snapping -- every run boundary is a real
    recorded change timestamp. This is a lossless, run-length-encoded
    reconstruction: no tick is skipped, none is duplicated, none is rounded.
    """
    empty_snap = {"bids": [], "asks": [], "best_bid": None, "best_ask": None}
    ts_list = [t for t, _ in tl]

    idx = bisect.bisect_right(ts_list, start_ts) - 1
    cur_snap = tl[idx][1] if idx >= 0 else empty_snap
    cur_start = start_ts

    runs = []
    for ts, snap in tl[idx + 1:]:
        if ts > end_ts:
            break
        if ts <= cur_start:
            cur_snap = snap
            continue
        runs.append({"t_start": round(cur_start, 6), "t_end": round(ts, 6), **cur_snap})
        cur_start = ts
        cur_snap = snap
    # Close out the final run. If the asset's last snapshot lands exactly on
    # end_ts (it does for whichever asset holds the globally-latest tick),
    # a further run would have zero length -- extend the previous one instead
    # of emitting an empty interval.
    if round(cur_start, 6) < round(end_ts, 6) or not runs:
        runs.append({"t_start": round(cur_start, 6), "t_end": round(end_ts, 6), **cur_snap})
    else:
        runs[-1] = {**runs[-1], "t_end": round(end_ts, 6), **cur_snap}

    return runs


def build_tick_times(timelines, start_ts, end_ts, grid_interval=0.1):
    """Shared, match-wide tick axis: a uniform grid_interval baseline PLUS
    every real change timestamp, merged. Grid alone can jump from one real
    change to the next real change with nothing in between during a long
    quiet stretch (the slider then has to leap minutes at a time); real
    change timestamps alone drop back to whatever the natural event rate
    is, which is far coarser than grid_interval during quiet stretches and
    finer than it during a burst. Merging both means: never coarser than
    grid_interval, and never coarser than what actually happened -- a real
    change that lands between two grid points still gets its own exact
    tick rather than being rounded onto the grid.
    """
    times = {round(start_ts, 6), round(end_ts, 6)}
    n_steps = int((end_ts - start_ts) / grid_interval) + 1
    for k in range(n_steps + 1):
        t = start_ts + k * grid_interval
        if t > end_ts:
            break
        times.add(round(t, 6))
    for tl in timelines.values():
        for ts, _ in tl:
            if start_ts <= ts <= end_ts:
                times.add(round(ts, 6))
    return sorted(times)


def build_game(slug, timelines, meta, window, window_mode="game"):
    all_ts = [ts for tl in timelines.values() for ts, _ in tl]
    if not all_ts:
        raise SystemExit(f"No events found for {slug}; nothing to rebuild.")

    if window and window_mode == "game":
        # Trim to kickoff -> actual game end (last completed play, from the
        # MLB Stats API -- NOT market close, which can trail the final out).
        start_ts, end_ts = window["kickoff_ts"], window["game_end_ts"]
    else:
        start_ts, end_ts = min(all_ts), max(all_ts)

    asset_ids = sorted(timelines.keys())
    runs = {aid: build_runs_exact(timelines[aid], start_ts, end_ts) for aid in asset_ids}
    tick_times = build_tick_times(timelines, start_ts, end_ts)

    # Verification: every real recorded change that falls inside the window
    # must appear on the tick axis. Nothing in-window is ever dropped -- the
    # only ticks the axis adds are the 100ms grid points.
    in_window = set()
    for tl in timelines.values():
        for ts, _ in tl:
            if start_ts <= ts <= end_ts:
                in_window.add(round(ts, 6))
    tick_set = set(tick_times)
    missing = in_window - tick_set
    if missing:
        raise SystemExit(
            f"{slug}: BUG -- {len(missing)} real in-window changes missing from the tick axis"
        )
    n_before = sum(1 for tl in timelines.values() for ts, _ in tl if ts < start_ts)
    n_after = sum(1 for tl in timelines.values() for ts, _ in tl if ts > end_ts)

    return {
        "event_slug": slug,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "kickoff_ts": window.get("kickoff_ts"),
        "game_end_ts": window.get("game_end_ts"),
        "game_status": window.get("status"),
        "tick_times": tick_times,
        "n_real_ticks_in_window": len(in_window),
        "n_real_ticks_before_window": n_before,
        "n_real_ticks_after_window": n_after,
        "assets": meta,
        "runs": runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None, help="subset of event_slugs; default: every game in data/depth_by_game or game_windows.json")
    ap.add_argument("--window", choices=("game", "full"), default="game",
                     help="'game' (default) trims to kickoff -> actual final out; "
                          "'full' keeps every recorded tick including the long pregame "
                          "session, which for most games holds MORE ticks than the game itself.")
    ap.add_argument("--depth", type=int, default=0,
                     help="max price levels per side to keep; 0 (default) keeps FULL depth. "
                          "Books here reach 88 levels/side (median ~30), so any small cap "
                          "silently truncates most of the real book.")
    ap.add_argument("--source", choices=("auto", "jsonl", "csv"), default="auto",
                     help="'jsonl' reads live_recorder.py output (data/live/books.jsonl); "
                          "'csv' reads the old multi_recorder.py depth CSV; "
                          "'auto' (default) prefers books.jsonl when it exists.")
    args = ap.parse_args()

    source = args.source
    if source == "auto":
        source = "jsonl" if os.path.exists(BOOKS_JSONL_PATH) else "csv"
    print(f"[source] {source}"
          f" ({BOOKS_JSONL_PATH if source == 'jsonl' else DEPTH_CSV_PATH})")

    windows = load_game_windows()

    if source == "jsonl":
        available = split_books_jsonl()
        slugs = args.games or available
        missing = [s for s in slugs if s not in available]
        if missing:
            raise SystemExit(f"no records in books.jsonl for: {', '.join(missing)}")
        load_timelines = build_timelines_jsonl
    else:
        if args.games:
            slugs = args.games
        elif windows:
            slugs = sorted(windows.keys())
        else:
            with open(DEPTH_CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                slug_idx = header.index("event_slug")
                slugs = sorted({row[slug_idx] for row in reader})
        split_depth_by_game(slugs)
        load_timelines = build_timelines

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for slug in slugs:
        window = windows.get(slug, {})
        if args.window == "game" and not window:
            print(f"\n[{slug}] no entry in game_windows.json -- keeping the full "
                  f"recorded span (run the MLB Stats API fetch after the game to trim)")
        timelines, meta = load_timelines(slug)
        n_points = sum(len(tl) for tl in timelines.values())
        max_lvls = max((max(len(s["bids"]), len(s["asks"])) for tl in timelines.values() for _, s in tl), default=0)
        print(f"\n[{slug}] {len(timelines)} assets, {n_points:,} real recorded snapshots "
              f"(max {max_lvls} levels/side)")

        # Optionally cap depth per snapshot (0 = keep the full recorded book)
        if args.depth:
            for aid, tl in timelines.items():
                for ts, snap in tl:
                    snap["bids"] = snap["bids"][: args.depth]
                    snap["asks"] = snap["asks"][: args.depth]

        out = build_game(slug, timelines, meta, window, window_mode=args.window)
        n_real = out["n_real_ticks_in_window"]
        n_total = len(out["tick_times"])
        if window and args.window == "game":
            print(f"  window: {(out['end_ts']-out['start_ts'])/60:.1f} min "
                  f"(kickoff -> actual final out, status={window['status']})")
        else:
            print(f"  window: {(out['end_ts']-out['start_ts'])/60:.1f} min "
                  f"(full recorded span, untrimmed)")
        print(f"  ticks: {n_total:,} total = {n_real:,} real recorded (all preserved) "
              f"+ {n_total - n_real:,} added 100ms grid points")
        print(f"  outside window (pregame/postgame, not in output): "
              f"{out['n_real_ticks_before_window']:,} before, {out['n_real_ticks_after_window']:,} after")

        out_path = os.path.join(OUTPUT_DIR, f"{slug}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
