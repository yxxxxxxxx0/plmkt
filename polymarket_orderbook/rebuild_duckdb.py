"""
DuckDB-powered rebuild of exact, tick-level order book histories from
data/live/by_game/<slug>.jsonl (live_recorder.py output).

Produces byte-for-byte-equivalent output to rebuild_multi.py's jsonl path
(same data/uniform/<slug>.json shape), but the load is done with DuckDB +
Arrow + numpy instead of a pure-Python line-by-line json.loads loop.

Why this exists: build_timelines_jsonl() in rebuild_multi.py calls
json.loads() once per recorded snapshot, then float() on every price/size in
every bid/ask level, then sorted() on every book. For a single busy game
that's 4-6M snapshot lines x ~70 levels/side, i.e. ~300-600M individual
float() conversions in the CPython interpreter -- minutes just to load one
game, before any of the actual run-building work. This module replaces only
that load step:

  1. DuckDB's read_ndjson() parses the whole file in C++ in one vectorized
     pass and sorts by (asset_id, seq) -- ~10-15s for a 4GB+ file.
  2. The bid/ask price and size arrays come back as Arrow ListArrays; calling
     .flatten() on those (not .to_pylist()!) gives one flat numpy float64
     array per column with zero per-row Python object creation. (.to_pylist()
     on a nested list column is exactly as slow as the pure-Python path it's
     replacing -- avoid it.)
  3. Per-asset run construction (identical semantics to
     rebuild_multi.build_runs_exact) works directly off numpy slices via
     np.stack, never converting a level to a Python tuple.
  4. Output is written with orjson (option=OPT_SERIALIZE_NUMPY), which
     serializes numpy arrays straight from C -- no .tolist() pass either.

Correctness is not asserted, it is tested: test_rebuild_parity.py runs both
this module and rebuild_multi.py over the same fixture and compares every run
boundary, every price level and every tick. (validate_rebuild.py cannot do this
job here -- its COVERAGE check reads data/depth_by_game/*.csv, which only the
retired multi_recorder.py ever wrote.)

Measured on mlb-col-wsh-2026-08-27 (4.32 GB, 4,389,164 records, 34 assets):
rebuild_multi.py had not finished in 20+ minutes; this runs end to end in
~145s, of which ~105s is load and ~40s is serialize+write.

CAVEAT -- output size. This preserves full fidelity, and full fidelity is
large: that game produces a 2.67 GB data/uniform/*.json. The cause is that
build_runs_exact emits one run per real change and each run carries the entire
book, while in this data only ~1 of ~23 price levels actually differs between
consecutive snapshots. So ~95% of the bytes are re-serialized unchanged levels.
A --depth cap exists but is both lossy and ineffective here (depth 20 still
keeps 77% of levels); the real fix is a format change (delta/RLE per level, or
a binary/Parquet payload). That is a viewer-affecting design decision and is
deliberately NOT made here.

Usage:
    python rebuild_duckdb.py mlb-col-wsh-2026-08-27
    python rebuild_duckdb.py --all
"""

import argparse
import os
import sys
import time

import duckdb
import numpy as np
import orjson

import rebuild_multi as rm  # reuse load_game_windows, build_tick_times, paths

JSONL_SPLIT_DIR = rm.JSONL_SPLIT_DIR
OUTPUT_DIR = rm.OUTPUT_DIR


# read_ndjson() only parses the fields named in `columns`, and the bid/ask
# arrays dominate that cost: on the 4.3 GB mlb-col-wsh file a scalar-only pass
# takes ~5s while a pass that also parses bids/asks takes ~35s. So the metadata
# query deliberately uses a NARROW column list -- it is a second pass over the
# file, but a cheap one, and it keeps the wide table free of five VARCHAR
# columns that would otherwise be materialised 4.4M times over.
# `line` must be DOUBLE, not VARCHAR: it is a JSON number (or null), and
# rebuild_multi carries it through json.loads as a float. Declaring it VARCHAR
# silently turns 3.5 into "3.5" in data/uniform/*.json, which the viewer would
# then compare or sort as a string.
SCALAR_COLS = ("'asset_id':'VARCHAR','market_type':'VARCHAR',"
               "'market_question':'VARCHAR','line':'DOUBLE',"
               "'outcome':'VARCHAR','condition_id':'VARCHAR'")

BOOK_COLS = ("'seq':'BIGINT','ts':'BIGINT','asset_id':'VARCHAR',"
             "'bids':'DOUBLE[2][]','asks':'DOUBLE[2][]'")


def load_columns(slug, split_dir=JSONL_SPLIT_DIR):
    """Parse the game's jsonl, sorted by (asset_id, seq), into flat numpy
    arrays (no per-row Python objects) plus per-asset market metadata.

    Metadata is verified constant per asset in the recorder's output, so
    any_value() over the group is exact, not a sample.
    """
    path = os.path.join(split_dir, f"{slug}.jsonl").replace("\\", "/")
    con = duckdb.connect()

    meta_rows = con.sql(f"""
        SELECT asset_id, any_value(market_type) AS market_type,
               any_value(market_question) AS market_question,
               any_value(line) AS line, any_value(outcome) AS outcome,
               any_value(condition_id) AS condition_id
        FROM read_ndjson('{path}', columns={{{SCALAR_COLS}}})
        GROUP BY asset_id
        ORDER BY asset_id
    """).fetchall()
    # ORDER BY, and hence a deterministic `assets` key order in the output:
    # GROUP BY alone returns rows in whatever order the hash aggregate produced,
    # which varies run to run. Sorting also matches the `runs` key order, since
    # the main query sorts by asset_id too.
    meta = {
        aid: {
            "market_type": mt, "market_question": mq or "",
            "line": line, "outcome": outcome, "condition_id": cid,
        }
        for aid, mt, mq, line, outcome, cid in meta_rows
    }

    tbl = con.sql(f"""
        SELECT asset_id, ts,
               list_transform(bids, x -> x[1]) AS bid_px,
               list_transform(bids, x -> x[2]) AS bid_sz,
               list_transform(asks, x -> x[1]) AS ask_px,
               list_transform(asks, x -> x[2]) AS ask_sz
        FROM read_ndjson('{path}', columns={{{BOOK_COLS}}})
        ORDER BY asset_id, seq
    """).arrow().read_all()

    asset_ids = tbl.column("asset_id").to_pylist()  # scalar column: cheap
    ts = tbl.column("ts").to_numpy(zero_copy_only=False)

    def flat(col):
        arr = tbl.column(col).combine_chunks()
        return arr.flatten().to_numpy(zero_copy_only=False), arr.offsets.to_numpy()

    bid_px, bid_off = flat("bid_px")
    bid_sz, _ = flat("bid_sz")
    ask_px, ask_off = flat("ask_px")
    ask_sz, _ = flat("ask_sz")

    return asset_ids, ts, bid_px, bid_sz, bid_off, ask_px, ask_sz, ask_off, meta


def peak_rss_gb():
    """Peak working set of this process, in GB. Worth printing on every run:
    the flat level arrays, the Arrow table they were copied from, the runs dict
    (which holds numpy VIEWS, so it pins those arrays alive) and the serialized
    payload are all live simultaneously near the end, and this machine is also
    running live_recorder.py against the same disk."""
    if sys.platform != "win32":
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        except Exception:
            return float("nan")
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    # argtypes/restype are required: without them GetCurrentProcess()'s HANDLE
    # is truncated to a 32-bit int and the call silently reports 0.
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]

    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return float("nan")
    # Peak commit as well as peak working set: Windows can trim the working set
    # under pressure, which would under-report how much this actually needed.
    return max(c.PeakWorkingSetSize, c.PeakPagefileUsage) / 1e9


def asset_boundaries(asset_ids):
    """[(asset_id, lo, hi), ...] contiguous ranges; input is sorted by asset_id."""
    out = []
    n = len(asset_ids)
    if n == 0:
        return out
    lo = 0
    cur = asset_ids[0]
    for i in range(1, n):
        if asset_ids[i] != cur:
            out.append((cur, lo, i))
            lo = i
            cur = asset_ids[i]
    out.append((cur, lo, n))
    return out


def dedup_nudge(ts_ms_slice):
    """Same 1us-per-collision nudge as rebuild_multi's loaders, vectorized.

    The reference implementation nudges whenever `ts <= prev`, which also
    repairs timestamps that go BACKWARDS. The vectorized form below only ranks
    within runs of consecutive equal values, so it is equivalent only if `ts`
    is non-decreasing within an asset once ordered by `seq`.

    That holds on mlb-col-wsh-2026-08-27 (4,389,164 records, zero backwards
    steps) but NOT universally -- another game in the same recording session
    had a single backwards step. So the property is checked per asset, and the
    rare asset that violates it falls back to the reference's exact sequential
    loop rather than being silently mis-nudged. (Globally, across interleaved
    assets, ts goes backwards constantly -- which is why the nudge is
    per-asset in the first place.)
    """
    ts_sec = ts_ms_slice.astype(np.float64) / 1000.0
    n = len(ts_sec)
    if n <= 1:
        return ts_sec

    if np.any(np.diff(ts_sec) < 0):
        # Exactly rebuild_multi's loop, including its rounding.
        out = np.empty(n, dtype=np.float64)
        prev = None
        for i in range(n):
            t = ts_sec[i]
            if prev is not None and t <= prev:
                t = round(prev + 1e-6, 6)
            out[i] = t
            prev = t
        return out

    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = ts_sec[1:] != ts_sec[:-1]
    group_id = np.cumsum(is_new) - 1
    idx = np.arange(n)
    first_idx_of_group = idx[is_new]
    rank_in_run = idx - first_idx_of_group[group_id]
    return ts_sec + rank_in_run * 1e-6


def build_asset_runs(lo, ts_adj, bid_px_off, bid_px, bid_sz, ask_px_off, ask_px, ask_sz,
                      start_ts, end_ts, depth=0):
    """numpy-native equivalent of rebuild_multi.build_runs_exact: same
    boundary/edge-case semantics, but every level array is a numpy slice
    (via np.stack), never a Python list of tuples."""
    empty = np.empty((0, 2), dtype=np.float64)

    def snap_at(row):
        if row is None:
            return empty, empty
        bp0, bp1 = bid_px_off[row], bid_px_off[row + 1]
        ap0, ap1 = ask_px_off[row], ask_px_off[row + 1]
        if depth:
            bp1 = min(bp1, bp0 + depth)
            ap1 = min(ap1, ap0 + depth)
        bids = np.stack((bid_px[bp0:bp1], bid_sz[bp0:bp1]), axis=1) if bp1 > bp0 else empty
        asks = np.stack((ask_px[ap0:ap1], ask_sz[ap0:ap1]), axis=1) if ap1 > ap0 else empty
        return bids, asks

    def make_run(t_start, t_end, row):
        bids, asks = snap_at(row)
        return {
            "t_start": round(t_start, 6), "t_end": round(t_end, 6),
            "bids": bids, "asks": asks,
            "best_bid": float(bids[0, 0]) if len(bids) else None,
            "best_ask": float(asks[0, 0]) if len(asks) else None,
        }

    idx = np.searchsorted(ts_adj, start_ts, side="right") - 1
    cur_row = (lo + idx) if idx >= 0 else None
    cur_start = start_ts

    j1 = np.searchsorted(ts_adj, end_ts, side="right")
    runs = []
    for j in range(max(idx + 1, 0), j1):
        ts = ts_adj[j]
        if ts <= cur_start:
            cur_row = lo + j
            continue
        runs.append(make_run(cur_start, ts, cur_row))
        cur_start = ts
        cur_row = lo + j

    # Close out the final run. When the asset's last snapshot lands exactly on
    # end_ts, a further run would have zero length -- extend the previous one
    # instead. Note this must keep runs[-1]'s ORIGINAL t_start (matching
    # rebuild_multi's `{**runs[-1], "t_end": ..., **cur_snap}`); rebuilding it
    # from cur_start would emit a zero-length [end_ts, end_ts] run and silently
    # drop the interval the previous run was covering.
    if round(cur_start, 6) < round(end_ts, 6) or not runs:
        runs.append(make_run(cur_start, end_ts, cur_row))
    else:
        runs[-1] = make_run(runs[-1]["t_start"], end_ts, cur_row)

    return runs


def prepare_game(slug, window, window_mode="game", split_dir=JSONL_SPLIT_DIR):
    """Everything except the runs: the loaded columns, the per-asset nudged
    time axes, the window, and the shared tick axis.

    Split out from build_game_duckdb so the writer can build and serialize the
    runs one asset at a time (see write_game_json). Computing the tick axis
    needs only the timestamps, which are ~35 MB in total -- it does NOT need
    the runs, which are the multi-GB part.
    """
    cols = load_columns(slug, split_dir=split_dir)
    asset_ids, ts = cols[0], cols[1]
    if not asset_ids:
        raise SystemExit(f"No events found for {slug}; nothing to rebuild.")

    boundaries = asset_boundaries(asset_ids)
    ts_adj_by_asset = {aid: dedup_nudge(ts[lo:hi]) for aid, lo, hi in boundaries}

    if window and window_mode == "game":
        start_ts, end_ts = window["kickoff_ts"], window["game_end_ts"]
    else:
        start_ts = min(a[0] for a in ts_adj_by_asset.values())
        end_ts = max(a[-1] for a in ts_adj_by_asset.values())

    n_before = n_after = 0
    # Distinct real in-window change timestamps, matching rebuild_multi's
    # `in_window` set: two assets that changed in the same microsecond are one
    # tick on the shared axis, so this is a set, not a running count.
    real_set = set()
    for aid, lo, hi in boundaries:
        ts_adj = ts_adj_by_asset[aid]
        n_before += int(np.sum(ts_adj < start_ts))
        n_after += int(np.sum(ts_adj > end_ts))
        in_win = ts_adj[(ts_adj >= start_ts) & (ts_adj <= end_ts)]
        real_set.update(np.round(in_win, 6).tolist())

    # Shared tick axis = 100ms grid baseline merged with every real change,
    # same as rebuild_multi.build_tick_times.
    tick_set = {round(start_ts, 6), round(end_ts, 6)}
    grid_interval = 0.1
    n_steps = int((end_ts - start_ts) / grid_interval) + 1
    for k in range(n_steps + 1):
        t = start_ts + k * grid_interval
        if t > end_ts:
            break
        tick_set.add(round(t, 6))
    tick_set |= real_set

    missing = real_set - tick_set
    if missing:
        raise SystemExit(
            f"{slug}: BUG -- {len(missing)} real in-window changes missing from the tick axis"
        )

    header = {
        "event_slug": slug,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "kickoff_ts": window.get("kickoff_ts"),
        "game_end_ts": window.get("game_end_ts"),
        "game_status": window.get("status"),
        "tick_times": sorted(tick_set),
        "n_real_ticks_in_window": len(real_set),
        "n_real_ticks_before_window": n_before,
        "n_real_ticks_after_window": n_after,
        "assets": cols[8],
    }
    return header, boundaries, ts_adj_by_asset, cols, start_ts, end_ts


def build_game_duckdb(slug, window, window_mode="game", split_dir=JSONL_SPLIT_DIR, depth=0):
    """Whole-game dict, runs included. Used by test_rebuild_parity.py, which
    needs the full structure in memory to diff it against rebuild_multi. For
    actually writing a big game, prefer write_game_json -- this holds every run
    for every asset live at once."""
    header, boundaries, ts_adj, cols, start_ts, end_ts = prepare_game(
        slug, window, window_mode=window_mode, split_dir=split_dir)
    _, _, bid_px, bid_sz, bid_off, ask_px, ask_sz, ask_off, _ = cols

    runs = {
        aid: build_asset_runs(lo, ts_adj[aid], bid_off, bid_px, bid_sz,
                              ask_off, ask_px, ask_sz, start_ts, end_ts, depth=depth)
        for aid, lo, hi in boundaries
    }
    return {**header, "runs": runs}


def write_game_json(slug, window, out_path, window_mode="game",
                    split_dir=JSONL_SPLIT_DIR, depth=0):
    """Same JSON as build_game_duckdb + orjson.dumps, written incrementally.

    Building the whole dict and serializing it in one shot peaked at 15.7 GB on
    the largest game -- on a 16.5 GB machine that only survived by spilling to
    the pagefile, while live_recorder.py was running against the same disk. Two
    terms dominated and both are avoidable: every asset's runs held live at
    once, and the single ~2.7 GB `bytes` object orjson returns.

    So the runs for one asset are built, serialized, written and dropped before
    the next asset starts. The output is byte-identical -- key order is the
    same, and orjson emits the same encoding either way.
    """
    header, boundaries, ts_adj, cols, start_ts, end_ts = prepare_game(
        slug, window, window_mode=window_mode, split_dir=split_dir)
    _, _, bid_px, bid_sz, bid_off, ask_px, ask_sz, ask_off, _ = cols

    n_bytes = 0
    n_runs = 0
    with open(out_path, "wb") as f:
        # OPT_SERIALIZE_NUMPY is needed here too, not just for the runs: in
        # window="full" mode start_ts/end_ts come from the numpy time axis and
        # are numpy.float64, which orjson rejects by default.
        head = orjson.dumps(header, option=orjson.OPT_SERIALIZE_NUMPY)
        assert head.endswith(b"}")
        f.write(head[:-1])          # drop the closing brace, append "runs"
        f.write(b',"runs":{')
        n_bytes += len(head) + 9
        for i, (aid, lo, hi) in enumerate(boundaries):
            asset_runs = build_asset_runs(lo, ts_adj[aid], bid_off, bid_px, bid_sz,
                                          ask_off, ask_px, ask_sz, start_ts, end_ts,
                                          depth=depth)
            n_runs += len(asset_runs)
            chunk = (b"," if i else b"") + orjson.dumps(aid) + b":" + \
                    orjson.dumps(asset_runs, option=orjson.OPT_SERIALIZE_NUMPY)
            f.write(chunk)
            n_bytes += len(chunk)
            del asset_runs, chunk
        f.write(b"}}")
        n_bytes += 2

    return header, n_bytes, n_runs


def rebuild_one(slug, windows, window_mode="game", depth=0):
    t0 = time.time()
    window = windows.get(slug, {})
    if not window:
        print(f"[{slug}] no entry in game_windows.json -- keeping full recorded span")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{slug}.json")
    out, n_bytes, n_runs = write_game_json(slug, window, out_path,
                                           window_mode=window_mode, depth=depth)
    n_real = out["n_real_ticks_in_window"]
    n_total = len(out["tick_times"])
    print(f"[{slug}] {len(out['assets'])} assets, {n_runs:,} runs")
    if window:
        print(f"  window: {(out['end_ts']-out['start_ts'])/60:.1f} min "
              f"(kickoff -> final out, status={window['status']})")
    else:
        print(f"  window: {(out['end_ts']-out['start_ts'])/60:.1f} min (full recorded span, untrimmed)")
    print(f"  ticks: {n_total:,} total = {n_real:,} real recorded + {n_total - n_real:,} added 100ms grid points")
    print(f"  outside window: {out['n_real_ticks_before_window']:,} before, {out['n_real_ticks_after_window']:,} after")

    print(f"  wrote {out_path} ({n_bytes/1e6:.1f} MB)")
    print(f"  total {time.time()-t0:.1f}s, peak RSS {peak_rss_gb():.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", help="event_slugs to rebuild")
    ap.add_argument("--all", action="store_true", help="rebuild every game found in data/live/by_game")
    ap.add_argument("--window", choices=("game", "full"), default="game")
    ap.add_argument("--depth", type=int, default=0,
                    help="max price levels per side to keep; 0 (default) keeps FULL depth. "
                         "Books here reach ~90 levels/side, so any small cap silently "
                         "truncates most of the real book -- it is opt-in for a reason.")
    args = ap.parse_args()

    windows = rm.load_game_windows()

    if args.all:
        slugs = sorted(
            f[:-6] for f in os.listdir(JSONL_SPLIT_DIR) if f.endswith(".jsonl")
        )
    elif args.games:
        slugs = args.games
    else:
        raise SystemExit("pass one or more event_slugs, or --all")

    for slug in slugs:
        rebuild_one(slug, windows, window_mode=args.window, depth=args.depth)


if __name__ == "__main__":
    main()
