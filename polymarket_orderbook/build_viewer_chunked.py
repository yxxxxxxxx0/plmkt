"""
Builds a FULL-FIDELITY viewer dataset: every recorded tick, every price level,
nothing resampled and nothing capped.

Why this exists
---------------
build_match_index.py delivers one JSON per match. At full fidelity that is
~2.6 GB for a single game (4.2M runs x ~46 levels), which no browser will
fetch. The previous answer was to resample onto a 1-second grid -- which shrank
it to 70 MB but threw away the microsecond-level change history that is the
entire point of this dataset.

So this changes the DELIVERY, not the DATA. The same runs are split into pieces
small enough to fetch on demand:

  index.json              manifest: matches, markets, tokens, chunk boundaries.
                          No per-tick bulk. A few hundred KB.

  <slug>/ticks.json       the match's complete tick axis, delta-encoded.
                          ~1.8M ticks -> ~7 MB instead of ~25 MB raw.

  <slug>/<asset>/series.json
                          best_bid/best_ask for EVERY run of that token,
                          columnar and delta-encoded. This is what the price
                          chart draws, so the chart shows every tick without
                          loading a single order book. ~2-4 MB per token.

  <slug>/<asset>/cNNNN.json
                          full order books for every run in one time chunk.
                          Fetched only when the playhead is inside that chunk.
                          ~10 MB each at the default 300s chunk.

The viewer therefore has every tick available at all times, and downloads a few
MB to show any given moment instead of 2.6 GB to show the first one.

Usage:
    python build_viewer_chunked.py
    python build_viewer_chunked.py --chunk-seconds 300 --games mlb-col-wsh-2026-08-27

Source of truth
---------------
Built from data/live/by_game/<slug>.jsonl via rebuild_duckdb, NOT by re-parsing
data/uniform/<slug>.json. Both produce identical runs (test_rebuild_parity.py
enforces that), but streaming 2.6 GB of JSON back in with ijson costs ~10
minutes per match, while rebuild_duckdb reloads the source in ~2. The uniform
files remain the archive; this just doesn't take the slow road to the same
numbers.
"""

import argparse
import glob
import json
import os
import shutil

import orjson

import build_match_index as bmi
import rebuild_duckdb as rd
import rebuild_multi as rm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UNIFORM_DIR = os.path.join(DATA_DIR, "uniform")
OUT_DIR = os.path.join(DATA_DIR, "viewer_full")

US = 1_000_000  # times are carried as integer microseconds


def enc_times(times):
    """Delta-encode a sorted time axis losslessly.

    Times have microsecond resolution (the rebuild nudges collisions apart by
    1us), so integer microseconds are exact -- no float drift. Consecutive
    deltas are small, which is where the size win comes from: a raw
    '1787850300.123456,' is 18 bytes, a delta is usually 3-6.
    """
    if not times:
        return {"t0": 0, "d": []}
    base = int(round(times[0] * US))
    out = []
    prev = base
    for t in times[1:]:
        cur = int(round(t * US))
        out.append(cur - prev)
        prev = cur
    return {"t0": base, "d": out}


def enc_series(runs):
    """best_bid / best_ask for every run, columnar + delta-encoded.

    Prices are integer thousandths (Polymarket quotes in 0.001 increments);
    the encoding is verified to round-trip exactly and falls back to raw floats
    if any price ever fails that check, so this can never silently distort a
    quote. `null` (empty side) is carried as a JSON null.
    """
    ts = [r["t_start"] for r in runs]
    enc = enc_times(ts)

    def px(vals):
        out = []
        for v in vals:
            if v is None:
                out.append(None)
                continue
            iv = int(round(v * 1000))
            if abs(iv / 1000.0 - v) > 1e-9:
                return None  # not representable -> caller keeps raw floats
            out.append(iv)
        return out

    bb = px([r["best_bid"] for r in runs])
    ba = px([r["best_ask"] for r in runs])
    if bb is None or ba is None:
        return {"t0": enc["t0"], "d": enc["d"], "raw": True,
                "bb": [r["best_bid"] for r in runs],
                "ba": [r["best_ask"] for r in runs],
                "t_end": runs[-1]["t_end"] if runs else None}
    return {"t0": enc["t0"], "d": enc["d"], "bb": bb, "ba": ba,
            "t_end": runs[-1]["t_end"] if runs else None}


def chunk_index(start_ts, end_ts, chunk_seconds):
    """Uniform wall-clock chunk edges covering [start_ts, end_ts]."""
    edges = []
    t = start_ts
    while t < end_ts:
        edges.append(t)
        t += chunk_seconds
    edges.append(end_ts)
    return edges


def split_runs_into_chunks(runs, edges):
    """Assigns runs to chunks, repeating the run in force at each boundary.

    A run that straddles a boundary appears in BOTH chunks. That duplication is
    deliberate and necessary: each chunk has to be independently sufficient to
    answer "what was the book at time t" for any t inside it, including the
    instant at its start. Without it, scrubbing to the first tick of a chunk
    during a quiet stretch would find no run at all.
    """
    out = [[] for _ in range(len(edges) - 1)]
    if not runs:
        return out
    k = 0
    for r in runs:
        # advance past chunks that end before this run starts
        while k < len(out) - 1 and edges[k + 1] <= r["t_start"]:
            k += 1
        j = k
        while j < len(out) and edges[j] < r["t_end"]:
            out[j].append(r)
            j += 1
        if not out or (j == k and k < len(out)):
            out[k].append(r)
    return out


def build_one(slug, out_root, chunk_seconds, window, window_mode="game", event=None):
    # prepare_game returns exactly the header shape build_match expects
    # (assets / start_ts / end_ts / tick_times / n_real_ticks_*), plus the
    # loaded columns needed to build each asset's runs one at a time.
    header, boundaries, ts_adj, cols, start_ts, end_ts = rd.prepare_game(
        slug, window, window_mode=window_mode)
    _, _, bid_px, bid_sz, bid_off, ask_px, ask_sz, ask_off, _ = cols
    match = bmi.build_match(slug, header, event, grid=0)

    slug_dir = os.path.join(out_root, slug)
    if os.path.isdir(slug_dir):
        shutil.rmtree(slug_dir)
    os.makedirs(slug_dir, exist_ok=True)

    tick_times = header["tick_times"]
    with open(os.path.join(slug_dir, "ticks.json"), "wb") as f:
        f.write(orjson.dumps(enc_times(tick_times)))
    ticks_mb = os.path.getsize(os.path.join(slug_dir, "ticks.json")) / 1e6

    edges = chunk_index(start_ts, end_ts, chunk_seconds)

    total_runs = 0
    total_bytes = 0
    series_bytes = 0
    biggest_chunk = 0
    token_info = {}

    for asset_id, lo, hi in boundaries:
        if asset_id not in match["tokens"]:
            continue
        runs = rd.build_asset_runs(lo, ts_adj[asset_id], bid_off, bid_px, bid_sz,
                                   ask_off, ask_px, ask_sz, start_ts, end_ts, depth=0)
        adir = os.path.join(slug_dir, asset_id)
        os.makedirs(adir, exist_ok=True)

        spath = os.path.join(adir, "series.json")
        with open(spath, "wb") as f:
            f.write(orjson.dumps(enc_series(runs), option=orjson.OPT_SERIALIZE_NUMPY))
        sb = os.path.getsize(spath)
        series_bytes += sb

        parts = split_runs_into_chunks(runs, edges)
        counts = []
        for i, part in enumerate(parts):
            cpath = os.path.join(adir, f"c{i:04d}.json")
            with open(cpath, "wb") as f:
                f.write(orjson.dumps(part, option=orjson.OPT_SERIALIZE_NUMPY))
            n = os.path.getsize(cpath)
            total_bytes += n
            biggest_chunk = max(biggest_chunk, n)
            counts.append(len(part))
        total_runs += len(runs)
        token_info[asset_id] = {"n_runs": len(runs), "series_bytes": sb}
        del runs, parts

    entry = dict(match)
    entry.pop("tick_times", None)
    entry["chunk_seconds"] = chunk_seconds
    entry["chunk_edges"] = edges
    entry["n_chunks"] = len(edges) - 1
    entry["n_runs"] = total_runs

    # Written LAST, so its presence means this game is complete. A build can be
    # interrupted (kill, reboot, full disk) after some tokens are written; the
    # half-built directory is otherwise indistinguishable from a finished one,
    # and a resumed run would skip it and ship a game missing tokens.
    with open(os.path.join(slug_dir, "entry.json"), "wb") as f:
        f.write(orjson.dumps(entry))

    return entry, {
        "runs": total_runs, "chunk_mb": total_bytes / 1e6,
        "series_mb": series_bytes / 1e6, "ticks_mb": ticks_mb,
        "biggest_chunk_mb": biggest_chunk / 1e6, "n_chunks": len(edges) - 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--window", choices=("game", "full"), default="game",
                    help="'game' (default) keeps kickoff -> actual final out and "
                         "trims ONLY the pregame/postgame tails; 'full' keeps every "
                         "recorded tick. Neither resamples or caps anything.")
    ap.add_argument("--chunk-seconds", type=float, default=300.0,
                    help="wall-clock seconds of order-book data per chunk file")
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--force", action="store_true",
                    help="rebuild games that already have a completion marker")
    ap.add_argument("--merge", action="store_true",
                    help="keep manifest entries for games not rebuilt in this run. "
                         "Without it index.json lists only what this run built, so "
                         "building slate-by-slate would drop every earlier slate "
                         "from the viewer.")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip the Gamma API; use cached metadata only")
    args = ap.parse_args()

    cache = {}
    if os.path.exists(bmi.DISCOVERY_CACHE_PATH):
        with open(bmi.DISCOVERY_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

    windows = rm.load_game_windows()
    slugs = args.games or sorted(
        f[:-6] for f in os.listdir(rm.JSONL_SPLIT_DIR) if f.endswith(".jsonl"))
    print(f"Found {len(slugs)} game(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    grand = 0.0
    for slug in slugs:
        event = None
        if not args.no_fetch:
            try:
                event = bmi.fetch_event_cached(slug, cache)
            except Exception as e:
                print(f"  [warn] {slug}: Gamma fetch failed ({e})")
        else:
            event = cache.get(slug)

        marker = os.path.join(args.out_dir, slug, "entry.json")
        if os.path.exists(marker) and not args.force:
            with open(marker, "rb") as f:
                manifest.append(orjson.loads(f.read()))
            print(f"  {slug}: already built, skipping (--force to rebuild)")
            continue

        window = windows.get(slug, {})
        if not window:
            print(f"  [warn] {slug}: no game_windows entry -- keeping full recorded span")
        entry, st = build_one(slug, args.out_dir, args.chunk_seconds,
                              window, args.window, event)
        manifest.append(entry)
        grand += st["chunk_mb"] + st["series_mb"] + st["ticks_mb"]
        print(f"  {slug}: {entry['match_name']}")
        print(f"    {entry['n_ticks']:,} ticks (ALL real changes kept), "
              f"{st['runs']:,} runs, {len(entry['tokens'])} tokens")
        print(f"    ticks {st['ticks_mb']:.1f} MB | series {st['series_mb']:.1f} MB | "
              f"books {st['chunk_mb']:,.0f} MB in {st['n_chunks']} chunks/token "
              f"(largest chunk {st['biggest_chunk_mb']:.1f} MB)")

    with open(bmi.DISCOVERY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    mpath = os.path.join(args.out_dir, "index.json")
    if args.merge:
        # Rebuild the manifest from every completion marker present, not from
        # the previous index.json. index.json is written only at the end of a
        # run, so a run killed partway leaves finished games absent from it --
        # the markers are the durable record of what is actually on disk.
        built = {m["event_slug"] for m in manifest}
        found = 0
        for mk in sorted(glob.glob(os.path.join(args.out_dir, "*", "entry.json"))):
            with open(mk, "rb") as f:
                e = orjson.loads(f.read())
            if e["event_slug"] not in built:
                manifest.append(e)
                built.add(e["event_slug"])
                found += 1
        print(f"[merge] adopted {found} game(s) already on disk")
    manifest.sort(key=lambda m: (m.get("date") or "", m.get("match_name") or ""))
    with open(mpath, "wb") as f:
        f.write(orjson.dumps({"matches": manifest}))
    print(f"\nWrote {mpath} ({os.path.getsize(mpath)/1e6:.2f} MB manifest, "
          f"{len(manifest)} matches)")
    print(f"Total on disk: {grand:,.0f} MB -- full fidelity, fetched a few MB at a time")


if __name__ == "__main__":
    main()
