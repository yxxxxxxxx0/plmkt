"""
Correctness cross-check: rebuild_duckdb.py must produce exactly what
rebuild_multi.py's jsonl path produces.

validate_rebuild.py can't do this job for the jsonl recorder -- its COVERAGE
check reads data/depth_by_game/*.csv, which only the retired multi_recorder.py
ever wrote. So instead of validating against a file that doesn't exist, this
compares the fast implementation against the slow reference implementation
directly, on a fixture small enough that the slow one finishes.

It builds the fixture by taking the first N records of a real game file, so
the data is genuine (real collisions, real depth, real asset interleaving) --
just less of it.

Usage:
    python test_rebuild_parity.py
    python test_rebuild_parity.py --slug mlb-ari-sf-2026-08-27 --records 300000
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import rebuild_multi as rm
import rebuild_duckdb as rd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_fixture(slug, n_records, out_dir):
    """First n_records lines of the real game file -> out_dir/<slug>.jsonl"""
    src = os.path.join(rm.JSONL_SPLIT_DIR, f"{slug}.jsonl")
    dst = os.path.join(out_dir, f"{slug}.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(src, "r", encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            fo.write(line)
            n += 1
            if n >= n_records:
                break
    print(f"[fixture] {n:,} records -> {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
    return dst


def as_levels(x):
    """Reference gives list[tuple]; duckdb gives an (n,2) numpy array."""
    if isinstance(x, np.ndarray):
        return [(float(p), float(s)) for p, s in x]
    return [(float(p), float(s)) for p, s in x]


def cmp_float(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def compare(ref, fast, label):
    fails = []

    def bad(msg):
        fails.append(f"[{label}] {msg}")

    for k in ("event_slug", "game_status"):
        if ref[k] != fast[k]:
            bad(f"{k}: {ref[k]!r} != {fast[k]!r}")
    for k in ("start_ts", "end_ts", "kickoff_ts", "game_end_ts"):
        if not cmp_float(ref[k], fast[k], 1e-6):
            bad(f"{k}: {ref[k]} != {fast[k]}")

    for k in ("n_real_ticks_in_window", "n_real_ticks_before_window", "n_real_ticks_after_window"):
        if ref[k] != fast[k]:
            bad(f"{k}: {ref[k]:,} != {fast[k]:,}")

    if len(ref["tick_times"]) != len(fast["tick_times"]):
        bad(f"tick_times length: {len(ref['tick_times']):,} != {len(fast['tick_times']):,}")
        only_ref = set(ref["tick_times"]) - set(fast["tick_times"])
        only_fast = set(fast["tick_times"]) - set(ref["tick_times"])
        bad(f"  only in reference: {len(only_ref):,} e.g. {sorted(only_ref)[:5]}")
        bad(f"  only in duckdb:    {len(only_fast):,} e.g. {sorted(only_fast)[:5]}")
    else:
        for i, (a, b) in enumerate(zip(ref["tick_times"], fast["tick_times"])):
            if not cmp_float(a, b, 1e-9):
                bad(f"tick_times[{i}]: {a} != {b}")
                break

    if set(ref["assets"]) != set(fast["assets"]):
        bad(f"asset id sets differ: ref-only={set(ref['assets'])-set(fast['assets'])} "
            f"fast-only={set(fast['assets'])-set(ref['assets'])}")
    for aid in sorted(set(ref["assets"]) & set(fast["assets"])):
        if ref["assets"][aid] != fast["assets"][aid]:
            bad(f"assets[{aid[:12]}..] meta: {ref['assets'][aid]} != {fast['assets'][aid]}")

    for aid in sorted(set(ref["runs"]) & set(fast["runs"])):
        r_runs, f_runs = ref["runs"][aid], fast["runs"][aid]
        if len(r_runs) != len(f_runs):
            bad(f"runs[{aid[:12]}..] count: {len(r_runs):,} != {len(f_runs):,}")
            continue
        for i, (r, f) in enumerate(zip(r_runs, f_runs)):
            if not cmp_float(r["t_start"], f["t_start"], 1e-9) or not cmp_float(r["t_end"], f["t_end"], 1e-9):
                bad(f"runs[{aid[:12]}..][{i}] interval: "
                    f"[{r['t_start']},{r['t_end']}] != [{f['t_start']},{f['t_end']}]")
                break
            if not cmp_float(r["best_bid"], f["best_bid"]) or not cmp_float(r["best_ask"], f["best_ask"]):
                bad(f"runs[{aid[:12]}..][{i}] best: "
                    f"({r['best_bid']},{r['best_ask']}) != ({f['best_bid']},{f['best_ask']})")
                break
            rb, fb = as_levels(r["bids"]), as_levels(f["bids"])
            ra, fa = as_levels(r["asks"]), as_levels(f["asks"])
            if rb != fb:
                bad(f"runs[{aid[:12]}..][{i}] bids differ: {rb[:3]} != {fb[:3]} (len {len(rb)} vs {len(fb)})")
                break
            if ra != fa:
                bad(f"runs[{aid[:12]}..][{i}] asks differ: {ra[:3]} != {fa[:3]} (len {len(ra)} vs {len(fa)})")
                break

    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="mlb-col-wsh-2026-08-27")
    ap.add_argument("--records", type=int, default=250_000)
    ap.add_argument("--keep", action="store_true", help="keep the fixture directory")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="parity_")
    try:
        make_fixture(args.slug, args.records, tmp)
        windows = rm.load_game_windows()

        for mode in ("full", "game"):
            window = windows.get(args.slug, {})
            if mode == "game" and not window:
                print(f"\n=== window={mode}: SKIPPED (no game_windows entry for {args.slug}) ===")
                continue

            print(f"\n=== window={mode} ===")
            t = time.time()
            timelines, meta = rm.build_timelines_jsonl(args.slug, split_dir=tmp)
            ref = rm.build_game(args.slug, timelines, meta, window, window_mode=mode)
            t_ref = time.time() - t
            print(f"  reference (rebuild_multi): {t_ref:.1f}s")

            t = time.time()
            fast = rd.build_game_duckdb(args.slug, window, window_mode=mode, split_dir=tmp)
            t_fast = time.time() - t
            print(f"  duckdb    (rebuild_duckdb): {t_fast:.1f}s  ({t_ref/max(t_fast,1e-9):.1f}x)")

            fails = compare(ref, fast, mode)
            if fails:
                print(f"  MISMATCH ({len(fails)}):")
                for f in fails[:25]:
                    print(f"    {f}")
                return 1
            n_runs = sum(len(v) for v in ref["runs"].values())
            print(f"  OK -- {len(ref['assets'])} assets, {n_runs:,} runs, "
                  f"{len(ref['tick_times']):,} ticks match exactly")

            # The streaming writer must produce byte-identical JSON to
            # serializing the whole dict in one shot -- it exists purely to cut
            # peak memory, so any difference in its output is a bug.
            import orjson
            whole = orjson.dumps(fast, option=orjson.OPT_SERIALIZE_NUMPY)
            streamed_path = os.path.join(tmp, f"{args.slug}.streamed.json")
            rd.write_game_json(args.slug, window, streamed_path,
                               window_mode=mode, split_dir=tmp)
            with open(streamed_path, "rb") as f:
                streamed = f.read()
            if streamed != whole:
                print(f"  STREAM MISMATCH: {len(streamed):,} bytes vs {len(whole):,} bytes")
                for i, (a, b) in enumerate(zip(streamed, whole)):
                    if a != b:
                        print(f"    first difference at byte {i}: "
                              f"{streamed[max(0,i-60):i+60]!r} != {whole[max(0,i-60):i+60]!r}")
                        break
                return 1
            print(f"  OK -- streaming writer byte-identical ({len(whole)/1e6:.1f} MB)")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"\nfixture kept at {tmp}")
    print("\nPARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
