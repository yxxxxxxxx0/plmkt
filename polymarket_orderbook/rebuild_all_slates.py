"""
Rebuilds every recorded slate into the viewer, one slate at a time.

Per slate: split -> reconstruct (data/uniform) -> chunk for the viewer
(data/viewer_full) -> drop that slate's by_game cache before the next slate.

Slate-at-a-time matters for two reasons:
  * disk -- holding every slate's by_game split at once would add ~120 GB of
    pure cache on top of the outputs; this keeps at most one slate's worth.
  * memory -- reconstruction peaks around 10 GB on a 16 GB machine, so the
    heavy steps must stay sequential.

--merge on the viewer build keeps slates already in index.json, so earlier
slates stay visible instead of being replaced by whichever ran last.

Usage:
    python rebuild_all_slates.py
    python rebuild_all_slates.py --sessions 2026-08-28 2026-08-29
"""
import argparse
import datetime as dt
import glob
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LIVE = os.path.join(BASE, "data", "live")
BY_GAME = os.path.join(LIVE, "by_game")


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(args, label):
    log(f">>> {label}")
    t0 = time.time()
    p = subprocess.Popen(args, cwd=BASE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1)
    for ln in p.stdout:
        print("    " + ln.rstrip(), flush=True)
    p.wait()
    log(f"<<< {label}: exit {p.returncode} in {time.time()-t0:.0f}s")
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="session tags (default: every data/live/books_*.jsonl)")
    ap.add_argument("--skip-uniform", action="store_true",
                    help="don't write data/uniform. The viewer never reads it -- "
                         "build_viewer_chunked reconstructs from by_game directly -- "
                         "so skipping saves ~55 min and ~5 GB per slate. Only needed "
                         "if you want the archive for simulation.")
    ap.add_argument("--force-split", action="store_true",
                    help="re-split even if by_game already holds this slate")
    ap.add_argument("--keep-cache", action="store_true",
                    help="don't delete each slate's by_game split afterwards")
    args = ap.parse_args()

    if args.sessions:
        files = [os.path.join(LIVE, f"books_{s}.jsonl") for s in args.sessions]
    else:
        files = sorted(glob.glob(os.path.join(LIVE, "books_*.jsonl")))
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"missing: {missing}")

    log(f"{len(files)} slate(s) to rebuild")
    t_all = time.time()

    for path in files:
        tag = os.path.basename(path)[len("books_"):-len(".jsonl")]
        log(f"===== SLATE {tag}  ({os.path.getsize(path)/1e9:.1f} GB) =====")

        sys.path.insert(0, BASE)
        import rebuild_multi as rm
        existing = sorted(os.path.basename(f)[:-len(".jsonl")]
                          for f in glob.glob(os.path.join(BY_GAME, f"*-{tag}.jsonl"))
                          if os.path.getsize(f) > 0)
        if existing and not args.force_split:
            slugs = existing
            log(f"reusing existing split -> {len(slugs)} games (skipped ~30 min)")
        else:
            slugs = rm.split_books_jsonl(path)
            log(f"split -> {len(slugs)} games: {', '.join(slugs[:3])}...")

        if args.skip_uniform:
            log("skipping data/uniform (viewer builds from by_game directly)")
        elif run([PY, "-u", "rebuild_duckdb.py", *slugs], f"reconstruct {tag}") != 0:
            log(f"!! reconstruct failed for {tag}; continuing to next slate")
            continue
        if run([PY, "-u", "build_viewer_chunked.py", "--merge", "--games", *slugs],
               f"viewer data {tag}") != 0:
            log(f"!! viewer build failed for {tag}")

        if not args.keep_cache:
            freed = 0
            for s in slugs:
                f = os.path.join(BY_GAME, f"{s}.jsonl")
                if os.path.exists(f):
                    freed += os.path.getsize(f)
                    os.remove(f)
            log(f"freed {freed/1e9:.1f} GB of by_game cache for {tag}")

    run([PY, "-u", "build_viewer_multi.py"], "viewer html")
    log(f"ALL DONE in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
