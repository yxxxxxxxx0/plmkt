"""
Decompress the .xz raw recordings back to their original .jsonl.

The archives were produced by compress_raw.py, which deleted each original
only after a full md5 round-trip. This is the reverse trip.

Safety choices, all deliberate:
  * refuses to overwrite an existing .jsonl unless --force -- these files are
    tens of GB and re-making one is not quick
  * checks free disk BEFORE writing, because xz stores the uncompressed size
    in its header, so running out of space part-way is avoidable rather than
    something to discover at 90%
  * verifies the archive with `xz -t` first; a corrupt archive is caught
    before it has written a partial file
  * deletes the partial output if decompression fails, so a truncated .jsonl
    can never be mistaken for a good one
  * keeps the .xz afterwards (use --remove-archive only if you are sure)

Usage:
    python decompress_raw.py --list
    python decompress_raw.py --all
    python decompress_raw.py data/live/books_2026-08-28.jsonl.xz
    python decompress_raw.py --all --dest D:/restore
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(BASE, "data", "live")
XZ = shutil.which("xz") or r"C:\Program Files\Git\mingw64\bin\xz.exe"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def uncompressed_size(path):
    """xz stores the original size in the stream footer, so this is exact."""
    try:
        out = subprocess.run([XZ, "--robot", "-l", path], capture_output=True,
                             text=True).stdout
        for line in out.splitlines():
            f = line.split("\t")
            if f and f[0] == "totals":
                return int(f[4])
    except Exception:
        pass
    return None


def human(n):
    return f"{n/1e9:.2f} GB" if n else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archives", nargs="*")
    ap.add_argument("--all", action="store_true", help="every .xz in data/live")
    ap.add_argument("--dest", default=None, help="output directory (default: alongside the .xz)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing .jsonl")
    ap.add_argument("--remove-archive", action="store_true",
                    help="delete the .xz after a successful decompression")
    ap.add_argument("--list", action="store_true", help="show archives and exit")
    args = ap.parse_args()

    if not os.path.exists(XZ):
        raise SystemExit(f"xz not found at {XZ}")

    archives = args.archives or (sorted(glob.glob(os.path.join(LIVE, "*.xz")))
                                 if args.all or args.list else [])
    if not archives:
        raise SystemExit("nothing to do: pass archive paths, --all, or --list")

    if args.list:
        print(f"{'archive':<44}{'packed':>10}{'unpacked':>12}")
        tot_c = tot_u = 0
        for a in archives:
            c = os.path.getsize(a)
            u = uncompressed_size(a)
            tot_c += c
            tot_u += u or 0
            print(f"  {os.path.basename(a):<42}{c/1e6:>8.0f} MB{human(u):>12}")
        print(f"  {'TOTAL':<42}{tot_c/1e6:>8.0f} MB{human(tot_u):>12}")
        return 0

    # Default alongside each archive, not a fixed directory: decompressing an
    # archive you copied elsewhere should land beside it, not silently in
    # data/live.
    def dest_for(a):
        d = args.dest or os.path.dirname(os.path.abspath(a))
        os.makedirs(d, exist_ok=True)
        return d

    need = sum(uncompressed_size(a) or 0 for a in archives)
    free = shutil.disk_usage(dest_for(archives[0])).free
    log(f"{len(archives)} archive(s), need {human(need)}, free {human(free)}")
    if need and free < need * 1.05:
        raise SystemExit(f"not enough free space: need ~{human(need)}, have {human(free)}")

    for a in archives:
        out = os.path.join(dest_for(a), os.path.basename(a)[:-3])  # strip .xz
        log(f"=== {os.path.basename(a)} -> {out} ===")
        if os.path.exists(out) and not args.force:
            log(f"  SKIP: {out} already exists (use --force to overwrite)")
            continue

        if subprocess.run([XZ, "-t", a]).returncode != 0:
            log("  !! archive failed integrity test; skipping")
            continue

        t = time.time()
        with open(out, "wb") as fh:
            rc = subprocess.run([XZ, "-d", "-c", a], stdout=fh).returncode
        if rc != 0:
            log(f"  !! decompression failed rc={rc}; removing partial output")
            try:
                os.remove(out)
            except OSError:
                pass
            continue

        sz = os.path.getsize(out)
        exp = uncompressed_size(a)
        if exp and sz != exp:
            log(f"  !! size mismatch: got {sz:,}, expected {exp:,}; removing")
            os.remove(out)
            continue
        log(f"  OK {human(sz)} in {time.time()-t:.0f}s")

        if args.remove_archive:
            os.remove(a)
            log("  removed archive")

    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
