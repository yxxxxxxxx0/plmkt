"""
Losslessly compress raw recordings, deleting an original ONLY after a
full bit-for-bit round-trip has been proven.

Sequence per file:
    1. md5 the original
    2. xz -3 -T0  ->  <name>.xz
    3. xz -t      -- verifies the CRC64 xz stores for every block
    4. decompress the WHOLE archive to a stream and md5 that
    5. compare; delete the original only on an exact match

Step 4 is deliberately a full decompression, not a sample. These recordings
cannot be re-made -- the feed they came from no longer exists -- so the cost
of reading the data one extra time is trivial against the cost of deleting an
original that turns out not to round-trip.

Usage:
    python compress_raw.py data/live/books_2026-08-28.jsonl ...
    python compress_raw.py --keep data/live/books.jsonl     # verify, don't delete
"""
import argparse
import hashlib
import os
import subprocess
import shutil

# Absolute path, not bare "xz": this runs under Task Scheduler, whose PATH
# is not the Git Bash PATH where xz lives. Resolving it up front turns a
# mid-run FileNotFoundError into a clear failure before any work starts.
XZ = shutil.which('xz') or r'C:\Program Files\Git\mingw64\bin\xz.exe'
# Same trap for gzip, which Git for Windows ships in usr\bin, not mingw64\bin.
GZIP = shutil.which('gzip') or r'C:\Program Files\Git\usr\bin\gzip.exe'
import sys
import time


def md5_file(path, bufsize=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def md5_stream(cmd, bufsize=1 << 22):
    h = hashlib.md5()
    n = 0
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while chunk := p.stdout.read(bufsize):
        h.update(chunk)
        n += len(chunk)
    p.wait()
    return h.hexdigest(), n, p.returncode


def xz_name(src):
    """books.jsonl -> books.jsonl.xz; books.jsonl.gz -> books.jsonl.xz"""
    return (src[:-3] if src.endswith(".gz") else src) + ".xz"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--keep", action="store_true", help="verify but never delete")
    ap.add_argument("--level", default="3")
    args = ap.parse_args()

    if not os.path.exists(XZ):
        raise SystemExit(f'xz not found at {XZ}')
    log(f'using {XZ}')

    total_before = total_after = 0
    for src in args.files:
        if not os.path.exists(src):
            log(f"SKIP {src}: missing")
            continue
        dst = xz_name(src)
        size = os.path.getsize(src)
        log(f"=== {os.path.basename(src)}  ({size/1e9:.1f} GB) ===")

        t = time.time()
        partial = False
        if src.endswith(".gz"):
            # recorded as gzip (live_recorder.py --gzip): the original is the
            # decompressed stream. A recorder killed mid-write leaves a
            # truncated last block; everything before it is kept.
            m_before, size, grc = md5_stream([GZIP, "-dc", src])
            if grc != 0:
                partial = True
                log(f"  gzip -dc exited {grc} (truncated tail?); archiving the {size/1e9:.2f} GB "
                    f"it decoded, and KEEPING the .gz")
            with open(dst, "wb") as out:
                g = subprocess.Popen([GZIP, "-dc", src], stdout=subprocess.PIPE)
                rc = subprocess.run([XZ, f"-{args.level}", "-T0", "-c"], stdin=g.stdout,
                                    stdout=out).returncode
                g.stdout.close()
                g.wait()
        else:
            m_before = md5_file(src)
            with open(dst, "wb") as out:
                rc = subprocess.run([XZ, f"-{args.level}", "-T0", "-c", src],
                                    stdout=out).returncode
        log(f"  md5 original    {m_before}  ({time.time()-t:.0f}s)")
        if rc != 0:
            log(f"  !! xz failed rc={rc}; leaving original alone")
            continue
        csize = os.path.getsize(dst)
        log(f"  compressed      {csize/1e6:.0f} MB  ({size/csize:.0f}x, {time.time()-t:.0f}s)")

        if subprocess.run([XZ, "-t", dst]).returncode != 0:
            log("  !! xz -t FAILED; leaving original alone")
            continue
        log("  xz -t           PASS")

        t = time.time()
        m_after, nbytes, rc = md5_stream([XZ, "-d", "-c", dst])
        log(f"  md5 round-trip  {m_after}  ({nbytes/1e9:.1f} GB, {time.time()-t:.0f}s)")

        if rc != 0 or m_after != m_before or nbytes != size:
            log("  !! MISMATCH -- original kept, archive suspect")
            continue
        log("  VERIFIED identical")

        total_before += size
        total_after += csize
        if args.keep or partial:
            log("  original retained")
        else:
            freed = os.path.getsize(src)
            os.remove(src)
            log(f"  deleted original, reclaimed {freed/1e9:.1f} GB")

    if total_before:
        log(f"TOTAL {total_before/1e9:.1f} GB -> {total_after/1e9:.2f} GB "
            f"({total_before/max(total_after,1):.0f}x)")


if __name__ == "__main__":
    sys.exit(main())
