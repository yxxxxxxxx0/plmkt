"""Pre-filter each session ONCE, one file at a time, to small on-disk caches.

Loading all built sessions at once and filtering in memory turned out to be
unsafe on this machine: reading four feat_*.parquet files simultaneously (even
before any tensor concat) drove free RAM down to 0.2-2GB in real time, close
enough to zero to be the kind of pressure that has already crashed the editor
once this session. That happens because pandas' parquet reader and the
subsequent .str/.isin operations hold multiple transient copies of each
column while parsing -- the on-disk size badly understates the peak.

The fix is to never hold more than one raw session in memory at a time. This
script does the filtering (market type, in-game trim, broken-match exclusion)
per session, writes the much smaller result to *_trimmed.parquet/.npy, and
frees everything before moving to the next file. jump_split.load() can then
read the trimmed caches directly with `use_trimmed=True`, at which point
combining several sessions is cheap because each one is already small.

Measured trim ratios (in-game vs raw), for reference:
    2026-08-28   16.9% kept   2026-08-30   43.7% kept
    2026-09-10   52.8% kept   2026-09-11   42.7% kept

Usage:
    python trim_sessions.py
    python trim_sessions.py --sessions 2026-08-28 2026-08-30
"""
from __future__ import annotations

import argparse
import gc
import glob
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.join(BASE, "data", "jump")


def log(m):
    print(m, flush=True)


def trim_one(feat_path, market_types, in_game_only, drop_broken, force=False,
             batch_rows=500_000):
    """Filter one session, reading it in ROW-GROUP BATCHES throughout.

    Even a "narrow" single-shot pass over just (series, ts) drove free RAM to
    0.02GB on the largest session (24.5M rows): a pandas object-dtype string
    column holds one Python object PER ROW, and 24.5M of those is several GB
    on its own before any .str/.isin temporaries are added, independent of
    column count. So both passes below are batch-wise -- nothing accumulates
    a full-length array of strings or objects at any point:

      pass 1  stream (series, ts) in small batches, updating only a per-slug
              max-timestamp dict (~50 entries, not 24.5M rows) plus, if
              market-type filtering only, a batch-local boolean mask that is
              applied and discarded per batch. drop_broken's decision needs
              the whole file's max ts per match, hence pass 1; in_game_only
              needs no global information at all (the window bounds come
              from data/game_windows.json), so it is applied inline in pass 2.

      pass 2  stream the FULL columns in the same batches, keep only rows
              that survive market-type + in-game-window + not-broken, write
              each surviving batch to its own small parquet part immediately,
              and gather that batch's tensor rows from the memmap right away.
    """
    from match_filter import (MAX_TRUNCATION_MIN, PREGAME_BUFFER_S,
                              load_windows)

    tag = os.path.basename(feat_path).replace("feat_", "").replace(
        ".parquet", "")
    out_feat = os.path.join(JD, f"feat_{tag}_trimmed.parquet")
    out_lob = os.path.join(JD, f"lob_{tag}_trimmed.npy")
    if not force and os.path.exists(out_feat) and os.path.exists(out_lob):
        log(f"  {tag}: trimmed cache already exists, skipping "
            f"(pass --force to rebuild)")
        return out_feat

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(feat_path)
    n0 = pf.metadata.num_rows
    log(f"  {tag}: {n0:,} rows across {pf.num_row_groups} row group(s), "
        f"batch size {batch_rows:,}")
    windows = load_windows()

    # ---- pass 1: per-slug max ts only, if a broken-match decision is needed
    broken_slugs = set()
    if drop_broken:
        max_ts = {}
        for batch in pf.iter_batches(columns=["series", "ts"],
                                     batch_size=batch_rows):
            series = batch.column("series").to_pylist()
            ts = batch.column("ts").to_pylist()
            for s, t in zip(series, ts):
                slug = s.split("|", 1)[0]
                if t > max_ts.get(slug, -1):
                    max_ts[slug] = t
            del series, ts, batch
        for slug, mx in max_ts.items():
            w = windows.get(slug)
            if w is None:
                broken_slugs.add(slug)     # unknown coverage: exclude
                continue
            _, en = w
            trunc_min = (en - mx) / 60000.0
            if trunc_min > MAX_TRUNCATION_MIN:
                broken_slugs.add(slug)
        if broken_slugs:
            log(f"  {tag}: excluding {len(broken_slugs)} broken match(es): "
                f"{', '.join(sorted(broken_slugs))}")
        del max_ts
        gc.collect()

    # ---- pass 2: stream full columns, filter, write, gather tensor rows
    lob_path = feat_path.replace("feat_", "lob_").replace(".parquet", ".npy")
    tm = np.load(lob_path, mmap_mode="r")

    part_paths, tensor_chunks = [], []
    row0 = kept_total = 0
    for batch in pf.iter_batches(batch_size=batch_rows):
        n = batch.num_rows
        bdf = batch.to_pandas()
        del batch
        slug = bdf["series"].str.split("|", n=1).str[0]
        mt = bdf["series"].str.split("|").str[1]

        bkeep = np.ones(n, bool)
        if market_types:
            bkeep &= mt.isin(market_types).to_numpy()
        if in_game_only:
            ts = bdf["ts"].to_numpy()
            lo = np.full(n, np.iinfo(np.int64).max, np.int64)
            hi = np.full(n, np.iinfo(np.int64).min, np.int64)
            # Keep PREGAME_BUFFER_S before kickoff. Those rows are history,
            # not training targets: the deep models need 200 contiguous grid
            # steps (40s) behind every sample, so cutting exactly at kickoff
            # leaves the first 40s of each game with no lookback and it gets
            # silently dropped by the contiguity check. The staleness and
            # contiguity gates in jump_split.load() still decide which rows
            # may serve as prediction points, so keeping the run-up does not
            # reintroduce pregame as a training target.
            buf_ms = int(PREGAME_BUFFER_S * 1000)
            for s in slug.unique():
                w = windows.get(s)
                m = (slug == s).to_numpy()
                if w is None:
                    bkeep &= ~m         # no window: exclude these rows
                    continue
                lo[m], hi[m] = w[0] - buf_ms, w[1]
            bkeep &= (ts >= lo) & (ts <= hi)
        if drop_broken and broken_slugs:
            bkeep &= ~slug.isin(broken_slugs).to_numpy()

        row0_this = row0
        row0 += n
        if not bkeep.any():
            del bdf, slug, mt
            continue
        bdf = bdf[bkeep].reset_index(drop=True)
        part_path = f"{out_feat}.part{len(part_paths)}.parquet"
        bdf.to_parquet(part_path, index=False)
        part_paths.append(part_path)
        idx = np.where(bkeep)[0] + row0_this
        tensor_chunks.append(np.asarray(tm[idx]))
        kept_total += len(bdf)
        del bdf, slug, mt, bkeep
        gc.collect()
    del tm
    log(f"  {tag}: {kept_total:,} of {n0:,} rows kept "
        f"({kept_total/max(n0,1):.1%})")

    if not part_paths:
        raise SystemExit(f"{tag}: nothing survived filtering")
    F_out = pd.concat([pd.read_parquet(p) for p in part_paths],
                      ignore_index=True)
    F_out.to_parquet(out_feat, index=False)
    for p in part_paths:
        os.remove(p)
    T_out = (tensor_chunks[0] if len(tensor_chunks) == 1
            else np.concatenate(tensor_chunks, axis=0))
    np.save(out_lob, T_out)
    log(f"  {tag}: wrote {out_feat} ({os.path.getsize(out_feat)/1e6:.0f} MB) "
        f"and {out_lob} ({os.path.getsize(out_lob)/1e6:.0f} MB)")
    assert len(F_out) == len(T_out), "row count mismatch after batching"

    del F_out, T_out, tensor_chunks
    gc.collect()
    return out_feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="substring filter on which feat_*.parquet to trim; "
                         "default: all (excluding *_trimmed.parquet itself)")
    ap.add_argument("--market-types", nargs="*",
                    default=["moneyline", "total", "spread"])
    ap.add_argument("--keep-pregame", action="store_true",
                    help="skip in-game trimming (not recommended)")
    ap.add_argument("--keep-broken", action="store_true",
                    help="skip broken-match exclusion (not recommended)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    fs = sorted(f for f in glob.glob(os.path.join(JD, "feat_*.parquet"))
               if "_trimmed" not in f)
    if a.sessions:
        fs = [f for f in fs if any(s in os.path.basename(f)
                                   for s in a.sessions)]
    if not fs:
        raise SystemExit("no sessions to trim")

    log(f"trimming {len(fs)} session(s), one at a time, "
        f"in_game_only={not a.keep_pregame} drop_broken={not a.keep_broken}")
    for f in fs:
        trim_one(f, a.market_types, not a.keep_pregame, not a.keep_broken,
                 force=a.force)

    tot_in = sum(os.path.getsize(f.replace("feat_", "lob_")
                                 .replace(".parquet", ".npy")) for f in fs)
    outs = glob.glob(os.path.join(JD, "*_trimmed.npy"))
    tot_out = sum(os.path.getsize(f) for f in outs
                 if any(os.path.basename(f).replace(".npy", "")
                        .replace("lob_", "").replace("_trimmed", "") in f2
                        for f2 in [os.path.basename(x) for x in fs]))
    log(f"\ndone. Original lob_*.npy total {tot_in/1e9:.2f} GB; "
        f"trimmed caches are the small files training now reads.")


if __name__ == "__main__":
    main()
