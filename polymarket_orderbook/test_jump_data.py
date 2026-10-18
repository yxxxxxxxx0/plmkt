"""Regression tests for the grid construction in jump_data.py.

The bug these exist for: `extract()` used to store the incoming book into
cur[key] and *then* back-fill every pending grid slot with it, so rows
timestamped throughout a quiet stretch carried the book from the END of that
stretch. It is invisible in any plot of the output -- the mid series looks
like a normal step function, just with the steps in the wrong place -- and it
made `stale` a near-deterministic predictor of "no jump", which is exactly
the kind of artifact that reads as a real finding.

    python test_jump_data.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import orjson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jump_data as jd


def book(ts, bid, ask, slug="g1", mt="moneyline", n=3):
    """One snapshot, `n` levels a cent apart on each side."""
    return dict(
        ts=ts, slug=slug, market_type=mt, outcome="YES", line=None,
        bids=[[round(bid - 0.01 * i, 4), 100.0 + i] for i in range(n)],
        asks=[[round(ask + 0.01 * i, 4), 200.0 + i] for i in range(n)],
    )


def write(recs):
    fh = tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False)
    for r in recs:
        fh.write(orjson.dumps(r) + b"\n")
    fh.close()
    return fh.name


def check(name, cond, detail=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


# --------------------------------------------------------------------------- #
def test_backfill_uses_previous_book():
    """A silence must be filled with the PRE-silence book, not the next one."""
    print("back-fill direction")
    # first message at 1000 (mid 0.50), next at 3000 (mid 0.60)
    path = write([book(1000, 0.49, 0.51), book(3000, 0.59, 0.61)])
    try:
        d = jd.extract(path, grid_ms=200, k=10)
    finally:
        os.unlink(path)

    ts, tsb = d["ts"], d["ts_book"]
    mid = (d["BP"][:, 0] + d["AP"][:, 0]) / 2.0
    ok = True

    # slots 1200..2800 inclusive, then 3000 landing exactly on the message
    ok &= check("row count", len(ts) == 10, f"got {len(ts)}")
    ok &= check("slots are contiguous 200ms apart",
                list(ts) == list(range(1200, 3200, 200)), f"{list(ts)}")

    filled, exact = ts < 3000, ts == 3000
    ok &= check("silence carries the pre-silence mid (0.50)",
                np.allclose(mid[filled], 0.50),
                f"got {np.unique(np.round(mid[filled], 4))}")
    ok &= check("slot landing on the message carries the new mid (0.60)",
                np.allclose(mid[exact], 0.60),
                f"got {np.unique(np.round(mid[exact], 4))}")
    ok &= check("back-filled rows are attributed to the old book time",
                (tsb[filled] == 1000).all(), f"got {np.unique(tsb[filled])}")
    ok &= check("the exact-hit row is attributed to the new book time",
                (tsb[exact] == 3000).all())
    return ok


def test_no_row_sees_its_own_future():
    """The invariant the bug violated, over a messier stream."""
    print("no lookahead")
    rng = np.random.default_rng(0)
    recs, t, mid = [], 5_000, 0.40
    for _ in range(400):
        # irregular arrivals, including long silences
        t += int(rng.choice([37, 111, 240, 1_500, 9_000], p=[.4, .3, .15, .1, .05]))
        mid = float(np.clip(mid + rng.choice([-0.01, 0, 0.01]), 0.05, 0.95))
        recs.append(book(t, round(mid - 0.005, 4), round(mid + 0.005, 4)))
    path = write(recs)
    try:
        d = jd.extract(path, grid_ms=200, k=10)
    finally:
        os.unlink(path)

    ts, tsb = d["ts"], d["ts_book"]
    ok = check("ts_book <= ts for every row", (tsb <= ts).all(),
               f"{(tsb > ts).sum()} violations of {len(ts)}")
    ok &= check("slots strictly increasing", (np.diff(ts) > 0).all())
    ok &= check("slots exactly one grid step apart",
                set(np.unique(np.diff(ts)).tolist()) == {200})
    ok &= check("some rows are genuinely stale (the fill is doing work)",
                (ts - tsb).max() > 1_000, f"max age {(ts-tsb).max()}ms")
    return ok


def test_series_are_independent():
    """Two markets interleaved must not contaminate each other's fill."""
    print("series independence")
    recs = [book(1000, 0.49, 0.51, slug="a"),
            book(1000, 0.19, 0.21, slug="b", mt="total"),
            book(2000, 0.59, 0.61, slug="a"),
            book(2000, 0.19, 0.21, slug="b", mt="total")]
    path = write(recs)
    try:
        d = jd.extract(path, grid_ms=200, k=10)
    finally:
        os.unlink(path)
    mid = (d["BP"][:, 0] + d["AP"][:, 0]) / 2.0
    sl = jd.key_column(d, "slug")
    a, b = sl == "a", sl == "b"
    ok = check("series a filled from a's own book",
               np.allclose(mid[a & (d["ts"] < 2000)], 0.50))
    ok &= check("series b filled from b's own book",
                np.allclose(mid[b], 0.20))
    ok &= check("both series emitted", a.sum() > 0 and b.sum() > 0)
    return ok


def test_path_max_excursion_persisted():
    """path_exc_ticks must be >= |signed_ticks| and drive the label."""
    print("path-max excursion")
    # a round trip: mid goes up 3 ticks then comes back, so the endpoint move
    # is 0 but a resting quote was picked off by 3 ticks on the way.
    recs, t = [], 1_000
    # build_features requires >=400 rows per series
    path_mids = [0.50] * 300 + [0.53] * 10 + [0.50] * 200
    for m in path_mids:
        recs.append(book(t, round(m - 0.005, 4), round(m + 0.005, 4)))
        t += 200
    p = write(recs)
    try:
        d = jd.extract(p, grid_ms=200, k=10)
        F, T, S = jd.build_features(d, horizon_s=5, jump_ticks=2)
    finally:
        os.unlink(p)

    if len(F) == 0:
        return check("series long enough to build", False,
                     "build_features needs >=400 rows per series")
    v = F[F.valid]
    ok = check("path_exc_ticks present", "path_exc_ticks" in F.columns)
    ok &= check("path max >= |endpoint| everywhere",
                (v.path_exc_ticks + 1e-6 >= v.signed_ticks.abs()).all())
    ok &= check("the round trip is caught by path but missed by endpoint",
                (v.path_exc_ticks > v.signed_ticks.abs() + 1).any())
    ok &= check("book_age_ms present and non-negative",
                "book_age_ms" in F.columns and (v.book_age_ms >= 0).all())
    return ok


def test_fill_cap_truncates_long_silences():
    """A silence past the cap must stop emitting and leave a real gap."""
    print("fill cap")
    # 5s cap at 200ms = 25 slots; the silence here is 20s
    path = write([book(1000, 0.49, 0.51), book(21_000, 0.59, 0.61)])
    try:
        d = jd.extract(path, grid_ms=200, k=10, max_fill_s=5.0)
    finally:
        os.unlink(path)
    ts = d["ts"]
    ok = check("emitted the cap, not the whole silence",
               len(ts) <= 27, f"got {len(ts)} rows (uncapped would be ~100)")
    ok &= check("filled exactly up to the cap",
                int((ts < 21_000).sum()) == 25,
                f"{int((ts < 21_000).sum())} filled rows")
    ok &= check("a real gap is left in the slot sequence",
                bool((np.diff(ts) > 200).any()),
                f"steps {sorted(set(np.diff(ts).tolist()))}")
    ok &= check("no row carries a future book",
                (d["ts_book"] <= ts).all())
    # a silence under the cap must still be filled contiguously
    path = write([book(1000, 0.49, 0.51), book(3000, 0.59, 0.61)])
    try:
        d2 = jd.extract(path, grid_ms=200, k=10, max_fill_s=60.0)
    finally:
        os.unlink(path)
    ok &= check("a silence under the cap is filled contiguously",
                set(np.unique(np.diff(d2["ts"])).tolist()) == {200})
    return ok


def test_two_assets_one_line_stay_separate():
    """Both legs of a spread bet must not collapse into one series.

    The two sides of a spread carry the SAME slug, market_type and line, and
    both are outcome "YES" -- they are distinguished only by asset_id. Keying
    without it merged them, so one forward-filled book alternated between two
    unrelated price levels (measured at 0.195 vs 0.545 on a real market) and
    manufactured a ~35-tick move on every switch. Spread markets were the
    only affected type, and they carried every apparent edge in the study.
    """
    print("two assets, one line")
    recs, t = [], 1_000
    for _ in range(300):
        # leg A near 0.20, leg B near 0.55, interleaved, identical key fields
        a = book(t, 0.195, 0.205); a["asset_id"] = "AAA"; a["line"] = -1.5
        a["market_type"] = "spread"
        recs.append(a)
        t += 200
        b = book(t, 0.545, 0.555); b["asset_id"] = "BBB"; b["line"] = -1.5
        b["market_type"] = "spread"
        recs.append(b)
        t += 200
    p = write(recs)
    try:
        d = jd.extract(p, grid_ms=200, k=10)
    finally:
        os.unlink(p)

    aid = jd.key_column(d, "asset_id")
    assets = set(a for a in aid if a)
    ok = check("both asset_ids survive into the rows", assets == {"AAA", "BBB"},
               f"got {assets}")
    mid = (d["BP"][:, 0] + d["AP"][:, 0]) / 2.0
    for tag, lo, hi in (("AAA", 0.15, 0.25), ("BBB", 0.50, 0.60)):
        m = aid == tag
        if m.any():
            ok &= check(f"{tag} rows stay near their own price",
                        bool(((mid[m] > lo) & (mid[m] < hi)).all()),
                        f"range {mid[m].min():.3f}-{mid[m].max():.3f}")
    # the giveaway symptom: a single series swinging across both levels
    for tag in ("AAA", "BBB"):
        m = aid == tag
        if m.any():
            ok &= check(f"{tag} shows no fabricated cross-level jump",
                        bool((np.abs(np.diff(mid[m])) < 0.10).all()))
    return ok


def test_scratch_spill_matches_in_memory():
    """Spilling blocks to disk must change nothing about the output.

    The largest session's row buffer is ~8 GB, which cannot sit in RAM beside
    the feature frames, so `extract(scratch=...)` writes each sealed block to
    disk and streams it back into a memmap. That path only ever runs on the
    sessions too big to check by eye, so it is checked here instead, against
    the in-memory path on the same input.
    """
    print("scratch spill == in memory")
    rng = np.random.default_rng(7)
    recs, t, mid = [], 1_000, 0.40
    for i in range(1500):
        t += int(rng.choice([200, 400, 1_100]))
        mid = float(np.clip(mid + rng.choice([-0.01, 0, 0.01]), 0.05, 0.95))
        r = book(t, round(mid - 0.005, 4), round(mid + 0.005, 4),
                 slug=f"g{i % 3}", mt="spread")
        r["asset_id"] = f"A{i % 2}"
        r["line"] = -1.5
        recs.append(r)
    path = write(recs)
    scratch = tempfile.mkdtemp()
    try:
        # BLOCK is 500k rows; shrink it so this input spans several blocks
        old = jd._RowBuffer.BLOCK
        jd._RowBuffer.BLOCK = 256
        try:
            mem = jd.extract(path, grid_ms=200, k=10)
            spl = jd.extract(path, grid_ms=200, k=10, scratch=scratch)
        finally:
            jd._RowBuffer.BLOCK = old
    finally:
        os.unlink(path)

    ok = check("same row count", len(mem["ts"]) == len(spl["ts"]),
               f"{len(mem['ts'])} vs {len(spl['ts'])}")
    ok &= check("same key table", mem["keys"] == spl["keys"])
    for key in ("code", "ts", "ts_book", "BP", "BS", "AP", "AS"):
        same = (np.array_equal(np.asarray(mem[key]), np.asarray(spl[key]),
                               equal_nan=key in ("BP", "AP")))
        ok &= check(f"{key} identical", same)
    ok &= check("the spill actually wrote files", len(spl["_scratch_files"]) > 0,
                f"{len(spl['_scratch_files'])} files")
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    return ok


if __name__ == "__main__":
    print("jump_data grid construction\n" + "=" * 60)
    results = [test_scratch_spill_matches_in_memory(),
               test_two_assets_one_line_stay_separate(),
               test_backfill_uses_previous_book(),
               test_no_row_sees_its_own_future(),
               test_series_are_independent(),
               test_path_max_excursion_persisted(),
               test_fill_cap_truncates_long_silences()]
    print("=" * 60)
    print(f"{sum(results)}/{len(results)} groups passed")
    sys.exit(0 if all(results) else 1)
