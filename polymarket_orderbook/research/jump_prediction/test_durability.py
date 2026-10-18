"""Correctness tests for the move decomposition in audit_durability.py.

The corrected result rests entirely on this function, so it gets the same
treatment as the rest of the pipeline: synthetic books with a known answer,
including the two artefact shapes that motivated the decomposition.

    python test_durability.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from audit_durability import per_series_moves  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print("  [%s] %s  %s" % ("ok  " if cond else "FAIL", name, detail))
    if not cond:
        FAIL.append(name)


def series(mids, spreads, start=1_000_000, step=1000):
    mid = np.asarray(mids, np.float32)
    sp = np.asarray(spreads, np.float32)
    ts = start + np.arange(len(mid), dtype=np.int64) * step
    return mid, sp, ts


def main():
    H, HOLD, TIGHT = 10, 3, 2.0

    print("ordering invariants")
    rng = np.random.default_rng(0)
    mid = np.cumsum(rng.normal(0, 0.005, 400)).astype(np.float32) + 0.5
    sp = rng.integers(1, 40, 400).astype(np.float32)
    ts = 1_000_000 + np.arange(400, dtype=np.int64) * 1000
    raw, clean, dur, endp, ok = per_series_moves(mid, sp, ts, H, HOLD, TIGHT)
    v = ok
    check("clean <= raw everywhere", bool(np.all(clean[v] <= raw[v] + 1e-6)),
          "max excess %.2e" % float(np.max(clean[v] - raw[v])))
    check("durable <= raw everywhere", bool(np.all(dur[v] <= raw[v] + 1e-6)),
          "max excess %.2e" % float(np.max(dur[v] - raw[v])))
    check("endpoint <= raw everywhere", bool(np.all(endp[v] <= raw[v] + 1e-6)))
    check("all four finite on valid rows",
          bool(np.all(np.isfinite([raw[v].sum(), clean[v].sum(),
                                   dur[v].sum(), endp[v].sum()]))))

    print("transient spike (the artefact shape)")
    # flat at 0.40, one single-second excursion to 0.60, flat again
    m = [0.40] * 12
    m[5] = 0.60
    mid, sp, ts = series(m, [1.0] * 12)
    raw, clean, dur, endp, ok = per_series_moves(mid, sp, ts, H, HOLD, TIGHT)
    check("raw sees the 0.20 spike", abs(float(raw[0]) - 0.20) < 1e-6,
          "raw[0]=%.3f" % raw[0])
    check("durable rejects a 1s spike", float(dur[0]) < 0.01,
          "durable[0]=%.3f" % dur[0])
    check("endpoint rejects it too", float(endp[0]) < 1e-6)

    print("sustained move (must survive)")
    m = [0.40] * 5 + [0.60] * 7
    mid, sp, ts = series(m, [1.0] * 12)
    raw, clean, dur, endp, ok = per_series_moves(mid, sp, ts, H, HOLD, TIGHT)
    check("raw sees it", abs(float(raw[0]) - 0.20) < 1e-6)
    check("durable keeps a sustained move", abs(float(dur[0]) - 0.20) < 1e-6,
          "durable[0]=%.3f" % dur[0])
    check("endpoint keeps it", abs(float(endp[0]) - 0.20) < 1e-6)

    print("quote vacuum (excursion only while the book is blown out)")
    # mid moves to 0.60 but ONLY while the spread is 40 ticks
    m = [0.40] * 5 + [0.60] * 4 + [0.40] * 3
    s = [1.0] * 5 + [40.0] * 4 + [1.0] * 3
    mid, sp, ts = series(m, s)
    raw, clean, dur, endp, ok = per_series_moves(mid, sp, ts, H, HOLD, TIGHT)
    check("raw counts the vacuum", abs(float(raw[0]) - 0.20) < 1e-6,
          "raw[0]=%.3f" % raw[0])
    check("clean rejects the vacuum", float(clean[0]) < 0.01,
          "clean[0]=%.3f" % clean[0])
    check("durable alone does NOT reject it (why both are needed)",
          abs(float(dur[0]) - 0.20) < 1e-6, "durable[0]=%.3f" % dur[0])

    print("gaps are never bridged")
    mid, sp, ts = series([0.40] * 20, [1.0] * 20)
    ts[10:] += 60_000                     # a one-minute hole after row 9
    raw, clean, dur, endp, ok = per_series_moves(mid, sp, ts, H, HOLD, TIGHT)
    check("rows whose window spans the hole are invalid",
          not bool(ok[:10].any()), "ok[:10]=%s" % ok[:10].astype(int))

    print("short series")
    mid, sp, ts = series([0.4] * 3, [1.0] * 3)
    raw, clean, dur, endp, ok = per_series_moves(mid, sp, ts, H, HOLD, TIGHT)
    check("series shorter than H yields no valid rows", not bool(ok.any()))

    print("=" * 62)
    if FAIL:
        print("FAILED: %s" % ", ".join(FAIL))
        return 1
    print("all durability tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
