"""Integrity checks on the built datasets, run against every session.

Written after the series-key bug: `extract()` keyed a market on
(slug, market_type, line), which is not unique, so both legs of a spread bet
merged into one forward-filled series whose "mid" alternated between two
unrelated price levels. Every positive result in the study came from that.

The lesson driving this file is that the bug was invisible to every aggregate
statistic being computed at the time, and obvious the moment anybody plotted
a price. So these checks look at the *structure* of each series rather than
its summary, and the headline one is CONTIGUITY: inside a single series, one
200ms grid step cannot move the mid by 20 ticks. A real market does not do
that. Two interleaved tokens do it constantly, and that is the signature.

Checks, per session:
  shape        parquet rows == tensor rows, tensor is finite
  identity     one asset_id per series, one series per asset_id, sid <-> series
  contiguity   no absurd mid jump between adjacent grid rows of one series
  causality    book_age_ms >= 0, ts strictly increasing on a grid multiple
  labels       path_exc_ticks >= |signed_ticks|, base rate plausible
  prices       mid in (0,1), spread > 0

Exit code is non-zero if any session fails, so this can gate a retrain.

    python audit_datasets.py
    python audit_datasets.py --sessions 2026-09-10 --max-step-ticks 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.join(BASE, "data", "jump")
GRID_MS = 200

# Columns needed for the checks, kept narrow because the biggest session is
# 30M rows and the machine this runs on has 16 GB.
COLS = ["mid", "spread_ticks", "signed_ticks", "path_exc_ticks", "valid",
        "book_age_ms", "ts", "series", "sid"]


class Report:
    def __init__(self, name):
        self.name = name
        self.fails, self.warns = [], []

    def check(self, ok, label, detail="", warn=False):
        if ok:
            print(f"    [ok  ] {label}" + (f"  {detail}" if detail else ""))
        else:
            tag = "WARN" if warn else "FAIL"
            print(f"    [{tag}] {label}  {detail}")
            (self.warns if warn else self.fails).append(f"{label}: {detail}")
        return ok


def audit(name, max_step_ticks=20.0, max_base_rate=0.35,
          tight_ticks=2.0, root=JD):
    fp = os.path.join(root, f"feat_{name}.parquet")
    tp = os.path.join(root, f"lob_{name}.npy")
    R = Report(name)
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    if not (os.path.exists(fp) and os.path.exists(tp)):
        R.check(False, "both files present", f"{fp} / {tp}")
        return R

    tbl = pq.read_table(fp, columns=COLS)
    # Dictionary-encode the series label before it reaches pandas. Converted
    # straight across it becomes one Python string object per row, which on
    # the 30M-row session is gigabytes for a column with ~50 distinct values.
    ser = (tbl.column("series").combine_chunks()
           .dictionary_encode().to_pandas())
    F = tbl.drop_columns(["series"]).to_pandas()
    del tbl
    T = np.load(tp, mmap_mode="r")

    # ---- shape ---------------------------------------------------------- #
    R.check(len(F) == T.shape[0], "parquet rows == tensor rows",
            f"{len(F):,} vs {T.shape[0]:,}")
    R.check(T.ndim == 3 and T.shape[1] == 4, "tensor is (n, 4, levels)",
            f"{T.shape}")
    # sample rather than scan 4.9 GB
    s = np.linspace(0, T.shape[0] - 1, min(200_000, T.shape[0])).astype(np.int64)
    R.check(np.isfinite(np.asarray(T[s])).all(), "tensor sample is finite")

    # The string work happens on the DISTINCT (sid, series) pairs -- a few
    # dozen rows -- not on the 30M-row column. Splitting a 30M-element string
    # column allocates 30M Python lists and is its own way to run out of
    # memory on this machine.
    pairs = (pd.DataFrame({"sid": F.sid.to_numpy(), "series": ser})
             .drop_duplicates())
    pairs["series"] = pairs.series.astype(str)
    parts = pairs.series.str.split("|")
    nfield = parts.str.len()
    R.check(bool((nfield == 4).all()), "every series id has 4 fields "
            "(slug|market_type|line|asset_id)",
            f"field counts {sorted(nfield.unique().tolist())}")
    pairs["mt"] = parts.str[1]
    pairs["asset"] = parts.str[3]

    # ---- identity: THE bug ---------------------------------------------- #
    # one asset per series, and -- the direction that actually broke -- no
    # asset appearing under two series ids.
    per_series = pairs.groupby("series", observed=True)["asset"].nunique()
    R.check(bool((per_series <= 1).all()), "one asset_id per series",
            f"{int((per_series > 1).sum())} series carry more than one asset")
    named = pairs[pairs.asset != ""]
    per_asset = named.groupby("asset", observed=True)["series"].nunique()
    R.check(bool((per_asset <= 1).all()), "one series per asset_id",
            f"{int((per_asset > 1).sum())} assets appear under several series")
    per_sid = pairs.groupby("sid", observed=True)["series"].nunique()
    R.check(bool((per_sid <= 1).all()), "sid maps to exactly one series",
            f"{int((per_sid > 1).sum())} sids straddle series")
    blank = int((pairs.asset == "").sum())
    R.check(blank == 0, "no series has an empty asset_id",
            f"{blank:,} series", warn=True)
    print(f"    ... {pairs.series.nunique():,} series, "
          f"{pairs.asset.nunique():,} assets, {pairs.sid.nunique():,} sids")
    # sid -> market type, so the contiguity report can name the guilty type
    mt_of_sid = pairs.set_index("sid")["mt"]

    # ---- contiguity: the signature of interleaved tokens ----------------- #
    # Rows are emitted grouped by series and ascending in time, so a diff
    # masked to "same sid and exactly one grid step apart" compares genuinely
    # adjacent observations of one book.
    sid = F.sid.to_numpy()
    ts = F.ts.to_numpy(np.int64)
    mid = F.mid.to_numpy(np.float64)
    same = np.r_[False, sid[1:] == sid[:-1]]
    step = np.r_[0, np.diff(ts)]
    adj = same & (step == GRID_MS)
    jump = np.zeros(len(F))
    jump[adj] = np.abs(np.r_[0, np.diff(mid)][adj]) / 0.01

    # A big mid step on its own is NOT evidence of corruption. When the touch
    # is pulled on a thin book and the next level sits 50 ticks away, the mid
    # legitimately gaps -- on 2026-09-10 all 1,796 such steps had a spread
    # above 10 ticks (median 53) and a 10ms book age, i.e. genuine fresh
    # quotes, and every one is already excluded by the spread filter used for
    # trading.
    #
    # What cannot be explained that way is a large step while the spread stays
    # TIGHT on BOTH sides: each merged token had its own healthy book, so the
    # interleaving moved the mid tens of ticks with a 1-2 tick spread
    # throughout. That conjunction is the signature, and it is what this
    # checks.
    sp = F.spread_ticks.to_numpy(np.float64)
    tight_both = np.zeros(len(F), bool)
    tight_both[adj] = (sp[adj] <= tight_ticks) & (sp[np.where(adj)[0] - 1] <= tight_ticks)
    suspect = (jump > max_step_ticks) & tight_both
    n_bad = int(suspect.sum())
    gapped = int(((jump > max_step_ticks) & ~tight_both).sum())
    worst = float(jump.max()) if len(jump) else 0.0

    # A handful of these are genuine. The clearest example is one `total`
    # market on 2026-08-28 -- a unique asset, so a merge is impossible --
    # repricing 0.925 -> 0.635 in a single step on 17ms-fresh quotes with a
    # 1-tick spread either side, after which the book blew out to a 29-tick
    # spread. A run scoring does that. So the test is on the RATE, not on
    # zero: corruption is systematic (4,530 suspect steps in 2.9M rows, 0.16%,
    # on the pre-fix build) while real repricings are one in tens of millions.
    tol = max(3, int(1e-5 * max(adj.sum(), 1)))
    rate = n_bad / max(int(adj.sum()), 1)
    R.check(n_bad <= tol,
            f"no systematic >{max_step_ticks:.0f}-tick mid step at a tight "
            f"(<={tight_ticks:.0f} tick) spread -- the merged-asset signature",
            f"{n_bad:,} suspect steps ({rate:.6%}), tolerance {tol:,}")
    print(f"    ... adjacent-step mid change: mean {jump[adj].mean():.4f}, "
          f"p99.9 {np.quantile(jump[adj], 0.999):.2f}, max {worst:.2f} ticks")
    print(f"    ... {gapped:,} large steps explained by a wide book "
          f"(touch pulled; excluded from trading anyway)")
    if n_bad:
        b = mt_of_sid.reindex(sid[suspect]).value_counts()
        print(f"    ... SUSPECT steps by market type: {b.to_dict()}")

    # ---- causality ------------------------------------------------------- #
    R.check(bool((F.book_age_ms.to_numpy() >= 0).all()),
            "no row carries a book from its own future")
    R.check(bool((step[same] > 0).all()), "ts strictly increasing within a series",
            f"{int((step[same] <= 0).sum()):,} non-increasing steps")
    R.check(bool((step[same] % GRID_MS == 0).all()),
            "every in-series time step is a grid multiple")

    # ---- labels ---------------------------------------------------------- #
    v = F.valid.to_numpy(bool)
    R.check(bool((F.path_exc_ticks.to_numpy()[v] + 1e-4
                  >= np.abs(F.signed_ticks.to_numpy()[v])).all()),
            "path max >= |endpoint move| on every valid row")
    br = float(F.loc[v, "signed_ticks"].abs().gt(0).mean())
    R.check(br < max_base_rate, "fraction of rows that move at all is plausible",
            f"{br:.3f}", warn=True)

    # ---- prices ---------------------------------------------------------- #
    R.check(bool(((mid > 0) & (mid < 1)).all()), "mid strictly inside (0, 1)",
            f"range {mid.min():.4f}-{mid.max():.4f}")
    R.check(bool((F.spread_ticks.to_numpy() > 0).all()), "spread is positive",
            f"min {F.spread_ticks.min():.2f}")

    del F, T, ser
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--max-step-ticks", type=float, default=20.0,
                    help="largest believable mid change across one 200ms grid "
                         "step inside a single series; the merged-asset bug "
                         "produced ~35 routinely")
    ap.add_argument("--quarantined", action="store_true",
                    help="audit the pre-fix datasets in _seriesbug_20260915 "
                         "instead -- used to prove these checks actually "
                         "catch the bug they were written for")
    a = ap.parse_args()

    names = sorted(f[len("feat_"):-len(".parquet")]
                   for f in os.listdir(JD)
                   if f.startswith("feat_books_") and f.endswith(".parquet")
                   and "_trimmed" not in f)
    if a.sessions:
        names = [n for n in names if any(s in n for s in a.sessions)]
    if not names:
        raise SystemExit("no built sessions found")

    root = os.path.join(JD, "_seriesbug_20260915") if a.quarantined else JD
    reps = [audit(n, a.max_step_ticks, root=root) for n in names]
    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    bad = 0
    for r in reps:
        state = ("PASS" if not r.fails else "FAIL")
        extra = f"  ({len(r.warns)} warn)" if r.warns else ""
        print(f"  {state:>4}  {r.name}{extra}")
        for f in r.fails:
            print(f"          {f}")
        bad += len(r.fails)
    print(f"\n{len(reps)} sessions, {bad} failed checks")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
