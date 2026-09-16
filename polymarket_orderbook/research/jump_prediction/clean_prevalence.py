"""Class prevalence for every (J, H), under each label definition.

Section 3 of the brief asks for prevalence at every threshold and horizon. The
original jump_prevalence.csv answered that for the raw max-|mid| label and the
answer was 0.45-0.78, which should have been the first sign that something was
wrong: a "jump" that happens two-thirds of the time is not a jump.

This reports the same grid for all four label definitions so the size of the
artefact is visible at every setting rather than only at J=0.02, H=30s:

    raw       max |mid_u - mid_t|                    -- the original label
    clean     ... restricted to u whose book is tight
    durable   ... requiring the displacement to persist
    both      clean and durable                      -- the corrected label

    python clean_prevalence.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "jump_prediction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="*", default=[10, 30, 60])
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[0.01, 0.02, 0.03, 0.05])
    ap.add_argument("--tight-ticks", type=float, default=2.0)
    a = ap.parse_args()

    ps = sorted(glob.glob(os.path.join(CACHE, "points_books_*.parquet")))
    base = pd.concat([pd.read_parquet(p, columns=["sid", "ts", "spread_ticks"])
                      for p in ps], ignore_index=True)

    rows = []
    for H in a.horizons:
        mv = os.path.join(CACHE, "moves_H%d.parquet" % H)
        if not os.path.exists(mv):
            print("skip H=%d (no moves table)" % H)
            continue
        M = pd.read_parquet(mv)
        D = base.merge(M[["sid", "ts", "raw_move", "clean_move",
                          "durable_move"]], on=["sid", "ts"], how="inner")
        tight = (D.spread_ticks <= a.tight_ticks).to_numpy()
        both = np.minimum(D.clean_move.to_numpy(), D.durable_move.to_numpy())
        for J in a.thresholds:
            for regime, m in (("all books", np.ones(len(D), bool)),
                              ("tight at t", tight)):
                n = int(m.sum())
                r = dict(H_s=H, J=J, regime=regime, n_valid=n)
                r["prev_raw"] = float((D.raw_move.to_numpy()[m] >= J).mean())
                r["prev_clean"] = float((D.clean_move.to_numpy()[m] >= J).mean())
                r["prev_durable"] = float((D.durable_move.to_numpy()[m] >= J).mean())
                r["prev_both"] = float((both[m] >= J).mean())
                r["n_pos_both"] = int((both[m] >= J).sum())
                r["artefact_share"] = 1.0 - (r["prev_both"] /
                                             max(r["prev_raw"], 1e-12))
                rows.append(r)

    out = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    f = os.path.join(RES, "jump_prevalence_clean.csv")
    out.to_csv(f, index=False)
    print(out.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\nwrote", f)


if __name__ == "__main__":
    sys.exit(main())
