"""Leakage-safe splitting, shared by every model in this study.

Section 6 of the brief, and the single most important file here. Adjacent
prediction points overlap almost completely -- at 1 Hz with a 60s horizon,
consecutive rows share 59 of 60 seconds of their label window -- so a random
split over rows would put near-duplicates of the same moment in train and
test and report a number that means nothing.

Rules enforced:
  * whole MARKETS (one game's slug) live entirely in one split
  * splits are chronological where the calendar allows it: earlier dates
    train, later dates validate and test
  * both legs of a market (the YES and NO tokens of a spread, say) share a
    slug and therefore always land in the same split -- they are near-
    complementary series and treating them as independent across a boundary
    would leak directly
  * normalisation is fitted on train only; `Normaliser` refuses to be fitted
    on anything else
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})$")


def market_date(slug):
    m = DATE_RE.search(str(slug))
    return m.group(1) if m else "9999-99-99"


def chronological_market_split(markets, train=0.70, val=0.15, test=0.15):
    """Assign whole markets to splits, earliest dates first.

    Markets from the same date are kept together as far as possible so that a
    split boundary falls between slates rather than through one.
    """
    df = pd.DataFrame({"market": sorted(set(map(str, markets)))})
    df["date"] = df.market.map(market_date)
    df = df.sort_values(["date", "market"]).reset_index(drop=True)
    n = len(df)
    n_tr = int(round(n * train))
    n_va = int(round(n * val))
    # snap the boundaries to date changes where that does not distort the
    # proportions by more than a couple of markets
    def snap(i):
        if i <= 0 or i >= n:
            return i
        d = df.date.to_numpy()
        for j in range(i, min(i + 3, n)):
            if d[j] != d[j - 1]:
                return j
        for j in range(i, max(i - 3, 0), -1):
            if d[j] != d[j - 1]:
                return j
        return i
    i1, i2 = snap(n_tr), snap(n_tr + n_va)
    i2 = max(i2, i1 + 1)
    df["split"] = "test"
    df.loc[:i1 - 1, "split"] = "train"
    df.loc[i1:i2 - 1, "split"] = "val"
    return df


def assign(D, train=0.70, val=0.15, test=0.15, verbose=True):
    """Attach a `split` column to a prediction-point frame."""
    S = chronological_market_split(D.market.unique(), train, val, test)
    m = dict(zip(S.market, S.split))
    out = D.market.astype(str).map(m).to_numpy()
    if verbose:
        print("\nsplit assignment (whole markets, chronological):")
        for s in ("train", "val", "test"):
            sub = S[S.split == s]
            dates = sorted(sub.date.unique())
            print(f"  {s:>5}: {len(sub):>3} markets, dates {dates[0]}..{dates[-1]}"
                  if len(sub) else f"  {s:>5}: empty")
            for d in dates:
                ms = sorted(sub[sub.date == d].market.tolist())
                print(f"          {d}  " + ", ".join(ms))
        # the check that matters
        for s1 in ("train", "val", "test"):
            for s2 in ("train", "val", "test"):
                if s1 >= s2:
                    continue
                a = set(S[S.split == s1].market)
                b = set(S[S.split == s2].market)
                assert not (a & b), f"market in both {s1} and {s2}: {a & b}"
        print("  verified: no market appears in more than one split")
    return out, S


class Normaliser:
    """z-scoring fitted on TRAIN ROWS ONLY, with train-only clipping bounds."""

    def __init__(self, cols):
        self.cols = list(cols)
        self.mu = None
        self.sd = None
        self.lo = None
        self.hi = None

    def fit(self, X, split):
        assert (np.asarray(split) == "train").any(), "fit needs train rows"
        A = np.asarray(X[np.asarray(split) == "train"], np.float64)
        self.lo = np.nanpercentile(A, 0.1, axis=0)
        self.hi = np.nanpercentile(A, 99.9, axis=0)
        A = np.clip(A, self.lo, self.hi)
        self.mu = np.nanmean(A, axis=0)
        self.sd = np.nanstd(A, axis=0)
        self.sd[~np.isfinite(self.sd) | (self.sd < 1e-8)] = 1.0
        self.mu[~np.isfinite(self.mu)] = 0.0
        return self

    def transform(self, X):
        assert self.mu is not None, "fit first"
        A = np.clip(np.asarray(X, np.float64), self.lo, self.hi)
        A = (A - self.mu) / self.sd
        return np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def to_json(self):
        return json.dumps(dict(cols=self.cols, mu=self.mu.tolist(),
                               sd=self.sd.tolist(), lo=self.lo.tolist(),
                               hi=self.hi.tolist()))
