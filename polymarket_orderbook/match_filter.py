"""Which rows and which matches are fit to train on.

Two filters, both applied at load time rather than by rebuilding, because both
are cheap masks over columns the datasets already carry.

IN-GAME TRIMMING
    jump_data.py keeps every recorded row, and the recorder starts before the
    FIRST game of a slate -- so a game beginning eight hours later accumulates
    eight hours of "pregame". Measured across four sessions, 53-83% of all
    rows are pregame, and jumps there are 4-7x rarer:

        2026-08-28   83.3% pregame   in-game 0.3895  vs out 0.0532  (7.3x)
        2026-08-30   58.3% pregame   in-game 0.3563  vs out 0.0611  (5.8x)
        2026-09-10   53.3% pregame   in-game 0.3277  vs out 0.0716  (4.6x)
        2026-09-11   59.8% pregame   in-game 0.3466  vs out 0.0819  (4.2x)

    Those rows are free negatives: a classifier earns credit for telling a
    dead book from a live one, which is trivial and untradeable. It is the
    same confound as market-type mixing, and removing THAT one dropped pooled
    AUC from 0.88 to 0.67.

BROKEN-MATCH EXCLUSION
    A match whose recording stops before the final out is missing the late
    innings -- the highest-information part of the game. Measured truncations
    range from a few seconds to 45 minutes. Matches without a window entry at
    all cannot be trimmed or judged, so they are excluded rather than guessed
    at.

Both are deliberately conservative: when coverage cannot be established, the
match is dropped.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
WIN = os.path.join(BASE, "data", "game_windows.json")

# A match is unusable for training if its recording stops more than this many
# minutes before the final out. Small negatives are the recorder's stop grace
# and cost nothing; tens of minutes means whole innings are missing.
MAX_TRUNCATION_MIN = 5.0

# Seconds of PREGAME kept before kickoff. Not training data -- these rows are
# excluded as prediction points elsewhere -- but history: the deep models need
# `lookback` (200 rows = 40s) of contiguous book state behind every sample, so
# cutting exactly at kickoff leaves the first 40s of every game with no history
# and silently drops it. 120s gives the 40s window plus slack for the gaps the
# forward-fill cap can leave.
PREGAME_BUFFER_S = 120.0


def load_windows(path=WIN):
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        if not (isinstance(v, dict) and v.get("start_utc") and v.get("end_utc")):
            continue
        out[k] = (pd.Timestamp(v["start_utc"]).value // 10**6,
                  pd.Timestamp(v["end_utc"]).value // 10**6)
    return out


def match_report(F, windows=None, max_truncation_min=MAX_TRUNCATION_MIN):
    """Per-match coverage and a keep/drop decision. F needs `series` and `ts`."""
    windows = load_windows() if windows is None else windows
    slug = F.series.astype(str).str.split("|").str[0]
    ts = F.ts.to_numpy()
    rows = []
    for s, idx in slug.groupby(slug).groups.items():
        i = np.asarray(idx)
        t = ts[i]
        w = windows.get(s)
        if w is None:
            rows.append(dict(slug=s, n=len(i), keep=False,
                             reason="no game window: coverage unknown",
                             ingame_frac=np.nan, truncation_min=np.nan))
            continue
        st, en = w
        ing = (t >= st) & (t <= en)
        trunc = (en - t.max()) / 60000.0      # >0 means data stops early
        keep, reason = True, ""
        if trunc > max_truncation_min:
            keep = False
            reason = f"stops {trunc:.1f} min before the final out"
        elif not ing.any():
            keep = False
            reason = "no rows inside the game window"
        rows.append(dict(slug=s, n=len(i), keep=keep, reason=reason,
                         ingame_frac=float(ing.mean()),
                         truncation_min=float(trunc)))
    return pd.DataFrame(rows).sort_values("slug").reset_index(drop=True)


def masks(F, windows=None, in_game_only=True, drop_broken=True,
          max_truncation_min=MAX_TRUNCATION_MIN,
          pregame_buffer_s=PREGAME_BUFFER_S, log=print):
    """Return (row_mask, report). True = usable for training/eval.

    `pregame_buffer_s` keeps that many seconds before kickoff so the first
    in-game samples have their lookback window. Those rows are retained as
    HISTORY only -- the staleness and contiguity gates in jump_split.load()
    still decide which rows may serve as prediction points, and the in-game
    base-rate gap (jumps are 4-7x rarer pregame) is why they must not be
    training targets themselves.
    """
    windows = load_windows() if windows is None else windows
    rep = match_report(F, windows, max_truncation_min)
    slug = F.series.astype(str).str.split("|").str[0].to_numpy()
    ts = F.ts.to_numpy()

    keep = np.ones(len(F), bool)
    if drop_broken:
        bad = set(rep.loc[~rep.keep, "slug"])
        if bad:
            keep &= ~np.isin(slug, list(bad))
            log(f"  dropping {len(bad)} match(es) as unusable:")
            for _, r in rep[~rep.keep].iterrows():
                log(f"    {r.slug}: {r.reason} ({r.n:,} rows)")
    if in_game_only:
        lo = np.full(len(F), np.iinfo(np.int64).max, np.int64)
        hi = np.full(len(F), np.iinfo(np.int64).min, np.int64)
        buf_ms = int(pregame_buffer_s * 1000)
        for s, (st, en) in windows.items():
            m = slug == s
            if m.any():
                lo[m] = st - buf_ms      # keep a pregame run-up for history
                hi[m] = en
        ing = (ts >= lo) & (ts <= hi)
        strict = (ts >= lo + buf_ms) & (ts <= hi)
        log(f"  in-game trim: {strict.mean():.1%} of rows are strictly "
            f"between kickoff and the final out; keeping {ing.mean():.1%} "
            f"including a {pregame_buffer_s:.0f}s pregame buffer for lookback "
            f"history")
        keep &= ing
    log(f"  usable rows after match filtering: {keep.sum():,} of {len(F):,} "
        f"({keep.mean():.1%})")
    return keep, rep
