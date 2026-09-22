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

    The rule originally checked only the TAIL, and that is not symmetric.
    A recording can also start late, or die in the middle and resume: on
    books_2026-09-21, mlb-tor-bal was captured for the last 20 minutes of a
    198-minute game and stopped exactly on time, so its truncation was zero
    and it would have been kept in full. Two further tests close that:
    `late_start_min` against first pitch, and `ingame_cover` -- the share of
    the window's minutes that carry any grid row at all. The same blind spot
    in match_quality.py's coverage metric was fixed in the same pass.

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

# ...and the mirror of it. The recorder normally starts before first pitch, so
# a positive late start of more than a few minutes means innings are missing
# from the FRONT of the game. Same tolerance, for the same reason.
MAX_LATE_START_MIN = 5.0

# Share of the game window's minutes that must carry at least one grid row.
# Catches the case neither endpoint can see: a recording that starts on time,
# ends on time, and is dead in between.
MIN_INGAME_COVER = 0.90

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


def judge_match(window, min_ts, max_ts, minute_buckets,
                max_truncation_min=MAX_TRUNCATION_MIN,
                max_late_start_min=MAX_LATE_START_MIN,
                min_ingame_cover=MIN_INGAME_COVER):
    """The keep/drop decision, from three per-slug summaries and the window.

    It lives here alone because it did not, and the copies drifted.
    `trim_sessions.py` streams sessions too large to hold in memory and so
    reimplemented the truncation test inline; when the late-start and coverage
    tests were added to `match_report` in 2026-09-22, the builder that
    actually writes the trimmed caches kept applying the old rule and the two
    disagreed about four matches. Both now call this.

    `minute_buckets` is any iterable of `ts // 60000`; only those inside the
    window are counted, so passing the whole session's is fine.
    """
    if window is None:
        return dict(keep=False, reason="no game window: coverage unknown",
                    truncation_min=float("nan"), late_start_min=float("nan"),
                    ingame_cover=float("nan"))
    st, en = window
    a, b = st // 60000, en // 60000
    total = b - a + 1
    seen = sum(1 for m in minute_buckets if a <= m <= b)
    cover = float(seen / total) if total > 0 else 0.0
    trunc = (en - max_ts) / 60000.0       # >0 means data stops early
    late = (min_ts - st) / 60000.0        # >0 means data starts late

    keep, reason = True, ""
    if seen == 0:
        keep, reason = False, "no rows inside the game window"
    elif trunc > max_truncation_min:
        keep, reason = False, f"stops {trunc:.1f} min before the final out"
    elif late > max_late_start_min:
        keep, reason = False, f"starts {late:.1f} min after first pitch"
    elif cover < min_ingame_cover:
        keep, reason = False, (f"only {cover:.0%} of the game covered "
                               f"({seen} of {total} minutes)")
    return dict(keep=keep, reason=reason, truncation_min=float(trunc),
                late_start_min=float(late), ingame_cover=cover)


def match_report(F, windows=None, max_truncation_min=MAX_TRUNCATION_MIN,
                 max_late_start_min=MAX_LATE_START_MIN,
                 min_ingame_cover=MIN_INGAME_COVER):
    """Per-match coverage and a keep/drop decision. F needs `series` and `ts`.

    Three independent ways a match can be incomplete, all measured against
    game_windows.json rather than against the recording's own extent:
    it stops early, it starts late, or it is hollow in the middle.
    """
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
        v = judge_match(w, int(t.min()), int(t.max()),
                        np.unique(t // 60000).tolist(),
                        max_truncation_min, max_late_start_min,
                        min_ingame_cover)
        rows.append(dict(slug=s, n=len(i), ingame_frac=float(ing.mean()), **v))
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
