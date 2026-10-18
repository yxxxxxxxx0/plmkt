"""Regression tests for the two coverage metrics that decide which games count.

The bug these exist for, found 2026-09-22 on `books_2026-09-21`: coverage was
measured against the recording's own extent rather than against the game. Both
places had the same blind spot.

  match_quality.py graded `mlb-tor-bal-2026-09-21` **GOOD at 99.9%** while 90%
  of the game was never recorded. Its metric is
  `1 - median(per-asset largest gap) / span`, and `span` runs from a match's
  first record to its last -- so a recording that starts three hours late is
  continuous over the sliver it caught, and scores as if it were complete.
  Only 2 of that match's 42 assets existed before the restart, so the median
  asset's largest gap was 16 seconds.

  match_filter.py dropped a match when the recording stopped early, and that
  test is not symmetric: the same match stopped exactly on time, so its
  truncation was zero and it would have trained the model on the last 20
  minutes of a 198-minute game.

The fix in both is to count the GAME WINDOW's minutes that carry data. These
tests pin that, and pin that the old metric really does miss the case -- a
test that only checked the new number would pass against the old code too.

    python test_match_coverage.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match_filter as mf
import match_quality as mq

MIN = 60_000
START = 1_790_000_000_000 // MIN * MIN      # a whole minute, for exact counts
END = START + 100 * MIN                      # a 100-minute game
WINDOW = (START, END)


def check(name, got, want):
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}: got {got}, want {want}")
    return ok


def approx(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}: got {got:.4f}, want {want:.4f}")
    return ok


def minutes(lo, hi):
    """Bucket ids for every minute in [lo, hi) of the window."""
    return {(START + m * MIN) // MIN for m in range(lo, hi)}


def frame(ts, slug="mlb-aaa-bbb-2026-09-21"):
    """The two columns match_report reads."""
    return pd.DataFrame(dict(series=[f"{slug}|moneyline||tok"] * len(ts),
                             ts=np.asarray(ts, dtype=np.int64)))


def grid(lo, hi, step_s=1):
    """Rows every `step_s` seconds across minutes [lo, hi) of the window."""
    return [START + lo * MIN + i * step_s * 1000
            for i in range((hi - lo) * 60 // step_s)]


# --------------------------------------------------------------------------
def test_late_start_is_visible():
    """The tor-bal shape: nothing until minute 90, then continuous to the end."""
    print("a late start is counted against the game, not against the data")
    ok = []
    cov, late, early = mq.window_coverage(minutes(90, 101), WINDOW)
    # 11 of the window's 101 minute buckets carry data (90 through 100)
    ok.append(approx("game cover", cov, 11 / 101))
    ok.append(approx("late start seconds", late, 90 * 60))
    ok.append(approx("early stop seconds", early, 0.0))

    r = mf.match_report(frame(grid(90, 100)), windows={"mlb-aaa-bbb-2026-09-21": WINDOW})
    ok.append(check("match_report keeps it", bool(r.keep.iloc[0]), False))
    ok.append(check("reason names the late start",
                    "after first pitch" in r.reason.iloc[0], True))
    # and the number the OLD rule looked at is clean, which is why it passed
    ok.append(approx("truncation the old rule saw", r.truncation_min.iloc[0], 0.0, tol=0.02))
    return all(ok)


def test_the_old_metric_really_does_miss_it():
    """Pins the failure mode itself: a late start cannot move the old number.

    Without this the fix could be reverted and the suite would still pass on
    the new column alone.
    """
    print("the stream metric is blind to it -- the reason the fix was needed")
    ok = []
    # one asset, live every second from minute 90 on: no internal gap at all
    ts = np.array(grid(90, 100), dtype=np.int64)
    gaps = np.diff(ts) / 1000.0
    span = (ts[-1] - ts[0]) / 1000.0
    stream_cover = 1.0 - (np.median(gaps) / span)
    ok.append(check("stream cover reads as complete", stream_cover > 0.99, True))
    cov, _, _ = mq.window_coverage(set(ts // MIN), WINDOW)
    ok.append(check("game cover does not", cov < 0.15, True))
    return all(ok)


def test_hollow_middle_is_caught():
    """Starts on time, ends on time, dead for the hour in between."""
    print("neither endpoint can see a recording that died in the middle")
    ok = []
    mins = minutes(0, 20) | minutes(80, 101)
    cov, late, early = mq.window_coverage(mins, WINDOW)
    ok.append(approx("game cover", cov, 41 / 101))
    ok.append(approx("late start", late, 0.0))
    ok.append(approx("early stop", early, 0.0))

    r = mf.match_report(frame(grid(0, 20) + grid(80, 101)),
                        windows={"mlb-aaa-bbb-2026-09-21": WINDOW})
    ok.append(check("match_report drops it", bool(r.keep.iloc[0]), False))
    ok.append(check("reason names the coverage",
                    "of the game covered" in r.reason.iloc[0], True))
    return all(ok)


def test_a_complete_recording_is_kept():
    """The control: a normal night must not be caught by any of the new rules."""
    print("a complete recording is still kept")
    ok = []
    cov, late, early = mq.window_coverage(minutes(-30, 101), WINDOW)
    ok.append(approx("game cover", cov, 1.0))
    ok.append(approx("late start", late, 0.0))
    ok.append(approx("early stop", early, 0.0))

    # pregame included, as the recorder really does
    r = mf.match_report(frame(grid(-30, 101)),
                        windows={"mlb-aaa-bbb-2026-09-21": WINDOW})
    ok.append(check("match_report keeps it", bool(r.keep.iloc[0]), True))
    ok.append(check("no reason given", r.reason.iloc[0], ""))
    return all(ok)


def test_no_window_is_not_a_pass():
    """A match with no window cannot be judged, so it must not be judged GOOD."""
    print("an unjudgeable match is not a passing match")
    ok = []
    cov, late, early = mq.window_coverage(minutes(0, 101), None)
    ok.append(check("game cover is None", cov is None, True))
    verdict, reasons = mq.judge(dict(
        reseeds=0, outage_s=0.0, coverage=1.0, lag_spread_ms=0.0,
        worst_gap_s=0.0, attributable=True, n=10_000,
        game_cover=None, game_minutes=0, game_minutes_seen=0,
        late_start_s=None, early_stop_s=None))
    ok.append(check("verdict", verdict, "SUSPECT"))
    ok.append(check("reason names the missing window",
                    any("no game window" in r for r in reasons), True))

    r = mf.match_report(frame(grid(0, 101)), windows={})
    ok.append(check("match_report drops it", bool(r.keep.iloc[0]), False))
    return all(ok)


def test_sub_minute_holes_need_the_exclusion_list():
    """The 08-30 shape: 40s of every minute missing for 25 minutes.

    Every minute still holds a record, so minute coverage is 100% and the
    match is kept -- 16.7 minutes of the game gone and nothing notices. Only
    the continuity scan sees it, and the exclusion list is how that verdict
    reaches the filter.
    """
    print("holes shorter than a minute pass minute coverage; the list catches them")
    import continuity_scan as cs
    ok = []
    slug = "mlb-aaa-bbb-2026-09-21"
    rows = [t for t in grid(0, 101)
            if not (10 * MIN <= t - START < 35 * MIN and (t - START) % MIN >= 20_000)]
    w = {slug: WINDOW}

    r = mf.match_report(frame(rows, slug), windows=w, excluded={})
    ok.append(check("minute coverage alone keeps it", bool(r.keep.iloc[0]), True))

    r = mf.match_report(frame(rows, slug), windows=w,
                        excluded={slug: "lost 16.7 of 100 min"})
    ok.append(check("the exclusion list drops it", bool(r.keep.iloc[0]), False))
    ok.append(check("reason says why", r.reason.iloc[0].startswith("excluded:"), True))

    v = mf.judge_match(WINDOW, START, END, minutes(0, 101), slug=slug,
                       excluded={slug: "x"})
    ok.append(check("judge_match honours it (trim_sessions path)", v["keep"], False))

    # the classifier's three bands, on the gaps the scan would report
    def row(*gaps):
        return dict(window=True, game_min=100.0, gaps=[
            dict(secs=s, disconnect=d, at_start=False, at_end=False) for s, d in gaps])
    ok.append(check("25 x 40s holes: severe",
                    cs.classify(row(*[(40.0, False)] * 25))[0], "severe"))
    ok.append(check("one 140s hole: slight",
                    cs.classify(row((140.0, True)))[0], "slight"))
    ok.append(check("a 13s reconnect: slight",
                    cs.classify(row((13.0, True)))[0], "slight"))
    ok.append(check("8s of quiet, no disconnect: continuous",
                    cs.classify(row((8.0, False)))[0], "continuous"))
    return all(ok)


def test_a_decided_game_is_not_an_early_stop():
    """hou-oak 2026-09-26: moneyline at 0.9975, grid ends 28 min early.

    The recording ran to the final out; the book went one-sided once the
    result was certain, and the grid keeps only two-sided rows. That must not
    read as a truncated recording -- but an UNDECIDED game that ends early
    still must.
    """
    print("a decided game's early end is kept; an undecided one is not")
    ok = []
    slug = "mlb-aaa-bbb-2026-09-21"
    rows = grid(0, 70)                      # ends 30 min before the final out
    w = {slug: WINDOW}

    def with_mid(last_mid):
        F = frame(rows, slug)
        F["mid"] = 0.6
        F.loc[F.index[-1], "mid"] = last_mid
        return F

    r = mf.match_report(with_mid(0.9975), windows=w, excluded={})
    ok.append(check("decided (0.9975): kept", bool(r.keep.iloc[0]), True))
    ok.append(approx("reports how early it settled", r.settled_min.iloc[0], 30.0, tol=0.1))
    r = mf.match_report(with_mid(0.004), windows=w, excluded={})
    ok.append(check("decided the other way (0.004): kept", bool(r.keep.iloc[0]), True))

    r = mf.match_report(with_mid(0.55), windows=w, excluded={})
    ok.append(check("undecided (0.55): dropped", bool(r.keep.iloc[0]), False))
    ok.append(check("reason is the early stop", "before the final out" in r.reason.iloc[0], True))
    r = mf.match_report(frame(rows, slug), windows=w, excluded={})
    ok.append(check("no mid column: nothing counts as settled", bool(r.keep.iloc[0]), False))

    r = mf.match_report(with_mid(0.9975), windows=w, excluded={slug: "raw died"})
    ok.append(check("the exclusion list still wins", bool(r.keep.iloc[0]), False))

    # the trim_sessions path: judge_match with the streamed flag
    v = mf.judge_match(WINDOW, START, START + 70 * MIN - 1000, minutes(0, 70),
                       settled=True)
    ok.append(check("judge_match(settled=True) keeps it", v["keep"], True))
    v = mf.judge_match(WINDOW, START, START + 70 * MIN - 1000, minutes(0, 70))
    ok.append(check("judge_match without the flag drops it", v["keep"], False))
    return all(ok)


if __name__ == "__main__":
    print("game-window coverage\n" + "=" * 70)
    results = [test_late_start_is_visible(),
               test_the_old_metric_really_does_miss_it(),
               test_hollow_middle_is_caught(),
               test_a_complete_recording_is_kept(),
               test_no_window_is_not_a_pass(),
               test_sub_minute_holes_need_the_exclusion_list(),
               test_a_decided_game_is_not_an_early_stop()]
    print("=" * 70)
    print(f"{sum(results)}/{len(results)} groups passed")
    sys.exit(0 if all(results) else 1)
