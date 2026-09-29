"""Regression tests for mark_single_jumps.py, on synthetic books.

Each case is a price path whose right answer is known by construction, one per
rule that the visual review of the real marks forced into the detector:

  * a clean jump between two quoted levels is ONE single mark, priced at the
    ask before and a bid that stood after;
  * two jumps separated by a calm, quoted pause are two jumps: neither alone
    breaks even here, and because the book went back to normal in between,
    the pair is NOT marked;
  * an emptied book standing still in the middle of a jump does not split it;
  * a bid resting part-way up while the ask flickers DOES split it;
  * a quote pull that comes back where it was is not a jump;
  * two small jumps with the book blown out throughout, bid resting between,
    are marked as COMBINED;
  * a recording silence inside a mark's window excludes it.

    python research/makinen/test_single_jumps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mark_single_jumps as M  # noqa: E402

NOGAP = np.empty((0, 2), np.int64)
T0 = 1_790_000_000_000


class Path_:
    """Build a 200ms book path segment by segment."""

    def __init__(self):
        self.bid, self.ask = [], []

    def hold(self, bid, ask, secs):
        n = int(round(secs * 5))
        self.bid += [bid] * n
        self.ask += [ask] * n
        return self

    def ramp(self, bid0, bid1, ask0, ask1, secs):
        n = int(round(secs * 5))
        self.bid += list(np.round(np.linspace(bid0, bid1, n), 2))
        self.ask += list(np.round(np.linspace(ask0, ask1, n), 2))
        return self

    def frame(self):
        b, a = np.array(self.bid), np.array(self.ask)
        n = len(b)
        return pd.DataFrame({
            "ts": T0 + np.arange(n, dtype=np.int64) * M.GRID_MS,
            "bid": b, "ask": a, "spread_ticks": np.round((a - b) / M.TICK, 6),
            "bid_usd": np.full(n, 500.0), "ask_usd": np.full(n, 500.0),
            "series": "mlb-test-2026-01-01|moneyline|None|123",
        })


def calm(p, bid, secs=60):
    return p.hold(bid, round(bid + 0.01, 2), secs)


def run(p, gaps=NOGAP):
    out = pd.DataFrame(M.scan_series(p.frame(), gaps))
    if len(out):
        out["slug"] = "mlb-test-2026-01-01"
        M.check_no_overlap(out)
    return out


def kept(out):
    return out[out.excluded == ""] if len(out) else out


def test_clean_single():
    p = calm(Path_(), 0.50)
    p.ramp(0.50, 0.60, 0.55, 0.66, 8)          # disturbed burst
    calm(p, 0.60)
    k = kept(run(p))
    assert len(k) == 1 and k.kind.iat[0] == "single", k
    assert abs(k.entry_ask.iat[0] - 0.51) < 1e-9
    assert abs(k.exit_bid.iat[0] - 0.60) < 1e-9


def test_calm_pause_splits_and_pair_not_combined():
    # two 2-cent jumps through a blown-out book, calm quoted pause between:
    # buy 0.51 -> sell 0.52 alone nets < 0; the pair would, but the book went
    # back to normal on the pause, so it is two separate jumps.
    p = calm(Path_(), 0.50)
    p.hold(0.40, 0.70, 3)
    calm(p, 0.52, 30)
    p.hold(0.40, 0.70, 3)
    calm(p, 0.54)
    k = kept(run(p))
    assert len(k) == 0, k[["kind", "entry_ask", "exit_bid", "net_ticks"]]


def test_still_empty_book_does_not_split():
    p = calm(Path_(), 0.50)
    p.hold(0.30, 0.80, 8)                      # emptied book, motionless > PLATEAU_S
    calm(p, 0.60)
    k = kept(run(p))
    assert len(k) == 1 and k.kind.iat[0] == "single", k


def test_bid_rest_part_way_splits():
    p = calm(Path_(), 0.50)
    # bid sits at 0.55 for 8s while the ask flickers between two far prices
    for _ in range(8):
        p.hold(0.55, 0.70, 0.4).hold(0.55, 0.90, 0.6)
    calm(p, 0.60)
    k = kept(run(p))
    # the move is cut at the rest: the first jump sells into the 0.55 bid.
    # The second (buy the rest's 0.70 ask, sell 0.60) loses and is not marked;
    # without the rest rule this would be one 0.51 -> 0.60 single.
    assert len(k) == 1 and k.kind.iat[0] == "single", k[["kind", "entry_ask", "exit_bid"]]
    assert abs(k.exit_bid.iat[0] - 0.55) < 1e-9


def test_quote_pull_is_not_a_jump():
    p = calm(Path_(), 0.50)
    p.hold(0.10, 0.95, 4)
    calm(p, 0.50)
    assert len(kept(run(p))) == 0


def test_combined_when_book_stays_disturbed():
    # the book is blown out through both steps and the pause; the bid rests
    # at 0.52 in between. Buy 0.51: sell 0.52 alone and 0.54 from the rest
    # each fail, 0.51 -> 0.54 clears.
    p = calm(Path_(), 0.50, 300)
    p.hold(0.45, 0.75, 3)
    p.hold(0.52, 0.75, 8)
    p.hold(0.45, 0.75, 3)
    calm(p, 0.54, 300)
    k = kept(run(p))
    assert len(k) == 1 and k.kind.iat[0] == "combined", k[["kind", "net_ticks"]]
    assert k.pause_regime.iat[0] == "disturbed"


def test_chained_combined_do_not_share_a_step():
    # a calm staircase of instant 2-cent steps with 10s pauses, the shape that
    # tripped check_no_overlap on 2026-09-17. No single step pays (buy 0.51,
    # sell 0.52), A->C pays and so does B->D; marking both would count B->C
    # twice, so only A->C may be marked.
    p = calm(Path_(), 0.50, 300)
    calm(p, 0.52, 10)
    calm(p, 0.54, 10)
    calm(p, 0.56, 300)
    k = kept(run(p))                 # run() also asserts no overlap
    assert len(k) == 1 and k.kind.iat[0] == "combined", k
    assert abs(k.entry_ask.iat[0] - 0.51) < 1e-9 and abs(k.exit_bid.iat[0] - 0.54) < 1e-9


def test_recording_gap_excludes():
    p = calm(Path_(), 0.50)
    p.ramp(0.50, 0.60, 0.55, 0.66, 8)
    calm(p, 0.60)
    f = p.frame()
    t_mid = int(f.ts.iat[len(f) // 2])
    out = run(p, np.array([[t_mid - 6000, t_mid]], np.int64))
    assert len(kept(out)) == 0 and "recording_gap" in out.excluded.iat[0]


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
            n += 1
    print(f"{n} passed")
