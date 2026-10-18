"""Correctness tests for the label and feature construction.

The label is the one thing in this study that is allowed to look forward, so
it is the one thing most likely to look forward by MORE than intended. These
tests pin down the exact window and prove the masking of non-contiguous rows.

    python test_features.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FE


def check(name, cond, detail=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def test_label_window_is_exactly_forward():
    """max over t+1..t+H inclusive, never including t, never beyond t+H."""
    print("label window")
    n, h = 40, 5                      # H = 1s on a 200ms grid
    mid = np.full(n, 0.50)
    mid[10] = 0.60                    # spike 10 steps in
    sid = np.zeros(n, np.int64)
    ts = np.arange(n, dtype=np.int64) * 200
    L = FE.jump_labels(mid, sid, ts, horizon_s=1.0, thresholds=[0.05])
    mv = L.future_move.to_numpy()
    ok = check("row 5 sees the spike at row 10 (5 steps ahead)",
               abs(mv[5] - 0.10) < 1e-9, f"got {mv[5]:.4f}")
    ok &= check("row 9 sees it (1 step ahead)", abs(mv[9] - 0.10) < 1e-9,
                f"got {mv[9]:.4f}")
    ok &= check("row 4 does NOT see it (6 steps ahead, beyond H)",
                abs(mv[4]) < 1e-9, f"got {mv[4]:.4f}")
    # Row 10 IS the spike and its future is back at 0.50, so a 0.10 move there
    # is correct -- it is the move back down. To test that t itself is excluded
    # the price has to STAY at the new level: then the row standing on the step
    # has no future move at all, while the rows before it do.
    step = np.where(np.arange(n) >= 10, 0.60, 0.50)
    L2 = FE.jump_labels(step, sid, ts, horizon_s=1.0, thresholds=[0.05])
    m2 = L2.future_move.to_numpy()
    ok &= check("the row standing ON a step change sees no further move",
                abs(m2[10]) < 1e-9, f"got {m2[10]:.4f}")
    ok &= check("the row just before the step does see it",
                abs(m2[9] - 0.10) < 1e-9, f"got {m2[9]:.4f}")
    ok &= check("a jump already underway cannot label itself",
                abs(m2[12]) < 1e-9, f"got {m2[12]:.4f}")
    return ok


def test_no_cross_series_contamination():
    """Two series concatenated must not label across the boundary.

    This is the failure the first implementation had: groupby().rolling()
    returns rows in GROUP order, so .to_numpy() silently permuted the labels
    and produced moves computed between unrelated markets -- which inflated
    every prevalence.
    """
    print("cross-series isolation")
    a = np.full(30, 0.20)
    b = np.full(30, 0.80)             # a wildly different level
    mid = np.concatenate([a, b])
    sid = np.concatenate([np.zeros(30, np.int64), np.ones(30, np.int64)])
    ts = np.concatenate([np.arange(30), np.arange(30)]).astype(np.int64) * 200
    L = FE.jump_labels(mid, sid, ts, horizon_s=1.0, thresholds=[0.05])
    mv, v = L.future_move.to_numpy(), L.valid_label.to_numpy()
    ok = check("no row reports a move between the two levels",
               not (mv[v] > 0.01).any(),
               f"max move {mv[v].max():.4f} (0.60 would mean contamination)")
    ok &= check("rows near the end of series 0 are invalid, not cross-labelled",
                not v[25:30].any(), f"valid flags {v[25:30]}")
    ok &= check("series 1 rows are labelled", v[30:50].any())
    return ok


def test_time_gaps_invalidate():
    """A gap in the grid must invalidate, not bridge."""
    print("time gaps")
    mid = np.full(40, 0.50)
    mid[25] = 0.70
    sid = np.zeros(40, np.int64)
    ts = np.arange(40, dtype=np.int64) * 200
    ts[20:] += 600_000                # ten-minute hole before row 20
    L = FE.jump_labels(mid, sid, ts, horizon_s=1.0, thresholds=[0.05])
    v = L.valid_label.to_numpy()
    ok = check("rows whose window spans the hole are invalid",
               not v[15:20].any(), f"valid flags {v[15:20]}")
    ok &= check("rows wholly after the hole are valid", v[20:34].any())
    return ok


def test_order_is_preserved():
    """Labels must line up with the rows they belong to, in input order."""
    print("row order")
    rng = np.random.default_rng(3)
    # interleave three series so any group-order permutation is visible
    sid = np.tile(np.arange(3), 40).astype(np.int64)
    order = np.argsort(sid, kind="stable")
    sid = sid[order]                              # grouped, as the builder emits
    mid = np.where(sid == 0, 0.20, np.where(sid == 1, 0.50, 0.80))
    mid = mid + rng.normal(0, 1e-6, len(mid))
    ts = np.concatenate([np.arange((sid == s).sum()) for s in range(3)]) * 200
    L = FE.jump_labels(mid, sid, ts.astype(np.int64), 1.0, [0.05])
    mv = L.future_move.to_numpy()
    ok = check("no fabricated large move anywhere",
               np.nanmax(mv[L.valid_label.to_numpy()]) < 0.01,
               f"max {np.nanmax(mv[L.valid_label.to_numpy()]):.4f}")
    ok &= check("output length matches input", len(L) == len(mid))
    return ok


def test_missing_levels_are_masked():
    """A level that does not exist must not read as a quote at a distance."""
    print("depth mask")
    n, K = 5, 10
    T = np.zeros((n, 4, K), np.float32)
    T[:, 1, :3] = np.log1p(np.array([100.0, 50.0, 25.0]))   # 3 bid levels
    T[:, 3, :2] = np.log1p(np.array([80.0, 40.0]))          # 2 ask levels
    T[:, 0, :] = -5.0                                        # fabricated dists
    T[:, 2, :] = +5.0
    d = FE.decode(T, np.full(n, 0.5))
    ok = check("bid mask marks exactly the 3 real levels",
               (d["bid_mask"][0] == np.r_[1, 1, 1, np.zeros(7)]).all(),
               f"{d['bid_mask'][0]}")
    ok &= check("ask mask marks exactly the 2 real levels",
                (d["ask_mask"][0] == np.r_[1, 1, np.zeros(8)]).all())
    ok &= check("missing levels carry zero distance, not a fake quote",
                (d["bid_dist"][0][3:] == 0).all() and (d["ask_dist"][0][2:] == 0).all())
    ok &= check("dollars recovered from log1p",
                abs(d["bid_usd"][0, 0] - 100.0) < 0.01,
                f"got {d['bid_usd'][0,0]:.2f}")
    S = FE.state_features(d, spread_ticks=np.ones(n))
    ok &= check("levels_bid counts 3", (S.levels_bid == 3).all())
    ok &= check("entropy of a 3-level book is below log(10)",
                bool((S.entropy_bid < np.log(10)).all()))
    return ok


def test_trajectory_is_causal():
    """Trailing changes only; a future spike must not move a past feature."""
    print("trajectory causality")
    n = 100
    T = np.zeros((n, 4, 10), np.float32)
    T[:, 1, 0] = np.log1p(100.0)
    T[:, 3, 0] = np.log1p(100.0)
    T[60:, 1, 0] = np.log1p(10_000.0)          # depth jumps at row 60
    sid = np.zeros(n, np.int64)
    d = FE.decode(T, np.full(n, 0.5))
    S = FE.state_features(d, spread_ticks=np.ones(n))
    Tr = FE.trajectory_features(S, sid, lags_s=(5,), grid_ms=200,
                                book_age_ms=np.zeros(n), mid=np.full(n, 0.5))
    c = Tr["d_log_bid_depth_5s"].to_numpy()
    ok = check("rows before the change see no change",
               np.allclose(np.nan_to_num(c[:59]), 0.0, atol=1e-5),
               f"max |change| before = {np.nanmax(np.abs(c[:59])):.4f}")
    ok &= check("rows after the change do see it",
                np.nanmax(np.abs(c[60:90])) > 1.0)
    return ok


if __name__ == "__main__":
    print("jump-prediction feature/label construction\n" + "=" * 62)
    r = [test_label_window_is_exactly_forward(),
         test_no_cross_series_contamination(),
         test_time_gaps_invalidate(),
         test_order_is_preserved(),
         test_missing_levels_are_masked(),
         test_trajectory_is_causal()]
    print("=" * 62)
    print(f"{sum(r)}/{len(r)} groups passed")
    sys.exit(0 if all(r) else 1)
