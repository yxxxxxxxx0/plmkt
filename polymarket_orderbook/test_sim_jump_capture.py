"""Unit tests for the jump-capture simulator's two load-bearing primitives.

Both are easy to get subtly wrong in the flattering direction -- a zigzag that
emits overlapping legs double-counts profit, and a ladder walk that ignores
the size at a level under-charges for depth. Neither error would look wrong in
the aggregate output, so they are pinned here instead.
"""
import numpy as np

from sim_jump_capture import walk_ladder, zigzag_up_legs, bucket_of


def test_zigzag_single_swing():
    v = np.array([0.50, 0.51, 0.52, 0.51, 0.50])
    assert zigzag_up_legs(v, 0.01) == [(0, 2)]


def test_zigzag_monotonic_emits_final_leg():
    v = np.array([0.50, 0.51, 0.52])
    assert zigzag_up_legs(v, 0.01) == [(0, 2)]


def test_zigzag_flat_no_legs():
    v = np.array([0.50, 0.501, 0.4995, 0.50])
    assert zigzag_up_legs(v, 0.01) == []


def test_zigzag_legs_never_overlap():
    """The core anti-double-count property: buy index of each leg must be at
    or after the previous leg's sell index."""
    rng = np.random.default_rng(0)
    v = 0.5 + np.cumsum(rng.normal(0, 0.002, 5000))
    for theta in (0.005, 0.01, 0.03):
        legs = zigzag_up_legs(v, theta)
        for (i0, j0), (i1, _) in zip(legs, legs[1:]):
            assert i0 < j0, "leg must buy before it sells"
            assert i1 >= j0, f"overlapping legs {(i0, j0)} then {i1}"


def test_zigzag_each_leg_clears_threshold():
    rng = np.random.default_rng(1)
    v = 0.5 + np.cumsum(rng.normal(0, 0.002, 5000))
    theta = 0.01
    for i, j in zigzag_up_legs(v, theta):
        assert v[j] - v[i] >= theta - 1e-12


def test_zigzag_skips_nan():
    v = np.array([0.50, np.nan, 0.52, 0.51, 0.50])
    legs = zigzag_up_legs(v, 0.01)
    assert legs == [(0, 2)]


def test_walk_ladder_partial_second_level():
    px = np.array([0.55, 0.56])
    sz = np.array([100.0, 200.0])
    off = np.array([0, 2])
    filled, notional, basis = walk_ladder(0, off, px, sz, 250.0)
    assert filled == 250.0
    assert abs(notional - (0.55 * 100 + 0.56 * 150)) < 1e-12
    assert abs(basis - (0.55 * 0.45 * 100 + 0.56 * 0.44 * 150)) < 1e-12


def test_walk_ladder_caps_at_available():
    px = np.array([0.55])
    sz = np.array([10.0])
    off = np.array([0, 1])
    filled, notional, _ = walk_ladder(0, off, px, sz, 1000.0)
    assert filled == 10.0
    assert abs(notional - 5.5) < 1e-12


def test_walk_ladder_empty_side():
    off = np.array([0, 0])
    assert walk_ladder(0, off, np.array([]), np.array([]), 100.0) == (0.0, 0.0, 0.0)


def test_walk_ladder_vwap_is_worse_than_touch():
    """Depth must cost money: the average fill can never beat the touch."""
    px = np.array([0.55, 0.56, 0.57])
    sz = np.array([5.0, 5.0, 5.0])
    off = np.array([0, 3])
    filled, notional, _ = walk_ladder(0, off, px, sz, 15.0)
    assert notional / filled > px[0]


def test_bucket_boundaries():
    assert bucket_of(0.0) == 0
    assert bucket_of(0.50) == 3
    assert bucket_of(1.0) == len(__import__("sim_jump_capture").PRICE_BUCKETS) - 1


def test_zigzag_leading_nan_still_finds_legs():
    """A one-sided book at the window start must not disable the whole token.

    Seeding the swing filter at index 0 when v[0] is NaN makes every
    comparison False, so i_lo/i_hi freeze and the token silently reports no
    swings at all. That is a wrong answer that looks like a quiet market.
    """
    v = np.array([np.nan, 0.50, 0.51, 0.52, 0.51, 0.50])
    assert zigzag_up_legs(v, 0.01) == [(1, 3)]


def test_zigzag_all_nan_is_empty():
    v = np.array([np.nan, np.nan, np.nan])
    assert zigzag_up_legs(v, 0.01) == []


def test_zigzag_nan_run_then_swing():
    v = np.array([np.nan, np.nan, 0.40, 0.42, 0.41])
    assert zigzag_up_legs(v, 0.01) == [(2, 3)]


def test_zigzag_last_tick_only_valid_is_empty():
    v = np.array([np.nan, np.nan, 0.40])
    assert zigzag_up_legs(v, 0.01) == []


def test_agg_move_is_share_weighted():
    """Per-share columns must all share one denominator.

    Averaging move per TRADE while averaging gross per SHARE is only
    equivalent when every trade is the same size. Depth caps trade size, so
    they are not, and the mismatch surfaces as execution cost that no one
    actually paid. This pins the two to the same weighting: a 1000-share trade
    moving 1c and a 1-share trade moving 100c average to (1000*1 + 1*100)/1001
    per share, not (1 + 100)/2.
    """
    from sim_jump_capture import Agg
    a = Agg()
    a.add(1000.0, 10.0, 0.0, 1.0, 0.01, 3, "x")
    a.add(1.0, 1.0, 0.0, 1.0, 1.00, 3, "x")
    per_share = a.move / a.shares
    assert abs(per_share - (1000 * 0.01 + 1 * 1.00) / 1001) < 1e-12
    assert abs(per_share - 0.011) < 1e-4      # share-weighted: ~1.1c
    assert abs((0.01 + 1.00) / 2 - 0.505) < 1e-12   # trade-weighted: 50.5c
