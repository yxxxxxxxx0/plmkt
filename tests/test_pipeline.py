"""Smoke tests for the pieces that are easy to break silently."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlbmodel.backtest import bet_sim, calibration_slope  # noqa: E402
from mlbmodel.baselines import log5, runs_model  # noqa: E402
from mlbmodel.odds import american_to_decimal, american_to_prob  # noqa: E402


def test_american_odds_conversion():
    # -110 and +100 are the two anchors every bettor knows
    assert abs(american_to_prob(np.array([-110.0]))[0] - 0.5238) < 1e-3
    assert abs(american_to_prob(np.array([100.0]))[0] - 0.5) < 1e-9
    assert abs(american_to_decimal(np.array([-110.0]))[0] - 1.9091) < 1e-3
    assert abs(american_to_decimal(np.array([150.0]))[0] - 2.5) < 1e-9
    # both branches must stay finite for either sign (np.where evaluates both)
    v = np.array([-250.0, -100.0, 100.0, 400.0])
    assert np.isfinite(american_to_prob(v)).all()
    assert np.isfinite(american_to_decimal(v)).all()
    print("  ok: american odds conversions")


def test_log5_properties():
    # equal teams on neutral ground -> 50%
    assert abs(log5(np.array([.6]), np.array([.6]), 1.0)[0] - 0.5) < 1e-9
    # the better team is favoured, and home advantage helps
    assert log5(np.array([.6]), np.array([.4]), 1.0)[0] > 0.5
    assert log5(np.array([.5]), np.array([.5]), 1.18)[0] > 0.5
    # symmetry: swapping teams complements the probability
    a, b = np.array([.62]), np.array([.44])
    assert abs(log5(a, b, 1.0)[0] + log5(b, a, 1.0)[0] - 1.0) < 1e-9
    print("  ok: log5 symmetry and monotonicity")


def test_runs_model_monotonic():
    lam = np.array([4.0, 5.0, 6.0])
    p = runs_model(lam, np.full(3, 4.5))
    assert np.all(np.diff(p) > 0), "more expected runs must raise win probability"
    assert abs(runs_model(np.array([4.5]), np.array([4.5]))[0] - 0.5) < 0.02
    assert np.all((p > 0) & (p < 1))
    print("  ok: run model is monotonic and centred")


def test_bet_sim_arithmetic():
    # one bet, model certain, price 2.0, and it wins -> +1.0 profit on 1 unit
    r = bet_sim(np.array([1.0]), np.array([1]), np.array([2.0]), np.array([2.0]),
                edge_thresh=0.0)
    assert r["bets"] == 1 and abs(r["flat_pnl"] - 1.0) < 1e-9
    # same bet, loses -> -1.0
    r = bet_sim(np.array([1.0]), np.array([0]), np.array([2.0]), np.array([2.0]),
                edge_thresh=0.0)
    assert abs(r["flat_pnl"] + 1.0) < 1e-9
    # a fair price with no edge places no bet
    r = bet_sim(np.array([0.5]), np.array([1]), np.array([2.0]), np.array([2.0]),
                edge_thresh=0.01)
    assert r["bets"] == 0
    print("  ok: betting simulation arithmetic")


def test_calibration_slope_recovers_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 20000)
    y = (rng.uniform(size=20000) < p).astype(int)   # perfectly calibrated
    assert abs(calibration_slope(p, y) - 1.0) < 0.12
    # an over-confident model must score below 1
    over = np.clip(0.5 + (p - 0.5) * 2.5, 1e-3, 1 - 1e-3)
    assert calibration_slope(over, y) < 0.85
    print("  ok: calibration slope recovers 1.0 and flags over-confidence")


def test_no_market_features_in_model_inputs():
    """The models must never see the closing line -- that would be the
    ultimate leak, since the market is the thing we are trying to beat."""
    from mlbmodel import features
    from mlbmodel.config import PROC
    df = pd.read_parquet(PROC / "dataset.parquet")
    cols = features.feature_columns(df, include_market=False)
    banned = ("mkt_", "ml_home", "ml_away", "dec_home", "dec_away",
              "pub_home", "snap_lag", "home_score", "away_score", "home_win")
    bad = [c for c in cols if any(b in c for b in banned)]
    assert not bad, f"market/outcome columns leaked into the feature set: {bad}"
    print(f"  ok: {len(cols)} features, none derived from odds or the final score")


if __name__ == "__main__":
    for fn in [test_american_odds_conversion, test_log5_properties,
               test_runs_model_monotonic, test_bet_sim_arithmetic,
               test_calibration_slope_recovers_one,
               test_no_market_features_in_model_inputs]:
        print(f"{fn.__name__}:")
        fn()
    print("\nall pipeline checks passed")
