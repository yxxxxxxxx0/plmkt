"""Classical, non-ML baselines that the sabermetrics literature actually uses.

These matter because they set the bar a machine-learning model has to clear.
In baseball that bar is high and the headroom above it is thin.

  log5          Bill James' odds-ratio matchup formula (a Bradley-Terry model)
  pythag_log5   the same, but on Pythagorean win expectation rather than W-L
  runs_nb       simulate the game from expected runs per side, using a negative
                binomial (Poisson over-disperses badly for baseball scoring)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom


def log5(p_a: np.ndarray, p_b: np.ndarray, hfa_odds: float = 1.18) -> np.ndarray:
    """P(A beats B) given each side's win probability vs a league-average team.

    `hfa_odds` multiplies A's odds; 1.18 corresponds to the ~54% MLB home rate.
    """
    p_a = np.clip(p_a, 1e-3, 1 - 1e-3)
    p_b = np.clip(p_b, 1e-3, 1 - 1e-3)
    num = p_a - p_a * p_b
    den = p_a + p_b - 2 * p_a * p_b
    p = num / den
    o = p / (1 - p) * hfa_odds
    return o / (1 + o)


def _nb_pmf(mean: np.ndarray, k: np.ndarray, disp: float) -> np.ndarray:
    """Negative binomial P(X=k) parameterised by mean and dispersion `disp`
    (variance = mean + mean^2/disp). disp -> inf recovers Poisson."""
    r = disp
    p = r / (r + mean[:, None])
    return nbinom.pmf(k[None, :], r, p)


def runs_model(lam_home: np.ndarray, lam_away: np.ndarray,
               disp: float = 6.0, max_runs: int = 25) -> np.ndarray:
    """P(home wins) from expected runs, integrating over the score matrix.

    Ties are impossible in baseball, so the tie mass (both sides same score,
    i.e. the game going to extras) is split by the relative run rates -- a
    good approximation to who wins an extra-inning game.
    """
    k = np.arange(max_runs + 1)
    ph = _nb_pmf(np.clip(lam_home, 0.5, 15), k, disp)
    pa = _nb_pmf(np.clip(lam_away, 0.5, 15), k, disp)
    ph /= ph.sum(axis=1, keepdims=True)
    pa /= pa.sum(axis=1, keepdims=True)

    cum_a = np.cumsum(pa, axis=1)
    lower_a = np.concatenate([np.zeros((len(pa), 1)), cum_a[:, :-1]], axis=1)
    p_win = (ph * lower_a).sum(axis=1)          # home scores strictly more
    p_tie = (ph * pa).sum(axis=1)
    share = lam_home / (lam_home + lam_away)
    return p_win + p_tie * share


def expected_runs(df: pd.DataFrame, lg_rpg: float = 4.45) -> tuple[np.ndarray, np.ndarray]:
    """Offence x opposing-defence, on the classic multiplicative scale, with a
    park adjustment and a starting-pitcher term."""
    def col(name, default):
        return df[name].to_numpy(dtype=float) if name in df.columns else np.full(len(df), default)

    ho = col("home_off_rpg_40", lg_rpg)
    ao = col("away_off_rpg_40", lg_rpg)
    hd = col("home_def_rpg_40", lg_rpg)
    ad = col("away_def_rpg_40", lg_rpg)
    park = col("park_factor", 1.0)

    # starter quality: FIP relative to league, weighted by the share of the
    # game a starter typically covers (~5.2 of 9 innings)
    h_fip = col("home_sp_fip_40", 4.15)
    a_fip = col("away_sp_fip_40", 4.15)
    h_ip = np.clip(col("home_sp_ip_40", 5.1), 2, 8)
    a_ip = np.clip(col("away_sp_ip_40", 5.1), 2, 8)
    sp_home_mult = 1 + (h_fip / 4.15 - 1) * (h_ip / 9.0)   # affects runs ALLOWED by home
    sp_away_mult = 1 + (a_fip / 4.15 - 1) * (a_ip / 9.0)

    lam_home = (ho * ad / lg_rpg) * park * sp_away_mult * 1.02   # small home boost
    lam_away = (ao * hd / lg_rpg) * park * sp_home_mult * 0.98
    return np.nan_to_num(lam_home, nan=lg_rpg), np.nan_to_num(lam_away, nan=lg_rpg)


class Log5Baseline:
    def __init__(self, col="wpct_std", hfa_odds=1.18):
        self.col, self.hfa = col, hfa_odds

    def fit(self, X, y, frame=None):
        return self

    def predict_proba_frame(self, frame):
        # KeyError here is caught upstream: a league without this column
        # (e.g. no Pythagorean expectation) just skips this baseline.
        h = frame[f"home_{self.col}"].fillna(0.5).to_numpy()
        a = frame[f"away_{self.col}"].fillna(0.5).to_numpy()
        return log5(h, a, self.hfa)


class RunsBaseline:
    def __init__(self, disp=6.0):
        self.disp = disp

    def fit(self, X, y, frame=None):
        return self

    def predict_proba_frame(self, frame):
        if "home_off_rpg_40" not in frame.columns:
            raise KeyError("home_off_rpg_40")   # not a baseball-shaped frame
        lh, la = expected_runs(frame)
        return runs_model(lh, la, self.disp)


EXTRA_BASELINES = {
    "log5_wpct": lambda: Log5Baseline("wpct_std"),
    "log5_pythag": lambda: Log5Baseline("pyth_std"),
    "runs_negbin": lambda: RunsBaseline(disp=6.0),
}
