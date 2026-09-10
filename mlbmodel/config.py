"""Shared paths and constants for the MLB prediction pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
OUT = ROOT / "reports"

for _p in (DATA, RAW, PROC, OUT):
    _p.mkdir(parents=True, exist_ok=True)

API = "https://statsapi.mlb.com/api/v1"

# Seasons pulled from the API. We need a multi-year burn-in before the test
# window so Elo ratings and rolling stats are warm by the time we score 2023+.
SEASONS = list(range(2015, 2027))

# Walk-forward test seasons -- "3 years ago until now" from the 2026 season.
TEST_SEASONS = [2023, 2024, 2025, 2026]

# 2020 was a 60-game COVID season; kept for Elo continuity but excluded from
# model training/eval by default because its run environment is an outlier.
ODD_SEASONS = {2020}

# --- Elo hyperparameters (MLB-tuned, in the spirit of FiveThirtyEight's model) ---
ELO_START = 1500.0
ELO_K = 4.0            # baseball has ~162 games and high per-game noise -> small K
ELO_HFA = 24.0         # home-field advantage in Elo points (~54% home win rate)
ELO_REVERT = 0.30      # fraction reverted to 1500 between seasons
ELO_MOV_BASE = 3.00    # margin-of-victory dampener
# Elo points per point of starting-pitcher game score above league average.
# All five values above were grid-searched on 2018-2022 only (see tune.py);
# the 2023-2026 test window had no influence on them.
SP_ELO_SCALE = 3.50

# Starting-pitcher adjustment
SP_PRIOR_STARTS = 10.0   # regression-to-mean weight for game-score averages
SP_ROLL_STARTS = 40      # exponential window over a pitcher's recent starts
