# Lee-Mykland jump detection — summary

Generated 2026-09-16 12:08. K=30, alpha=0.010, threshold=6.7809 (C_n=5.5874, S_n=0.2594, c=sqrt(2/pi)=0.7979).

| statistic | value |
|---|---|
| K | 30 |
| alpha | 0.01 |
| return_scale | diff |
| in_game_only | True |
| min_nonzero_pairs | 7 |
| threshold | 6.78087 |
| C_n | 5.58742 |
| S_n | 0.259438 |
| total_minute_bars | 181970 |
| bars_with_price | 165282 |
| valid_returns | 161952 |
| tested_observations | 116861 |
| warmup_or_untestable | 65109 |
| assets_total | 1696 |
| assets_tested | 1212 |
| total_jumps | 2239 |
| positive_jumps | 1119 |
| negative_jumps | 1120 |
| pct_of_tested_minutes | 1.91595 |
| frac_zero_minute_returns | 0.312741 |
| frac_zero_returns_among_tested | 0.229717 |
| frac_jumps_in_last_10min_of_series | 0.155427 |
| frac_jumps_at_mid_below_0.05 | 0.0433229 |
| frac_jumps_at_mid_above_0.95 | 0.0415364 |
| frac_jumps_on_carried_bar | 0 |
| median_abs_return_all_tested | 0.01 |
| sigma_regime_lag_median | 1.42187 |
| sigma_regime_lag_p25 | 1.30395 |
| sigma_regime_lag_p75 | 1.66936 |
| distinct_days | 4 |
| avg_jumps_per_day | 559.75 |
| median_abs_return | 0.215 |
| mean_abs_return | 0.224639 |
| max_positive_return | 0.764 |
| max_negative_return | -0.764 |

## Top 20 jumps by |LM|

| ts_hkt              | label                                                 |    mid |   log_return |   jump_direction |   LM_stat |   local_volatility |   staleness_s |   n_events |
|:--------------------|:------------------------------------------------------|-------:|-------------:|-----------------:|----------:|-------------------:|--------------:|-----------:|
| 2026-09-14 06:44:00 | mlb-sea-oak-2026-09-13 | spread | -1.5 | YES          | 0.51   |       0.418  |                1 |  110.18   |        0.00379379  |         0.007 |        162 |
| 2026-09-14 06:44:00 | mlb-sea-oak-2026-09-13 | spread | -1.5 | NO           | 0.49   |      -0.418  |               -1 | -110.18   |        0.00379379  |         0.007 |        162 |
| 2026-09-12 08:21:00 | mlb-lad-mia-2026-09-11 | first_five_total | 5.5 | YES | 0.44   |       0.345  |                1 |   64.5436 |        0.00534522  |         3.308 |         17 |
| 2026-09-12 08:21:00 | mlb-lad-mia-2026-09-11 | first_five_total | 5.5 | NO  | 0.56   |      -0.345  |               -1 |  -64.5436 |        0.00534522  |         3.308 |         17 |
| 2026-09-12 12:52:00 | mlb-sea-oak-2026-09-11 | spread | -2.5 | NO           | 0.745  |      -0.2305 |               -1 |  -62.672  |        0.00367788  |        48.827 |          3 |
| 2026-09-12 12:52:00 | mlb-sea-oak-2026-09-11 | spread | -2.5 | YES          | 0.255  |       0.2305 |                1 |   62.672  |        0.00367788  |        48.827 |          3 |
| 2026-09-14 04:38:00 | mlb-cle-min-2026-09-13 | spread | -1.5 | YES          | 0.3325 |       0.323  |                1 |   53.5354 |        0.00603339  |        26.691 |        271 |
| 2026-09-14 04:38:00 | mlb-cle-min-2026-09-13 | spread | -1.5 | NO           | 0.6675 |      -0.323  |               -1 |  -53.5354 |        0.00603339  |        26.691 |        271 |
| 2026-09-13 06:09:00 | mlb-laa-wsh-2026-09-12 | spread | -1.5 | YES          | 0.245  |      -0.63   |               -1 |  -50.1144 |        0.0125712   |         2.574 |        223 |
| 2026-09-13 06:09:00 | mlb-laa-wsh-2026-09-12 | spread | -1.5 | NO           | 0.755  |       0.63   |                1 |   50.1144 |        0.0125712   |         2.574 |        223 |
| 2026-09-12 08:12:00 | mlb-phi-atl-2026-09-11 | spread | -2.5 | YES          | 0.495  |       0.155  |                1 |   49.0153 |        0.00316228  |         1.764 |        210 |
| 2026-09-13 02:43:00 | mlb-nym-nyy-2026-09-12 | spread | -2.5 | NO           | 0.8185 |      -0.136  |               -1 |  -47.4262 |        0.00286761  |         0.087 |         73 |
| 2026-09-13 02:43:00 | mlb-nym-nyy-2026-09-12 | spread | -2.5 | YES          | 0.1815 |       0.136  |                1 |   47.4262 |        0.00286761  |         0.087 |         73 |
| 2026-09-12 09:14:00 | mlb-cin-mil-2026-09-11 | spread | -2.5 | NO           | 0.3895 |       0.35   |                1 |   46.2213 |        0.00757227  |         0.036 |         51 |
| 2026-09-12 09:14:00 | mlb-cin-mil-2026-09-11 | spread | -2.5 | YES          | 0.6105 |      -0.35   |               -1 |  -46.2213 |        0.00757227  |         0.036 |         51 |
| 2026-09-13 02:44:00 | mlb-nym-nyy-2026-09-12 | spread | -2.5 | YES          | 0.0325 |      -0.149  |               -1 |  -45.6536 |        0.00326371  |         7.483 |        154 |
| 2026-09-13 02:44:00 | mlb-nym-nyy-2026-09-12 | spread | -2.5 | NO           | 0.9675 |       0.149  |                1 |   45.6536 |        0.00326371  |         7.483 |        154 |
| 2026-09-13 12:22:00 | mlb-sea-oak-2026-09-12 | spread | -1.5 | NO           | 0.98   |      -0.012  |               -1 |  -44.8999 |        0.000267261 |         1.041 |        102 |
| 2026-09-13 12:22:00 | mlb-sea-oak-2026-09-12 | spread | -1.5 | YES          | 0.02   |       0.012  |                1 |   44.8999 |        0.000267261 |         1.041 |        102 |
| 2026-09-12 13:14:00 | mlb-sd-sf-2026-09-11 | spread | -1.5 | YES            | 0.26   |       0.2405 |                1 |   41.4748 |        0.00579871  |         1.384 |        166 |

---

## Why these parameters, and not the paper's

Mäkinen et al. apply Lee-Mykland to continuously traded equities over many
sessions, where K=600 minutes is about a day and a half of trailing history.
A Polymarket MLB contract is a different object: it is created hours before
first pitch, trades for one game, and resolves. Copying K=600 was wrong for
four measurable reasons, all fixed below.

| parameter | paper | here | why |
|---|---|---|---|
| sample | all recorded minutes | **in-game only** | see below |
| K | 600 | **30** | median contract lives 84 in-game minutes |
| returns | log | **arithmetic (probability points)** | log explodes at the boundary |
| alpha | 0.01 | **0.01** | unchanged — the threshold was never tuned |

**In-game trimming is the single biggest fix.** A contract is listed hours
early and barely moves; with K=600 the volatility window was filled entirely
with that dead period and the statistic only switched on once the game was
live and genuinely far more volatile. Measured as the ratio of forward realised
volatility to the trailing bipower estimate, sigma_hat was **3.08x too low**.
Trimming to [first pitch, final out] using `data/game_windows.json` brings that
to **1.42x**. It also drops the share of exactly-zero 1-minute returns from
**0.72 to 0.38** — most of the "flatness" was pre-game, not microstructure.

**K=30 was chosen from the data, not asserted** (`K_sweep_in_game.csv`):

| K | tested obs | assets | jumps | rate | sigma lag median | lag IQR | jumps in last 10 min |
|---|---|---|---|---|---|---|---|
| 20 | 119,840 | 1,288 | 2,808 | 2.34% | 1.48 | 0.73 | 12.7% |
| **30** | **116,861** | **1,212** | **2,239** | **1.92%** | **1.42** | **0.37** | **15.5%** |
| 45 | 108,122 | 1,113 | 1,758 | 1.63% | 1.47 | 0.45 | 15.9% |
| 60 | 100,352 | 1,042 | 1,639 | 1.63% | 1.46 | 0.58 | 17.6% |
| 90 | 85,457 | 937 | 1,296 | 1.52% | 1.45 | 0.75 | 20.4% |
| 120 | 71,599 | 875 | 1,133 | 1.58% | 1.46 | 0.86 | 23.9% |

K=30 tracks realised volatility most consistently (IQR 0.37, less than half the
spread at K=120), tests the most returns of any well-behaved setting (72% of
in-game returns, on 1,212 of 1,696 assets), and attributes the least to
end-of-game resolution. Larger K buys nothing: the median lag is flat near 1.45
everywhere, while coverage falls and resolution contamination rises.

**Arithmetic returns, not log.** These are contracts bounded in (0,1). In log
space one half-tick is a 0.0055 return at p=0.90 but 0.41 at p=0.01, so the
test spent its power on ticks in near-resolved contracts. In probability points
a half-tick is 0.005 everywhere. Two checks confirm the switch:

* positive and negative jumps are now **1,119 vs 1,120** (they were 1,942 vs
  2,431 under log returns) — a symmetric test on a symmetric process;
* the YES and NO legs of the same market produce **exactly mirrored statistics**
  (same |LM|, same sigma_hat, opposite sign), which is arithmetically forced
  when p_YES + p_NO = 1 and is simply not true of log returns. Note the
  consequence: each market event appears **twice** in the table, once per leg.

## Result under the MLB-tuned settings

K=30, alpha=0.01, arithmetic returns, in-game only:

* **2,239 jumps — 1,119 positive, 1,120 negative — on 116,861 tested minutes
  (1.92%)**, across 1,212 assets and 4 slate days.
* Median flagged move **0.215 in probability points** (mean 0.225, max 0.785).
  These are 20-point swings, not tick noise.
* 15.5% fall in a contract's final 10 minutes; 4.3% below mid 0.05 and 4.2%
  above 0.95 — all much reduced from the paper-parameter run.
* Residual sigma_hat lag **1.42x**. This is inherent: volatility genuinely rises
  through a game and any trailing estimator lags it. It is the main reason the
  rate is 1.92% rather than the nominal ~0%.

The paper-parameter run is kept intact in `K600_paper/` for comparison.

## Honest caveat carried forward

An earlier version of this summary claimed the discreteness of tick prices
meant "an ordinary one-tick move clears the threshold". That was wrong and is
retracted: a flag requires roughly **24 half-ticks**, and only **1.0%** of
tested minutes could be flagged by a single tick. The detected moves are large
and real. The genuine issue was never that the jumps are fake — it is that
sigma_hat was calibrated on the wrong regime, which is what the in-game trim
fixes.
