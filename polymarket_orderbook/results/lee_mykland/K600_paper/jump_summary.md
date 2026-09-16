# Lee-Mykland jump detection — summary

Generated 2026-09-16 11:48. K=600, alpha=0.010, threshold=6.5741 (C_n=5.3293, S_n=0.2706, c=sqrt(2/pi)=0.7979).

| statistic | value |
|---|---|
| K | 600 |
| alpha | 0.01 |
| min_nonzero_pairs | 30 |
| threshold | 6.57406 |
| C_n | 5.32934 |
| S_n | 0.270583 |
| total_minute_bars | 648819 |
| bars_with_price | 506413 |
| valid_returns | 487958 |
| tested_observations | 45582 |
| warmup_or_untestable | 603237 |
| assets_total | 1766 |
| assets_tested | 580 |
| total_jumps | 4373 |
| positive_jumps | 1942 |
| negative_jumps | 2431 |
| pct_of_tested_minutes | 9.5937 |
| frac_zero_minute_returns | 0.724759 |
| frac_zero_returns_among_tested | 0.248431 |
| frac_jumps_in_last_10min_of_series | 0.198719 |
| frac_jumps_at_mid_below_0.05 | 0.147039 |
| frac_jumps_at_mid_above_0.95 | 0.0137206 |
| frac_jumps_on_carried_bar | 0 |
| median_abs_return_all_tested | 0.0253178 |
| distinct_days | 4 |
| avg_jumps_per_day | 1093.25 |
| median_abs_return | 0.434286 |
| mean_abs_return | 0.608144 |
| max_positive_return | 3.89182 |
| max_negative_return | -6.30992 |

## Top 20 jumps by |LM|

| ts_hkt              | label                                                 |   mid |   log_return |   jump_direction |   LM_stat |   local_volatility |   staleness_s |   n_events |
|:--------------------|:------------------------------------------------------|------:|-------------:|-----------------:|----------:|-------------------:|--------------:|-----------:|
| 2026-09-12 09:33:00 | mlb-hou-tb-2026-09-11 | spread | -1.5 | NO            | 0.01  |    -4.34381  |               -1 | -186.333  |         0.0233121  |        34.014 |         11 |
| 2026-09-12 09:40:00 | mlb-sea-oak-2026-09-11 | first_five_total | 3.5 | YES | 0.39  |    -0.640503 |               -1 | -167.545  |         0.00382288 |        47.788 |         30 |
| 2026-09-12 10:36:00 | mlb-sea-oak-2026-09-11 | first_five_total | 3.5 | NO  | 0.001 |    -4.44852  |               -1 | -144.729  |         0.0307368  |         9.344 |          1 |
| 2026-09-12 09:54:00 | mlb-tex-ari-2026-09-11 | extra_innings | - | YES      | 0.5   |     1.20397  |                1 |  138.199  |         0.00871188 |        18.363 |          1 |
| 2026-09-14 04:41:00 | mlb-lad-mia-2026-09-13 | moneyline | - | YES          | 0.015 |    -2.87168  |               -1 | -127.491  |         0.0225246  |        27.558 |        415 |
| 2026-09-14 07:30:00 | mlb-sea-oak-2026-09-13 | moneyline | - | YES          | 0.015 |    -3.69718  |               -1 | -127.191  |         0.0290678  |        14.845 |       1060 |
| 2026-09-12 12:13:00 | mlb-tex-ari-2026-09-11 | extra_innings | - | YES      | 0.1   |    -1.58924  |               -1 | -122.715  |         0.0129506  |        48.872 |          1 |
| 2026-09-12 12:06:00 | mlb-tex-ari-2026-09-11 | extra_innings | - | YES      | 0.49  |     1.58924  |                1 |  122.342  |         0.0129901  |        34.788 |          4 |
| 2026-09-12 09:32:00 | mlb-hou-tb-2026-09-11 | moneyline | - | YES           | 0.015 |    -3.01226  |               -1 | -121.358  |         0.0248213  |         2.53  |        260 |
| 2026-09-12 12:14:00 | mlb-tex-ari-2026-09-11 | extra_innings | - | YES      | 0.48  |     1.56862  |                1 |  108.75   |         0.0144241  |        17.649 |          4 |
| 2026-09-12 12:52:00 | mlb-sea-oak-2026-09-11 | moneyline | - | YES          | 0.01  |    -4.11904  |               -1 | -107.619  |         0.0382742  |        33.929 |        490 |
| 2026-09-11 06:02:00 | mlb-tex-sea-2026-09-10 | total | 6.5 | NO             | 0.015 |    -3.24519  |               -1 | -105.302  |         0.030818   |        36.596 |        146 |
| 2026-09-13 05:57:00 | mlb-sd-sf-2026-09-12 | total | 9.5 | NO               | 0.001 |    -5.34711  |               -1 | -102.856  |         0.0519866  |        15.841 |          2 |
| 2026-09-12 11:33:00 | mlb-tex-ari-2026-09-11 | extra_innings | - | YES      | 0.49  |     1.32687  |                1 |  101.396  |         0.013086   |        58.721 |          4 |
| 2026-09-13 10:41:00 | mlb-cws-stl-2026-09-12 | total | 9.5 | NO             | 0.001 |    -6.1203   |               -1 |  -99.7951 |         0.0613286  |         3.471 |          2 |
| 2026-09-12 09:53:00 | mlb-nym-nyy-2026-09-11 | total | 9.5 | NO             | 0.01  |    -4.29046  |               -1 |  -96.7049 |         0.0443665  |        10.505 |        130 |
| 2026-09-12 11:09:00 | mlb-phi-atl-2026-09-11 | moneyline | - | YES          | 0.01  |    -3.7013   |               -1 |  -89.9606 |         0.0411436  |         4.721 |        848 |
| 2026-09-14 04:40:00 | mlb-lad-mia-2026-09-13 | total | 7.5 | NO             | 0.006 |    -4.09434  |               -1 |  -89.7144 |         0.0456376  |         0.402 |        108 |
| 2026-09-13 10:39:00 | mlb-cws-stl-2026-09-12 | moneyline | - | NO           | 0.185 |    -1.60944  |               -1 |  -88.7563 |         0.0181332  |         0.2   |        499 |
| 2026-09-13 06:26:00 | mlb-laa-wsh-2026-09-12 | total | 8.5 | NO             | 0.02  |    -3.47352  |               -1 |  -88.5631 |         0.0392208  |        39.036 |        247 |

---

## Reading these results

### What was used

`data/live/top_of_book_*.csv` — the recorder's event-level top-of-book feed, five
files (four full MLB slates 2026-09-10..09-13 plus a partial 09-16 recording),
**21.3M rows, 1,766 tradable tokens, 52 game markets**, median gap between
consecutive updates of the same token **27 ms**. Opened read-only.

`mid = (best_bid + best_ask)/2`, recomputed from the raw columns. The feed's own
`midpoint` column was verified to equal that in **100.0%** of rows.

1-minute bars: the mid of the **last valid event at or before the end of each
minute** — strictly backward-looking. A minute with no event carries the previous
quote forward for at most 5 minutes; past that the bar is left missing and no
return is computed across the gap. 648,733 bars, 506,413 with a price.

### The two things that make this dataset hard for Lee-Mykland

**1. The K=600 warm-up is longer than most contracts live.** These are per-game
contracts, not continuously traded stocks. Median asset lifetime is **286 valid
minutes**; only **46 of 1,766 assets reach 600**. At K=600 just 65,735 of 506,413
bars (13%) are ever testable, on 580 assets.

Worse, the warm-up lands on the wrong regime. A contract is listed hours before
first pitch and barely moves; the bipower window is filled with that quiet
pre-game period, and by the time the statistic switches on the game is live and
genuinely far more volatile. The time-of-day chart shows the consequence
directly: **every tested minute falls between 04:00 and 13:30 HKT**, i.e. live
play only. The volatility baseline is calibrated on the quiet regime and applied
to the loud one.

**2. The price is discrete and flat most of the time.** **72.5% of 1-minute
returns are exactly zero** — mid moves in half-tick steps of 0.005. Bipower
variation multiplies *adjacent* absolute returns, so a product is zero unless two
consecutive minutes both moved (~7.6% of pairs if independent). sigma_hat is
therefore biased well below the size of a single tick, and an ordinary one-tick
move clears the threshold.

In the first unguarded run this produced statistics as large as **4.6e8**, driven
by sigma_hat collapsing to 5.8e-11. A window carrying fewer than
`--min-nonzero-pairs` (30) non-zero products now emits no statistic, since
Lee-Mykland's asymptotics assume sigma > 0. That alone cut detections from 8,210
to 4,373 and removed every degenerate statistic. **The threshold itself was never
adjusted.**

### So are the detected jumps real?

**The large ones, yes — clearly.** The top-20 table is dominated by contracts
resolving: moneylines and spreads collapsing from ~0.9 to ~0.01 in a single
minute as a game is decided, and `extra_innings` doubling from 0.50 to 0.95.
`examples/jump_001.png` is typical: a flat book at 0.90, one minute to 0.01, flat
at 0.01 thereafter. Those are genuine, economically meaningful price jumps.

**The count as a whole, no.** 9.6% of tested minutes flagged at alpha=0.01 is
roughly two orders of magnitude more than the test should produce. That is the
two problems above, not 4,373 genuine jumps. Concretely:

* **19.9%** of detections fall in the last 10 minutes of a contract's life —
  these are resolution events. Real jumps, but not tradeable and not what a
  forecasting study wants to predict.
* **14.7%** occur at mid < 0.05, where a one-tick move is a huge *log* return
  (0.010 -> 0.005 is a log return of -0.69). Log returns are the wrong scale for
  a bounded probability near its boundary.
* 0% sit on a carried-forward bar, so staleness is *not* the driver.

### K=120 comparison

Rerunning with a 2-hour window (`K120/`) raises coverage to 939 assets and 85,064
tested observations and drops the rate to **2.89%** — still high, but the larger
tested sample and the shorter, better-matched volatility window both help. This
is a documented change of window length for data feasibility, not threshold
tuning; alpha stays 0.01 and the critical value is recomputed from n as the
method requires.

### Recommendation before any modelling

Do not use this jump table as a label set as it stands. It needs, at minimum:
contracts truncated before resolution; a price scale that does not explode near 0
and 1 (arithmetic or logit rather than log of a raw probability); and a sampling
frequency at which the price actually moves, so the diffusion null is not
violated by 72.5% zero returns.
