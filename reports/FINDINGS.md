# Findings

Walk-forward backtest, MLB regular season 2023-2026 (9,455 games).
Trained on 2015 through the season before each test year; nothing from a test
season ever reaches its own model fit.

## Headline

| | |
|---|---|
| Best model win rate | **56.6%** (`ensemble_elo`) |
| Closing line win rate | **56.8%** |
| Best model log loss | 0.6799 |
| Closing line log loss | 0.6790 |
| Does the free data beat the market? | **No** |
| Is there a profitable betting edge? | **No** (every CI includes zero on clean odds) |

The models work. They land squarely in the 55-62% band the published
literature reports for MLB, they are well calibrated, and they beat every
classical baseline. They just do not beat the betting market, and on
Polymarket you are betting against that market.

## Win rate by season

| model | 2023 | 2024 | 2025 | 2026 | mean |
|---|---|---|---|---|---|
| **market (closing line)** | 0.5712 | 0.5783 | 0.5610 | 0.5600 | **0.5676** |
| ensemble_elo | 0.5714 | 0.5646 | 0.5688 | 0.5562 | 0.5653 |
| elo_sp (pitcher-adjusted Elo) | 0.5702 | 0.5626 | 0.5675 | 0.5599 | 0.5651 |
| gbm_elo_offset | 0.5640 | 0.5675 | 0.5647 | 0.5558 | 0.5630 |
| margin_gbm (run-margin regression) | 0.5698 | 0.5695 | 0.5523 | 0.5521 | 0.5609 |
| ensemble (8 ML models) | 0.5694 | 0.5626 | 0.5581 | 0.5530 | 0.5607 |
| elo (team only) | 0.5571 | 0.5613 | 0.5688 | 0.5474 | 0.5586 |
| lightgbm | 0.5665 | 0.5650 | 0.5581 | 0.5456 | 0.5588 |
| logit_l2 | 0.5644 | 0.5584 | 0.5560 | 0.5493 | 0.5570 |
| runs_negbin | 0.5706 | 0.5617 | 0.5577 | 0.5215 | 0.5529 |
| log5_wpct (Bill James) | 0.5525 | 0.5486 | 0.5519 | 0.5405 | 0.5484 |

Full table with log loss, Brier, AUC and calibration slope in `summary.csv`
and `results_by_season.csv`.

## What actually mattered

Ranked by standardised logistic coefficient, fit on 2015-2022 only:

1. `d_def_fip_15` — recent team pitching quality (FIP over 15 games)
2. `d_sp_era_40` — starting pitcher's rolling ERA
3. `elo_sp_logit` — pitcher-adjusted Elo (**45% of all LightGBM gain**)
4. `d_elo` — team Elo difference
5. `d_sp_season_starts`, `d_sp_k_40` — starter workload and strikeout rate

Two structural observations:

- **Pitching dominates.** Seven of the top ten features are pitching. The
  best offence feature (`d_off_woba_std`) ranks 13th.
- **Elo does nearly all the work.** A single pitcher-adjusted Elo number
  (56.5%) is within 0.02 points of the full 64-feature ensemble (56.6%).
  Eight ML models, schedule-adjusted ridge ratings, park factors, bullpen
  fatigue and rest all together buy ~0.1 percentage points.

Grid-searching Elo on 2018-2022 gave K=4.0, home field 24 Elo points, 30%
between-season regression, and 3.5 Elo points per point of starting-pitcher
game score above league average.

## The result that killed the betting case

A naive read of the backtest looks profitable: betting whenever the model
saw >10% expected value returned **+9.2% ROI over 1,189 bets, profitable in
all four seasons**, with a bootstrap CI of [+2.2%, +16.1%].

Three controls dismantled it.

**1. It is not favourite-longshot bias.** The bets averaged 2.46 decimal
odds, so the obvious explanation is that the model just bets underdogs and
underdogs are underpriced. They are not, in this data:

| control strategy (closing line only, no model) | bets | ROI |
|---|---|---|
| bet every underdog | 9,443 | −1.7% |
| bet every favourite | 9,443 | −5.7% |
| bet dogs priced > 2.40 | 2,741 | −2.0% |
| bet dogs priced > 2.60 | 1,556 | −2.9% |

So the price band alone loses money, and the model beat a price-matched
random selection in 100% of 3,000 draws. That control *passed*.

**2. The odds were not closing lines.** Action Network timestamps every
odds row. Only 54% land within 15 minutes of first pitch; the rest are
recorded hours later. The market's *own* accuracy collapses across those
slices, which is the signature of a stale or mis-stamped price:

| snapshot age | n | market log loss | market accuracy |
|---|---|---|---|
| ≤15 min (clean) | 3,683 | 0.6747 | 58.4% |
| 15-60 min | 1,395 | 0.6681 | 58.7% |
| 60-180 min | 3,445 | 0.6831 | 55.3% |
| >180 min | 936 | 0.6976 | 52.9% |

A real closing line does not go from 58.4% to 52.9% accurate. The two stale
slices are bad prices, and that is exactly where the profit lived.

**3. On clean odds the edge disappears.**

| edge threshold | bets | ROI | 95% CI | profitable seasons |
|---|---|---|---|---|
| >5% | 1,028 | +3.9% | [−3.2%, +10.7%] | 1 of 4 |
| >10% | 476 | +6.6% | [−4.1%, +17.2%] | 2 of 4 |
| >15% | 209 | +13.4% | [−3.6%, +30.3%] | 2 of 3 |

Every interval includes zero. Per season at the 10% threshold: 2023 +14.2%,
2024 −0.2%, 2025 −53.9%, 2026 +2.0%. That is noise, not an edge.

**The decisive test.** Fit a model with the closing-line logit as a *fixed
offset*, so it can only learn what the market missed. If the free data
carried independent information, this would beat the market:

| season | market log loss | +free features (GBM) | difference |
|---|---|---|---|
| 2024 | 0.6761 | 0.6764 | −0.0003 |
| 2025 | 0.6778 | 0.6783 | −0.0005 |
| 2026 | 0.6833 | 0.6844 | −0.0010 |

Negative every year. The free public data contains **nothing** the closing
line has not already priced. And when the model disagreed with the close by
more than 3 points, it was right only **48.2%** of the time — worse than a
coin flip, which is what you would expect if the market is right and the
model is noise around it.

## Can bet sizing rescue it?

No. Expected profit is linear in stake — `E[profit] = Σ stake_i × edge_i` —
so if every edge is negative, no set of positive stakes makes the sum
positive. Sizing controls the variance and growth rate of an edge that
already exists; it cannot create one. Kelly says the same thing itself: the
optimal fraction `(p·d − 1)/(d − 1)` goes negative exactly when the bet is
−EV, and the correct stake there is zero.

`mlbmodel/staking.py` tests this rather than assuming it, resolving bets in
daily batches with per-bet and per-day exposure caps.

**Control — bet every favourite (known −5.7% edge), sized six ways:**

| scheme | final | return | ruined | P(profit) |
|---|---|---|---|---|
| flat stake | 0.009× | −99.1% | yes | 0.00 |
| flat fraction | 0.010× | −99.0% | yes | 0.00 |
| tenth Kelly | 0.010× | −99.0% | yes | 0.00 |
| quarter Kelly | 0.010× | −99.0% | yes | 0.00 |
| half Kelly | 0.010× | −99.0% | yes | 0.00 |
| full Kelly | 0.010× | −99.0% | yes | 0.00 |

Every scheme is wiped out, in 100% of bootstrap paths. Sizing changed only
the route to zero.

**The trap in my own code.** `bet_sim` originally reported a `kelly_growth`
figure of **+504%**, which looked like sizing rescuing the strategy. It was
computed as `prod(1 + pnl)` — compounding bets one at a time. Baseball plays
~15 games simultaneously; staking 5% on eight of today's games risks 40% at
once, not 5% eight times in sequence. That metric is now removed.

**The sample is not stationary.** A pooled block bootstrap on clean odds
reports quarter-Kelly at +86% with an 75% chance of profit. That number is an
artifact: the clean-snapshot subset is 97% concentrated in 2023-2024.

| season | clean games | flat stake | quarter Kelly |
|---|---|---|---|
| 2023 | 1,982 | **+63.5%** | **+185.9%** |
| 2024 | 1,590 | −9.5% | −20.7% |
| 2025 | 62 | −6.7% | −10.5% |
| 2026 | 45 | −7.3% | −8.4% |

You cannot bet a season you have already seen. The only honest experiment is
to deploy *after* the year that looked good:

| scheme | final | return | max drawdown | P(profit) | 5th pct |
|---|---|---|---|---|---|
| flat stake | 0.76× | −23.5% | 34.6% | 0.15 | −65.2% |
| tenth Kelly | 0.89× | −11.2% | 24.9% | 0.29 | −39.8% |
| quarter Kelly | 0.65× | −34.9% | 55.5% | 0.20 | −74.8% |
| half Kelly | 0.26× | −73.7% | 84.2% | 0.06 | −94.3% |

Losses grow monotonically with stake size. That ordering is the signature of
a negative edge — with a real edge, larger Kelly fractions raise the median
return until variance drag takes over. Here bigger sizing is strictly worse,
which is what sizing does to a losing proposition: it finds the bottom faster.

## Other sports

Same pipeline, Elo plus schedule and form features (no player-level data,
because no other league has an open equivalent of the MLB Stats API).

**NBA** — 6,070 games, tested 2023-2025:

| model | win rate | log loss | AUC | betting ROI |
|---|---|---|---|---|
| **market** | **69.4%** | **0.5775** | 0.7599 | — |
| logit_l2 | 67.5% | 0.6038 | 0.7315 | −13.0% |
| margin_ridge | 67.1% | 0.6031 | 0.7318 | −11.7% |
| ensemble_elo | 66.9% | 0.6054 | 0.7270 | −14.6% |

The NBA closing line is impeccably calibrated — predicted-versus-actual gaps
stay inside ±0.02 across all ten deciles, log loss 0.5930, vig 4.4% — and
every naive strategy loses (bet all dogs −6.2%, all favourites −3.6%, all
home teams −5.8%). So unlike MLB there is no data-quality confound here; the
market is simply better.

**NHL** — 6,540 games, tested 2023-2025:

| model | win rate | log loss | betting ROI |
|---|---|---|---|
| **market** | **59.3%** | **0.6650** | — |
| lightgbm | 57.3% | 0.6779 | −8.0% |
| logit_l2 | 57.3% | 0.6738 | −7.9% |
| margin_ridge | 57.1% | 0.6735 | −6.7% |

**NFL** — 2,623 games, tested 2022-2025:

| model | win rate | log loss | betting ROI |
|---|---|---|---|
| **market** | **67.6%** | **0.6080** | — |
| ensemble | 66.0% | 0.6364 | −12.3% |
| logit_l2 | 65.7% | 0.6380 | −11.2% |
| lightgbm | 65.4% | 0.6353 | −9.3% |

### The pattern across all four

| sport | market | best model | gap | model betting ROI |
|---|---|---|---|---|
| MLB | 56.8% | 56.6% | 0.2 pts | −1% to +7% (not significant) |
| NHL | 59.3% | 57.3% | 2.0 pts | −7% to −10% |
| NBA | 69.4% | 67.5% | 1.9 pts | −11% to −15% |
| NFL | 67.6% | 66.0% | 1.6 pts | −8% to −15% |

Predictability varies enormously — an NBA game is far more forecastable than
a baseball game, because a 48-minute contest lets talent gaps express
themselves where nine innings drown them in variance. But the market gap does
*not* shrink with predictability; MLB has the smallest gap precisely because
there is so little to know. More predictable does not mean more beatable.

Full tables in `nba_summary.csv`, `nhl_summary.csv`, `nfl_summary.csv`.

## Leakage controls

MLB papers reporting 90%+ accuracy are computing season statistics that
include the game being predicted. `tests/test_leakage.py` guards against
this and all six checks pass:

- rolling team windows recomputed by brute force — exactly prior-only
- pitcher game-score windows exclude the current start
- Elo ratings are pre-game and chain correctly game to game
- opening-day rolling stats sit at the league prior (sd = 0.60)
- **shuffled-label control: 52.9% accuracy** (chance)
- best real win rate 56.8% sits inside the published band

Calibration is good: the top model's slope is 0.89 and predicted-versus-
actual gaps are within ±0.05 across all ten probability deciles.

## Lead-lag arbitrage: game market → futures market

The idea: a team's game result mechanically changes their championship
probability. Game markets reprice in seconds; the World Series market is thin
and traded by people not watching every game. If the futures leg lags, you
trade it on information already public in the game leg — **no predictive
model required**, only a speed difference between two linked markets.

`mlbmodel/leadlag.py` measures the three quantities that decide it.

**The futures leg is expensive to trade** (30 World Series markets, live books):

| | |
|---|---|
| Median relative spread | **9.1%** (range 1.4% – 66.7%) |
| Median $ at best bid | **$8** |
| Median depth, 5 levels | $71 |

Absolute spreads look tight (median $0.0010) but prices are low, so crossing
costs 9% of position value at the median.

**Event study — 225 team-days, 15 live teams, Aug 22 – Sep 7:**

| | |
|---|---|
| Mean futures move on a day the team **won** | +0.00063 |
| Mean move on a day the team **lost** | +0.00041 |
| **Win-minus-loss effect** | **+0.00022** (SE 0.00053, t = 0.41) |
| Median tick size | 0.0010 |
| Median quoted spread | 0.0030 |
| **Signal ÷ tick** | **0.22** |
| **Signal ÷ spread** | **0.07** |

**One regular-season game moves the championship price by about one-fifth of
the smallest expressible price increment, and one-fourteenth of the round-trip
cost.** That is a *granularity* problem, not a latency problem — the market
cannot represent the update even in principle, so being faster buys nothing.
The effect is also not statistically distinguishable from zero at t = 0.41,
swamped in daily data by other teams' results and injury news.

Futures books are close to static: the Yankees' title price did not move on
**100%** of days in the sample, and sat at exactly 0.0950 for 1,441
consecutive minutes. The Brewers' moved on 94% of days. Liquidity varies
enormously by team.

**Where this could actually work: the playoffs.** The arithmetic inverts when
one game carries real championship weight. A playoff game can swing a team's
title probability by 0.05 or more against the same ~0.003–0.010 spread — a
signal-to-cost ratio of 5–15x rather than 0.07. That is the version of this
idea worth preparing for, and it is measurable in advance: record the game
market *and* the futures market simultaneously through October and measure
the realised lag at tick resolution before committing capital.

## Market making in small markets

Tested directly against the hypothesis that thinner markets pay wider
spreads. **They do not** (135 live sports moneylines):

| volume bucket | n | median spread | as % of mid | $ at best bid | $50 = share of queue | rewards funded |
|---|---|---|---|---|---|---|
| $1k – 10k | 57 | 0.0100 | 2.0% | $1,706 | 2.9% | 0% |
| $10k – 100k | 67 | 0.0100 | 2.2% | $4,358 | 1.2% | 1.5% |
| $100k – 1M | 11 | 0.0100 | 2.4% | $10,591 | 0.5% | 0% |

The spread is pinned at **exactly one tick in every bucket**. Small markets
pay you no more per round trip — roughly $1 gross on a $50 position
regardless of size. What improves is queue share: 2.9% in small markets
versus 0.5% in large, about 6x better but still marginal.

**Volume is the wrong screen.** The right one is top-of-book dollar depth,
and the two come apart badly. 21 of 135 books had $50 exceeding 25% of the
best bid, and several were *high* volume with near-empty books at that
moment — `lol-t1a-ktc` had $139,767 of volume and **$70** at the best bid.

Genuinely wide spreads do exist (`atp-wild-ferrar` at 0.18, 116% of mid;
`chi1-ccu-cdh` at 0.39) but these are stale quotes on markets nobody is
trading. You would sit unfilled, or be filled only by someone who knows the
current game state — adverse selection in its purest form.

## Realised adverse selection: where is the mid 30s after a fill?

Measured from the recorded order books — 3 days (Aug 28–30), 116 GB of
snapshots, 13.3M touch updates, 124,808 inferred fills on tight books.
`polymarket_orderbook/adverse_selection.py`.

The maker's whole economics reduce to one identity:

    kept per share  =  half_spread  −  adverse_selection

**Averaged over every fill at a one-tick touch:**

| | per share | as % of half-spread |
|---|---|---|
| Half-spread available | 0.00464 | 100% |
| Adverse selection (mid runs away) | **0.00288** | **62%** |
| **Net kept** | **+0.00176** | 38% |

That is +0.43% per round trip, t = 15.9. So making markets at the touch is
*mildly* profitable on average — the mid moves against you by about
six-tenths of what you collect, and you keep the rest.

**But adverse selection scales with the size of the trade that fills you:**

| size consumed | n | adverse | net kept | t |
|---|---|---|---|---|
| <$50 | 71,536 | 0.00281 | +0.00174 | 13.0 |
| $50–200 | 22,140 | 0.00387 | +0.00085 | 3.2 |
| $200–1k | 25,412 | 0.00194 | +0.00283 | 9.9 |
| $1k–5k | 4,473 | 0.00404 | +0.00085 | 1.1 |
| $5k–20k | 1,167 | 0.00387 | +0.00102 | 0.8 |
| **>$20k** | 80 | **0.00900** | **−0.00410** | −1.5 |

On the largest sweeps, adverse selection is roughly **twice** the half-spread.
Big trades are informed trades, and they are exactly the ones that reach deep
into the book.

**Which answers the $50 question.** A resting order fills only once the size
ahead of it is gone, so the average fill is the wrong benchmark — the right
one is the tail a small order can actually reach. Live books show a median
**$4,870** at the best MLB moneyline bid, so $50 sits behind essentially all
of it:

| fills only on sweeps ≥ | n | adverse | net kept | t | ROI |
|---|---|---|---|---|---|
| (all fills) | 34,374 | 0.00315 | +0.00086 | 5.2 | +0.18% |
| $500 | 3,857 | 0.00346 | +0.00058 | 1.0 | +0.08% |
| $1,000 | 1,542 | 0.00556 | −0.00087 | −0.9 | −0.14% |
| $2,000 | 702 | 0.00733 | **−0.00266** | −1.9 | **−0.44%** |
| $5,000 | 272 | 0.00610 | −0.00145 | −0.8 | −0.23% |

The edge is entirely in the small sweeps, and it is gone by $1,000. Front of
the queue earns +0.18%; back of the queue — where $50 necessarily sits —
earns nothing or loses. **Capital in market making buys queue position, and
queue position decides which half of this distribution you are filled in.**
That is the mechanism, measured rather than assumed.

**One genuinely useful side finding — the derivative markets pay better:**

| market type | n | adverse | net kept | ROI | t |
|---|---|---|---|---|---|
| spread | 50,047 | 0.00259 | +0.00225 | **+0.62%** | 12.3 |
| total | 38,080 | 0.00286 | +0.00206 | +0.50% | 9.2 |
| moneyline | 34,374 | 0.00315 | +0.00086 | +0.18% | 5.2 |
| first_five_total | 1,641 | 0.00504 | −0.00004 | −0.01% | −0.1 |

Run-line and totals markets show 3x the maker edge of the moneyline, on
comparable volume. The moneyline is the most efficiently priced market on
the board, which is consistent with everything else in this report.

**Caveats that matter.** `live_recorder.py` deliberately drops
`last_trade_price` (line 324), so there are no trade prints: a size decrease
at the touch may be a fill *or* a cancel. Cancels carry no adverse selection,
so their presence biases these estimates **optimistically** — the true edge is
weaker than shown, not stronger. P&L is marked to the mid 30s later, not to a
realised exit. The tail t-statistics (−0.8 to −1.9) support "the edge
vanishes", not "the loss is significant". And this is three days of one sport.

## Can a book-shape filter rescue small-scale market making?

The natural idea is to stop quoting when the book is thin, on the theory
that a thin book is fragile. Measured on the same 124,808 fills, that is
**the wrong variable**. Three things were tested — depth behind your touch,
depth on the far side, and the imbalance between them.

**Absolute thinness barely matters.** Adverse selection by depth behind the
touch: 0.00243 (<$50 of support) versus 0.00290 (>$5k). Essentially flat, and
if anything the thin books were *better*.

**Relative depth matters a lot.** Adverse selection by your side's share of
top-3 depth:

| imbalance | n | adverse | net kept | t |
|---|---|---|---|---|
| <0.30 (your side thin vs far) | 29,327 | **0.00409** | +0.00035 | 1.5 |
| 0.30–0.45 | 21,248 | 0.00363 | +0.00110 | 4.2 |
| 0.45–0.55 | 38,589 | 0.00252 | +0.00238 | 11.6 |
| 0.55–0.70 | 14,447 | 0.00195 | +0.00270 | 8.2 |
| >0.70 (your side thick vs far) | 21,197 | **0.00172** | +0.00262 | 10.4 |

**And gap size matters most of all** — how far the touch moved when it was
taken:

| gap | n | adverse | net kept | t | ROI |
|---|---|---|---|---|---|
| 1 tick | 102,399 | 0.00220 | +0.00237 | 20.7 | +0.58% |
| 2 ticks | 12,744 | 0.00370 | +0.00120 | 3.0 | +0.29% |
| 3–5 ticks | 7,642 | 0.00684 | **−0.00193** | −3.5 | −0.50% |
| >5 ticks | 2,023 | **0.01675** | **−0.01181** | −8.2 | **−2.79%** |

Adverse selection spans 7.6x across that range. The losses are entirely in
flow that *walks* the book. So the correct rule is not "avoid thin books" but
**"avoid being in the book when large flow arrives"** — and imbalance is the
early warning, not thinness.

Stacking the filters (all market types):

| filter | n | adverse | net kept | t | ROI |
|---|---|---|---|---|---|
| all sweeps | 124,808 | 0.00288 | +0.00176 | 15.9 | +0.43% |
| + queue ahead ≤ $50 | 71,536 | 0.00281 | +0.00174 | 13.0 | +0.51% |
| + imbalance ≥ 0.45 | 40,575 | 0.00209 | +0.00250 | 14.3 | +0.75% |
| + gap ≤ 1 tick | 35,279 | **0.00163** | **+0.00294** | 16.1 | **+0.88%** |

Adverse selection falls 43% and ROI doubles. On run-line markets the filtered
figure reaches **+1.19%** per fill (n = 16,804, t = 12.8).

### Then the contamination check, which cuts it roughly in half

A *pulled* quote is indistinguishable from a fill in snapshot data, and it
carries no adverse selection — precisely the signature of the 1-tick bucket
where the edge concentrates. Discriminator: does the touch snap back to its
old price within 5 seconds? A reprice cycle usually does; consumed inventory
usually does not.

**45.3% of 1-tick events snap back**, and they carry the flattering half:

| | n | adverse | net kept | t | ROI |
|---|---|---|---|---|---|
| snapped back (likely cancels) | 22,893 | **−0.00208** | +0.00652 | 30.3 | +1.54% |
| did not (likely real trades) | 27,634 | **+0.00322** | +0.00156 | 6.3 | +0.39% |

Negative adverse selection is the tell — a cancel cannot move the mid against
you because nobody traded. Restricting to non-reverting events, the stacked
filter still holds up but at half the size: **+0.00283 per share, ROI +0.85%,
t = 7.1**, with adverse selection 0.00196 against 0.00322 unfiltered.

And on the levels a $50 order could realistically join, non-reverting only:

| level size | n | net kept | t | ROI |
|---|---|---|---|---|
| $0–50 | 15,979 | +0.00142 | 4.9 | +0.41% |
| $50–200 | 3,622 | +0.00213 | 2.9 | +0.51% |
| $200–1,000 | 6,629 | +0.00196 | 3.4 | +0.40% |

### What this does and does not establish

It establishes that **book shape predicts adverse selection strongly and
usably** — imbalance and gap size are real signals, and they survive the
cancel screen. It does not establish that $50 can harvest them, for one
reason that snapshot data cannot resolve:

**The joining perturbation.** Every number above is measured on levels where
roughly $9–50 was already resting. Joining with $50 makes that level $59–100,
so the flow that would actually reach you is *not* the flow measured here.
Adding liquidity changes which trades hit you, and it changes it in the
unfavourable direction: you are filled by the larger orders, which are the
ones with 3–7x the adverse selection.

Realistic economics if it does hold: about +0.002/share on a ~$0.20 contract,
so roughly **$0.50 per fill on $50**. Reaching $10/day needs ~20 fills a day
while holding only one or two positions at a time — which implies a daily
return high enough to be implausible on its face, and that implausibility is
itself evidence that the perturbation or the exit-at-mid assumption is still
too generous.

**The one-line fix that would settle it:** record `last_trade_price` in
`live_recorder.py` (currently dropped at line 324). That removes the
fill/cancel ambiguity entirely and turns every number in this section from a
biased estimate into a clean measurement.

## Sharpe ratio and return — and why no single number is honest

Per-fill P&L is not a Sharpe ratio: 124,808 fills across 250 markets cannot
all be taken with $50. `polymarket_orderbook/maker_sim.py` walks the clock —
merges every market's events into one time line, takes the next qualifying
fill whenever flat, holds 30s, marks out, frees the capital — which produces
a real return series.

$50 capital, one position at a time, 3 days (Aug 28–30):

| scenario | fills/day | mean stake | P&L/fill | hit rate | total return | daily σ | **Sharpe (ann.)** | max DD |
|---|---|---|---|---|---|---|---|---|
| exit at mid, no screens | 848 | $15.29 | +$0.086 | 54% | +388% | 0.71 | **+26.0** | 43% |
| + cancel screen | 829 | $15.05 | +$0.063 | 46% | +276% | 1.00 | **+13.2** | 84% |
| + cross half the spread to exit | 829 | $15.05 | −$0.409 | 35% | **−1804%** | 3.73 | **−23.1** | — |
| + flow must clear your own size | **1.6** | $50.00 | −$0.333 | 33% | −2% | 0.07 | −1.8 | 8% |

**The Sharpe spans +26 to −23 on one assumption: what it costs to get out.**
That is not a measurement, it is a statement that the question is unresolved
by this data — and the reason is arithmetic, not sample size:

> the measured edge is **0.002–0.003 per share**, and the half-spread you
> cross to exit is **0.005**. Marking to the mid assumes a free exit. Paying
> anything close to the real exit cost flips the sign.

A Sharpe of 13–26 is a red flag rather than a finding. Serious HFT market
making runs Sharpe 3–10 with colocation and dedicated infrastructure; 26 from
a snapshot backtest means the backtest is missing costs. A daily σ of 0.71
alongside a 0.97 mean daily return — 97% per day — is self-evidently not a
real strategy.

Three further constraints the simulation exposes:

1. **Only ~$15 of the $50 ever deploys.** The filter that creates the edge
   requires a thin queue ahead, and thin levels hold a median of $15. You
   cannot size into the very condition that makes it profitable.
2. **Requiring the flow to actually fill $50 leaves 1.6 fills per day** —
   3 events in 3 days. The joining perturbation does not merely shrink the
   edge, it removes the opportunity set.
3. **Max drawdown is 43% even in the optimistic run**, 84% with the cancel
   screen. At $50 that is not survivable in any meaningful sense.

**What would resolve it.** Market making only works if you exit *passively* —
post the other side and get filled, earning the spread instead of paying it.
Whether that works depends on the passive exit fill rate and the adverse
selection on the exit leg, neither of which is observable without trade
prints. `live_recorder.py:324` drops `last_trade_price`; recording it is the
prerequisite for any credible Sharpe here.

**Best honest estimate of the round-trip economics**, stated as arithmetic
rather than a backtest: gross spread capture of one tick on a ~$0.40 contract
is ~2.5%; measured one-sided adverse selection is 0.0029, so a round trip
gives up roughly 0.006; net ≈ 0.004 per share, about **1% per completed round
trip** on a ~$15 position — around **$0.15**. Reaching $10/day requires ~67
completed round trips a day, and the risk that dominates is being filled on
one side and not the other, which is precisely the part that cannot be
measured here.

## Cross-market lead-lag: moneyline vs run line vs total vs NRFI

Same-game markets are mechanically linked and repriced by the same makers, so
any lag between them is a real dislocation rather than a difference of
opinion. This is a much better-posed question than game → futures, which
failed on tick granularity. `polymarket_orderbook/crossmarket.py`, 14 games.

**Result: no reliable lead-lag.** Cross-correlations of mid changes peak at
0.004–0.015 — indistinguishable from zero — and `spread → total` peaks at
lag 0, i.e. simultaneous.

A jump event study at a 250ms grid initially looked promising: after a
moneyline move of ≥2 ticks, the run line showed only +0.53 ticks at that
instant and then +1.85 to +2.01 ticks over the following 250ms–1s, implying a
~1.4-tick transient dislocation. **That did not survive a resolution check.**
Re-run on a 50ms grid the profile reverses sign and flattens, and the tell is
visible in the pre-event window: the run line was already moving −0.95 and
−1.52 ticks *before* the moneyline jumped. A genuine causal response cannot
precede its cause, so the 250ms result was forward-fill smearing across a
coarse grid, not a lead.

Two lessons worth keeping: pre-event drift is the cheapest diagnostic for a
spurious event study, and a lead-lag claim must be shown to survive a finer
time grid than the one that produced it.

## Does a CNN-Transformer on raw order books beat simple baselines?

Target: P(|mid(t+5s) − mid(t)| ≥ 2 ticks | book state up to t), from 10.6M
grid samples (200ms) with top-10 depth. `jump_data.py` builds both a LOB
tensor and hand-crafted order-flow features from the same window, so the
comparison is fair. Temporal split with a 120s gap.

The economic test, which is the one that matters. A maker quoting both sides
at the touch is filled on whichever side the market moves toward, so per
quote `pnl = half_spread − |mid(t+H) − mid(t)|`: earn the half-spread when
the market sits still, lose the excursion when it runs. Gating on predicted
jump probability is then directly actionable.

| model | AUC | Brier | P&L/quote | gain vs always-quote | quote frac |
|---|---|---|---|---|---|
| **gate on rv_150 only** (1 feature, no learning) | 0.884 | 0.169 | **+0.00635** | **+0.03524** | 0.75 |
| lightgbm, 20 hand features | **0.927** | **0.097** | +0.00374 | +0.03263 | 0.53 |
| CNN-Transformer, raw LOB | 0.874 | 0.146 | +0.00291 | +0.03180 | 0.24 |
| logistic regression, hand features | 0.898 | 0.118 | +0.00282 | +0.03171 | 0.45 |
| gate on rv_25 only | 0.844 | 0.176 | +0.00011 | +0.02900 | 0.80 |
| gate on spread only | 0.735 | 0.229 | −0.03462 | −0.00573 | 0.05 |
| always quote (no gating) | — | — | **−0.02889** | 0 | 1.00 |

**A single realised-volatility feature, rank-transformed with no model at
all, beats both the GBM and the CNN-Transformer on P&L** — while having a
*worse* AUC than either. The rank orderings are close to inverted:

- by AUC: lightgbm > logreg > rv_150 > cnn_transformer
- by P&L: rv_150 > lightgbm > cnn_transformer > logreg

This is precisely the failure mode the research framing anticipated. The
models optimise jump classification; the objective is "avoid quoting when the
excursion will exceed the half-spread". Sharpening the classifier moves the
selected subset in a direction that is worse economically.

The GBM's own feature importances say the same thing: **rv_150 is 56% of
total gain, staleness 17%, rv_25 9%** — 81% of the signal is volatility and
time-since-last-move. Every order-book microstructure feature combined
(imbalance at four depths, concentration, reach, microprice deviation)
contributes about 10%.

So the answer to the central question — is the extra information a deep model
extracts from order-book dynamics economically useful relative to simple
baselines? — is **no, not here**. The one genuinely valuable finding is the
bottom row: quoting indiscriminately loses 2.9 cents per share, and almost
any volatility filter fixes most of that.

**Caveats that bound this conclusion.** The CNN-Transformer (125k params) got
2 epochs on 400k samples on CPU and its loss was still falling (0.737 →
0.633), so this is a comparison at a modest fixed compute budget, not an
asymptotic one — a GPU run with more epochs could close the AUC gap, though
it would still have to overturn an inverted AUC-to-P&L ordering to change the
economic verdict. The temporal split also produced a base-rate shift (train
11.4%, test 30.4%), which penalises calibration-sensitive models. One day of
data, and the spread markets dominate the sample.

## Conclusion

The honest answer to "can free data beat Polymarket on sports?" is no — for
MLB, NBA, NHL or NFL game winners.

The market's closing price is the best single predictor available, the free
public data adds nothing to it, and the one apparently profitable strategy
in the backtest evaporated once the odds timestamps were checked. That last
step is the reusable lesson: a backtest that shows profit against "closing
lines" is worth nothing until you have verified those really are closing
lines.

Where this leaves something useful:

- The pipeline is a working, leak-free, well-calibrated probability model
  that runs on ~5 free API calls a day. `python -m mlbmodel.predict` scores
  today's slate and prices it against live Polymarket markets.
- Any real edge would have to come from information the market prices
  slowly, not from better modelling of information it already has: late
  scratches, weather, or bullpen availability in the minutes before first
  pitch. That is a latency problem, not a machine-learning one — and it is
  the opposite of mid-frequency. The `polymarket_orderbook/` recorder in
  this repo is aimed at exactly that question, and these results suggest it
  is the more promising of the two directions.
- Polymarket's own inefficiency, if any, is more likely to sit in thin
  markets than in the $2M MLB games, where the order book tracks the
  sportsbook consensus closely.

One caveat worth stating plainly: this tests **pre-game moneylines only**.
It says nothing about in-game markets, derivative markets (NRFI, spreads,
totals), or market-making rather than directional betting — all of which
have different economics and are untouched here.
