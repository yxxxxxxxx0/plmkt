# Methods and experiments, in full

Polymarket in-play order-book study, 2026-08 to 2026-09. This is the permanent
record of **what was tried, how it was built, and what it measured** — written
so that the implementation code for the closed methods can be deleted without
losing the work.

Every method below was implemented, run on real recorded data, and evaluated on
held-out games. The code for the closed ones was removed on **2026-09-21** and
is recoverable in full from git at commit **`40437ab`**:

```bash
git show 40437ab:polymarket_orderbook/taker_signal.py > taker_signal.py
git show 40437ab --stat            # everything that existed before the tidy
```

Companion documents: `RESEARCH_PROGRESS.md` (the conclusion),
`polymarket_orderbook/results/METHODOLOGY.md` (the surviving detection path),
`polymarket_orderbook/results/RULED_OUT.md` (the ledger, with figures).

---

## 0. The data and how it is built

**Recording.** `live_recorder.py` opens a websocket per contract for every MLB
moneyline, run-line and total on the slate and writes every book update as
JSONL, plus a flattened top-of-book CSV. It is driven from the real first pitch
(`collect_days.py` reads the MLB schedule, sleeps until 45 minutes before, and
refuses to start a slate more than an hour late on the grounds that a partial
recording is worse than none). Nightly via the `PolymarketSlateDaily` task,
whose trigger repeats every 30 minutes as a reboot watchdog.

Typical night: ~11M messages, ~22M book snapshots, 12 GB raw, ~100:1 under xz.

**Grid.** Raw updates are resampled onto a **200 ms grid**, trimmed to in-game
only (first pitch to final out, from `data/game_windows.json`). Trimming is not
cosmetic: a contract is listed hours before the game and barely moves, so
including pre-game fills every trailing volatility window with a quieter regime
than the one being tested. Measured as the ratio of forward realised volatility
to the trailing estimate, the bias was **3.08× before trimming and 1.42×
after.**

**Series identity is `asset_id`.** See defect 1 in section 7.

**Tick size is 0.01.** The sports taker fee is `0.05 · p · (1 − p)` dollars per
share, i.e. ~1.25 ticks a side at p = 0.5 and ~1.9 ticks for a round trip.
Makers are never charged.

**Sessions behind every number here:** 2026-08-28, 08-30, 09-10, 09-11, 09-12,
09-13 — six sessions, 75 games, 601 contracts, 28.0M in-game grid rows.

---

## 1. Jump arrival prediction

### 1.1 First pass — CNN/LSTM/Transformer on a raw mid label

*Implemented in* `jump_model.py`, `jump_split.py`, `collapse_cnn.py`,
`model_registry.py`, `research/jump_prediction/`.

Label: `future_move(t,H) = max over t < u ≤ t+H of |mid_u − mid_t|`, thresholded
at J. Architectures: CNN, CNN-LSTM, CNN-LSTM-Attention, CNN-Transformer, plus
gradient-boosted trees on hand-made features. Grid of J ∈ {0.01, 0.02, 0.03,
0.05} and H ∈ {10, 30, 60}s.

**Result: PR-AUC 0.93 — and invalid.** See defect 2. The label was largely a
function of the spread; a one-feature model on **spread alone** scored PR-AUC
0.817 against a 0.616 floor, two-thirds of the way to the full model.

### 1.2 Second pass — the corrected clean-and-durable label

*Implemented in* `research/jump_prediction/` (retained).

A move counts only if the displaced price is observed **while the book is still
tight** (clean) and the displacement **persists ≥ 3 s** (durable). Prediction
points restricted to books tight at `t`, so the model cannot win by detecting
"this book is already broken". Prevalence 0.1031.

| model | PR-AUC | lift | ROC-AUC |
|---|---|---|---|
| prevalence floor | 0.1031 | 1.00× | 0.500 |
| spread alone (control) | 0.1148 | 1.11× | 0.554 |
| current state only | 0.1607 | 1.56× | 0.648 |
| + price path only | 0.1923 | 1.87× | 0.695 |
| **+ book path only** | **0.1945** | **1.89×** | 0.694 |
| **+ all trajectory** | **0.2193** | **2.13×** | **0.723** |

**The finding that survived:** the book path alone (1.89×) matches the price
path alone (1.87×) and the two are complementary. On the contaminated label
that ordering was reversed, so the correction changed which family does the
work, not just the magnitude. Consistent across **14 of 14** held-out games,
lift range 1.57–2.82×.

**Architecture comparison, matched at 120,000 rows:** CNN 0.2097, trees 0.2058,
CNN-Transformer 0.2014, CNN-LSTM 0.1982 — all within 0.0115 of each other, and
the Transformer second-worst while carrying the most parameters. Given all 369k
rows the trees reach 0.2193, above every deep model. **More data helps; more
architecture does not.**

**Lead time.** Gapped labels (predict the label as computed at `t+g` from
features at `t`): 2.13× at 0s, 2.17× at 5s, 1.78× at 10s, 1.54× at 30s — real
information 5–30 s ahead. But first-alert lead time at a 5% alert rate has a
**median of 1 second**: the confident calls arrive at the move. Information
exists and is diffuse; confidence is concentrated where it is useless.

### 1.3 Mäkinen 1-minute replication

*Implemented in* `research/makinen/build_panel.py`, `label_jumps.py`,
`build_sequences.py`, `train_eval.py`, `models.py`.

Replicated the published 1-minute-bar setup on MLB. Lift 7.01× but only **109
test positives**, which left every architecture comparison unresolved — the
motivation for the reframing in section 2. The paper's strongest baseline,
**time-of-day**, scored lift 0.96× here, *below prevalence*: it does not
transfer to in-play sports.

### 1.4 Resolution sweep on esports

*Implemented in* `research/sports/build_bars.py`, `run_resolution.py`
(retained). 1,039 live contracts, 21 consecutive days. Bar sizes 1 s to 300 s.
Peak **6.21× at 15-second bars**. Real, and the economics never cleared.

### 1.5 Lee–Mykland jump detection

*Implemented in* `research/lee_mykland/` — `build_minute_bars.py`,
`detect_lee_mykland_jumps.py`, `plot_match_examples.py`, `inspect_data.py`.

Return ÷ trailing bipower volatility against the extreme-value threshold
(6.7809 at α = 0.01, K = 30; also run at K = 120 and K = 600). Used for
labelling and **external validation**, not in the live path.

**Validation result, and it is the reason the events are believed real:** on two
whole games, **95.5% and 97.0%** of Lee–Mykland jump-minutes were corroborated
by a *second market on the same game* firing in the same minute, and 6/9 and 8/9
actual runs scored produced a jump within ±2 minutes.

---

## 2. Collapse detection and classification — the surviving method

*Implemented in* `research/makinen/collapse_events.py`, `collapse_classify.py`,
`collapse_tree.py`, `live_replay.py` (all retained). Fully specified in
`results/METHODOLOGY.md`.

Reframing (Justin's): stop hunting a rare jump across all instants; **detect
every liquidity collapse with a rule, then classify which ones are real.** This
turns a 1–2% needle-hunt into a candidate set with a 26.9% base rate.

**Stage 1, no model.** Per contract, per side: near-touch liquidity = dollars
within 2 ticks of the mid; baseline = its trailing 60-second median; fire when
it drops below **25%** of baseline; guards are baseline ≥ $200 and a 30 s
refractory. **134,428 events** over six sessions. Deliberately *not* keyed on
the spread — 44.7% of collapses happen at a spread ≤ 2 ticks, so a spread
trigger would miss half of them and fire late on the rest.

**Stage 2.** The last 20 s of book history ending at the collapse instant — 50
snapshots × 22 features — reduced to 132 summary features (value now, 2 s
change, whole-window change, min, max, std) and fed to class-weighted
gradient-boosted trees.

**Outcome definition.** From the collapse instant the mid moves ≥ 0.02 within
10 s **and is still that far away 3 s after the peak**. REAL 26.9%, FAKE (moved
then snapped back) 32.7%, nothing 40.5%.

| model | PR-AUC | ROC-AUC |
|---|---|---|
| prevalence floor | 0.319 | 0.500 |
| logistic on the instant | 0.593 | 0.723 |
| CNN | 0.609 | 0.745 |
| CNN-LSTM | 0.610 | 0.746 |
| CNN-LSTM-Attention | 0.614 | 0.749 |
| **GBM, instant + trajectory** | **0.627** | **0.756** |

24,608 held-out events. **The tree wins again**, trains in 8 seconds, and is
interpretable. Dominant feature by ~8×: **the spread at the moment the book
breaks** — a collapse in an already-wide book sticks, one in a tight book snaps
back.

**Live tick-by-tick replay**, 112 contracts, 10.3 hours, state maintained
incrementally: ROC-AUC **0.665** (honestly lower than batch, because a live
system cannot score an event until it holds a full 20 s buffer), 82% precision
at a 2% alert rate with 0.05 false alerts per contract-hour, at 35× real time.

---

## 3. Direction — eight methods, all closed

Direction is the binding constraint. Every attempt is listed with what it
achieved, because three of them achieved genuine skill and lost money anyway.

| # | method | implemented in | skill | P&L |
|---|---|---|---|---|
| 1 | momentum (`dmid25`) | `diag_direction.py` | hit rate 58.4% | **−3.73 t** |
| 2 | GBM direction | `taker_signal.py` | sign AUC **0.662** | **−3.24 t** |
| 3 | CNN direction | `collapse_cnn.py` | AUC 0.634, **hit rate 73.5%** | **−5.60 t** |
| 4 | pre-jump book asymmetry | `research/makinen/pre_jump_seconds.py` | Cliff's δ max 0.305, *larger at −15 s than −1 s* | no timing content |
| 5 | imbalance at a 5 s lead | `research/makinen/pre_jump_5min.py` | ROC 0.397–0.547 | at/below chance |
| 6 | collapsing side → direction | `research/makinen/pre_jump_direction.py` | **50.2%** vs a 55.2% baseline | worse than guessing |
| 7 | direction from sequences | `research/makinen/train_eval.py` | ROC 0.88 | ~0.75 of it is mean-reversion of a spike already in the input |
| 8 | direction after a tight collapse | `research/makinen/direction_hold.py` | ROC **0.659** at +10 s | **−3.13 to −3.49 t** |

**Method 3 is the decisive one.** A 73.5% directional hit rate still lost 5.60
ticks per trade. Directional skill was achieved and was not enough.

**Method 6 kills the mechanical story.** A bid collapse is followed by a price
fall only **45.1%** of the time; signed-move distributions for bid-side and
ask-side collapses are near-identical. The mechanical tick when a side empties
is directional but transient; where the price *settles* is unrelated to which
side blinked. **Direction is not in the order book.**

---

## 4. Execution as a taker — six formulations

*Implemented in* `simulate_taker.py`, `taker_signal.py`, `diag_breakeven.py`,
`horizon_sweep.py`, `capital_required.py`, `pnl_by_game.py`,
`research/makinen/direction_hold.py`, and (2026-09-21)
`research/makinen/oracle_jump_scan.py` (retained).

| # | test | result |
|---|---|---|
| 1 | 5 s taker on predicted moves | break-even accuracy **≥ 100% at every move size** |
| 2 | taker on the collapse signal | **move ÷ spread ≈ 0.32** at every operating point |
| 3 | taker with *perfect* direction | EV −0.094 (top 50%) to −1.225 (top 1%) |
| 4 | the fee | a further ~1–1.75 ticks |
| 5 | tight entry, hold swept 10–600 s | negative at every horizon |
| 6 | every jump in the graph, classifier bypassed | **1.1% of 82,120 pay**; mean −7.98 t |

**Why cost beats the move.** A large move destroys the liquidity needed to exit
it, so cost scales with the prize:

| move size | share of rows | cost to trade |
|---|---|---|
| 0–1 ticks | 90.2% | 3.08 t |
| 4–6 ticks | 1.0% | 10.20 t |
| 10–20 ticks | 0.3% | 20.51 t |
| > 20 ticks | 0.2% | 33.00 t |

**Test 5 in detail (the strongest honest formulation).** Entry restricted to
collapses where the spread is already ≤ 3 ticks (76,976 events), exit after a
swept hold, round trip charged as `(spread_in + spread_out)/2` with **both ends
measured**:

| hold | E\|move\| | E[cost] | **oracle EV** | direction ROC | model EV |
|---|---|---|---|---|---|
| +10 s | 1.31 t | 2.62 t | **−1.31 t** | **0.659** | −3.39 t |
| +60 s | 2.63 t | 2.54 t | **+0.09 t** | 0.563 | −3.36 t |
| +600 s | 8.12 t | 2.55 t | +5.57 t | 0.552 | −3.13 t |

**The two ends close each other off.** At +10 s, the only horizon where the
signal carries direction, *perfect foresight still loses.* By +60 s the move
outruns the cost but direction has decayed to 0.563 against a 68.9%
requirement. Lengthening the hold does not trade the signal, it trades the
game: a *random* tight entry captures 3.0 of the 4.0 ticks a collapse entry
gets at +300 s.

---

## 5. Execution as a maker

*Implemented in* `maker_sim.py`, `paper_maker_sim.py`, `maker_regime_scanner.py`,
`adverse_selection.py`, `market_arb_scan.py`, `strategy_screen.py`.

124,808 candidate fills, assumptions layered on one at a time:

| assumption | fills (3 days) | P&L/fill |
|---|---|---|
| exit at mid, no screens | 4,428 | +$0.15 |
| + drop reprice cycles | 4,080 | +$0.07 |
| + pay the spread to exit | 4,080 | **−$0.40** |
| + flow must clear your size | **3** | −$0.33 |
| live paper maker, one session | **0 fills** | — |

By market type at a realistic exit: spread −$0.99/fill, total −$0.93, moneyline
−$0.47. Hit rates 12–17%. **Capital does not help** — mean stake saturates at
~$15, and $350 and $5,000 lose the same $75 over three days. The constraint is
the size of available fills, not the bankroll.

**Closed on fills, not on economics.** That distinction matters: it is a
different failure from the taker case.

---

## 6. The 2026-09-21 session — bypassing the classifier

*Implemented in* `research/makinen/oracle_jump_scan.py`,
`mark_breakeven_jumps.py`, `can_you_select_payers.py`,
`selector_realistic_exit.py`, `direction_precision_grid.py` (all retained).
Written up in `results/makinen/oracle_jumps/SUMMARY.md` and `SELECTOR.md`.

**6.1 Every jump, priced.** 82,120 durable moves read straight off the price
graph, each charged at its real entry and exit spread plus fees. **935 break
even — 1.1%.** The payers span 315 contracts, all 75 games, all six sessions.
Median payer moves 5.0 ticks, *identical to the median jump*; what separates
them is a 1-tick entry spread against 7 and a 2-tick exit against 8.

**6.2 The omniscient-exit trap.** Letting the oracle also pick the best exit
within 300 s turns the mean positive (+0.59 t). Against matched random in-game
entries scored identically, a random entry scores **+1.04 t** and a jump entry
**+0.59 t** — entering at a jump is *worse*. See defect 8.

**6.3 Moneyline clears its spread.** At a 1-tick entry: +0.242 t before fees,
game-clustered CI [+0.117, +0.364] — the only configuration in the project
where perfect foresight beats the spread. Moneyline books widen to ~5 ticks
when hit; spread and total widen to 22 and 16. The fee (1.94 t) then takes it
away, eight times over, to −1.697 t.

**6.4 Selection is solved.** A GBM on the book state at the jump's onset, split
by whole game with the last two sessions held out: **ROC-AUC 0.865, 22.1× lift,
52.9% precision at the top 0.1%.** Against 4,000 random draws of equal size the
model sits at the 100th percentile — real skill, not luck.

**6.5 The two dials.** `EV = (2a − 1)·E|move| − E[cost + fee]` on a 30 s clock:

| direction accuracy | loosest profitable cut | precision needed | EV | 95% CI |
|---|---|---|---|---|
| ≤ 70% | **nothing works at any precision** | — | — | — |
| 75% | top 0.1% | 54.5% | +0.03 t | [−0.43, +0.55] |
| 80% | top 0.1% | 54.5% | +0.38 t | [−0.17, +1.01] |
| 90% | top 0.5% | 38.9% | +0.42 t | **[+0.10, +0.78]** |
| 100% | top 1% | 29.9% | +0.54 t | **[+0.08, +0.99]** |

**The precision needed is 30–55% and the selector already delivers it.** The
constraint is entirely the row label. Nothing is positive below 75% directional
accuracy at any precision.

---

## 7. Defects found — every one produced a convincing false positive first

1. **Series key omitted `asset_id`.** `(slug, market_type, line)` collides
   across the two legs of a spread, which differ only by `asset_id`. The merged
   "price" alternated between two unrelated levels, manufacturing large,
   almost perfectly predictable moves. **A one-line rule using nothing but the
   current level scored 76% directional accuracy and beat the CNN.** Found by
   plotting the price. Guarded by `check_series_key.py` (retained) and
   documented in `polymarket_orderbook/FINDINGS_SERIES_KEY.md`.

2. **Raw `max |mid|` jump label.** 37–99% of "jumps" were quote vacuums — one
   side of the book is cancelled, the midpoint moves mechanically, a quote
   returns and it reverts, with no trade at any point. Worst where the event is
   rarest and most interesting: tight books, H = 10 s, J = 0.05 is **98.8%
   artefact.** Regression-tested in
   `research/jump_prediction/test_durability.py`.

3. **`future_move_H*` leaked past a literal-name feature filter.** ROC-AUC
   **1.0000**. Filter now bans by prefix *and* suffix, with an assertion.

4. **Event-study controls matched at the anchor.** Selected control moments
   that were themselves depleted, hiding the effect entirely — δ ≤ 0.17 instead
   of ≤ 0.67.

5. **`asset_id` read as float64 by pyarrow.** Lossy on 77-digit integers.

6. **pandas `groupby` silently drops NaN keys.** Moneyline carries a null
   `line` and vanished from a figure.

7. **A fat-tailed exit spread summarised by its median.** Median 1.0 tick, mean
   3.8, p90 9, p99 43. Quoting the median gave a 1.5-tick round trip instead of
   2.5–2.6 and briefly showed a profit that does not exist. **Expected value
   runs on means.**

8. **An oracle exit that picks the best of ~1500 future instants.** Positive on
   a pure random walk by construction. Meaningless without a matched random
   control — and against that control the jump entries were *worse* than
   arbitrary ones.

9. **A mean carried by one observation.** A single 75-tick move was 38% of a
   34-trade set's total, and made a directional requirement look like 65.3%
   when the trimmed figure is 75.3%.

**The standing rules these produced.** Prove any key used to group, join or
forward-fill a series is unique against the raw source. Plot the price and
compute a model-free oracle bound before believing a model score. Treat a
result concentrated in one slice as a corruption signature, not a finding.
Audit sibling code for the same class of bug once one instance is found.

---

## 8. What the deleted code did, file by file

Recoverable at `40437ab`. Grouped by the method it implemented.

**Taker / direction line (section 3, 4).** `jump_split.py` (dataset assembly
and grouped splits), `jump_model.py` (model zoo), `taker_signal.py` (GBM
direction + taker sim), `collapse_cnn.py` (CNN direction), `model_registry.py`,
`score_session.py`, `simulate_taker.py`, `cv_grouped.py` (game-clustered CV),
`capital_required.py`, `pnl_by_game.py`, `horizon_sweep.py`, `forward_test.py`,
`diag_direction.py`, `diag_reversal.py`, `diag_slices.py`, `diag_breakeven.py`,
`diag_timing_fit.py`, `visualize_jumps.py`, `collapse_summary_plot.py`,
`show_sim.py`.

**Post-mortems on the series-key bug.** `why_it_looked_profitable.py`,
`compare_seriesfix.py`. Their conclusions are preserved verbatim in
`FINDINGS_SERIES_KEY.md`.

**Maker line (section 5).** `maker_sim.py`, `paper_maker_sim.py`,
`maker_regime_scanner.py`, `adverse_selection.py`, `liquidity_collapse.py`,
`market_arb_scan.py`, `strategy_screen.py`.

**Lee–Mykland validation (section 1.5).** `research/lee_mykland/*.py`. Results
and figures retained under `results/lee_mykland/`.

**Root-level one-off audits.** `cross_market_lead_lag.py`,
`cross_category_lead_lag_backtest.py`, `build_linked_cross_market_universe.py`,
`discover_cross_market_candidates.py`, `inspect_cross_market.py`,
`kaggle_cross_market_extract.py`, `maker_feasibility_audit.py`,
`match_game_delay_audit.py`, `match_game_maker_audit.py`,
`taker_delay_check.py`. Their outputs are retained under
`polymarket_sports/reports/`.

**Note on cross-market.** The root-level `cross_market_*` scripts tested
lead-lag *across categories* on the Kaggle sports dataset. They are **not** the
open cross-market question, which is whether moneyline, run line and total on
the *same game* cohere. `polymarket_orderbook/crossmarket.py` was the start of
that and carried the series-key defect; it is deleted deliberately, because
that work should be rebuilt on the fixed key rather than patched.
