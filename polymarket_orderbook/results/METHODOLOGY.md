# Methodology: how the model detects price jumps

2026-09-16. The live path, end to end, with every parameter. Companion to
`PRICE_JUMP_REPORT.md` (what it achieves) and `RULED_OUT.md` (what it cannot
do). Code in `research/makinen/`.

## The design idea

Most jump-prediction work scans every instant asking "will a jump happen in the
next interval?" On this data that is a 1–2% prevalence needle hunt, and it is
what made the first version of this study unreadable — 109 positives in the
test set and no comparison that resolved statistically.

This pipeline splits the problem in two:

1. **A mechanical rule finds every candidate.** Cheap, causal, no training,
   catches everything.
2. **A model decides which candidates matter.**

That converts a rare-event problem into a classification problem with a
**26.9% base rate** on 134,428 candidates, which is why the numbers here are
the best-evidenced in the project.

Only stage 2 is a model. Stage 1 is arithmetic.

---

## Stage 1 — The price series

Source: the project's 200 ms in-game grid,
`data/jump/feat_books_*_trimmed.parquet` + `lob_books_*_trimmed.npy`
(10 depth levels per side).

```
mid = (best_bid + best_ask) / 2        # jump_data.py:406
```

**In-game only**, first pitch to final out, from `data/game_windows.json`. This
is not cosmetic: a contract is listed hours before the game and barely moves,
so including pre-game fills every trailing volatility window with a quieter
regime than the one being tested. Measured as the ratio of forward realised
volatility to the trailing estimate, that bias was **3.08×** before trimming
and **1.42×** after.

Series identity is `asset_id`. Slug + market_type + line alone collide across
the two legs of a spread — the defect recorded in `FINDINGS_SERIES_KEY.md`.

## Stage 2 — Detect the candidate (`collapse_events.py`)

Run continuously, per contract, per side (bid and ask independently):

| step | rule |
|---|---|
| measure | **near-touch liquidity** = dollars resting within **2 ticks** of the mid |
| baseline | trailing **60-second median** of that quantity, strictly backward-looking |
| trigger | near-touch **< 25%** of baseline |
| guard | baseline must be **≥ $200** (ignore books that were already dead) |
| debounce | **30 s** refractory, so one event cannot fire repeatedly |

That is the whole detector. It fired **134,428 times** across six sessions
(67,915 bid-side, 66,513 ask-side).

**What it is deliberately not keyed on: the spread.** Depth can vanish while
the best bid and ask prices stay exactly where they were — **44.7% of all
collapses occur at a spread of ≤ 2 ticks.** A spread-based trigger would miss
nearly half of them, and would only fire once the damage was already visible.

## Stage 3 — Classify the candidate (`collapse_classify.py`, `collapse_tree.py`)

**Input.** The last **20 seconds** of book history *ending at* the collapse
instant — 50 snapshots (200 ms grid, stride 2) × **22 features** per snapshot:

| group | features |
|---|---|
| price | `spread_ticks` |
| touch | `l1_bid_usd`, `l1_ask_usd` |
| depth | `log_bid_depth`, `log_ask_depth`, `log_total_depth` |
| balance | `imbalance`, `imbalance_l1` |
| shape | `hhi_bid`, `hhi_ask`, `entropy_bid`, `entropy_ask` |
| slope/extent | `slope_bid`, `slope_ask`, `levels_bid`, `levels_ask`, `reach_bid`, `reach_ask` |
| near touch | `log_bid_usd_within_2t`, `log_ask_usd_within_2t`, `near_frac_bid_2t`, `near_frac_ask_2t` |

Nothing after the collapse instant enters the input. The detector is itself a
trailing rule, so the candidate set is causal too.

**The model.** Gradient-boosted trees (LightGBM, class-weighted) on that window
reduced to **132 features** — for each of the 22 channels: the value now, the
change over the last ~2 s, the change across the whole window, and the window
min, max and standard deviation.

A CNN, a CNN-LSTM and the paper's CNN-LSTM-Attention were built on the same
sequences. **The trees won** (PR-AUC 0.627 vs 0.614 for attention), train in
8 seconds instead of minutes, and are interpretable — consistent with every
other architecture comparison in this project.

**Dominant feature, by roughly 8×: the spread at the moment the book breaks.**
A collapse in an already-wide book sticks; a collapse in a tight book snaps
back.

**The decision.** The probability is compared to a threshold chosen on training
sessions only and then frozen. Nothing about the held-out session informs it.

## Stage 4 — What counts as a "real" jump

The definition everything rests on:

> From the collapse instant, the mid moves **≥ 0.02** (2 probability points)
> within **10 seconds**, **and is still at least that far away 3 seconds after
> the peak.**

The persistence clause does all the work. It splits the candidates into:

| outcome | count | share |
|---|---|---|
| **REAL** — moved and stayed moved | 36,112 | **26.9%** |
| **FAKE** — moved, then snapped back as liquidity returned | 43,924 | 32.7% |
| nothing happened | 54,392 | 40.5% |

Without it, REAL and FAKE are indistinguishable. Conflating them is exactly the
defect that made the first pass report a meaningless PR-AUC of 0.93: on the raw
`max |mid_u − mid_t|` label, **37–99% of "jumps" were quote vacuums** — one
side of the book empties, the midpoint moves mechanically, a quote returns and
it reverts, with no trade at any point.

## Two different "jump" definitions in this project

| | where it is used | definition |
|---|---|---|
| **Lee–Mykland jump** | the earlier studies; labelling; validation | return ÷ trailing bipower volatility exceeds the extreme-value threshold (6.70 at α=0.01) — *statistically surprising* |
| **Durable move** | this pipeline | ≥ 2 ticks within 10 s, held 3 s — *economically real* |

**The live path does not use Lee–Mykland.** It was how the project arrived
here, and it is what validated the detector externally — on two whole games,
95.5% and 97.0% of Lee–Mykland jump-minutes were corroborated by a second
market on the same game firing in the same minute, and 6/9 and 8/9 actual runs
produced a jump within ±2 minutes. That established the events are real.
The live classifier then works directly on the durable-move definition.

## What it achieves

**Batch**, held-out session, 24,608 events, base rate 31.9%:

| model | PR-AUC | ROC-AUC |
|---|---|---|
| prevalence floor | 0.319 | 0.500 |
| logistic, book at the instant | 0.593 | 0.723 |
| CNN-LSTM-Attention | 0.614 | 0.749 |
| **GBM (instant + trajectory)** | **0.627** | **0.756** |

**Live tick-by-tick replay** (`live_replay.py`), 112 contracts, 10.3 hours,
all state maintained incrementally in Python:

| | |
|---|---|
| collapse events detected live | 14,449 |
| genuinely real | 35.2% |
| alerts at the frozen top-2% threshold | 309 |
| **precision** | **0.819** |
| recall | 0.050 |
| false alerts per contract-hour | 0.05 |
| **ROC-AUC** | **0.665** |

The live figure is **lower than batch (0.665 vs 0.756)** and is the honest one:
a live system cannot score an event until it holds a full 20-second buffer, so
it sees a harder mix. Throughput was 19,600 ticks/sec — about **35× real time**
— so inference latency is not a constraint.

Precision and recall are a **dial**, not a property. At the top-2% threshold:
82% precision, 5% recall. At the top 10%: 73% precision, 21% recall, still only
0.33 false alerts per contract-hour.

## Reproducing it

```bash
python research/makinen/build_panel.py        # 1-minute panel + 10-level book
python research/makinen/label_jumps.py        # Lee-Mykland labels (validation)
python research/makinen/collapse_events.py    # stage 2: 134,428 candidates
python research/makinen/collapse_classify.py  # stage 3: CNN family
python research/makinen/collapse_tree.py      # stage 3: the GBM that wins
python research/makinen/live_replay.py        # tick-by-tick honest evaluation
```

Leakage controls are asserted in `research/makinen/test_leakage.py` and the
label decomposition is regression-tested in
`research/jump_prediction/test_durability.py`.

## One line

> Watch near-touch liquidity. When it drops below a quarter of its own
> 60-second normal, ask a tree model whether the last 20 seconds of book
> structure say this one will stick.
