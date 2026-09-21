# Research progress

Polymarket in-play order-book study. Last updated 2026-09-21.

Companion documents: `REPRODUCE.md` (what to copy, re-download and rebuild),
`polymarket_orderbook/results/METHODOLOGY.md` (the detection path end to end),
`polymarket_orderbook/results/RULED_OUT.md` (the ledger of closed doors).

---

## The conclusion in one paragraph

We can detect liquidity collapses with a trailing rule, tell the real ones from
the fake ones at ROC-AUC 0.756, and — as of today — pick out the individual
jumps that would be profitable to trade at 22× better than chance. None of it
makes money, and the reason has not changed across four months and eight
independent attacks: **we can predict *when* the price will move and we cannot
predict *which way*.** A trade needs both. The final measurement, made today,
is that the selection problem is now solved well enough and the direction
problem needs 75–90% accuracy against roughly 62% achieved. That gap is the
project.

---

## What was built

**A recorder.** `live_recorder.py` captures full order books for every MLB
moneyline, run-line and total contract on the slate, tick by tick, driven off
the real first pitch. It has run unattended nightly since mid-September; the
task, its watchdog and its failure modes are documented in `REPRODUCE.md`.
Typical night: ~11M messages, ~22M book snapshots, 12 GB raw, ~100:1 under xz.

**A research pipeline.** Raw JSONL → 200 ms in-game grid → features → labels →
models, with regression tests on the label decomposition and asserted leakage
controls. Everything is keyed on `asset_id`, and there is a test that fails if
that ever regresses.

**A jump detector and classifier.** Watch near-touch liquidity; when it falls
below a quarter of its own 60-second median, ask a gradient-boosted tree
whether the last 20 seconds of book structure say the move will stick.

**A trade-level economic harness.** Every result is priced at the spread
actually standing at entry and at exit, with the Polymarket sports taker fee
(`0.05 · p · (1−p)`, ~1.9 ticks a round trip) charged on both legs, and
confidence intervals bootstrapped over whole games.

---

## What genuinely works

These are real, replicated, held-out results. They are also all *statistical*
rather than *economic*.

| result | figure | where |
|---|---|---|
| collapse detection, no model at all | 134,428 events, 26.9% real | `METHODOLOGY.md` |
| real-vs-fake collapse classification | **ROC-AUC 0.756**, PR-AUC 0.627 vs 0.319 floor, 24,608 held-out events | `results/makinen/collapse/SUMMARY.md` |
| live tick-by-tick replay | ROC-AUC 0.665, 82% precision at a 2% alert rate, 35× real time | `METHODOLOGY.md` |
| jump-arrival from LOB trajectory | 2.13× lift over prevalence, 14/14 held-out games | `results/jump_prediction/summary.md` |
| book path vs price path | 1.89× vs 1.87× — liquidity carries as much as volatility | same |
| **picking the jumps that would pay** | **ROC-AUC 0.865, 22.1× lift, 52.9% precision at the top 0.1%** | `results/makinen/oracle_jumps/SELECTOR.md` |

Two structural lessons run through all of it. **Architecture never mattered** —
gradient-boosted trees on hand-made features matched or beat CNN, CNN-LSTM and
CNN-LSTM-Attention in every matched comparison. And **more data helped where
more model did not**.

---

## What does not work, and why

### Direction — eight attempts, all closed

Momentum, GBM, CNN, pre-jump book asymmetry, imbalance at a 5s lead, the
collapsing side itself, sequence models, and direction after a tight-spread
collapse. Three produced genuine statistical skill. All lost money.

The decisive one: a CNN reaching a **73.5% directional hit rate still lost 5.60
ticks per trade.** Direction was partly solved and it was not enough.

### The structural reason

The model predicts jumps by noticing that liquidity has gone, and "liquidity
has gone" and "trading is expensive" are the same sentence. At the model's most
confident predictions the median spread is 72 ticks against a median move of
23. **The signal is a measurement of untradeability.**

### The arithmetic

`EV = P(real) × (2·accuracy − 1) × move − cost`. At 50% direction accuracy the
middle term is zero and EV is just `−cost`, no matter how good the detector is.
Cost also grows faster than the move, because a large move destroys the
liquidity you need to exit it: 0–1 tick moves cost 3.08 ticks to trade, >20
tick moves cost 33.00. Break-even accuracy is ≥100% at every move size.

### Making markets instead

Closed on fills, not on economics. Priced with a realistic exit it is −$0.40 a
fill, and once incoming flow actually has to clear your size, three days of
data yield **three fills**. Capital does not help — $350 and $5,000 lose the
same $75.

---

## What was settled on 2026-09-21

Four questions were asked and answered in one session.

**1. Can you go around the classifier?** Yes, and it changes nothing. Skipping
the detector entirely and reading every durable move straight off the price
graph — 82,120 jumps, 601 contracts, 75 games — then pricing each at its real
entry and exit spread plus fees: **935 of 82,120 break even, 1.1%.** Perfect
detection of every jump that exists still loses on 98.9% of them. The
classifier's recall was never the constraint.

**2. What do the paying jumps look like?** The median payer moves 5.0 ticks —
*identical to the median jump overall*. Size of move is irrelevant. What
separates them is cost: 1-tick entry spread against 7, 2-tick exit against 8.
They span 315 contracts, all 75 games, all six sessions, so it is not an
artefact. Together they are worth 2,230 ticks, about 0.6 per contract-hour.

**3. One slice clears its spread.** Moneyline entered at a 1-tick spread nets
**+0.242 ticks before fees**, CI [+0.117, +0.364] — the only configuration
found anywhere in this project where perfect foresight beats the spread.
Moneyline books widen to ~5 ticks when hit; spread and total books widen to
22 and 16. The **fee** then takes it away, at 1.94 ticks, eight times the gross
edge. This is the first place the binding constraint is the fee rather than the
spread.

**4. What precision would you need, given direction?** This is the cleanest
statement the project has produced:

| direction accuracy | loosest profitable cut | precision needed | EV | 95% CI |
|---|---|---|---|---|
| ≤ 70% | **nothing works at any precision** | — | — | — |
| 75% | top 0.1% | 54.5% | +0.03 t | [−0.43, +0.55] |
| 80% | top 0.1% | 54.5% | +0.38 t | [−0.17, +1.01] |
| 90% | top 0.5% | 38.9% | +0.42 t | **[+0.10, +0.78]** |
| 100% | top 1% | 29.9% | +0.54 t | **[+0.08, +0.99]** |

**The precision required is 30–55%, and the selector already delivers it.**
Selection is solved. Nothing is positive below 75% directional accuracy at any
precision, and only at 90%+ does a confidence interval exclude zero. Measured
direction skill is ROC-AUC 0.659 — a ranking statistic weaker than a 66% hit
rate.

For the first time the two problems are cleanly separated, and only one of them
is left.

---

## Data and method defects found (do not repeat)

These cost more time than the modelling did, and every one produced a
convincing false positive first.

1. **Series key omitted `asset_id`** — merged the two legs of a spread into one
   price series and manufactured near-perfectly predictable moves. A one-line
   rule scored 76% on the corrupted data and beat the CNN.
2. **Raw `max |mid|` jump label** — 37–99% of "jumps" were quote-vacuum
   artefacts. Produced a meaningless PR-AUC of 0.93.
3. **`future_move_H*` leaked past a literal-name feature filter** — ROC-AUC
   1.0000.
4. **Event-study controls matched at the anchor** — selected control moments
   that were themselves depleted, hiding the real effect.
5. **`asset_id` read as float64** — lossy on 77-digit integers.
6. **pandas `groupby` silently drops NaN keys** — moneyline vanished from a
   figure.
7. **Summarising a fat-tailed exit spread by its median** — median 1.0 tick,
   mean 3.8, p99 43. Briefly produced a profit that did not exist. EV runs on
   means.
8. **An oracle exit that picks the best of ~1500 future instants** — positive
   on a pure random walk. Only meaningful against matched random entries, and
   against that control, entering at a jump is *worse* than entering at an
   arbitrary moment.
9. **A mean carried by one observation** — a single 75-tick move was 38% of a
   34-trade set's total, and made a directional requirement look like 65% when
   it is 75%.

The standing rule that came out of this: **plot the price and compute a
model-free oracle bound before believing any model score**, and treat a result
concentrated in one slice as a corruption signature rather than a finding.

---

## The data asset

| session | status |
|---|---|
| 2026-08-28, 08-30, 09-10, 09-11, 09-12, 09-13 | analysed; feature and LOB caches built |
| 2026-09-16 (partial daytime run) | **deleted 2026-09-21** — 2 hours only, never used |
| 2026-09-17 | recorded, compressed, not yet processed |
| 2026-09-18, 09-19, 09-20 | recorded, **uncompressed (~63 GB), unverified** |

Six sessions, 75 games and 601 contracts are behind every number in this
document. Four further nights are banked and unused — going from 6 to 10
sessions is a ~67% increase in training data, and this project's own finding is
that data helps where architecture does not.

Immediate housekeeping: compress and verify 09-18, 09-19 and 09-20. That
reclaims ~60 GB and tells us whether the nights are clean.

---

## What is still open

Ranked by what the evidence actually supports.

1. **Re-measure the direction requirement on ten sessions.** The only operating
   point where the gap is small rests on 33 trades over two held-out sessions,
   with one game supplying 37% of the P&L at the next cut down. That is not a
   sample anyone can conclude from. Doubling the held-out data is the single
   highest-value next step and requires no new ideas.

2. **Cross-market consistency.** Moneyline, run line and total on the same game
   must cohere; any inconsistency is a fair value needing no latency edge and
   no direction model. All three are recorded on one clock. **Never tested.**
   `crossmarket.py` exists but predates the series-key fix and must be repaired
   first.

3. **Signed trade flow.** The aggressor side is not currently recorded. It is
   the one plausible remaining source of direction, and it is free to start
   collecting.

4. **The moneyline-and-fee result.** The closest thing to break-even is
   moneyline at a tight entry, failing by a fee that makers are never charged.
   The maker path was closed on *fills*, not economics, so this is a different
   door — but it is a narrow one and should not be opened without the fill
   analysis redone on moneyline specifically.

5. **The esports dataset** — 1,039 live contracts over 21 consecutive days,
   only partly exploited. It fixes the sample-size problem that left the MLB
   architecture comparisons unresolved.

**Not open:** anything that tries to profit from knowing when or where the price
moves inside a single contract's order book. Eight direction methods, seven
jump-arrival methods, six taker formulations and five maker assumptions are
closed, several of them having achieved real statistical skill first. The
untested directions are all about **knowing what a contract is worth**, not when
it will move.

---

## Reproducing the 2026-09-21 work

```bash
cd polymarket_orderbook
python research/makinen/oracle_jump_scan.py          # every jump, priced
python research/makinen/mark_breakeven_jumps.py      # only the ones that pay
python research/makinen/can_you_select_payers.py     # can a model pick them
python research/makinen/selector_realistic_exit.py   # on a runnable clock
python research/makinen/direction_precision_grid.py  # both dials together
```

Artefacts land in `results/makinen/oracle_jumps/`, with `SUMMARY.md` and
`SELECTOR.md` carrying the written findings and four figures.
