# Price jump prediction — what was achieved, and what it is good for

2026-09-16. Consolidates work across `results/jump_prediction/`,
`results/lee_mykland/`, `results/makinen/` and `results/sports/`. Written after
the economics came out negative, so it separates **what was established** from
**what it is worth**.

Every claim below distinguishes **OBSERVATION** (measured) from
**INTERPRETATION** (what it might mean).

---

## 1. A working, validated jump detector

Lee & Mykland (2008) implemented faithfully — bipower variation, strictly
trailing window, extreme-value threshold. One implementation in the repo;
everything else imports it. Method in `results/makinen/lee_mykland_method.md`.

Parameters were **re-derived from the data rather than copied from the paper**,
and the reasons were measured:

| parameter | paper | here | why |
|---|---|---|---|
| K | 600 min | **30 bars** | contracts live ~161 in-game minutes; K=600 leaves nothing testable |
| returns | log | **arithmetic** | at p=0.01 one half-tick is a 0.41 log return vs 0.0055 at p=0.90 |
| sample | all minutes | **in-game only** | pre-game is motionless and poisoned the volatility baseline (σ̂ 3.08× too low → 1.42×) |
| α | 0.01 | **0.01** | never tuned |

**OBSERVATION.** 1,516 jumps on 81,458 tested minutes = **1.86%**, median jump
**0.18 in probability points**.

**OBSERVATION — external validation.** Against MLB play-by-play on two whole
games: **95.5% and 97.0% of jump-minutes were corroborated by a second market
on the same game** firing in the same minute, and **6/9 and 8/9 actual runs**
produced a jump within ±2 minutes.

**INTERPRETATION.** Two independent order books moving together in the same
minute, aligned with a real scoring play, is not a book artifact. The detector
finds real events. This is the most solid thing in the project.

## 2. A large label defect, found and fixed

**OBSERVATION.** The original label, `max |mid_u − mid_t|`, counted any mid
excursion however brief and however broken the book. Measured artefact share:
**37%–99% depending on (J, H)**, worst where the event is rarest (tight books,
H=10s, J=0.05: **98.8% artefact**).

The mechanism: one side of the book empties, the next resting order is far
away, the midpoint moves mechanically and reverts when a quote returns — no
trade need occur. Visible in `results/jump_prediction/plots/examples_jumps_tight_*.png`.

**Fix.** A move counts only if the displaced price is observed while the book is
still tight (`clean`) AND it persists (`durable`). Regression-tested in
`research/jump_prediction/test_durability.py`.

**INTERPRETATION.** Any result built on the raw label was measuring quote
flicker. This is the single most important correction of the day and it
invalidated the first pass's headline.

## 3. Prediction: real but modest, and the architecture never mattered

### MLB, 1-minute horizon (`results/makinen/`)

| model | PR-AUC | 95% CI | lift |
|---|---|---|---|
| prevalence | 0.0258 | — | 1.00× |
| **time-of-day only** | 0.0247 | [0.018, 0.050] | **0.96×** |
| logistic (book) | 0.0904 | [0.055, 0.160] | 3.51× |
| MLP history (flattened) | 0.0448 | [0.036, 0.078] | 1.74× |
| CNN-LSTM-Attention | 0.1806 | [0.116, 0.252] | **7.01×** |

Only two comparisons resolved statistically (cluster bootstrap over games):
**the book beats time-of-day** (P = 0.999) and **flattening history hurts**
(P = 0.029). Attention vs LSTM: P = 0.610 — a coin flip.

**INTERPRETATION.** The paper's strong time-of-day effect does **not** transfer.
That is an artefact of exchange opening hours; these contracts trade
continuously and jumps follow the game clock.

### Esports, 21 days, resolution sweep (`results/sports/`)

| bar | windows | best lift | ROC-AUC |
|---|---|---|---|
| 1 s | 9,882 | 1.98× | 0.488 |
| 5 s | 88,814 | 3.27× | 0.706 |
| **15 s** | **142,971** | **6.21×** | 0.743 |
| 60 s | 57,031 | 4.99× | **0.856** |
| 300 s | 24,384 | 4.86× | 0.788 |

**OBSERVATION.** Signal peaks at **15-second bars** — four times finer than the
1-minute bar the paper uses. The 1s row is a *sparsity* artefact (only 13% of
one-second slots contain an update), not evidence against fine resolution.

## 4. The precursor is real, and it is ~6 seconds

Measured at 200 ms resolution on 430 clean large jumps against controls matched
at the *start* of the window (matching at the anchor was a bug that hid the
effect entirely).

| τ | spread | near-touch $ | price moved so far |
|---|---|---|---|
| −10s | 3.0 ticks | normal | 0.010 |
| **−6s** | **4.0** | **collapses** | 0.010 |
| −2s | 8.6 | drained | 0.020 |
| 0 | 13.0 | drained | 0.030 |
| **+0.4s** | **41.0** | — | **0.230** |

Effect sizes (Cliff's δ): **0.01 at −30s, 0.10 at −10s, 0.27 at −5s, 0.47 at
−2s, 0.67 at 0.**

**OBSERVATION.** The book visibly breaks while the price has moved ~3 ticks;
the jump of ~23 ticks follows. Per-event warning time is highly variable —
across 8 hand-inspected examples: **2.4s, 2.8s, 3.8s, 5.4s, 15.0s**.

**INTERPRETATION.** Near-touch liquidity withdrawal precedes repricing. This is
why every 1-minute model showed "zero lead time" — a 6-second precursor is
invisible in a 60-second bar.

## 5. The best-posed version: classify collapses

Reframing: stop hunting a rare jump at every instant; **detect every liquidity
collapse (mechanical, no model), then classify which are real.**

**OBSERVATION.** 134,428 collapse events over 6 sessions:

| outcome | count | share |
|---|---|---|
| **REAL** — durable move | 36,112 | **26.9%** |
| **FAKE** — moved then reverted | 43,924 | 32.7% |
| never moved | 54,392 | 40.5% |

Held-out session, **24,608 events**, base rate 31.9%:

| model | PR-AUC | ROC-AUC |
|---|---|---|
| prevalence | 0.319 | 0.500 |
| logistic (book at the instant) | 0.593 | 0.723 |
| CNN-LSTM-Attention | 0.614 | 0.749 |
| **GBM (instant + trajectory)** | **0.627** | **0.756** |

Dominant feature by 8×: **the spread at the moment the book breaks.**

**INTERPRETATION.** This is the strongest, best-evidenced predictive result in
the project — 24,608 held-out events rather than 109. A collapse in an already
wide book sticks; one in a tight book snaps back. And once again the tree beats
the neural net.

## 6. Direction: tested three ways, all negative

| test | result | baseline |
|---|---|---|
| pre-jump book asymmetry (Cliff's δ) | max 0.305, *larger at −15s than −1s* | — |
| imbalance features at a 5s lead | ROC-AUC 0.397–0.547 | 0.500 |
| **collapsing side = direction** | **50.2%** | **55.2%** |

**OBSERVATION.** Even the mechanical story fails: a bid collapse is followed by
a price fall only **45.1%** of the time. Signed-move distributions for bid-side
and ask-side collapses are near-identical.

**INTERPRETATION.** The mechanical tick when a side empties is directional but
transient; where the price *settles* is informational and unrelated to which
side blinked. **Direction is not in the order book.**

## 7. Economics: negative on both sides

**Taker.** EV = `P(real) × (2·accuracy − 1) × move − cost`. At 50% direction
accuracy the middle term is exactly zero, so **EV = −cost regardless of how good
the detector is**. Break-even would need 135% direction accuracy at a 4-tick
cost — no solution. Polymarket sports taker fees (~1–1.75 ticks) make it worse.

**Maker.** `maker_sim.py` on 124,808 candidate fills:

| assumption | fills (3 days) | P&L/fill |
|---|---|---|
| exit at mid, no screens | 4,428 | +$0.15 |
| + drop reprice cycles | 4,080 | +$0.07 |
| + pay the spread to exit | 4,080 | **−$0.40** |
| + flow must clear your size | **3** | −$0.33 |

**Capital saturates at ~$350.** Mean stake is $14.85 at both $350 and $5,000,
and both lose **the same $75** over three days.

**INTERPRETATION.** The apparent maker edge existed only under a free exit at
mid. Priced honestly it is negative, and the fills barely exist — which matches
the live paper-maker's zero fills.

---

## So what is the jump work actually good for?

**It is half a strategy.** It answers **WHEN** the price will move. It says
nothing about **WHERE TO**. On its own that is not tradeable — section 7 is
arithmetic, not pessimism.

Three genuine uses:

1. **Execution timing.** If you are going to trade for some *other* reason, do
   not cross the spread into a predicted collapse. The model flags, at ROC
   0.756, the moments when the book is about to break and the spread is about
   to go from 3 to 13 ticks. Avoiding those is worth real money to anyone
   trading these markets for any reason.

2. **A risk filter on resting orders.** If you ever do quote, pull on a high
   score. Filtering would cut adverse selection from 26.9% to roughly 7%. That
   does not rescue market making here — exit costs kill it first — but it is
   the correct use of the signal.

3. **The second half of a strategy whose first half is a fair-value model.**
   This is the real answer. Combine **WHEN** (this work) with **WHERE TO** from
   an independent source — cross-market consistency between moneyline, run line
   and total, which is recorded and untested, or a win-probability model, or
   signed order flow. Neither half trades on its own.

## What would change the conclusion

* **Signed trade flow** (aggressor side) — not currently recorded, the most
  likely source of direction, free to start collecting.
* **Cross-market consistency** — moneyline / spread / total must cohere; any
  inconsistency is a fair value that does not require latency. Untested.
* **More sessions** — the MLB architecture comparisons are unresolved purely
  for want of test data; the esports set (1,039 contracts, 21 days) already
  fixes this and is only partly exploited.

## What is safe to rely on

* The Lee–Mykland detector, validated against play-by-play.
* The clean/durable label; never the raw `max |mid|` label.
* The collapse classifier at ROC 0.756 on 24,608 held-out events.
* The ~6-second precursor, with the caveat that per-event warning varies 2–15s.
* That direction is not in the book, and that neither taker nor maker is
  profitable on these costs at any capital level tested.
