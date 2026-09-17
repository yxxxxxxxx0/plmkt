# Ledger of what has been tested and found not tradable

2026-09-17. A record of closed doors, so none of them get re-opened by accident.
Each entry gives the measured result and where it lives. Where a method *works
statistically* but still is not tradable, both facts are stated — that
distinction is the main lesson of this project.

---

## A. Direction prediction — 7 attempts, all closed

Direction is the binding constraint. It has been attacked seven independent
ways and every one has failed, including three that produced genuine
statistical skill.

| # | method | skill achieved | P&L / verdict |
|---|---|---|---|
| 1 | momentum (`dmid25`) | hit rate 58.4% | **−3.73 ticks/trade** |
| 2 | GBM direction | sign AUC **0.662** | **−3.24 ticks/trade** |
| 3 | CNN direction | sign AUC 0.634, **hit rate 73.5%** | **−5.60 ticks/trade** |
| 4 | pre-jump book asymmetry (Cliff's δ) | max 0.305, but *larger at −15s than −1s* | no timing content |
| 5 | imbalance features at a 5s lead | ROC-AUC 0.397–0.547 | at/below chance |
| 6 | collapsing side → direction of move | **50.2%** | baseline 55.2% — worse than guessing |
| 7 | jump-direction from sequences | ROC-AUC 0.88 | ~0.75 of it is mean-reversion of a spike already visible in the input |
| 8 | direction after a tight-spread collapse, holds 10s–600s | ROC **0.659** at +10s, decaying to 0.544 by +300s | **−3.13 to −3.49 ticks/trade**, every horizon — see C5 |

**The decisive one is #3.** A 73.5% directional hit rate still lost 5.60 ticks
per trade. Directional skill was *achieved* and was *not enough*, because the
moves are too small relative to costs. Do not assume that solving direction
solves the problem — it has already been partly solved and did not.

`data/jump/taker_comparison.csv`, `results/makinen/pre_jump_direction/`,
`results/makinen/collapse/SUMMARY.md`.

## B. Jump arrival prediction — works statistically, does not pay

| # | method | result | status |
|---|---|---|---|
| 1 | CNN/LSTM/Transformer on 200ms grid, raw-mid label | PR-AUC 0.93 | **invalid** — 37–99% of labels were quote-vacuum artefacts |
| 2 | same, corrected clean/durable label | lift 2.13× | real but weak; zero usable lead time |
| 3 | Mäkinen 1-min replication (MLB) | lift 7.01× | only 109 test positives; every architecture comparison unresolved |
| 4 | **time-of-day baseline** | lift **0.96×** | *below prevalence* — the paper's strongest baseline does not transfer |
| 5 | MLP on flattened history | lift 1.74× | **worse** than the snapshot MLP (P = 0.029) |
| 6 | resolution sweep 1s–300s (esports, 21 days) | peak **6.21× at 15s bars** | real; economics never cleared |
| 7 | collapse classification | **ROC-AUC 0.756** on 24,608 held-out events | best model in the project — and untradeable, see D2 |

Architecture never mattered. Gradient-boosted trees matched or beat CNN,
CNN-LSTM and CNN-LSTM-Attention in **every** comparison run.

## C. Execution as a taker — closed by arithmetic

| # | test | result |
|---|---|---|
| 1 | 5-second taker on predicted moves | break-even accuracy **≥100% at every move size**; perfect foresight still loses ~3 ticks |
| 2 | taker on the collapse signal | **move ÷ spread ≈ 0.32** at every operating point |
| 3 | taker with *perfect* direction on the collapse signal | EV **−0.094** (top 50%) to **−1.225** (top 1%) |
| 4 | Polymarket sports taker fee | adds a further ~1–1.75 ticks |
| 5 | **tight entry + wait for the spread to re-tighten, hold swept 10s–600s** | **negative at every horizon**, see below |

**The structural reason.** The model predicts jumps by detecting that liquidity
has already gone. "Liquidity has gone" and "trading is expensive" are the same
sentence. At the model's most confident predictions the median spread is **72
ticks** against a median move of **23 ticks**. The signal *is* a measurement of
untradeability.

### C5 in full — the strongest version of the taker case, and it still fails

Entry restricted to collapses where the spread is already **≤ 3 ticks**
(76,976 events), exit after a swept holding period, and the round trip charged
correctly as `(spread_in + spread_out) / 2` with **both ends measured**, not
assumed. This is the most favourable honest formulation found, and it was built
to answer the objection that earlier tests paid the wide mid-move spread.

Two corrections to earlier numbers are folded in here:

* A round trip is the **average** of the entry and exit spread, not twice the
  entry spread. Earlier taker tests over-charged.
* The exit spread must be summarised by its **mean, not its median**. Its
  median is 1.0 tick but its mean is 3.8, with p90 = 9 ticks and p99 = 43. An
  interim version of this analysis quoted the median, reported a 1.5-tick round
  trip, and briefly showed a profit that does not exist.

| hold | E\|move\| | E[cost] | **oracle EV** | direction ROC | model EV (held out) |
|---|---|---|---|---|---|
| +10s | 1.31t | 2.62t | **−1.31t** | **0.659** | −3.39t |
| +20s | 1.66t | 2.64t | −0.98t | 0.610 | −3.49t |
| +30s | 1.90t | 2.54t | −0.64t | 0.604 | −3.23t |
| +60s | 2.63t | 2.54t | **+0.09t** | 0.563 | −3.36t |
| +120s | 3.65t | 2.44t | +1.21t | 0.549 | −3.25t |
| +300s | 5.76t | 2.46t | +3.30t | 0.544 | −3.29t |
| +600s | 8.12t | 2.55t | +5.57t | 0.552 | −3.13t |

Move and cost are means over all 74,661–76,976 events; ROC and model EV are on
the held-out session (`books_2026-09-13`, n ≈ 14,000). Oracle EV is
`E|move| − E[cost]`, the ceiling for *any* direction model, and it does not
turn positive until **+60 s**. Full sample in
`results/makinen/direction_hold/oracle_bound_full_sample.csv`.

**The two ends of the sweep close each other off.** At +10s — the only horizon
where the signal carries real direction information, and ROC 0.659 is the best
honest direction figure in this project — **perfect foresight still loses 1.31
ticks per trade.** No model can rescue a negative oracle. At the horizons where
the move finally outruns the cost, direction skill has decayed to 0.544–0.563
against a 68.9–74.8% requirement.

Every variant loses, held out, with game-clustered 95% CIs entirely below zero:
model on every collapse −3.13 to −3.49t; confidence-gated −1.39 to −4.16t;
always buy −2.79 to −3.34t; coin flip −3.12 to −3.32t.

**Do not re-open this by lengthening the hold.** Stretching the horizon does not
trade the signal, it trades the game: entering at a *random* tight moment
captures 3.0 of the 4.0 ticks a collapse entry gets at +300s (Cliff's δ ≈
+0.12). Mean **signed** move decays monotonically from +0.21t at +10s to
+0.02t at +600s — i.e. indistinguishable from zero. The mid is efficient; the 35% up-rate is skew in
the distribution, not drift.

`research/makinen/direction_hold.py`, `results/makinen/direction_hold/`.

## D. Execution as a maker — closed by fills and exit costs

| # | assumption | fills (3 days) | P&L/fill |
|---|---|---|---|
| 1 | exit at mid, no screens | 4,428 | +$0.15 |
| 2 | + drop reprice cycles | 4,080 | +$0.07 |
| 3 | + pay the spread to exit | 4,080 | **−$0.40** |
| 4 | + flow must clear your size | **3** | −$0.33 |
| 5 | live paper maker, one session | **0 fills** | — |

By market type (realistic exit): spread −$0.99/fill, total −$0.93, moneyline
−$0.47. Hit rates 12–17%.

**Capital does not help.** Mean stake saturates at ~$15; $350 and $5,000 lose
**the same $75** over three days. The constraint is the size of available
fills, not the bankroll.

## E. Data and labelling defects found (not methods, but do not repeat)

1. **Series key omitted `asset_id`** — merged the two legs of a spread into one
   price series, manufacturing near-perfectly predictable moves. A one-line
   rule scored 76% on the corrupted data. (`FINDINGS_SERIES_KEY.md`)
2. **Raw `max |mid|` jump label** — 37–99% quote-vacuum artefacts.
3. **`future_move_H*` leaked past a literal-name feature filter** — produced
   ROC-AUC 1.0000.
4. **Event-study controls matched at the anchor** — selected control moments
   that were themselves depleted, hiding the effect entirely (δ ≤ 0.17 instead
   of ≤ 0.67).
5. **`asset_id` read as float64 by pyarrow** — lossy on 77-digit integers.
6. **pandas `groupby` silently drops NaN keys** — moneyline (null `line`)
   vanished from a figure.

---

## What is NOT ruled out

Four things remain genuinely open. None of them are order-book microstructure.

1. **Cross-market consistency.** Moneyline, run line and total on the same game
   must cohere. Any inconsistency is a fair value that needs no latency edge
   and no direction model. All three are recorded on one clock. **Never
   tested.** `crossmarket.py` exists but carries the series-key defect and
   needs fixing first.
2. **Signed trade flow** (aggressor side). Not currently recorded. The one
   plausible remaining source of direction, and free to start collecting.
3. **Hold to resolution.** Every negative above assumes a round trip. Entering
   and holding to settlement pays the spread once instead. But that is betting
   on the game outcome and needs a fair-value edge; the jump signal contributes
   nothing to it — C5 shows its direction content is gone within a minute.
4. **The esports dataset** — 1,039 live contracts over 21 consecutive days,
   only partly exploited. It fixes the sample-size problem that left the MLB
   architecture comparisons unresolved.

## The one-line summary

Everything tested that tries to profit from **knowing when or where the price
moves** has failed, including methods that achieved real statistical skill. The
untested directions are all about **knowing what a contract is worth**, not
when it will move.
