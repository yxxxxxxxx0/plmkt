# Ledger of what has been tested and found not tradable

2026-09-16. A record of closed doors, so none of them get re-opened by accident.
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

**The structural reason.** The model predicts jumps by detecting that liquidity
has already gone. "Liquidity has gone" and "trading is expensive" are the same
sentence. At the model's most confident predictions the median spread is **72
ticks** against a median move of **23 ticks**. The signal *is* a measurement of
untradeability.

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
3. **Hold to resolution.** Every negative above assumes a round trip, paying
   the spread twice — which is what makes `move < 2 × spread` fatal. Entering
   and holding to settlement pays the spread once. But that is betting on the
   game outcome and needs a fair-value edge; the jump signal contributes
   nothing to it.
4. **The esports dataset** — 1,039 live contracts over 21 consecutive days,
   only partly exploited. It fixes the sample-size problem that left the MLB
   architecture comparisons unresolved.

## The one-line summary

Everything tested that tries to profit from **knowing when or where the price
moves** has failed, including methods that achieved real statistical skill. The
untested directions are all about **knowing what a contract is worth**, not
when it will move.
