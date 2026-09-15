# Does LOB trajectory predict price jumps? — research summary

2026-09-15. Six recorded sessions, 75 MLB prediction markets, 28.0M in-game
200ms grid rows, 4.69M prediction points at 1 Hz. Splits are by whole market
and chronological; the held-out test markets are all from the latest date.

**Headline.** Temporal history does help, consistently and by a clear margin,
and liquidity depletion before jumps is large and real. But the effect is
almost entirely a persistent *regime* difference rather than a pre-jump
*dynamic*, the best model is gradient-boosted trees on handcrafted features
rather than any neural sequence model, and at a usable alert rate the warning
lead time is **zero seconds**. The model tells you which books are fragile,
not when they are about to break.

---

## 1. Is there visible evidence of LOB deterioration before jumps?

**Yes, large — but it is a level difference, not a trend.**

Effect sizes (Cliff's δ, jump windows vs controls matched within market on
spread bucket, price bucket and game phase; |δ| > 0.33 is large), J=0.02,
H=30s, tight books only:

| feature | τ = −30s | τ = −10s | τ = 0 |
|---|---|---|---|
| spread | **+0.42** | **+0.45** | +0.18 |
| bid book slope | **−0.41** | **−0.42** | −0.33 |
| ask dollars within 2 ticks | **−0.37** | **−0.39** | −0.19 |
| bid dollars within 2 ticks | **−0.35** | **−0.38** | −0.17 |
| ask book slope | −0.35 | **−0.37** | −0.31 |
| total depth | −0.25 | −0.26 | −0.16 |
| ask entropy | −0.24 | −0.23 | −0.29 |
| ask concentration (HHI) | +0.22 | +0.22 | +0.27 |
| **depth imbalance** | **+0.03** | **+0.03** | **+0.05** |

The plots (`plots/event_*.png`) are unambiguous about the shape: the jump and
control curves are **separated by a large, roughly constant gap across the
entire 60-second window**, with only a modest additional move in the final
~10 seconds. Near-touch bid dollars sit about 2 log units (~7x) lower in jump
windows at τ = −60s, and stay there.

So the honest statement is not "the book deteriorates before a jump". It is
**"books that are already thin, wide, flat and concentrated are the ones that
jump"**, plus a small genuine last-10-second dynamic.

## 2. Which variables change most?

Spread and book *slope* (how fast depth builds away from the touch) dominate,
followed by near-touch dollar depth and then concentration/entropy. Deeper
aggregate depth matters less than near-touch depth.

**Depth imbalance — the classic microstructure predictor — does essentially
nothing here (δ ≈ 0.03).** Direction of pressure is uninformative; the
*fragility* of the book is what matters. That is worth knowing because
imbalance is the first feature most order-book studies reach for.

## 3. Can a current-state model beat baseline?

**Yes, comfortably.** J=0.02, H=30s, held-out markets:

| model | PR-AUC | ROC-AUC | precision | recall | Brier skill |
|---|---|---|---|---|---|
| A prevalence floor | 0.6158 | 0.500 | 0.616 | 1.000 | −0.027 |
| B logistic, state only | 0.8462 | 0.781 | 0.722 | 0.903 | 0.212 |
| C trees, state only | 0.8713 | 0.815 | 0.754 | 0.889 | 0.295 |

## 4. Does recent history improve prediction?

**Yes — this is the clearest positive result in the study.** Same trees, same
data, same split; the only difference is 42 trailing-change features:

| model | PR-AUC | ROC-AUC | precision | Brier skill |
|---|---|---|---|---|
| C trees, state only | 0.8713 | 0.8151 | 0.754 | 0.295 |
| **D trees + trajectory** | **0.9299** | **0.8949** | **0.839** | **0.468** |

Consistent across **all 12 (J, H) settings**, and the gain grows as the task
gets harder:

| gain in PR-AUC | H=10s | H=30s | H=60s |
|---|---|---|---|
| J=0.01 | +10.1% | +5.1% | +2.9% |
| J=0.02 | +13.0% | +6.7% | +4.1% |
| J=0.03 | +15.5% | +8.2% | +4.9% |
| **J=0.05** | **+20.2%** | +11.2% | +7.0% |

ROC-AUC moves 0.80–0.82 → 0.88–0.90 uniformly. The hypothesis is supported.

Caveat on interpretation: given §1, much of what the trajectory features add
is probably a better estimate of the *current regime* (a 30-second average is
a less noisy read on "is this book thin" than one snapshot) rather than
genuine anticipation of a specific event.

## 5. Does CNN + Transformer beat CNN and CNN + LSTM?

**No. There is no measurable difference between the three, and none of them
beats the gradient-boosted trees.**

| model | PR-AUC | ROC-AUC | params |
|---|---|---|---|
| **D trees + trajectory** | **0.9299** | **0.8949** | — |
| CNN + LSTM | 0.9229 | 0.8840 | 65,985 |
| CNN only | 0.9220 | 0.8849 | 80,129 |
| CNN + Transformer | 0.9219 | 0.8833 | 105,985 |

The three deep models are within **0.001 PR-AUC** of one another. Attention
over the pre-jump trajectory buys nothing over a plain temporal convolution,
and recurrence buys nothing either.

**Fairness caveat, stated because it cuts against this conclusion:** the deep
models were trained on 80k samples against the trees' 800k, for IO reasons.
The deep-vs-trees comparison is therefore *not* matched and the deep models
are handicapped. The CNN vs LSTM vs Transformer comparison **is** matched —
identical budget, data, split and encoder — and that is the comparison showing
no difference.

### Ablations

| variant | PR-AUC | vs base |
|---|---|---|
| CNN+Transformer, full | 0.9219 | — |
| without depth mask | 0.9208 | −0.0011 |
| without elapsed-time channel | 0.9185 | −0.0034 |
| CNN+LSTM without elapsed-time | 0.9188 | −0.0041 |

Staleness/elapsed-time is worth a little; the explicit depth mask almost
nothing at this horizon.

## 6. Precision and false-alert rate at useful thresholds

J=0.02, H=30s, model D, prevalence 0.616:

| operating point | precision | alert rate | false alerts / market-hour |
|---|---|---|---|
| F1-optimal threshold | 0.839 | 65.0% | 376 |
| top 10% of probabilities | 0.991 | 10% | — |
| top 5% | 0.997 | 5% | — |
| top 1% | **1.000** | 1% | — |

Precision in the confident tail is essentially perfect. But read that against
the 61.6% base rate: guessing "jump" blindly is already right 62% of the time,
and 376 false alerts per market-hour at the F1 threshold reflects a model that
is alerting two-thirds of the time.

## 7. How much warning time?

**Effectively none at a usable alert rate.**

| alert rate | detection rate | median lead | lead < 2s |
|---|---|---|---|
| 67% (F1 threshold) | 89% | 42.0s | 12% |
| **5%** | **8.1%** | **0.0s** | **90%** |

The 42-second figure is an artefact: a model alerting two-thirds of the time
will be "already alerting" before almost any event. Tighten to a 5% alert rate
and the median lead collapses to **zero seconds**, with 90% of detections
arriving within 2 seconds of the jump.

**When the model is confident, it is confident at the instant the move
begins.** It is reacting, not anticipating. This is entirely consistent with
§1: the signal is a standing regime property, so it cannot time anything.

## 8. Is the result consistent across held-out markets?

**Yes.** Per-market, on the 14 held-out games (CNN+LSTM):

* PR-AUC lift over each market's own prevalence: median **1.44**, range
  **1.33 – 1.98**
* **14 of 14** markets have lift > 1.3
* **14 of 14** have ROC-AUC > 0.85, minimum 0.859

Not driven by a handful of games. See `per_market_consistency_J0.02_H30.csv`.

## 9. Why this could still fail in live trading

1. **Zero lead time.** §7 is disqualifying on its own for any strategy that
   needs to act before a move.
2. **The economics are already known to be negative.** Separate analysis in
   `../../FINDINGS_SERIES_KEY.md`: the average 5-second move is 0.44 ticks
   against a 3.99-tick round trip, and break-even accuracy is 100% or worse at
   *every* move size because a large move blows the exit spread out to 20–60
   ticks. Perfect foresight still loses ~3 ticks per opportunity. Prediction
   skill is not the binding constraint.
3. **The label is a mid-price move, not a tradeable event.** A mid moves when
   a quote is *cancelled*, with no trade and no liquidity to hit. Much of what
   is being predicted may be un-executable by construction.
4. **The 200ms grid discards 77–92% of updates** (median inter-update gap is
   8–17ms). The models see roughly one message in ten, and the discarding is
   worst when the book is busiest. Live inference would see everything — a
   train/serve mismatch, and an argument that these numbers understate the
   book's information content.
5. **Prevalence is regime-dominated.** At >30-tick spreads P(jump) ≈ 0.96, so
   a pooled model partly learns "is this book illiquid". Tight-book numbers
   are the meaningful ones and are weaker.
6. **Six sessions, 75 games, one sport, one month.** All from MLB in-play
   markets in Aug–Sep 2026; nothing here tests another sport, another season,
   or a different liquidity environment.
7. **Latency.** Measured round trip to the venue is ~150ms against ~2–8ms of
   inference, so any edge that decays inside ~150ms is unreachable regardless.
8. **Clock skew between sessions.** `recv - ts` has median +85 to +143ms on
   the August sessions and −133 to −1227ms on the September ones, so the two
   clocks are not on a common basis across the dataset. All work here uses
   exchange `ts` consistently, but anything relying on our receive time would
   be unsafe.

---

## Leakage controls that were run

* Whole-market chronological splits; verified no market in two splits.
* Both tokens of a market share a slug, so complementary legs cannot straddle
  a split boundary.
* Normalisation and every threshold fitted on train/validation only.
* Label window starts at t+1, so a move underway cannot label itself; tested.
* Non-contiguous forward windows and history windows are invalidated, never
  bridged across a gap.
* Post-resolution and pre-game rows excluded; truncated matches dropped.
* **Label shuffling within market**: PR-AUC 0.5620 vs the 0.5503 prevalence
  floor, ROC-AUC 0.5163. Collapses to chance as required. The small residual
  is expected — shuffling within market preserves per-market prevalence.

### One leak was found and fixed during this study

The first baseline run returned **ROC-AUC 1.0000**. The feature filter
excluded the literal column name `future_move` while the cache actually stored
`future_move_H10/H30/H60`, so every model was handed a continuous version of
its own label. The filter is now a prefix match with an assertion
(`baselines.py`), and a perfect score is treated as evidence of a bug. Reported
here because it is the single most likely way a result like this goes wrong.

## Bottom line

The hypothesis as stated is **supported**: trajectory carries materially more
information than the instantaneous book, consistently across thresholds,
horizons and held-out markets.

Three findings qualify it heavily:

1. The mechanism is **fragility, not deterioration** — a standing property of
   thin, wide, flat, concentrated books, visible a full minute ahead and
   roughly constant, not a build-up.
2. **Handcrafted trees beat every neural sequence model**, and CNN, LSTM and
   Transformer are indistinguishable from one another. Nothing in this dataset
   justifies the larger architecture.
3. At a usable alert rate there is **no warning time at all**.

Useful as a regime/risk filter — "this book is fragile, do not rest size
here". Not useful as a jump alarm.
