# Does LOB trajectory predict price jumps? — research summary (second pass)

2026-09-16. Supersedes `summary_firstpass_20260915.md`, whose headline numbers
were mostly an artefact. Six recorded sessions, 75 MLB prediction markets,
28.0M in-game 200ms grid rows, 4.68M prediction points at 1 Hz. Splits are by
whole market and chronological; held-out test markets are all from the latest
date.

**Headline.** The hypothesis is **supported, but far more weakly than the first
pass claimed, and for a different reason.** Most of what the first pass measured
was not price movement at all: **71–99% of labelled "jumps" are transient mid
excursions caused by one side of the book being briefly emptied**, not
repricings. On an artefact-free label the picture is honest and much smaller:
recent LOB history does beat the instantaneous book (PR-AUC lift 1.56× → 2.13×
over prevalence), the liquidity path matters at least as much as realised
volatility, no neural architecture beats gradient-boosted trees, and there is
real but modest information 5–30 seconds ahead.

---

## 0. What was wrong with the first pass

The label was

```
future_move(t,H) = max over t < u <= t+H of |mid_u - mid_t|
```

which counts a mid excursion however briefly it existed and however broken the
book was when it happened. Three findings, in the order they were made:

**(a) The label was largely a function of the spread.** Prevalence rises
monotonically with book width, and so does the median move:

| spread at t | share of rows | P(jump) J=0.02 H=30 | median 30s move |
|---|---|---|---|
| ≤ 1 tick | 25.7% | 0.423 | 0.010 |
| 2–3 | 10.6% | 0.654 | 0.050 |
| 10–20 | 8.3% | 0.910 | 0.065 |
| 20–30 | 3.1% | 0.981 | 0.100 |
| > 50 | 2.9% | 0.959 | 0.245 |

A one-feature LightGBM on **spread alone** scored PR-AUC 0.817 against a 0.616
floor — about two-thirds of the way to the full model's 0.929
(`audit_label_spread_J0.02_H30.csv`).

**(b) Plotting the price showed why.** `plots/examples_jumps_tight_J0.02_H30.png`
draws twelve randomly chosen labelled jumps from the test split. In almost every
panel the mid spike coincides *exactly* with the spread exploding from ~2 ticks
to 20–80 ticks, and the mid snaps straight back. That is a quote vacuum: the
best quote is cancelled, the next resting order is far away, the midpoint moves
mechanically, and it reverts when a quote returns. No trade need occur.
Restricting to tight books does not fix it, because the tightness test was
applied at `t` while the artefact happens later, inside the label window.

**(c) Quantified, the artefact is most of the label.** Each move was decomposed
(`audit_durability.py`) into `raw`, `clean` (the displaced price must be
observed while the book is still tight), `durable` (the displacement must
persist ≥3s) and `endpoint`. For tight books at J=0.02, H=30s:

| | prevalence | share of raw jumps surviving |
|---|---|---|
| raw | 0.479 | — |
| durable (≥3s) | 0.339 | 70.7% |
| still displaced at t+H | 0.205 | 42.8% |
| **clean (book tight at u)** | **0.118** | **24.5%** |

Across every (J, H) the **artefact share is 37%–99%**
(`jump_prevalence_clean.csv`), and it is *worst where the event is rarest and
most interesting*: tight books, H=10s, J=0.05 is **98.8% artefact**, leaving
2,421 real positives out of 1.85M rows.

This decomposition is a **strict lower bound on the problem**: the recomputation
runs on the 1 Hz points while the cached label was computed on the 200ms grid,
and cached ≥ recomputed in **100.0%** of 503,159 compared rows (median excess
0.010). The extra excursions the cached label catches are sub-second, so they
cannot be durable repricings either.

Everything below uses the corrected label — **clean AND durable** — with
prediction points restricted to books that are tight at `t`, so the model
cannot win by detecting "this book is already broken". Prevalence 0.1031.

---

## 1. Is there visible evidence of LOB deterioration before jumps?

**Yes, but it is small, and it is a standing level difference rather than a
build-up.** Against controls matched within market on spread bucket, price
bucket and game phase (`event_study_effects_J0.02_H30_tight_both.csv`):

| feature | δ at τ=−30s | δ at τ=−10s | δ at τ=0 |
|---|---|---|---|
| bid book slope | −0.149 | **−0.156** | −0.117 |
| ask dollars within 2 ticks | −0.130 | −0.134 | −0.084 |
| spread | +0.110 | +0.123 | +0.034 |
| bid dollars within 2 ticks | −0.103 | −0.102 | −0.045 |
| log bid depth | −0.100 | −0.099 | −0.058 |
| total depth | −0.072 | −0.071 | −0.041 |
| L1 imbalance | +0.045 | +0.047 | +0.058 |
| depth imbalance | −0.046 | −0.045 | −0.015 |

**Compare the first pass, which reported δ = +0.42 for spread and −0.41 for bid
slope — "large" by any convention.** On real repricings the same quantities are
−0.16 at best, i.e. small-to-negligible. The dramatic pre-jump collapse was the
artefact.

What survives is real but modest: `plots/event_slope_bid_J0.02_H30_tight_both.png`
shows jump windows sitting at a persistently lower bid slope (~115 vs ~175)
across the entire 60s with non-overlapping bootstrap bands on the median, then
rising in the last ~10s. Median curves are clearly separated while the
distributions overlap heavily — which is exactly why δ is small and why
per-event discrimination is weak.

## 2. Which variables change most before jumps?

**Bid-side book slope** (how fast depth builds away from the touch), then
**near-touch dollar depth** on both sides, then spread. Aggregate deep depth and
**depth imbalance** (δ ≈ −0.05) do essentially nothing — the classic
microstructure predictor is uninformative here, as in the first pass. Fragility
of the book, not direction of pressure.

## 3. Can a current-state model beat baseline?

**Yes, modestly.** Tight books, clean+durable label, held-out markets,
prevalence 0.1031:

| model | PR-AUC | lift | ROC-AUC | Brier skill |
|---|---|---|---|---|
| A prevalence floor | 0.1031 | 1.00× | 0.500 | 0 |
| **spread alone (control)** | **0.1148** | **1.11×** | **0.554** | 0.004 |
| B/C state only | 0.1607 | 1.56× | 0.648 | 0.025 |

The spread-only control is the number that matters: on the corrected label it
collapses to near-useless (1.11×), confirming that the first pass's spread-driven
skill was the artefact. Genuine current-state skill is 1.56×.

## 4. Does recent LOB history improve prediction?

**Yes — this remains the clearest positive result, and it is now attributable to
the order book rather than to volatility clustering.**

| model | PR-AUC | lift | ROC-AUC | P@1% | Brier skill |
|---|---|---|---|---|---|
| C state only | 0.1607 | 1.56× | 0.648 | 0.297 | 0.025 |
| D + price path only (`ret_`, `rv_`) | 0.1923 | 1.87× | 0.695 | 0.346 | 0.044 |
| **D + book path only** (`d_depth`, `d_spread`, `d_hhi`, `d_entropy`, `d_l1`) | **0.1945** | **1.89×** | 0.694 | 0.353 | 0.045 |
| D + book path + activity | 0.2042 | 1.98× | 0.706 | 0.350 | 0.051 |
| **D + all trajectory** | **0.2193** | **2.13×** | **0.723** | **0.393** | **0.061** |

Two things matter here:

1. **The book path alone (1.89×) matches the price path alone (1.87×)**, and the
   two are complementary (together 2.13×). This is the decomposition the first
   pass never ran. **On the contaminated label the ordering was reversed** —
   price/realised-volatility 0.855 vs book 0.829 — so the correction changes not
   just the magnitude but which family is doing the work.
2. Trajectory beats current state by **+36% relative PR-AUC**, larger in relative
   terms than the first pass's +6.7% on the contaminated label.

## 5. Does CNN + Transformer beat CNN and CNN + LSTM?

**No.** The first pass flagged that trees saw 800k rows and the deep models 80k;
here every model is trained on **120,000 rows**, with identical splits, label and
calibration, so the comparison is matched for the first time.

| model | PR-AUC | lift | ROC-AUC | params |
|---|---|---|---|---|
| CNN only | **0.2097** | 2.04× | 0.710 | 80,129 |
| D trees + trajectory (120k, matched) | 0.2058 | 2.00× | 0.711 | — |
| CNN + Transformer | 0.2014 | 1.94× | 0.693 | 105,985 |
| CNN + LSTM | 0.1982 | 1.92× | 0.701 | 65,985 |

All four sit within **0.0115 PR-AUC** of one another, and the two best — a plain
CNN and gradient-boosted trees on handcrafted features — differ by 0.004, which
is not a real difference at this sample size. Attention over the pre-jump
trajectory buys nothing over a temporal convolution, and recurrence buys nothing
either: **the Transformer is the second-worst model while carrying the most
parameters.** Nothing in this dataset justifies the larger architecture.

Note this is the matched comparison only. Given all 369k available training rows
the trees reach 0.2193 (2.13×), above every deep model at 120k — so on this
dataset the cheapest way to improve the result is more data, not more
architecture.

### Ablations

| variant | PR-AUC | vs its base |
|---|---|---|
| CNN + Transformer, full | 0.2014 | — |
| … without depth mask | 0.2108 | **+0.0094** |
| … without elapsed-time channel | 0.1904 | −0.0110 |
| CNN + LSTM, full | 0.1982 | — |
| … without elapsed-time channel | 0.2065 | +0.0083 |

Elapsed-time helps the Transformer and *hurts* the LSTM; the depth mask *helps*
when removed. Both effects are within the spread of the architectures
themselves, so the honest reading is that **none of these ablations resolves
above noise at this sample size** — not that masking is harmful.

## 6. Precision and false-alert rate at useful thresholds

Model D, tight books, prevalence 0.1031, on 81,526 held-out points:

| operating point | precision | recall | alert rate | false alerts / market-hour |
|---|---|---|---|---|
| F1-optimal (validation-chosen) | 0.190 | 0.614 | 33.3% | 969 |
| top 10% of probabilities | 0.268 | — | 10% | — |
| top 5% | 0.311 | — | 5% | — |
| top 1% | 0.393 | — | 1% | — |

Precision in the confident tail is **0.39 against a 0.10 base rate** — a real
4× enrichment, but a long way from the "1.000" the first pass reported on the
artefact label. Nearly a thousand false alerts per market-hour at the F1
threshold reflects a model alerting a third of the time.

Probabilities are isotonic-calibrated on validation only. Uncalibrated
(`class_weight="balanced"`) Brier skill is **−1.30**; calibration moves it to
**+0.061**. The ranking is unchanged — this only affects the probabilities.

## 7. How much warning time?

**Some, and more than the first pass found — but not at a usable alert rate.**

Two independent measurements agree:

**(a) Gapped labels.** Predicting the label as computed at `t+g` using only
features at `t`, so the move has not started (labels shifted within series on
the exact-timestamp 1 Hz grid):

| gap | prevalence | C state | D trajectory |
|---|---|---|---|
| 0s | 0.103 | 1.56× | 2.13× |
| 5s | 0.174 | 1.52× | **2.17×** |
| 10s | 0.191 | 1.38× | 1.78× |
| 30s | 0.178 | 1.37× | 1.54× |

Skill decays but **does not collapse** — there is genuine information about
moves 5–30s ahead. On the contaminated label this test was much less meaningful
because the "moves" were instantaneous artefacts.

**(b) First-alert lead times** (`lead_time_J0.02_H30_tight_both.csv`):

| alert rate | detection rate | median lead | lead < 2s | lead > 10s |
|---|---|---|---|---|
| 29.6% (F1 threshold, CNN) | 54.9% | 15.0s | 26% | 56% |
| **5.0%** | **13.9%** | **1.0s** | **54%** | **22%** |

Tighten to a 5% alert rate and the median lead is **1 second** — better than the
first pass's 0 seconds, and 22% of detections do arrive more than 10s ahead, but
the model's most confident calls still cluster at the move.

**Reconciling (a) and (b):** information about the next 5–30s genuinely exists in
the book, but it is diffuse. The model's *high-confidence* predictions are the
ones where a move is already beginning.

## 8. Is the result consistent across held-out markets?

**Yes.** Per-market on the 14 held-out games, each scored against **its own**
prevalence (`per_market_clean_J0.02_H30_both.csv`):

* PR-AUC lift: median **2.26×**, range **1.57× – 2.82×**
* **14 of 14** markets have lift > 1.2
* **14 of 14** have ROC-AUC > 0.6, minimum 0.666

Not driven by a handful of games.

## 9. Why this could still fail in live trading

1. **The economics are already known to be negative.** `FINDINGS_SERIES_KEY.md`:
   the average 5-second move is 0.44 ticks against a 3.99-tick round trip, and
   break-even accuracy is 100% or worse at every move size. Perfect foresight
   still loses ~3 ticks per opportunity. Prediction skill is not the binding
   constraint.
2. **A 0.39 precision at a 1% alert rate is a weak edge** to carry those costs.
3. **The label is still a mid move, not a tradeable event.** Even cleaned, a mid
   moves when a quote is cancelled. The `clean` filter requires a tight book at
   the moved price, which helps, but nothing here requires a *trade*.
4. **The 200ms grid discards 77–92% of updates** (median inter-update gap
   8–17ms), worst when the book is busiest. Live inference would see everything —
   a train/serve mismatch.
5. **Latency**: ~150ms round trip to the venue against ~2–8ms inference, so any
   edge decaying inside ~150ms is unreachable.
6. **Six sessions, 75 games, one sport, one month**, all MLB in-play Aug–Sep 2026.
7. **Clock skew between sessions**: `recv - ts` median +85…+143ms in August,
   −133…−1227ms in September. All work here uses exchange `ts` consistently.
8. **The tight-book restriction removes half the data.** Results apply to books
   with spread ≤ 2 ticks; wide books are excluded precisely because their mid is
   not meaningful, so this says nothing about them.

---

## Leakage controls

* Whole-market chronological splits; `splits.assign` asserts no market appears
  in two splits. Exact market lists printed in every run log.
* **Complementary-token check**: only 0.56% of same-`(market, market_type, line)`
  pairs have mids summing to ~1.0 (median sum 0.67). These are the two sides of a
  spread at one line — genuinely distinct contracts, not YES/NO duplicates. No
  double counting. (They *were* merged by the old series key; see
  `FINDINGS_SERIES_KEY.md`.)
* Normalisation, isotonic calibration and every threshold fitted on
  train/validation only; `Normaliser` refuses to fit on anything else.
* Feature filter bans future-derived columns by **prefix and suffix**, with an
  assertion — the first pass leaked `future_move_H30` past a literal-name filter
  and scored ROC-AUC 1.0000.
* Label windows exclude `t`; gapped labels join on exact `(sid, ts)` with no
  interpolation; windows spanning a gap are invalidated, never bridged.
* Post-resolution and pre-game rows excluded; truncated matches dropped.
* **Label shuffling within market**: PR-AUC 0.1701 vs the 0.1695 floor, ROC-AUC
  0.5016, Brier skill −0.0007. Collapses to chance exactly as required.
* `test_features.py` 6/6 groups pass (labels, masks, causality, gap handling).
* `test_durability.py` — new — verifies the decomposition on synthetic books:
  a 1-second spike is rejected by `durable`, a quote vacuum is rejected by
  `clean` but **not** by `durable` alone (which is why both filters are needed),
  and gaps are never bridged.

## Bottom line

1. **The first pass's headline was mostly an artefact.** 71–99% of its "jumps"
   were transient mid excursions during quote vacuums. Its large event-study
   effects (δ ≈ 0.42) shrink to δ ≈ 0.16 on real repricings, and its "perfect"
   top-1% precision falls to 0.39.
2. **The hypothesis still holds, at a smaller size.** Trajectory beats current
   state 2.13× vs 1.56× over prevalence — a larger *relative* gain than the first
   pass reported, and consistent across 14/14 held-out markets.
3. **The liquidity path is doing real work.** Book-path features alone match
   realised-volatility features alone; on the contaminated label that ordering
   was reversed. This is the part of the original hypothesis that is genuinely
   supported.
4. **No architecture earns its keep.** At a matched 120k-row budget a plain CNN
   (0.2097) and gradient-boosted trees (0.2058) are indistinguishable, and the
   Transformer is second-worst while carrying the most parameters. More training
   data helps (trees reach 0.2193 on all 369k rows); more architecture does not.
5. **Warning time is real but diffuse**: skill persists at 5–30s gaps, yet the
   confident alerts still arrive at the move.

Useful as a fragility/risk filter. Still not a jump alarm, and still not
tradeable on these costs.
