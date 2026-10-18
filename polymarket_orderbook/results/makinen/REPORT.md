# Can the order book predict a Lee–Mykland jump before it happens?

Adaptation of Mäkinen, Kanniainen, Gabbouj & Iosifidis (2018), *Forecasting of
Jump Arrivals in Stock Prices*, to Polymarket in-play MLB order books.
2026-09-16. Code in `research/makinen/`, all artefacts under `results/makinen/`.

Every conclusion separates **OBSERVATION** (what was measured) from
**INTERPRETATION** (what it might mean). No causal claim is made anywhere.

---

## 1. Dataset

`data/jump/feat_books_*_trimmed.parquet` + `lob_books_*_trimmed.npy` — the
project's existing 200 ms in-game grid with **10 depth levels per side**,
opened read-only. Chosen over the raw `top_of_book_*.csv` because that feed
carries only level 1, and this study needs the depth the paper's features are
built on; the grid also covers six sessions rather than four.

| | |
|---|---|
| sessions | 6 (2026-08-28 … 2026-09-13 US dates) |
| contracts (series) | 601 |
| 1-minute bars | 94,358 |
| HKT dates | 2026-08-29 … 2026-09-14 |
| depth levels | 10 per side |
| snapshots | event-driven underneath, resampled to a 200 ms grid, then 1 minute |
| price | `mid = (best_bid + best_ask)/2`, verified identical to the feed's own `midpoint` in 100.0% of raw rows |

Bars are the **last valid grid row at or before each minute close**, per
contract; minutes with no valid row are left missing and the series is broken
there. Series identity is `asset_id` — the first three fields of the series key
collide across the two legs of a spread (`FINDINGS_SERIES_KEY.md`).

Full audit: `results/lee_mykland/data_inspection.json`, `panel_build_stats.csv`.

## 2. Lee–Mykland implementation

Full derivation in **`results/makinen/lee_mykland_method.md`**. Summary:
arithmetic 1-minute returns; bipower local volatility
$\hat\sigma^2(i)=\frac{1}{K-2}\sum_{j=i-K+2}^{i-1}|r_j||r_{j-1}|$ over a
**trailing** window; statistic $L(i)=r_i/\hat\sigma(i)$; extreme-value threshold
$C_n+S_n\beta^*$ with $c=\sqrt{2/\pi}$, $\alpha=0.01$, giving **6.7021** on
$n=81{,}458$ tested minutes. One implementation exists in the repository; this
study imports it.

Two documented departures, both forced by the data and both measured:
**K = 30, not 600** (contracts live a median ~161 in-game minutes, so K=600
leaves nothing testable; K chosen on a volatility-tracking sweep), and
**arithmetic returns, not log** (these are probabilities bounded in (0,1); in
log space one half-tick is 0.0055 at p=0.90 but 0.41 at p=0.01). α was never
tuned.

## 3. Jump visualisation and sanity check

`results/makinen/jumps/` — full series with ▲/▼ markers, the $L(i)$ series with
±threshold, 10 zoomed examples, and jump frequency by time-of-day and by day.

**OBSERVATION.** 1,516 jumps on 81,458 tested minutes = **1.86%**. Verdict
**PLAUSIBLE**; the pipeline proceeded automatically. An earlier run under the
paper's own settings gave 9.6%, which failed this check and is why the
parameters were re-derived.

## 4. Detected jump statistics

| | |
|---|---|
| total minute bars | 94,358 |
| tested observations | 81,458 |
| warm-up / untestable | 12,900 |
| **total jumps** | **1,516** (873 positive, 643 negative) |
| % of tested minutes | 1.86% |
| jumps per day | 252.7 |
| median abs jump return | **0.180** probability points |
| mean / p90 abs jump return | 0.199 / 0.345 |

**INTERPRETATION.** A median jump of 18 probability points is a large,
economically real move, not tick noise. The 20 strongest (`top20_jumps.csv`)
are dominated by run-scoring moments and contract resolutions.

## 5. Available LOB features

**`results/makinen/features_available.md`** carries the full table. 69 features
per minute. Headline: this is a **snapshot feed, not an order-event feed**, so
every order-flow intensity in the paper — new-order rates, cancellation rates,
market-order intensities, signed order-flow imbalance — is **unavailable and
was not approximated**. Depth decreasing between two snapshots cannot separate
a cancel from a trade.

**INTERPRETATION.** This is a strictly weaker input than the paper had. If the
models underperform its reported numbers, missing order flow is a live
candidate explanation and should not be read as the method failing.

## 6. Prediction target

$y_t = 1$ if a Lee–Mykland jump occurs in $(t, t+H]$ on the same contract.
$X_t$ covers minutes $[t-119, t]$. Main experiment $H=1$, lookback $L=120$ as
in the paper. 18,579 windows, **403 positives (2.169%)**.

## 7. Leakage controls

`research/makinen/test_leakage.py` — **all pass**:

* input window ends strictly before the target minute (0/4000 violations)
* window is 120 contiguous minutes of one contract (0/4000)
* target interval contiguous and length H (0/4000)
* sessions disjoint across splits; **no contract in two splits** (292/54/94)
* splits chronologically ordered
* train features standardised (max|mean| 0.0021), **test not re-centred**
  (max|mean| 1.4950) — proving scalers were fitted on train only
* label reproduces the LM jump flag on the target minute (0/1500 mismatches)

Sample alignment is printed, e.g. `input 08-29 08:19 → 10:18 | predict (10:18,
10:19] | label=1 jump at 10:19`.

## 8. Split

Whole sessions, chronological. A session is one night and its contracts exist
nowhere else, so a session boundary is also a contract boundary.

| split | sessions | windows | positives |
|---|---|---|---|
| train | 08-28, 08-30, 09-10, 09-11 | 12,027 | 241 (2.00%) |
| val | 09-12 | 2,323 | 53 (2.28%) |
| test | 09-13 | 4,229 | **109 (2.58%)** |

**This is the study's binding limitation.** Six sessions give a 4/1/1 split and
the test set is a single night — 14 games, 109 positives. All uncertainty below
follows from that.

## 9. Models

Models 0–8 as specified, identical splits and target, PyTorch, CPU, seed 0,
early stopping on **validation** PR-AUC, threshold frozen from **validation**
F1, natural prevalence preserved in val and test (no oversampling). The
attention in Model 8 is **feature attention** — one weight per feature, applied
across all time steps of that sample — implemented in
`models.FeatureAttention` and saved to `attention_weights.csv`.

## 10. Main results

Test set, 109 positives, prevalence 2.577%. CI = 95% cluster bootstrap
resampling the 14 **games** (a day-level bootstrap is undefined here — the test
split is one date).

| model | Prec | Rec | F1 | PR-AUC | 95% CI | lift | ROC-AUC | FP/hr |
|---|---|---|---|---|---|---|---|---|
| 0 prevalence | — | — | — | 0.0258 | [0.020, 0.035] | 1.00 | 0.500 | 0 |
| 1 time-of-day only | 0.000 | 0.000 | 0.000 | 0.0247 | [0.018, 0.050] | **0.96** | 0.484 | 0 |
| 2 logistic (snapshot) | 0.159 | 0.156 | 0.157 | 0.0904 | [0.055, 0.160] | 3.51 | 0.703 | 1.28 |
| 3 MLP snapshot | 0.113 | 0.248 | 0.156 | 0.0965 | [0.055, 0.192] | 3.74 | 0.676 | 2.99 |
| 4 MLP history | 0.158 | 0.028 | 0.047 | 0.0448 | [0.036, 0.078] | 1.74 | 0.618 | 0.23 |
| 5 CNN | 0.134 | 0.248 | 0.174 | 0.1351 | [0.078, 0.222] | 5.24 | **0.746** | 2.47 |
| 6 LSTM | 0.247 | 0.174 | 0.204 | 0.1599 | [0.111, 0.261] | 6.20 | 0.738 | 0.82 |
| 7 CNN-LSTM | **0.309** | 0.156 | 0.207 | 0.1326 | [0.084, 0.199] | 5.14 | 0.693 | **0.54** |
| **8 CNN-LSTM-Attention** | 0.267 | **0.257** | **0.262** | **0.1806** | [0.116, 0.252] | **7.01** | 0.705 | 1.09 |

### The research questions

**Q1 — Does LOB information beat the time-of-day baseline? YES, and it is the
only firmly resolved result.**
*OBSERVATION.* Logistic on the book beats time-only by +0.0685 PR-AUC, CI
**[+0.021, +0.133]**, P(A beats B) = **0.999**. Time-of-day alone scores lift
**0.96 — below prevalence**, i.e. useless.
*INTERPRETATION.* This is the sharpest contrast with the paper, which found
time-of-day predicts many equity jumps. That effect is an artefact of exchange
hours; these contracts trade continuously and jumps follow the *game* clock,
not the wall clock. Any equity-jump study's time baseline should not be assumed
to transfer.

**Q2 — Does history help over a snapshot? Not when flattened.**
*OBSERVATION.* MLP history is **worse** than MLP snapshot: −0.0560,
CI [−0.142, +0.002], P = 0.029 — the one comparison besides Q1 that nearly
resolves.
*INTERPRETATION.* 120×69 = 8,280 flattened inputs against 241 training
positives is an unfavourable ratio; the architecture, not the information, is
the likely problem. Q3 supports that.

**Q3 — Does explicitly modelling temporal structure help? Point estimates say
yes; the test set cannot confirm it.**
*OBSERVATION.* CNN (0.135) and LSTM (0.160) both exceed MLP history (0.045)
and MLP snapshot (0.097), but CNN vs MLP snapshot is +0.035, CI [−0.068,
+0.130], **P = 0.761** — unresolved.
*INTERPRETATION.* Consistent with history helping only when a model can exploit
temporal ordering, but not demonstrated at this sample size.

**Q4 — CNN vs LSTM vs CNN-LSTM? Indistinguishable.**
*OBSERVATION.* LSTM vs CNN: +0.031, CI [−0.071, +0.139], P = 0.729. CNN-LSTM
(0.133) is not above either individually.

**Q5 — Does feature attention add value? Best point estimate, not a resolved
difference.**
*OBSERVATION.* Attention has the top PR-AUC (0.1806, lift 7.0×) and top F1
(0.262). But against CNN-LSTM: +0.048, CI **[−0.039, +0.135]**, P = 0.854;
against LSTM: +0.012, CI [−0.100, +0.102], **P = 0.610** — a coin flip.
*INTERPRETATION.* The paper's architecture is the best on this data by point
estimate, and its CI **[0.116, 0.252] excludes the prevalence floor (0.026)**,
so it is clearly better than baseline. It is **not** shown to be better than a
plain LSTM. Following the brief's instruction, that is reported as unresolved
rather than claimed as a win.

## 11. Precision–recall analysis

`pr_curves.png`. At the frozen validation threshold the attention model catches
**28 of 109** jumps at **26.7% precision** and **1.09 false alerts per contract-
hour**. CNN-LSTM trades recall for precision (30.9% precision, 17 caught, 0.54
FP/hr).

**INTERPRETATION.** A 27% precision alarm at 2.6% prevalence is a real
enrichment (~10×) but is not, on its own, an actionable trading signal —
especially against the ~4-tick round-trip cost measured elsewhere in this repo.

## 12. Time-of-day baseline comparison

Covered in Q1. `jumps/jump_frequency.png` shows jumps spread across the live
window with no exchange-open spike.

## 13. MLP vs CNN vs LSTM interpretation

Covered in Q2–Q4. Summary: architectures that respect time order (CNN, LSTM,
CNN-LSTM, attention: 0.133–0.181) sit above those that do not (logistic 0.090,
MLP snapshot 0.097, MLP history 0.045), but only the flattened-history
degradation resolves statistically.

## 14. CNN-LSTM-Attention result

Best point estimate on every headline metric (PR-AUC 0.1806, F1 0.262, lift
7.0×), clearly above baseline, not separable from LSTM. 

## 15. Lead-time experiment

Same architecture, same split, $y_t(H)=1$ if ≥1 jump in $(t,t+H]$:

| horizon | prevalence | Prec | Rec | F1 | PR-AUC | **lift** | FP/hr |
|---|---|---|---|---|---|---|---|
| **1 min** | 2.58% | 0.267 | 0.257 | 0.262 | 0.181 | **7.01** | 1.09 |
| 2 min | 4.53% | 0.306 | 0.181 | 0.227 | 0.167 | 3.69 | 1.11 |
| 5 min | 9.89% | 0.174 | 0.510 | 0.259 | 0.266 | 2.69 | 14.4 |
| 10 min | 17.60% | 0.248 | 0.639 | 0.357 | 0.381 | 2.16 | 20.4 |

*OBSERVATION.* Raw PR-AUC rises with the horizon, but only because prevalence
rises. **Lift falls monotonically 7.01 → 3.69 → 2.69 → 2.16.**
*INTERPRETATION.* The book's information about jumps is concentrated in the
immediately following minute and decays quickly. This is evidence against a
usable multi-minute early-warning signal.

## 16. Attention analysis

Mean feature-attention weights over the test set (uniform = 1.0), top 10:

| feature | weight |
|---|---|
| `rv_5` (5-min realised volatility) | **3.15** |
| `d_log_ask_depth` | 2.98 |
| `bid_gap_1` | 2.44 |
| `l1_ask_usd` | 2.42 |
| `hhi_ask` | 2.16 |
| `ask_dist_1` | 2.09 |
| `rv_15` | 1.59 |
| `imbalance_l1` | 1.56 |
| `ask_dist_4` | 1.44 |
| `log_bid_usd_within_2t` | 1.38 |

*OBSERVATION.* The top weight is recent realised volatility; the rest are
ask-side depth dynamics and near-touch book geometry. Depth imbalance —
microstructure's default feature — ranks 8th.
*INTERPRETATION.* Partly volatility clustering (a known statistical effect, not
an order-book insight) and partly genuine book-shape information. The two are
not separated here; the earlier study in this repo
(`results/jump_prediction/summary.md`) ran that decomposition and found both
contribute.

## 17. Failure cases

`prediction_examples/`. The decisive one is `true_positive_1.png`: predicted
probability is **elevated and oscillating violently for the entire 60 minutes
before the jump** (0.05 → 0.8 → 0.3 → 0.9), not rising cleanly into it. Only a
very high frozen threshold (0.956) keeps the alert rate down.

*INTERPRETATION.* The model appears to identify a **jumpy regime** rather than
time a specific event. This agrees independently with the earlier study, which
concluded the signal is book *fragility* (a standing property) rather than a
pre-jump dynamic. It also explains the lead-time decay in §15.

## 18. Limitations

1. **Test set is one session** — 14 games, 109 positives. This is why almost
   every architecture comparison is unresolved.
2. **No order-flow features** (§5); a whole half of the paper's inputs is absent.
3. **Six sessions, one sport, one month.** No test of another sport or regime.
4. K=30 and arithmetic returns depart from the paper; justified and measured,
   but they mean this is an adaptation, not a reproduction.
5. Single seed per model. Seed variance is not quantified and could plausibly
   cover the unresolved gaps in §10.
6. Labels inherit the mid-price's vulnerability to quote vacuums; the in-game
   trim and the non-zero-product guard reduce but do not eliminate this.

## 19. What is genuinely supported by the data

**Supported:**
* Order-book information predicts next-minute Lee–Mykland jumps far better than
  chance — lift up to 7×, CI excluding the prevalence floor.
* It **decisively beats the time-of-day baseline** (P = 0.999). The paper's
  time-of-day effect does not transfer to continuously traded contracts.
* Flattening history into an MLP **hurts** (P = 0.029).
* Predictive information **decays quickly** with horizon (lift 7.0 → 2.2).

**Not supported:**
* That attention beats a plain LSTM (P = 0.610).
* That any sequence architecture beats a snapshot model (P = 0.761).
* That the probability rises *before* a jump in a usable way (§17).

## 20. Recommended next experiment

The binding constraint is **109 test positives**, not the architecture. In
order of expected value:

1. **Record more sessions.** Everything in §10 is unresolved for want of test
   data. The nightly recorder is already scheduled; ~10 more slates would allow
   a genuine multi-session test split.
2. **Repeat with multiple seeds** and report seed variance alongside the
   bootstrap CI — cheap, and it would tell you whether the §10 ordering is even
   stable.
3. **Separate volatility clustering from book information** by ablating `rv_*`,
   since it carries the top attention weight.
4. **Reframe the target.** §15 and §17 both suggest the book identifies a
   regime rather than an event. Predicting *jump intensity over the next 10
   minutes* may be the question this data can actually answer.
