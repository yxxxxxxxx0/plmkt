# Does the order book's recent trajectory predict a jump?

A self-contained experiment testing one hypothesis:

> `P(jump soon | recent LOB trajectory)` carries materially more predictive
> information than `P(jump soon | current LOB state alone)`.

Nothing here writes to the raw recordings. The raw `.jsonl.xz` files are
opened read-only; every generated artefact lands in `cache/` (datasets) or
`../../results/jump_prediction/` (reports, metrics, plots).

## Layout

| file | role |
|---|---|
| `config.yaml` | every parameter of the experiment in one place |
| `inspect_data.py` | phase 1: data-quality report over raw and gridded data |
| `features.py` | LOB decoding, state features, trajectory features, labels |
| `build_dataset.py` | phase 2: prediction-point cache + prevalence table |
| `test_features.py` | correctness tests for labels, masks and causality |
| `splits.py` | leakage-safe market-level chronological splitting |
| `event_study.py` | phase 3: pre-jump book behaviour vs matched controls |
| `baselines.py` | phase 4: prevalence / logreg / trees / trees+trajectory |
| `models.py` | phase 5: CNN, CNN+LSTM, CNN+Transformer (shared encoder) |
| `train.py` | phase 5 runner, plus ablations |
| `evaluate.py` | rare-event metrics; the only place a threshold is chosen |
| `leadtime.py` | phase 6: how much warning a correct prediction gives |

### Second pass: the label audit and the corrected experiment

The first pass concluded that the hypothesis was supported. Re-auditing the
label showed that most of that result was an artefact, so these files were
added. **They supersede the first pass; read `summary.md` for what changed.**

| file | role |
|---|---|
| `audit_label.py` | is the label measuring price movement, or the spread? |
| `plot_examples.py` | draws labelled jumps — this is what exposed the artefact |
| `audit_durability.py` | decomposes each move into raw / clean / durable / endpoint and caches `moves_H*.parquet` |
| `audit_mechanism.py` | which feature family carries the gain; anticipation gaps |
| `clean_prevalence.py` | prevalence and artefact share for every (J, H) |
| `run_clean.py` | the corrected headline experiment |
| `per_market.py` | per-market consistency on held-out games |

`train.py`, `event_study.py` and `leadtime.py` all take `--label
{raw,clean,durable,both}`; `raw` reproduces the first pass, `both` is the
corrected label. Every output is tagged with the label, so a corrected run
never silently overwrites an original one.

## Order to run

```bash
python inspect_data.py                       # data quality report
python test_features.py                      # must be 6/6 before anything else
python build_dataset.py                      # ~10 min, writes cache/

# --- label audit: do this before believing any model number ---
python audit_label.py --J 0.02 --H 30        # spread-only control
python plot_examples.py --J 0.02 --H 30      # look at the actual price
python audit_durability.py --H 10            # writes cache/moves_H10.parquet
python audit_durability.py --H 30
python audit_durability.py --H 60
python clean_prevalence.py

# --- corrected experiment ---
python event_study.py --J 0.02 --H 30 --tight-only --label both
python run_clean.py --J 0.02 --H 30 --label both --shuffle-control
python run_clean.py --J 0.02 --H 30 --label both --max-train 120000   # matched
python per_market.py --J 0.02 --H 30 --label both
python train.py --smoke                      # plumbing check
python train.py --J 0.02 --H 30 --tight-only --label both --ablations
python leadtime.py --J 0.02 --H 30 --tight-only --label both
```

## Data

The source is the recorder's event stream: one JSON record per websocket book
update, with `asset_id`, `ts` (exchange time), and top-of-book depth. Six
sessions survive quality filtering, 28.0M in-game grid rows across 75 baseball
games.

**The modelling dataset is built on the existing 200ms grid** produced by
`../../jump_data.py`, which is a backward-only as-of resampling: each slot
carries the most recent book at or before it, never the next one, and silences
beyond 60s are left as genuine gaps rather than filled. That pipeline has been
audited by `../../audit_datasets.py` (all six sessions pass) after a serious
defect was found and fixed — see `../../FINDINGS_SERIES_KEY.md`.

### The resolution caveat, stated up front

`inspect_data.py` measures the interval between consecutive updates of the
same token. The median is **8–17 ms**, and **77–92% of updates arrive faster
than one 200ms grid slot**. The grid therefore discards the large majority of
events, and it discards them preferentially when the book is busiest.

Consequences, honoured throughout:

* horizons below 10s are **not** evaluated on the grid; claiming H=1s on data
  whose median book is seconds old would be dishonest
* any negative result here bounds "a model fed a 200ms view of the book", not
  "the book contains no information"
* an event-level track is specified in `config.yaml` for H = 1/5/10s and is
  the natural next step

### Representation

Per level, per side: **price distance from the current mid in ticks** (not
absolute price), **log1p resting dollars**, and an explicit **existence
mask**. The mask is not cosmetic — the stored tensor encodes a missing level
as price 0, which after mid-centring becomes a large negative distance that a
model would otherwise read as a real quote far from the touch. Only 66–96% of
books carry all ten levels, so this matters on a third of the data.

Mid is kept as a separate scalar, because mid-centring is exactly translation
invariance and erases the price level.

### Irregular sampling and `delta_t`

On a uniform 200ms grid the spacing between frames is constant, so a literal
`delta_t` channel would carry nothing. The quantity that actually varies is
**how stale each frame's book is** (`book_age_ms`), and that is what is fed to
the models as the elapsed-time channel. The `--ablations` run removes it. The
transformer's time encoding uses the real seconds-before-now offset rather
than the token index, so the same code stays correct on the irregular
event-level track.

## Labels

```
future_move(t, H) = max over t < u <= t+H of |mid_u - mid_t|
jump = future_move >= J
```

The window **excludes t**, so a move already underway cannot label itself.
A row is valid only if its entire forward window is present and contiguous in
time. `J ∈ {0.01, 0.02, 0.03, 0.05}` on the 0–1 probability scale;
`H ∈ {10, 30, 60}s`. Three-class up/flat/down labels are also stored, with
binary as the primary task.

**Prevalence is high and strongly regime-dependent.** Pooled, it runs 19–78%
depending on (J, H). Broken out by spread on one session at J=0.05, H=10s:

| spread | P(jump) |
|---|---|
| 0–2 ticks | 0.103 |
| 5–10 ticks | 0.158 |
| >30 ticks | **0.961** |

In a 60-tick-spread book essentially every moment qualifies — that is an
illiquid quote wobbling, not a repricing. Results are therefore reported both
pooled and restricted to tight books (`--tight-only`), and the tight-book
numbers are the economically meaningful ones.

## Splitting

Whole **markets** (one game), assigned chronologically — earliest dates to
train, latest to test. Never a random split over rows: at 1 Hz with a 30s
horizon, consecutive prediction points share 29 of 30 seconds of their label
window, so a random split would place near-duplicates of the same moment on
both sides and report a meaningless number.

Both tokens of a market share its slug and therefore always land in the same
split, which prevents the complementary-leg leak.

Normalisation is fitted on train rows only (`splits.Normaliser` refuses
otherwise). Decision thresholds are chosen on validation only
(`evaluate.pick_threshold`). The exact market list per split is printed on
every run and saved to `split_markets.csv`.

## Leakage controls

Beyond the split rules:

* **Label-shuffle control.** Labels are permuted within each market,
  preserving prevalence and market composition while destroying the timing
  link. Every model must collapse to the prevalence floor. It does.
* **Future-derived columns are banned by prefix, not by name.** An early
  version of this code excluded the literal string `future_move` while the
  cache actually stored `future_move_H10/H30/H60`, so the models were handed
  a continuous version of their own label and scored ROC-AUC **1.0000**. The
  filter is now a prefix match with an assertion, and perfect scores are
  treated as evidence of a bug rather than a result.
* **Post-resolution quotes** cannot enter: only kickoff→final-out rows survive
  `trim_sessions.py`, and matches with truncated recordings are dropped.
* **Jump-already-underway** is excluded by the label's window starting at t+1,
  and tested in `test_features.py`.

## Metrics

PR-AUC is primary, with `pr_auc_lift` (PR-AUC divided by prevalence) as the
comparable figure across settings with very different base rates. Raw accuracy
is deliberately absent. Also reported: ROC-AUC, precision/recall/F1 at the
validation-selected threshold, precision within the top 1/5/10% of predicted
probabilities, Brier score and Brier skill, and false alerts per market-hour.
