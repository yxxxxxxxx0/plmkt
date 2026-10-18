# More data, and does a coarser bar give a cleaner signal?

2026-09-16. Data: `polymarket_sports` live subset (1,039 in-play esports/match
tokens, 21 consecutive days, 62M tick rows). Code: `research/sports/`.

Two questions were asked. Short answers: **yes, this is materially more data
than the MLB set**, and **yes, coarser helps — but only up to ~15 seconds,
after which it gets worse again.**

## Can this feed the CNN-LSTM-Attention model?

**Not by pooling with the MLB data.** The `polymarket_sports` orderbook table
carries `best_bid`/`best_ask` but **no bid/ask sizes and no depth levels**, so
the 69 MLB features (10-level depth, near-touch dollars, book slope) do not
exist here and were not fabricated. The two feature sets barely overlap.

**But the new data alone is bigger, and it is better structured:**

| | MLB (`makinen`) | sports, 15s |
|---|---|---|
| windows | 18,579 | **142,971** (7.7×) |
| contracts | 601 | **1,039** |
| calendar days | 6 sessions | **21 consecutive** |
| test split | **1 session, 109 positives** | 14,879 windows |
| features | 69 (incl. 10-level depth) | 17 (L1 + **order flow**) |

The 21-day span is the real prize: the MLB study's binding constraint was a
single held-out session, which left every architecture comparison statistically
unresolved. It also supplies something MLB never had — a genuine **delta
stream**, so update intensity, buy/sell counts and flow imbalance are real
features here rather than absent ones.

## The resolution sweep

Everything held fixed except bar size: K = 60 bars, lookback = 120 bars,
horizon = 1 bar, α = 0.01, chronological split over whole dates.

| bar | lookback | windows | prevalence | best deep model | **PR-AUC lift** | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 s | 2 min | 9,882 | 1.65% | CNN-LSTM | **1.98×** | 0.488 |
| 5 s | 10 min | 88,814 | 1.12% | CNN-LSTM | 3.27× | 0.706 |
| **15 s** | **30 min** | **142,971** | 1.26% | **CNN-LSTM-Attention** | **6.21×** | 0.743 |
| 60 s | 2 h | 57,031 | 1.65% | CNN-LSTM | 4.99× | **0.856** |
| 300 s | 10 h | 24,384 | 2.92% | CNN-LSTM | 4.86× | 0.788 |

**OBSERVATION.** Lift rises 1.98 → 3.27 → **6.21** as bars coarsen from 1s to
15s, then falls back to ~4.9 at 60s and 300s. The peak is at **15 seconds**.
The two metrics disagree about where the optimum sits: PR-AUC lift peaks at
15s, ROC-AUC at 60s.

**INTERPRETATION.** There appears to be a genuine sweet spot where the bar is
long enough to contain real information but short enough not to average the
event away. It is *far* finer than the 1-minute bar the paper uses and the MLB
study inherited.

## Why 1-second failed — and why that is not a conclusion

At 1s the model is barely above chance (ROC 0.488) on only 9,882 windows. That
is **not** evidence that fine resolution is wrong; it is a density artifact:

| bar | share of bar slots that contain an update |
|---|---|
| 1 s | **13.0%** |
| 5 s | 30.3% |
| 15 s | 43.8% |
| 60 s | 64.5% |

Windows here require **120 strictly contiguous bars**. At 1s, 87% of slots are
empty, so almost no contiguous 120-bar window exists and 11.3M bars collapsed
to 9,882 usable windows. The MLB 200 ms work did not hit this because that grid
is an as-of forward fill with no holes.

**So the 1s row measures sparsity, not signal.** To test 1s properly the bars
need the same as-of forward-fill treatment (with a staleness cap), which is a
build change rather than a finding. Until then, do not read the 1s row as
evidence against the ~6-second precursor measured earlier on the MLB grid.

## Compared with the MLB result

MLB best was 7.01× lift at 1-minute horizon — but on 109 test positives with a
bootstrap CI of [0.116, 0.252] that left everything unresolved. Sports at 15s
gives **6.21× on 7.7× more windows**. Comparable effect, far more evidence
behind it.

Attention beats CNN-LSTM at 15s only (6.21 vs 6.15) and loses at every other
resolution. That is consistent with the MLB finding that attention is not
reliably better than a plain recurrent model, and it should not be called a win
without the bootstrap test being run here too.

## Recommended next step

1. Rebuild the bars with as-of forward-fill plus a staleness cap so 1s and 5s
   are testable on equal terms with 15s.
2. Run the full model zoo and the game-level bootstrap CI at 15s, as was done
   for MLB, before claiming any architecture ordering.
3. Keep the live-subset filter — 88% of that dataset is dead longshot futures
   and would swamp the result.
