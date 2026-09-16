# Real collapses vs fake ones — and can the side tell you direction?

2026-09-16. Reframing suggested by Justin: stop hunting a rare jump across all
instants, and instead **detect every liquidity collapse, then classify which
ones are real.** Code: `research/makinen/collapse_events.py` and
`collapse_classify.py`.

## Why the reframing is better

The collapse is the physical thing that precedes the mid moving, and it is
detectable with a trailing rule and no model at all. That turns a ~1-2%
prevalence needle-hunt into a candidate set with a usable base rate.

134,428 collapse events over 6 sessions (67,915 bid-side, 66,513 ask-side),
detected as near-touch dollars falling below 25% of their own trailing 60s
median, debounced 30s:

| outcome within 10s | count | share |
|---|---|---|
| **REAL** — moved >= 0.02 and held 3s | 36,112 | **26.9%** |
| **FAKE** — moved then reverted | 43,924 | **32.7%** |
| never moved | 54,392 | 40.5% |

The FAKE class is exactly the confusion worth modelling: the book empties, the
mid ticks mechanically, liquidity returns, the price snaps back.

## Can a CNN tell them apart? Yes, partly.

Input = the book in the 20 seconds ENDING AT the collapse instant (50 steps at
200ms, 22 features). Chronological split by whole session; test is one
held-out session, **24,608 events**, base rate 31.9%.

| model | PR-AUC | lift | ROC-AUC | precision | recall | F1 |
|---|---|---|---|---|---|---|
| prevalence floor | 0.3189 | 1.00 | 0.500 | — | — | — |
| logistic, book at the collapse instant | 0.5928 | 1.86 | 0.723 | 0.517 | 0.567 | 0.541 |
| CNN | 0.6089 | 1.91 | 0.745 | 0.471 | 0.723 | 0.571 |
| CNN-LSTM | 0.6100 | 1.91 | 0.746 | 0.489 | 0.671 | 0.566 |
| CNN-LSTM-Attention | 0.6140 | 1.93 | 0.749 | 0.449 | 0.778 | 0.570 |
| GBM, book at the collapse instant | 0.5949 | 1.87 | 0.726 | 0.481 | 0.640 | 0.549 |
| **GBM, instant + trajectory summaries** | **0.6267** | **1.97** | **0.756** | 0.468 | 0.742 | **0.574** |

**The tree wins.** A gradient-boosted model on hand-made features beats the
CNN-LSTM-Attention by +0.013 PR-AUC and +0.007 ROC-AUC, trains in 8 seconds
rather than minutes, and is interpretable. The architecture does not earn its
keep here either.

Top features by gain, and the story is blunt:

| feature | gain |
|---|---|
| **spread at the collapse instant** | **92,641** |
| spread max over the window | 11,172 |
| spread change over 20s | 10,744 |
| near-touch bid fraction now | 8,615 |
| spread std over the window | 7,985 |
| near-touch bid dollars now | 7,342 |

Spread at the moment the book breaks dominates everything else by roughly 8x.
How wide the book already is when it empties is most of what separates a real
collapse from one that snaps back.

**OBSERVATION.** PR-AUC 0.614 against a 0.319 floor, ROC-AUC 0.749, roughly 50%
precision at 67-78% recall — on 24,608 held-out events rather than the 109
positives that made every earlier comparison unreadable.

**OBSERVATION.** A logistic regression on the book at the single collapse
instant reaches 0.593. The full 20-second sequence and the attention
architecture add **+0.021 PR-AUC** on top of that.

**INTERPRETATION.** Most of what separates a real collapse from a fake one is
visible in the instantaneous state of the book at the moment it breaks — how
much is left, how wide, how concentrated — not in the trajectory leading up to
it. The deep models are worth little here, consistent with every other
architecture comparison in this project.

## Direction: still no, and now tested a third way

The mechanical story is appealing: if the BID collapses there is no support
beneath, so the price should fall. It is false.

Real collapses only (n = 36,112):

| collapsing side | price went UP | price went DOWN |
|---|---|---|
| bid | 0.5487 | **0.4513** |
| ask | **0.5553** | 0.4447 |

* rule "collapsing side = direction of the move": **50.2%** accuracy
* majority-class baseline (always guess up): **55.2%**
* held-out last session alone: rule **50.1%**, baseline **57.1%**

The signed-move distributions for bid-side and ask-side collapses are
effectively identical (mean +0.0158 vs +0.0150, same quartiles).

**INTERPRETATION.** The *mechanical* move when a side empties is directional but
transient — the mid ticks toward the vacated side and comes back. The *durable*
move is informational, and appears unrelated to which side happened to blink
first. Direction has now been tested three independent ways — pre-jump book
asymmetry, imbalance features at a 5s lead, and the collapsing side — and all
three land at or below the naive baseline.

## What this means

You can now do two things reliably:

1. **Detect a liquidity collapse** — mechanical, no model, no latency cost.
2. **Score whether it is real** — ~50% precision at ~70% recall, ROC-AUC 0.75.

You still cannot tell which way the price will go. So this remains a risk
signal, not a directional one — but it is a much sharper risk signal than
anything earlier in this project, and it rests on 24,608 held-out events rather
than 109.

## Next

* The sequence adds almost nothing over the instant. Before investing further in
  architecture, test whether a gradient-boosted tree on the instantaneous book
  matches the CNN — on this project's record it probably will.
* Direction needs an input the book does not contain. Signed trade flow
  (aggressor side) is the obvious candidate and is not currently recorded.
