# Does one side of the book vanish before a large jump?

2026-09-21. Justin's question: "are there any patterns of what happened before
a large jump — e.g. bid volume vanishes, ask volume still remains". Code:
`research/makinen/pre_jump_asymmetry.py`. 81,749 durable jumps with 81,533
matched controls, six sessions.

## Short answer

**Yes, the pattern is real, and it gets dramatically stronger with jump size.
And it still cannot tell you which way the price will go.**

Those two facts sit together because the asymmetry shifts the *average* book
without separating *individual* events — and because most of it arrives in the
last two seconds.

## 1. What actually happens in the 30 seconds before a jump

Taking the largest jumps (≥20 ticks, n = 7,785), change in log resting dollars
relative to t−30s:

| | t−20s | t−10s | t−5s | t−2s | t0 |
|---|---|---|---|---|---|
| side the price moves **into** | +0.12 | +0.50 | +0.36 | −0.07 | **−0.68** |
| the other side | +0.14 | +0.59 | +0.56 | +0.24 | −0.32 |

Two things, and the first was not expected:

**Liquidity builds before it collapses.** Both sides *gain* roughly 50–60% in
resting dollars between t−30s and t−10s, then evaporate. A large jump is not
preceded by a quietly thinning book; it is preceded by a book that fattens and
then breaks.

**The side the price moves into breaks harder.** At t0 it has lost 0.68 in log
dollars against 0.32 on the other side. That is Justin's hypothesis, confirmed
— the bids vanish and the asks hold, and then the price falls.

## 2. The asymmetry scales sharply with jump size

`into − away` at each lead, by size. Negative means the move's own side emptied
more:

| jump size | n | t−10s | t−5s | t−2s | t0 | Cliff's δ vs control |
|---|---|---|---|---|---|---|
| 2–3 ticks | 21,199 | −0.025 | −0.030 | −0.031 | −0.030 | −0.021 |
| 3–5 ticks | 20,759 | −0.030 | −0.038 | −0.034 | −0.055 | −0.023 |
| 5–10 ticks | 21,215 | −0.006 | −0.031 | −0.073 | −0.147 | −0.047 |
| 10–20 ticks | 10,791 | −0.039 | −0.090 | −0.146 | −0.209 | −0.061 |
| **≥20 ticks** | 7,785 | −0.094 | −0.197 | **−0.305** | **−0.361** | **−0.086** |

A **12× spread** between the smallest and largest buckets at t0. This is a
genuinely new result: `RULED_OUT` A4 measured pre-jump asymmetry across all
jumps and found no timing content, and pooling is exactly what hid it — the
median jump has essentially none, and the tail has a lot.

**But it arrives late.** For the ≥20-tick bucket the gap is −0.094 at t−10s,
−0.197 at t−5s, −0.305 at t−2s. Two thirds of it appears inside the final five
seconds, which is the same "no usable lead time" wall the rest of the project
hit.

And the effect sizes stay small: Cliff's δ of −0.086 at the largest size is
below the 0.147 conventional threshold for "small".

## 3. The test that decides it: can it call the side?

Rule, using only information before the jump: **the side that lost more resting
dollars is the side the price will move into.**

| jump size | lead | depletion rule | imbalance rule | **always guess the majority side** |
|---|---|---|---|---|
| 2–3 ticks | −2s | 50.5% | 51.0% | **52.4%** |
| 3–5 ticks | −2s | 50.3% | 50.7% | **50.2%** |
| 5–10 ticks | −2s | 51.2% | 50.2% | **51.9%** |
| 10–20 ticks | −2s | 52.2% | 51.0% | **55.0%** |
| ≥20 ticks | −2s | **55.0%** | 52.0% | **56.3%** |

The depletion rule climbs to 55.0% on the largest jumps, which looks
encouraging until you put the right benchmark next to it. **Those jumps are
56.3% up-moves anyway.** A rule that ignores the book entirely and always
guesses "up" beats it.

**The rule loses to the majority class in every size bucket.** At −5s and −10s
it is worse still. Both rules are beaten by a constant.

## 4. Why both things are true at once

The asymmetry is real in the mean and useless per event, because the
distributions overlap almost completely. A Cliff's δ of −0.086 means that if
you draw one jump and one control, the jump's book is more lopsided only about
54% of the time.

The more useful reading: **what the book knows before a jump is the *size*, not
the *side*.** That is consistent with everything measured today — the selector
that predicts *which jumps are worth taking* reaches ROC-AUC 0.865 and a 22×
lift, while eight separate attacks on *which way* have all failed. The book
broadcasts that a large move is coming. It does not broadcast its sign.

## 5. What this closes and what it does not

**Closes:** the depletion-asymmetry rule as a direction signal, now tested on
the large-jump tail specifically rather than pooled. Ninth direction method,
same answer.

**Does not close:** the signed trade flow already sitting in the recordings
(~100k prints, 86% aggressor-resolved, each with the full book). Resting-order
asymmetry is *passive* information — who is willing to wait. Aggressor flow is
*active* — who is willing to pay. They are different quantities, and only the
first has been tested here.

**Worth keeping** from this run: the build-then-break shape. Liquidity rising
~55% and then collapsing is a cleaner precursor signature than steady thinning,
and it is a size signal the current detector does not use — it triggers on the
fall alone, with no memory of the build.
