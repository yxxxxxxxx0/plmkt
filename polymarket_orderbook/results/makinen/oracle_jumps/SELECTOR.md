# Is 77.3% precision reachable? And is precision even the binding constraint?

> **SUPERSEDED IN PART, 2026-09-22. Read this first.**
>
> Every held-out number below was produced with `TEST_SESSIONS` hardcoded to
> `books_2026-09-12` and `books_2026-09-13`. Four sessions recorded *after*
> those dates — 09-17 through 09-20 — were later built and went into
> **training**, so the split stopped being chronological and the model was
> fitted on the future of its own test period. The script's leakage assertion
> only checks that no game straddles the split, and none did, so nothing
> flagged it.
>
> Re-run on ten sessions with the split derived as the last two present
> (09-19, 09-20):
>
> | | below | corrected |
> |---|---|---|
> | ROC-AUC | 0.865 | 0.888 |
> | PR-AUC lift | 22.1× | 20.7× |
> | **best precision anywhere** | **52.9%** | **34.2%** (73 trades) |
> | held-out payer base rate | 0.94% | 0.60% |
> | nothing profitable below | 75% direction | **80% direction** |
>
> **The conclusion of section 4c is reversed.** "Precision is solved" does not
> survive the corrected split: the requirement is 71.8% and the selector
> reaches 34.2%. The structure of the argument — that direction and precision
> trade off against each other, and that a wrong-side trade costs about twice
> what a right-side trade earns — stands unchanged. Only the claim that one of
> the two dials was already delivered does not.
>
> Current figures: `RESEARCH_PROGRESS.md`, section "What changed on
> 2026-09-22", and the regenerated CSVs beside this file.

2026-09-21. Justin asked: if I beat 77.3% precision at a 1.1% base rate, I
capture the 935 paying jumps and make money. The arithmetic is right. This tests
whether the precision is reachable, and finds that precision stops being the
binding constraint before you get there.

Code: `research/makinen/can_you_select_payers.py` (can a model pick them) and
`selector_realistic_exit.py` (re-priced on a clock you can run, plus the
controls).

## 1. A model can pick them, and it is genuinely skilful

Every durable jump becomes one row, labelled by whether it nets positive after
fees. Features are the book and price state **at the jump's onset only**. Split
is by whole game and chronological — the last two sessions are held out, and an
assertion enforces that no game appears on both sides.

| | |
|---|---|
| train | 48,616 jumps, 621 payers (1.28%) |
| test | 33,504 jumps, 314 payers (0.94%) |
| ROC-AUC | **0.865** |
| PR-AUC | 0.207 against a 0.0094 floor — **22.1× lift** |
| best precision | **52.9%** at the top 0.1% (34 trades) |

22× is by a distance the largest lift in this project. And it is not luck:
drawing the same number of jumps at random 4,000 times, the random mean is
−9.27 ticks and the model's selection sits at the **100th percentile** at every
size tested. The selector is real.

It is also short of 77.3%.

## 2. But 77.3% was computed against an oracle exit

That bar came from the marked set, which exits at the *peak* of the move — a
point located with future information. Re-priced on a fixed clock, exit at
entry + T seconds, the picture is different, because the spread re-tightens
faster than the price gives back:

| top | n | precision | +5s | +10s | +30s | +60s | +300s |
|---|---|---|---|---|---|---|---|
| 0.1% | 34 | 52.9% | −1.69 | −0.46 | **+3.45** | +3.49 | +6.28 |
| 0.5% | 168 | 38.7% | −2.17 | −0.78 | **+1.17** | +1.04 | +1.23 |
| 1.0% | 335 | 29.9% | −2.59 | −1.23 | +0.20 | −0.04 | +0.17 |
| 2.0% | 670 | 20.0% | −2.95 | −1.98 | −0.37 | −0.70 | −0.83 |

Game-clustered 95% CIs at the top operating point exclude zero from +20s
onward — +3.45 t at 30s, CI [+0.46, +9.15].

**Do not stop reading here.** Every number in that table grants direction for
free.

## 3. Take the free direction away and it all goes negative

Same picks, same 30s hold, the only change being who chooses the side:

| top | n | oracle direction | always long | always short | coin flip |
|---|---|---|---|---|---|
| 0.1% | 34 | **+3.45** | −0.30 | −3.21 | −1.67 |
| 0.5% | 168 | +1.17 | −1.96 | −2.66 | −2.31 |
| 1.0% | 332 | +0.20 | −2.31 | −3.75 | −3.03 |
| 2.0% | 665 | −0.37 | −3.20 | −4.07 | −3.64 |

**Every operating point is negative without the oracle direction.** The entire
apparent edge is the free side.

## 4. So the real remaining bar is directional, not precision

For the model's picks at a 30s hold:

| top | n | E\|move\| | cost + fee | perfect-direction EV | **accuracy needed** |
|---|---|---|---|---|---|
| 0.1% | 34 | 5.73 t | 1.75 t | +3.98 t | **65.3%** |
| 0.5% | 168 | 4.08 t | 2.31 t | +1.77 t | 78.3% |
| 1.0% | 332 | 4.27 t | 3.03 t | +1.24 t | 85.5% |
| 2.0% | 665 | 4.18 t | 3.63 t | +0.55 t | 93.4% |

This is the first configuration found in the project where the requirement lands
anywhere near measured skill. At the very top the picks need **65.3% directional
accuracy**, and the best direction result on record here is ROC-AUC 0.659.

**Those two numbers are not the same quantity.** ROC-AUC 0.659 is a ranking
statistic and corresponds to a materially lower hit rate than 65.3% at any
single operating point, so this is not "we are already there". It is the first
time the gap has been small enough to be worth another measurement.

## 4b. CORRECTION — the 65.3% figure was one trade

The 65.3% in the table above is **not robust and should not be quoted**. In the
top-34 set there is a single 75-tick move, and it is **38% of that set's entire
total move**. Remove it and the requirement moves to 74.1%; on a 5–95% trimmed
mean it is 75.3%. The whole "closest we have come" impression was one
observation.

| cut | n | mean \|move\| | median | robust accuracy needed |
|---|---|---|---|---|
| top 34 | 34 | 5.73 (raw) | 3.00 | **74.1% – 75.3%** |
| top 67 | 67 | 4.32 (raw) | 2.35 | 80.0% – 83.7% |
| top 167 | 167 | 4.09 (raw) | 3.00 | 81.3% – 84.3% |
| top 334 | 334 | 4.26 (raw) | 3.00 | 87.6% – 92.9% |

`direction_precision_grid.py` uses the trimmed estimate throughout for exactly
this reason.

## 4c. The two dials together — what precision do you need, given direction?

Asked the other way round: fix directional accuracy, and ask how tightly you
must select. `EV = (2a − 1) × E|move| − E[cost + fee]`, on a 30s clock, held
out, trimmed:

| how tightly you select | n | selector precision | 60% | 70% | **75%** | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|
| top 0.1% | 33 | 54.5% | −1.03 | −0.33 | **+0.03** | +0.38 | +1.09 | +1.80 |
| top 0.2% | 67 | 47.8% | −1.34 | −0.74 | −0.43 | −0.13 | +0.47 | +1.08 |
| top 0.5% | 167 | 38.9% | −1.62 | −0.94 | −0.60 | −0.26 | +0.42 | +1.10 |
| top 1% | 334 | 29.9% | −2.32 | −1.61 | −1.25 | −0.89 | −0.18 | +0.54 |
| top 2% | 669 | 19.7% | −2.93 | −2.23 | −1.88 | −1.53 | −0.83 | −0.13 |
| everything | 33,434 | 0.9% | −12.48 | −11.56 | −11.09 | −10.63 | −9.70 | −8.78 |

The loosest cut that profits at each accuracy, with game-clustered CIs:

| direction accuracy | feasible? | cut | trades/session | precision needed | EV | 95% CI |
|---|---|---|---|---|---|---|
| ≤ 70% | **no cut works** | — | — | — | — | — |
| 75% | yes | top 0.1% | 16.5 | 54.5% | +0.03 t | [−0.43, +0.55] |
| 80% | yes | top 0.1% | 16.5 | 54.5% | +0.38 t | [−0.17, +1.01] |
| 90% | yes | top 0.5% | 83.5 | 38.9% | +0.42 t | **[+0.10, +0.78]** |
| 100% | yes | top 1% | 167 | 29.9% | +0.54 t | **[+0.08, +0.99]** |

**Read the "precision needed" column: 30–55%. The selector already delivers
that** — 54.5% at the top 0.1%, 38.9% at the top 0.5%. Precision is solved.

What is not solved is the row label. **Nothing is positive below 75% directional
accuracy, at any selection precision**, and only at 90% and above does a
confidence interval exclude zero. The best direction result in this project is
ROC-AUC 0.659, a ranking statistic weaker than a 66% hit rate, let alone 75%.

## 5. What would have to be true

Three separate predictions are needed, and only the first is solved:

1. **pick a jump that pays** — done, 52.9% precision, 22× lift, verified
   against a placebo.
2. **know which way it goes** — needs ~65% accuracy at the top operating point.
   Unsolved; eight attempts are logged in `RULED_OUT.md` section A.
3. **hold for ~30s and exit on the clock** — no foresight required, priced here
   honestly.

## 6. Why this is not yet a result

* **n = 34** at the only operating point where the requirement is near reach.
  Two held-out sessions. Nothing can be concluded from 34 trades.
* **Concentration.** At the top 0.5%, one game out of 28 supplies 37% of the
  ticks.
* **Volume.** 0.1% of jumps is roughly 17 trades a session — about 2 per game.
  Even if it worked, it is a small book.
* **Size is ignored.** Fills at the touch are assumed at both ends.
* **The CIs are wide and skewed** — [+0.46, +9.15] is the signature of a mean
  carried by a few observations.

## 7. The honest answer to the question asked

Yes, beating 77.3% precision would profit — but that bar was set against an
oracle exit, and on a clock you can actually run the precision requirement drops
a long way. What replaces it is the same wall as always: **direction**. The
selector works; the side does not.

The specific, testable thing this produces: at the top 0.1% of picks the
directional accuracy needed falls to ~65%. That is the narrowest the gap has
ever been here, and the right response is more data rather than more modelling
— four nights recorded since 09-17 are not in this, which would roughly double
the held-out sample.
