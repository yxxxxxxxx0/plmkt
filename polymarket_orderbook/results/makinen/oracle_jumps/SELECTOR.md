# Is 77.3% precision reachable? And is precision even the binding constraint?

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
