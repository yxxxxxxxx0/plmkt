# Can you go around the classifier? Every jump in the price graph, priced

2026-09-21. Justin's question: stop asking a model which collapses are real —
just look at the price series, find all the jumps, and tell me which ones are
big enough to break even. Code: `research/makinen/oracle_jump_scan.py`.

This bypasses the collapse detector and the classifier entirely. Every durable
move in the 200ms in-game grid is enumerated directly from the mid — 82,120
jumps over six sessions, 601 contracts, 75 games — and each is priced at the
spread actually standing at entry and at the exit, plus the Polymarket sports
taker fee on both legs.

Because the entry is placed at the jump *before it happens* and the direction
is assumed known, every number here is an **oracle bound**. No classifier,
however good, can beat it.

## The headline

| exit rule | mean P&L | jumps that pay |
|---|---|---|
| at the peak of the move, no fee | −6.15 t | 4,999 / 82,120 = **6.1%** |
| **at the peak, fee charged** | **−7.98 t** | **935 / 82,120 = 1.1%** |
| 3s after the peak, fee charged | −7.21 t | 2,009 = 2.4% |

**935 jumps out of 82,120 break even.** They are not concentrated — they span
315 contracts, all 75 games and all six sessions, so this is not a data
artefact. Taking only those 935, with perfect foresight, earns 2,230 ticks
across six nights.

The median paying jump moves 5.0 ticks, exactly the same as the median jump
overall. **What separates them is not the size of the move, it is the cost of
the round trip**: median entry spread 1 tick against 7 for the population, and
median spread at the exit 2 ticks against 8.

## The trap in the omniscient exit

Letting the oracle also pick the best exit instant within 300s turns the mean
*positive*, +0.59 ticks with fees, 44% of jumps paying. That number is worthless
on its own: choosing the best of ~1500 future instants is positive on a pure
random walk by construction.

The control settles it. Scoring **random in-game instants** the identical way:

| entries | P&L, exit at peak | P&L, omniscient exit |
|---|---|---|
| at a jump | −7.98 t | +0.59 t |
| at a random moment | −5.50 t | **+1.04 t** |

Unconditionally, **entering at a jump is 0.45 ticks worse than entering at an
arbitrary moment**, because jumps happen where the book is already wide and you
pay for that on the way in.

Within a matched entry spread the ordering reverses — at a 1-tick entry a jump
beats a random moment by 2.5 ticks — but that is the oracle monetising the
higher volatility around a jump, and it is only collectable by someone who knows
which of 1500 future instants to sell into. It is not a signal anyone can trade.

## The one slice with any life in it

Restricting to entries at a **1-tick spread** and exiting at the peak, which is
the best exit a real system could aim at:

| market | n | spread at exit | move | cost | fee | before fee | after fee |
|---|---|---|---|---|---|---|---|
| **moneyline** | 2,739 | 4.8 t | 3.13 t | 2.89 t | 1.94 t | **+0.24 t** | **−1.70 t** |
| spread | 5,709 | 21.9 t | 7.77 t | 11.46 t | 1.98 t | −3.69 t | −5.67 t |
| total | 5,709 | 16.1 t | 5.91 t | 8.53 t | 2.13 t | −2.62 t | −4.74 t |

Moneyline entered at a 1-tick spread is **the only configuration found anywhere
in this project where perfect foresight clears the spread**: +0.242 ticks,
game-clustered 95% CI [+0.117, +0.364], with 57.8% of those jumps paying. The
reason is visible in the second column — when a moneyline book is hit it widens
to ~5 ticks, while a spread book widens to ~22.

And then the fee takes it away. At 1.94 ticks for the round trip it is **eight
times** the gross edge, and the slice lands at −1.697 ticks, CI [−1.847,
−1.545].

## What this answers

**You can go around the classifier — it makes no difference.** The classifier's
recall was never the binding constraint. Enumerating every jump directly, with
perfect knowledge of timing and direction, still loses on 98.9% of them.

**The binding constraint moved.** Everywhere else in this project the answer was
"the spread eats the move". In the one slice that escapes that — moneyline at a
1-tick entry — the spread *is* cleared and the **Polymarket taker fee** is what
makes it negative. That is a different wall from the one C5 documented, and it
is worth stating precisely because makers are never charged the fee at all.

Do not read that as a live maker idea: section D already closed the maker path
on fills, with three fills in three days once flow has to clear your size. But
it does mean the honest statement is now "the closest thing to break-even is
moneyline, and it fails by the fee" rather than "nothing comes close".

## Caveats

* Everything here is an oracle. A real system would need to capture essentially
  all of a +0.24-tick gross edge, and the best honest direction figure in the
  project is ROC 0.659.
* Size is ignored — the oracle is assumed to fill at the touch at both ends.
* Six sessions, 75 MLB games, Aug–Sep 2026. The four nights recorded since
  09-17 are not in this.
* `mid` is the midpoint, so a jump can be a quote move with no trade behind it.
  The durability clause (still displaced 3s after the peak) is the filter, the
  same one the classifier uses.
