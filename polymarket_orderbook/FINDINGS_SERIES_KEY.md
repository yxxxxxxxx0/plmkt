# The series-key bug, and what survived it

> **Note (2026-09-21).** Some scripts named below were deleted in the codebase
> tidy. They are recoverable at commit `40437ab` (`git show 40437ab:<path>`),
> and what each one did is recorded in `METHODS_AND_EXPERIMENTS.md`.

2026-09-15. Every positive result this study produced before this date was an
artifact of one line in `jump_data.py`. This is the record of what happened,
how it was found, and what the corrected data says.

## The bug

A market time-series was keyed as `(slug, market_type, line)`:

```python
key = (r.get("slug"), mt, r.get("line"))                    # wrong
key = (r.get("slug"), mt, r.get("line"), r.get("asset_id")) # right
```

That key is not unique. A **spread** bet has two tradeable tokens -- e.g.
"Rockies -1.5" near $0.195 and "Yankees +1.5" near $0.545. They share slug,
market type, line, and both carry `outcome == "YES"`. They differ only by
`asset_id`.

Keyed together, their updates interleaved into one forward-filled series whose
"mid" alternated between two unrelated levels. The result is not a price. It
is a square wave, and a square wave is trivially predictable: if you are on the
low level the next move is up, and vice versa. One bit of information.

10 of 35 series on 2026-09-10 were affected. All 10 were spread markets.
Moneyline and totals have unique keys and were never touched.

## Why it was not caught sooner

The edge lived **entirely** in spread markets while moneyline sat at chance.
That was read repeatedly as "spread markets are more predictable" rather than
as a corruption signature. No aggregate statistic being computed at the time
could distinguish the two readings.

It was found in minutes by *plotting the price* (`visualize_jumps.py`). The mid
oscillated between 0.70 and 0.02 with constant +/-60-tick moves, which no real
market does.

## How large the artifact was

Measured with `compare_seriesfix.py`, on tradeable rows only (fresh book,
spread <= 2 ticks). Moneyline and totals are bit-for-bit identical before and
after, which makes this a clean controlled experiment -- only spread changed.

| market | | avg 5s move | round trip cost | oracle P&L |
|---|---|---|---|---|
| moneyline | before | 0.107 | 3.04 | -2.94 |
| moneyline | after | 0.107 | 3.04 | -2.94 |
| total | before | 0.213 | 3.50 | -3.29 |
| total | after | 0.213 | 3.50 | -3.29 |
| **spread** | before | **4.54** | 3.41 | **+1.13** |
| **spread** | after | **0.29** | 3.47 | **-3.17** |

"Oracle P&L" is what a trader who knows the direction of every move in advance
still earns after paying the spread and both taker fees. It bounds every
possible model. Before the fix spread markets showed +1.13 ticks of headroom;
after, -3.17.

The reported CNN result of +0.3208 ticks/opp sat comfortably inside that fake
+1.13 ceiling.

## Proof that the model was learning the artifact

`why_it_looked_profitable.py` runs a rule using **nothing but the current
price level** -- no order book, no training, no parameters. Buy if the mid is
below the series median, sell if above.

| | direction correct | avg move | P&L per trade |
|---|---|---|---|
| broken data | **76.0%** | 4.54 ticks | **+0.54** |
| fixed data | 51.0% | 0.29 ticks | -3.42 |

On the broken data a one-line rule **beat** the CNN (+0.54 vs +0.32). That is
the tell: the architecture bought nothing, because what was there to capture
was a bookkeeping artifact. On fixed data the same rule is a coin flip.

## What the corrected data says

Rebuilt all six sessions (160 GB raw -> 97M grid rows, 28M in-game after
trimming). `audit_datasets.py` passes all six: series identity exactly 1:1, no
lookahead, no unexplained price steps. That check was validated by running it
against the pre-fix data, where it fires on 4,530 steps, 100% spread.

**There is genuine predictive signal, and it cannot pay for the spread.**

| model | sign AUC | P&L per trade |
|---|---|---|
| momentum control | 0.5530 | -3.73 |
| GBM on hand features | **0.6617** | -3.24 |

0.6617 is well above chance. The P&L is negative anyway, because the average
5-second move is 0.44 ticks against a 3.99-tick round trip.

## Why the 5-second taker case cannot work

Cost is not constant. It grows faster than the move does, because a large move
destroys the liquidity needed to exit it:

| move size | % of rows | exit spread | cost | accuracy needed |
|---|---|---|---|---|
| 0-1 ticks | 90.2% | 1.5 | 3.08 | impossible |
| 4-6 ticks | 1.0% | 15.2 | 10.20 | impossible |
| 10-20 ticks | 0.3% | 35.3 | 20.51 | impossible |
| >20 ticks | 0.2% | 60.6 | 33.00 | impossible |

Break-even accuracy is 100% or worse at **every** move size. There is no
threshold at which being right often enough is sufficient.

## The one direction that is not closed

`horizon_sweep.py`, two sessions, oracle P&L per opportunity:

| horizon | 09-10 | 09-13 |
|---|---|---|
| 5s | -3.08 | -3.37 |
| 30s | -2.61 | -2.89 |
| 120s | -1.16 | -1.31 |
| **300s** | **+0.77** | **+0.94** |
| **600s** | **+2.70** | **+3.24** |

The move grows 15x from 5s to 600s while cost stays near 3.9, because fees are
fixed and a 10-minute exit is no longer inside the gap the move opened. The
crossover is around four minutes.

The binding constraint moves to accuracy: 91.6% at 300s, 79.5% at 600s. And a
10-minute label on a 200ms grid overlaps 3,000x, so the independent sample is
small -- fertile ground for exactly the class of false positive documented
above. Treat any positive result there with more suspicion than usual.

## Caution on maker-exit numbers

The same sweep under a passive exit returns +24.65 ticks on moves >= 6 ticks,
with **negative** break-even accuracies -- i.e. it claims profit while being
wrong. That is not an edge, it is proof the assumption is unphysical: it
credits the trader with earning half of a 60-tick post-move spread. These
recordings contain 8k-20k trade prints per session with `aggressor`, `price`,
`size` and the book at that instant, so fill probability and adverse selection
*can* be measured -- but until they are, maker P&L here is an assumption, not
a result.

## Still outstanding

`crossmarket.py` has the identical defect (`crossmarket.py:74`, plus the
`groupby(["mt", "line"])` at :144 and :190). It never captures `asset_id` at
all, so `data/adverse/xm_books_*.parquet` and every conclusion drawn from its
lead-lag and jump-event analysis are suspect and require re-extraction.
