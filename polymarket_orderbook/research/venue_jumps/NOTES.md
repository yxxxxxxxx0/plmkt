# Two-venue jump study (Polymarket + Kalshi), 12 games of 2026-10-18/19

Run every script from `polymarket_orderbook/`. Each reads `data/view/` (built by
`prep_venue_view.py`) plus `data/view/score_marks.json` from `mark_scores.py`.

## Findings, in order

1. **Goal labels** (`../../mark_scores.py`). Goal/run times from match reports,
   fitted to price moves. 32 of 39 scores show a clear move; the rest were late
   goals with the price already above 95%. Every 6c / 30 s jump is labelled score,
   event or noise. 135 jumps: 41 score, 5 event, 89 noise.
2. **Cross-venue agreement** (`agree.py`, `agree2.py`). When one venue jumps 6c+,
   the other moved 6c+ the same way within 5 s for 88% of goal jumps, 49% of
   non-goal jumps and 3% of random moments. One-venue jumps are mostly noise.
3. **Price correlation** (`corr.py`, `corr_pre.py`). Correlation of 5 s changes,
   1 minute either side of the jump: 0.83 goals, 0.68 non-goals, 0.00 random.
   It is a weaker filter than move size. In the minute *before* a jump it is
   0.36 / 0.42, so it gives no early warning.
4. **Good vs bad jumps** (`good_jumps.py`; buy at ask before the jump, sell at
   bid 30 s after, with fees; perfect foresight). Good / bad counts: goal 31/12,
   match event (red card, VAR, penalty, woodwork) 9/3, 85'+ 10/4, no cause
   found 40/81. Writes `data/view/jump_events.json`.
5. **Taker after both books collapse and prices agree** (`joint_collapse.py`).
   Loses 4-7c per trade at any entry delay from 0 to 3 s. The move is mostly
   done (6c of it) and the spread is about 7c at entry.
6. **Taker at the collapse, before the price moves** (`collapse_corr.py`).
   268 joint collapses, 20% became good jumps. Even with hindsight-perfect
   direction the average is about 0c (early entries -1.3c). Depth correlation
   is above 0.8 for 208 of the 268, so it doesn't separate real from noise.
   Joint collapses look like market makers protecting themselves on both venues.

## Next
Maker strategy: quote at the touch and use the joint-collapse detector to cancel
(`maker_sim.py`). Bankroll $100.
