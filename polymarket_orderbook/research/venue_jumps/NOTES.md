# Polymarket strategy search, 2026-10-29

Started as a two-venue jump study (Polymarket + Kalshi, 12 games of 2026-10-18/19)
and widened into a search for anything a $100 Polymarket-only bot could trade.
Run every script from `polymarket_orderbook/`. The two-venue scripts read
`data/view/` (built by `prep_venue_view.py`) plus `data/view/score_marks.json`
from `mark_scores.py`.

## Where it stands

| Idea | Verdict | Why |
|---|---|---|
| Taker on jumps (any timing, any filter) | Dead | Can't enter before the move. After it, what's left doesn't cover the spread. |
| Kalshi moves first, trade Polymarket | Dead | Polymarket follows only 14-24% of the time. -5 to -7c per trade. |
| Latency trading off a data feed (Tank01) | Dead | Polymarket's MLB price moves about 1.7 s *before* MLB's own play timestamp. |
| Maker at the touch, cancel on collapse | Not profitable | 30 s edge is negative. The cancel switch makes Polymarket worse. |
| Polymarket vs Kalshi arbitrage | Small, and needs Kalshi | About 3 catchable windows a game, about $5 a game. Kalshi is US-only. |
| 3-way football arbitrage (home/draw/away) | Out of reach | 95% of the profit is during jumps, when the 1 s delay lets makers cancel. |
| Multi-outcome arbitrage (politics, IPO ranges) | Tiny | 2 of 82 events had a gap, worth $0.11 and $2.60, with money locked for months. |
| MLB arbitrage between main market and props | None | 3.2M checks over 2 World Series games, zero violations. |
| Futures arbitrage (champion vs conference) | None | NFL and NBA snapshot: closest gap -0.1c to -0.25c. |
| **Market making props off the moneyline** | **Open, untested** | Props are priced consistently but with wide spreads (see 13). |

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
7. **Maker with a collapse cancel switch** (`maker_cancel.py`; 5-share quotes at
   the touch, inventory cap 20, $100 bankroll). The edge per share 30 s after a
   fill is negative in every mode on both venues. Polymarket on its own: -$4 to
   -$18 over 12 games once each fill is valued 30 s later. With no switch it's
   nearly break-even (-0.09c a share); every switch makes it worse. The +$27
   totals in some modes come from positions held to the end of the game, not
   from edge. Kalshi results assume a maker fee of 0.0175 x p(1-p).
8. **MLB latency** (one-off check against MLB's free stats API, game 849817).
   Half of each Polymarket move happened a median 1.7 s *before* MLB's official
   end-of-play time; 82% of moves were half done before it. A resold feed like
   Tank01 arrives after the price has moved. Polymarket's matching engine is in
   AWS eu-west-2 (London); eu-west-1 (Dublin) is the usual nearby host.
9. **Kalshi moves first, trade Polymarket** (`kalshi_leads.py`). 149-800 alerts
   depending on threshold. Polymarket followed 14-24% of the time. -5 to -7c
   per trade, -5.1c on MLB with no delay, and -4.8c even on alerts that became
   real jumps.
10. **Polymarket vs Kalshi arbitrage** (`cross_arb.py`). 473 windows with fresh
    quotes; 279 last only 1 s. 38 lasted 2 s+ with 2c+ edge: about 3 a game at
    a median 3.2c on about 49 shares, about $5 a game. Legging risk on the
    delayed Polymarket leg; settlement-rule mismatch; Kalshi needs US residency.
11. **3-way football arbitrage within Polymarket** (`pm_internal_arb.py`). The
    three YES asks normally sum to $1.01-1.02. 90 windows, 41 lasting 2 s+,
    $120 in total, but $114 of that was within 60 s of a 6c+ jump, exactly when
    the 1 s taker delay lets makers cancel. About $6 in calm periods. MLB can't
    do this: it's one two-outcome market with a shared YES/NO book.
12. **Taker delay by market** (Gamma `secondsDelay`, snapshot 2026-10-29).
    Football, NFL and NHL: 1 s on the main market *and* every prop. NBA and MLB:
    0 s. Futures and awards (season champions, MVP, Ballon d'Or-type markets):
    no delay (8,059 of 8,075 markets unset, the rest 0).
13. **MLB logical arbitrage** (`mlb_logic_arb.py`, recorded top of book
    2026-10-26/27). Over L1 >= Over L2, Team -k <= Team wins, Team -k1 >=
    Team -k2, A -k + B -k <= 1. 3.2M checks, zero windows; the closest was
    -0.7c after fees. But the props have wide spreads. Median spread: moneyline
    1c, totals 6-11c, run line 8-12c, first-five / first-inning / extra
    innings 13-28c, with about 50 shares at the best bid.
14. **Futures arbitrage** (snapshot, inline check). Team champion <= its
    conference champion, NFL and NBA 2027: 65 pairs, closest -0.1c / -0.25c.
    Multi-outcome sum-of-prices check over 82 open events: 2 gaps, Waymo cities
    (4.7 shares, about $0.11) and OpenAI IPO market cap (71 shares, about $2.60,
    locked until end-2027).

## Goal (set 2026-10-29)
Arbitrage in ordinary markets (politics, weather, economics, ...) and in futures
(season champions, awards), all on Polymarket, all of which have no taker delay.

Tool: `arb_scan.py`, a snapshot scan of every open event for (1) outcomes
summing to under $1 (or bids over $1) and (2) nested futures (champion <=
conference). First full run, 2026-10-29: 2,100 open events, 763 multi-outcome,
**4 with a positive edge**:

| Edge | Trade | Event | Ends |
|---|---|---|---|
| +2.91c | buy all YES | How many cities will Waymo operate in by Dec 31? (4.7 shares at best price) | 2027-01-01 |
| +1.37c | buy all NO | SC-02 House Election Winner | **2026-11-04** |
| +0.40c | buy all YES | Brazil Annual Inflation 2026 | 2027-01-13 |
| +0.37c | buy all YES | OpenAI IPO Closing Market Cap (71 shares) | 2028-01-01 |

Nested futures: NFL closest -0.11c, NBA -0.25c, no arbitrage.

Before trading any hit: confirm it on the CLOB order book (Gamma prices are a
summary and can lag), check how many shares are available, and read the
resolution rules ("buy all YES" is only safe if exactly one outcome must win).
SC-02 is the interesting lead: it settles in 6 days, so the money isn't locked
for months.

Next for this goal:
1. Run `arb_scan.py` on a schedule for a week and log every gap (size, shares,
   how long it lasted). That shows how often gaps open and whether they're
   catchable, before building an order-sending bot.
2. Add more nested relationships: a team's league title vs. making the playoffs,
   division vs. conference, an election's party winner vs. its candidates.

## Next (sports props, parked)
Market-make MLB/NBA props (no delay) around a fair value derived from the 1c
moneyline and main total, quoting inside the 6-28c spreads and repricing when
the moneyline moves. The test should report fills, edge 30 s after each fill,
and P&L at $100 scale. The risk is the one that closed the earlier maker work:
too few fills, or fills only from better-informed traders.
