# Polymarket strategy screen for a $50 bankroll

> **Note (2026-09-21).** Some scripts named below were deleted in the codebase
> tidy. They are recoverable at commit `40437ab` (`git show 40437ab:<path>`),
> and what each one did is recorded in `METHODS_AND_EXPERIMENTS.md`.

Research date: 2026-09-08. This is a research screen, not a promise of profit.

## Bottom line

No tested strategy is ready for real-money deployment. The best next project is
a paper-traded, maker-only microstructure strategy in short-duration markets,
with a hard adverse-selection filter. With $50, reward farming and diversified
market making are too capital intensive.

## Tests completed

### 1. Binary complete-set arbitrage: fails the live screen

For a binary market, one YES plus one NO pays exactly $1. A risk-free gross
opportunity therefore requires `best_ask_yes + best_ask_no < 1`.

Live public-API snapshot:

- 1,000 active binary markets fetched
- 856 had executable asks on both outcomes
- 0 had a gross ask sum below $1
- best observed ask sum: $1.001, before taker fees
- typical reported minimum-order cost for the best rows: about $5 for both legs

Verdict: mechanically valid and compatible with $50, but no opportunity was
available. Keep it as a scanner, not as an expected source of daily income.

### 2. Cancellation / liquidity-collapse signal: detects activity, not direction

The retained event study contains 3,759 selected windows from 51 MLB games over
four dates. Models were fit only on earlier dates and evaluated on the next date.

| signal | mean test AUC | interpretation |
|---|---:|---|
| book imbalance | 0.572 | weak |
| directional withdrawal + imbalance | 0.637 | modest separation |
| symmetric withdrawal activity | 0.924 | detects an impending repricing episode |
| all retained features | 0.970 | detects selected jump windows extremely well |

This is **not** a tradeable 0.97-AUC result. The archive compares upward-jump
windows with sampled flat controls and omits the full population of downward
moves. Both sides of the book collapse together. It can say “the book is about
to move/become unsafe,” but cannot yet say which side to buy. No fill-based PnL
can be computed from the retained aggregate alone.

Verdict: useful as a maker cancel/risk-off filter. Not approved as a directional
entry signal.

### 3. Favourite-longshot / public-data directional betting: fails

The existing walk-forward sports study already tests this family much more
deeply: 18,075 held-out games across MLB, NHL, NBA and NFL. Every best model
underperformed the closing line. In MLB, betting every underdog returned -1.7%,
dogs above 2.40 returned -2.0%, and favourites returned -5.7% against sportsbook
prices. Apparent model profits disappeared under timestamp and stationarity
controls.

Verdict: do not spend the $50 bankroll on generic favourites, longshots, or the
existing pre-game model.

## Strategies ranked for this bankroll

### 1. Maker-only short-duration quoting, paper trade first

Quote one small order on one side only when the spread is at least two ticks,
the book is stable, and symmetric withdrawal activity is low. Cancel immediately
when activity rises. Use a market-derived fair value and avoid carrying inventory
through scheduled announcements or decisive sports events.

Why first: makers are not charged trading fees under the current CLOB V2 model,
and filled maker orders may receive rebates. Why not live yet: the present data
does not estimate queue position, fill probability, or post-fill adverse
selection. Classical market-making theory explicitly treats inventory and fill
risk; $50 leaves little room to diversify those risks.

Initial paper limits:

- $1 maximum loss-equivalent per quote
- $3 maximum total open exposure
- one market at a time
- no averaging down
- require at least 1,000 simulated fills and positive PnL after marking exits to
  the future executable bid/ask

### 2. Logical/cross-market arbitrage scanner

Extend the included binary scanner to mutually exclusive and exhaustive market
families: nomination winners, election winners, exact ranges, and nested
thresholds. Evaluate executable depth on every leg and include fees. This has a
sound mechanism, but simultaneous execution and resolution wording are serious
risks. Long-dated opportunities are unattractive for $50 because capital turns
over too slowly.

### 3. External-anchor latency

For short scheduled events, compare Polymarket with a faster authoritative feed
or a liquid external venue. Trade only deviations that exceed spread, fee,
latency and model-error buffers. The existing pre-game sportsbook comparison is
not enough; this requires synchronized live feeds and executable quotes.

### Rejected for now

- liquidity-reward farming: qualifying sizes can be around 50 shares and rewards
  are competitive/pro-rata, too large and uncertain for $50
- taker momentum: spread, dynamic taker fees and adverse selection dominate
- copy-trading wallets: historical wallet performance is selection-biased and
  public trades arrive after the relevant decision
- long-duration calibration bets: too much capital lock-up for a small bankroll

## Research basis

- Cont, Kukanov and Stoikov find that short-horizon price changes are primarily
  related to order-flow imbalance, including cancellations:
  https://academic.oup.com/jfec/article-abstract/12/1/47/816163
- Avellaneda and Stoikov formalize spread capture versus inventory and execution
  risk in limit-order market making:
  https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf
- Recent prediction-market research warns that calibration effects depend on
  domain, time and venue rather than providing a universal longshot rule:
  https://arxiv.org/abs/2602.19520
- Prediction-market designs with related or interval securities explicitly need
  arbitrage removal across submarkets:
  https://arxiv.org/abs/2102.07308
- Polymarket's current fee/rebate and reward rules:
  https://docs.polymarket.com/market-makers/maker-rebates
  https://docs.polymarket.com/market-makers/liquidity-rewards

## Reproducibility

Run:

```powershell
.\.venv\Scripts\python.exe polymarket_orderbook\strategy_screen.py
.\.venv\Scripts\python.exe polymarket_orderbook\market_arb_scan.py --max-markets 1000
```

The first command uses retained local data. The second is a read-only live scan;
its result changes with the order book.
