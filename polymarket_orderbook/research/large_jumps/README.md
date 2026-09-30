# Large jumps: can they be detected, predicted and traded?

Study of 2026-09-30 on the 18 recorded sessions (204 games, 11,884 marked break-even jumps from
`research/makinen/mark_single_jumps.py`). Everything is on the recorded YES token; trades are
$10, taker only, with the spread and the sports fee on both legs.

## Headline

**Not tradable with information available before the pitch.** The best selection makes about 5% of
trades large jumps; break-even needs about 34%. Every held-out session loses money.

| step | result |
|---|---|
| "large" threshold | log-odds move >= 0.46: above it a jump is more likely than not a scoring play (95% CI 0.43-0.49; AUC 0.81 vs statsapi scoring plays) |
| book before large vs other jumps | no difference before entry (all AUC <= 0.54); large ones only last longer and empty the book more once under way |
| when the book changes | near-touch depth drains ~75% in the last 4-5s before a jump while prices are still; identical for large and small |
| cross-market confirmation | "moves in >=2 other markets" does NOT mean real (19.7% vs 23.7% real); only moves on a quoted book help |
| depth-drain alert | fires ~59/market-hour (about once per pitch); a jump follows ~9% of alerts, a large one ~0.45% |
| + game state (runners, outs, count, RE24, margin) | top-1% alert precision for any real jump 23% -> 36% (leak check with state 10s older: unchanged) |
| trade confirmation | the confirming trade *is* the jump: only ~4% of marks get one within 2s, and half the move is gone by entry |
| direction | "up if the YES team is batting": 79% on large real jumps (moneyline+spread), 72-85% in every session; totals: 82% of large jumps go up. Drain asymmetry carries ~no direction |
| combined backtest, large jumps only | top 0.25% of alerts: 10 of 192 trades were large jumps (+10.7t each), 169 had no upward jump (-6.2t each); -4.7t per trade, CI [-5.5, -4.0] |
| event-level replay (raw books, entry at alert+50ms) | same as the 200ms grid within 0.1 tick: -4.1t (top 5%), -4.65t (top 0.25%) |
| CNN vs trees, learning curve | trees win at every size (large-jump AUC 0.87 vs 0.85 hybrid CNN vs 0.70 book-only CNN); more data helps slowly (top-1% large precision 3% -> 5% over 10x games) |

Why it cannot reach 34%: whether *this* pitch produces a run is close to random even in the best game
states. Information that could change that is news (the play's outcome) arriving before the book
reprices, which statsapi does not provide (it lags the market by ~8s on `endTime`).

## Alert used throughout

On a YES token's 200ms grid: the book has been still for 5s (bid and ask each within 1c, spread <=
normal + max(2, normal), no gaps), and the dollars within 5 ticks of the mid are now < 25% of their
value 4s earlier. 10s debounce; not in the first 2 minutes of a series.

## Run order (from `polymarket_orderbook/`)

Inputs (small, committed under `results/large_jumps/inputs/`):

    python research/large_jumps/fetch_plays.py      # statsapi plays for any game missing from results/makinen/mlb_plays
    python research/large_jumps/fetch_state.py      # per-pitch game state timeline -> game_state.csv
    python research/large_jumps/fetch_tokens.py     # YES/NO token pairs and team sides -> token_pairs.csv, yes_meta.csv
    sh     research/large_jumps/extract_trades.sh   # trades from the raw recordings -> trades/*.csv

Analyses (intermediates go to `research/large_jumps/cache/`, git-ignored):

    thr2.py                     large-jump threshold against scoring plays
    micro.py, cmp.py            book features of large vs other jumps
    es.py, es2.py               event study around the jump start
    xmkt.py, xmkt2.py           cross-market confirmation hypothesis
    prec2.py, alertpnl2.py      depth-drain alert precision and naive P&L (needs xmkt.py)
    alerts_ds.py, alerts_an.py  per-alert dataset and what separates converting alerts
    newmodel.py                 + game state and trades (needs alerts_ds.py)
    capture.py                  trade confirmation and latency
    direction.py, direction_an.py  batting-team and drain direction
    combined.py, large.py       held-out backtests (large.py: large jumps, book-walked fills, stabilisation exit)
    replay.py select|extract <session>   event-level replay of large.py's picks from the raw .jsonl.xz
    cnn_data.py, cnn_train.py   CNN tensor and learning curves -> results/large_jumps/learning_curve/

Limits: trades exist only from 09-10; run-expectancy values are approximate league averages;
"large" is defined on the realised move; the fill model assumes you are first to the ask.
