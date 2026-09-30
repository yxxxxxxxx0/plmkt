# Cross-market study: does the rest of the game tell you a large jump is coming?

Every earlier model saw one market at a time. These five methods see **every market of the game at
once** -- moneyline, spreads and totals -- and ask whether that predicts a large upward YES jump on
one market **more than 200 ms and at most 5 s ahead**.

| method | script | what it learns |
|---|---|---|
| 1 trees on summary features | `m1_trees.py` | own market vs + other markets vs + game state; the difference is the value of the other markets |
| 2 multi-market CNN | `m2_cnn.py` | all markets' last 20 s stacked as channels (target market first) + game state |
| 3 attention across markets | `m3_attention.py` | shared per-market CNN encoder, transformer over the set of markets, target token + game state |
| 4 lead-lag | `m4_leadlag.py` | Granger (1-5 s and 3-8 s lags), who moved first around each large jump, Hawkes cross-excitation |
| 5 unsupervised | `m5_unsupervised.py` | PCA / k-means "modes" of the whole game's book, autoencoder "unusual book" score, and whether they add to a tree |

## Scoring (same for methods 1, 2, 3 and 5)

Every 200 ms of the 7 test sessions (09-21 .. 09-27) is scored. At each cut (top 5% .. 0.1% of
scores) a trade is the first sample a market clears it, then 10 s out on that market. Each trade buys
the YES ask after a fill delay and sells the bid once the book has been quoted and still for 3 s
(at least 5 s after entry; 90 s timeout); spread and fees are charged; >= $10 must be at the ask.

**P&L is shown at three fill delays: 0.2 s, 1 s and 3 s.** Polymarket's docs say marketable orders on
live sports markets wait a configured delay before matching (reported as 1-3 s); the 0.2 s column is
what the signal is worth if that delay does not bite, the 1 s and 3 s columns if it does.

The table, per method and cut: trades, trades per day, how many became large jumps, the rest, and at
each delay the P&L per trade (overall, on the large ones, on the rest) and the break-even count.

## Settings (environment variables)

| variable | default | meaning |
|---|---|---|
| `XM_H_LO_MS`, `XM_H_HI_MS` | 200, 5000 | label: a large jump starts in (t + H_LO, t + H_HI] |
| `XM_EVERY_MS` | 200 | sample spacing; 1000 makes the deep models ~5x faster to score |
| `XM_DELAYS_MS` | 200,1000,3000 | fill delays the trades are priced at |

Example: `XM_H_LO_MS=2000 XM_H_HI_MS=8000 sh research/large_jumps/cross_market/run_all.sh`

## Run

    sh research/large_jumps/cross_market/run_all.sh --smoke    # check it works (2 sessions, 12 games)
    sh research/large_jumps/cross_market/run_all.sh            # full run: roughly 10 h on a CPU at 200 ms

Results land in `results/large_jumps/cross_market/` (`reports.csv`, feature importances, lead-lag
tables, clusters). Panels, logs and smoke outputs go to the git-ignored `research/large_jumps/cache/xm/`.

Needs the grid (`data/jump/feat_*_trimmed.parquet`, `lob_*_trimmed.npy`), `results/makinen/single_jumps/marks.csv`
and `results/large_jumps/inputs/`. Pricing uses the panel's top of book (1 level), not a book walk.
