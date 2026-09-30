# Cross-market study: does the rest of the game tell you a large jump is coming?

Every earlier model saw one market at a time. These five methods see **every market of the game at
once** -- moneyline, spreads and totals -- and ask whether that predicts a large upward YES jump on
one market **2-8 s ahead**. The 2 s gap is deliberate: Polymarket delays marketable orders on live
sports markets (reported 1-3 s), so a warning shorter than that cannot be traded as a taker.

| method | script | what it learns |
|---|---|---|
| 1 trees on summary features | `m1_trees.py` | own market vs + other markets vs + game state; the difference is the value of the other markets |
| 2 multi-market CNN | `m2_cnn.py` | all markets' last 20 s stacked as channels (target market first) + game state |
| 3 attention across markets | `m3_attention.py` | shared per-market CNN encoder, transformer over the set of markets, target token + game state |
| 4 lead-lag | `m4_leadlag.py` | Granger (1-5 s and tradable 3-8 s lags), who moved first around each large jump, Hawkes cross-excitation |
| 5 unsupervised | `m5_unsupervised.py` | PCA / k-means "modes" of the whole game's book, autoencoder "unusual book" score, and whether they add to a tree |

## Scoring (same for methods 1, 2, 3 and 5)

Every second of the 7 test sessions (09-21 .. 09-27) is scored. At each cut (top 5% .. 0.1% of
scores) a trade is the first second a market clears it, then 10 s out on that market. Each trade buys
the YES ask **1 s after the signal** (needs >= $10 at the ask) and sells the bid once the book has been
quoted and still for 3 s (at least 5 s after entry; 90 s timeout); spread and fees are charged.
The table, per method and cut: trades, how many became large jumps, the rest, P&L per trade (overall,
on the large ones, on the rest), and the break-even share and count.

Results land in `results/large_jumps/cross_market/` (`reports.csv`, feature importances, lead-lag
tables, clusters). Panels and smoke outputs go to the git-ignored `research/large_jumps/cache/xm/`.

## Run

    sh research/large_jumps/cross_market/run_all.sh --smoke    # check it works, ~5 min
    sh research/large_jumps/cross_market/run_all.sh            # full run, several hours on a CPU

Needs the grid (`data/jump/feat_*_trimmed.parquet`, `lob_*_trimmed.npy`), `results/makinen/single_jumps/marks.csv`
and `results/large_jumps/inputs/`. Pricing uses the panel's top of book (1 level), not a book walk.
