# Which Mäkinen et al. features this dataset can support

Source: `data/jump/feat_books_*_trimmed.parquet` + `lob_books_*_trimmed.npy`
(200ms in-game grid, 10 depth levels per side), reduced to 1-minute bars by
`research/makinen/build_panel.py`. 69 features per minute, 120 minutes per
sample.

The single most important line in this table is the last block: **this is a
periodic snapshot feed, not an order-event feed.** The recorder stores the book
state, not the messages that produced it, so every order-flow intensity feature
in the paper is unavailable. They are listed as absent rather than
approximated, because a "cancellation rate" inferred from depth decreasing
between two snapshots cannot distinguish a cancellation from a trade and would
be a fabricated feature.

| Paper feature | Available? | My implementation | Source columns | Notes |
|---|---|---|---|---|
| Bid/ask prices, 10 levels | **Yes** | `bid_dist_1..5`, `ask_dist_1..5` | `lob_*.npy` ch. 0/2 | Stored as distance from that row's mid in ticks, which is the representation the study wants; levels 6–10 exist but were dropped from the model input to keep the flattened MLP tractable |
| Bid/ask quantities, 10 levels | **Yes** | `bid_usd_1..10`, `ask_usd_10` | `lob_*.npy` ch. 1/3 | `log1p` dollars; a level that does not exist is stored as 0 size and carries an explicit mask |
| Spread | **Yes** | `spread_ticks` | grid | |
| Mid-price | **Yes** | `mid` | `(best_bid+best_ask)/2` | verified identical to the feed's own `midpoint` in 100.0% of raw rows |
| Price gaps between levels | **Yes** | `bid_gap_1..3`, `ask_gap_1..3` | derived | zeroed where either level is absent |
| Price/volume averages | **Yes** | `log_bid_depth`, `log_ask_depth`, `log_total_depth`, `l1_bid_usd`, `l1_ask_usd` | derived | |
| Bid–ask differences | **Yes** | `imbalance`, `imbalance_l1` | derived | (bid−ask)/(bid+ask) |
| Book shape / concentration | **Yes** (extra) | `hhi_bid/ask`, `entropy_bid/ask`, `slope_bid/ask`, `reach_bid/ask`, `levels_bid/ask`, `near_frac_*_2t` | derived | not in the paper; included because the project's earlier study found book slope to be its most informative geometry feature |
| Changes / derivatives | **Yes** | `d_mid`, `d_spread_ticks`, `d_imbalance`, `d_log_total_depth`, `d_log_bid_depth`, `d_log_ask_depth` | derived | 1-minute differences, `NaN` across a gap so a change never bridges missing data |
| Short-horizon realised volatility | **Yes** | `rv_5`, `rv_15` | derived | trailing rolling sd of 1-minute mid changes |
| Time of day | **Yes** | `tod_sin`, `tod_cos` | timestamp | cyclic encoding; these contracts trade continuously, so there is no exchange open/close |
| **New limit-order arrival counts/rates** | **No** | — | — | snapshot feed carries no order messages |
| **Cancellation counts/rates** | **No** | — | — | same; depth decreasing between snapshots cannot be attributed to a cancel vs a trade |
| **Market-order / aggressive-trade intensities** | **Partly, unused** | — | `books_*.jsonl` has ~3k trade prints per session | far too sparse at 1-minute resolution across 601 contracts to form a rate; excluded rather than approximated |
| **Bid-side vs ask-side order-flow imbalance** | **No** | — | — | requires signed order flow, not available |
| **Changes in those intensities** | **No** | — | — | follows from the above |

## Consequence for the replication

The paper's feature set splits into book-state features and order-flow
intensity features. **This dataset supports the first group in full and the
second group not at all.** Any result here is therefore a test of whether
*book state and its recent trajectory* predict jumps — a strictly weaker input
than the paper had. If the models underperform the paper's reported numbers,
missing order flow is one of the candidate explanations and should not be
confused with the method failing.

## Deliberate additions

Book-shape features (slope, concentration, entropy, reach) are not in the
paper. They were added because the earlier study in this repository
(`results/jump_prediction/summary.md`) found book slope to carry more signal
than depth imbalance, which is the feature most microstructure work reaches
for first. They are flagged here so the input set is not mistaken for a
faithful reproduction of the paper's.
