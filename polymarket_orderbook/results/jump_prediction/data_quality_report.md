# Data quality report -- LOB jump prediction

Generated 2026-09-15 17:16:30.
Raw recordings sampled at up to 4,000,000 records each.

## Raw event stream (source of truth, opened read-only)

| session | books | trades | tokens | markets | span h | obs/token |
|---|---|---|---|---|---|---|
| books_2026-08-27.jsonl.jsonl.xz | 3,828,390 | 0 | 126 | 7 | 8.4 | 30,384 |
| books_2026-08-28.jsonl.xz | 3,636,479 | 0 | 270 | 15 | 9.3 | 13,468 |
| books_2026-08-29.jsonl.xz | 3,836,318 | 0 | 270 | 15 | 3.1 | 14,209 |
| books_2026-08-30.jsonl.xz | 4,000,000 | 0 | 254 | 14 | 3.6 | 15,748 |
| books_2026-09-10.jsonl.xz | 3,912,188 | 3,508 | 90 | 5 | 4.5 | 43,469 |
| books_2026-09-11.jsonl.xz | 3,668,751 | 3,989 | 270 | 15 | 5.8 | 13,588 |
| books_2026-09-12.jsonl.xz | 3,825,721 | 3,065 | 270 | 15 | 2.9 | 14,169 |
| books_2026-09-13.jsonl.xz | 3,836,356 | 3,096 | 270 | 15 | 2.8 | 14,209 |

### Timestamp resolution

Interval between consecutive updates **of the same token**.

| session | p1 | p25 | p50 | p75 | p90 | p99 | mean | < 200ms | > 60s |
|---|---|---|---|---|---|---|---|---|---|
| books_2026-08-27.jsonl.jsonl.xz | 1 | 2 | **8** | 37 | 458 | 31,802 | 1,257 | **87.0%** | 0.47% |
| books_2026-08-28.jsonl.xz | 1 | 2 | **17** | 137 | 3283 | 71,986 | 3,072 | **77.0%** | 1.28% |
| books_2026-08-29.jsonl.xz | 1 | 2 | **9** | 42 | 429 | 21,836 | 989 | **87.0%** | 0.32% |
| books_2026-08-30.jsonl.xz | 1 | 2 | **12** | 49 | 567 | 18,966 | 986 | **85.6%** | 0.27% |
| books_2026-09-10.jsonl.xz | 1 | 2 | **10** | 33 | 129 | 6,664 | 400 | **92.0%** | 0.12% |
| books_2026-09-11.jsonl.xz | 1 | 3 | **17** | 62 | 1338 | 41,583 | 1,782 | **82.0%** | 0.46% |
| books_2026-09-12.jsonl.xz | 1 | 2 | **10** | 38 | 264 | 20,594 | 867 | **88.8%** | 0.31% |
| books_2026-09-13.jsonl.xz | 1 | 3 | **13** | 39 | 342 | 19,985 | 823 | **87.8%** | 0.28% |

All values in milliseconds. The **< 200ms** column is the share of updates that arrive faster than one grid slot: those events are collapsed by the 200ms resampling and are invisible to any model trained on it.

### Integrity

| session | crossed | locked | empty side | dup ts | backwards ts | full 10 levels | feed lag p50 |
|---|---|---|---|---|---|---|---|
| books_2026-08-27.jsonl.jsonl.xz | 560 | 5,028 | 7,196 | 840,881 | 1 | 92.6% | 85ms |
| books_2026-08-28.jsonl.xz | 38 | 286 | 3,940 | 713,740 | 23 | 96.4% | 143ms |
| books_2026-08-29.jsonl.xz | 22 | 220 | 9,680 | 891,413 | 0 | 88.9% | 122ms |
| books_2026-08-30.jsonl.xz | 9,424 | 2,206 | 13,842 | 748,052 | 4 | 83.4% | 113ms |
| books_2026-09-10.jsonl.xz | 62 | 262 | 72,648 | 569,974 | 0 | 66.5% | -136ms |
| books_2026-09-11.jsonl.xz | 5,514 | 1,618 | 68,656 | 628,496 | 10 | 73.5% | -200ms |
| books_2026-09-12.jsonl.xz | 4,950 | 2,800 | 31,190 | 663,696 | 0 | 72.9% | -133ms |
| books_2026-09-13.jsonl.xz | 338 | 490 | 89,010 | 632,176 | 0 | 67.0% | -1227ms |

Legacy-key collision check: **708** (slug, market_type, line) keys map to more than one token across the sampled records -- the defect documented in `FINDINGS_SERIES_KEY.md`. The current pipeline keys on `asset_id` as well, so these no longer merge; the count is reported to show the collision is real in the raw feed.

## Built 200ms grid datasets

Backward-only as-of resampling: each slot carries the most recent book at or before it. Silences beyond 60s are left as real gaps rather than filled.

| session | grid rows | in-game rows | series | markets | grid hours |
|---|---|---|---|---|---|
| books_2026-08-28 | 30,481,811 | 5,599,167 | 120 | 15 | 1,693 |
| books_2026-08-30 | 11,653,622 | 4,459,338 | 97 | 12 | 647 |
| books_2026-09-10 | 4,857,353 | 1,784,419 | 40 | 5 | 270 |
| books_2026-09-11 | 18,105,340 | 5,474,888 | 112 | 14 | 1,006 |
| books_2026-09-12 | 14,404,892 | 5,269,717 | 120 | 15 | 800 |
| books_2026-09-13 | 11,812,566 | 5,414,754 | 112 | 14 | 656 |

Total 91,315,584 grid rows, 28,002,283 in-game.

