# Audit: `polymarket_sports` — is this data usable?

2026-09-16. Everything below was measured from the files, not taken from the
existing `run_manifest.json`. Files opened read-only.

## Verdict

**Yes — for about 12% of it, and that subset is arguably better than the MLB
in-play data for the question this project has ended up asking.** The other
88% is dead longshot futures that never move. You must filter; used whole, this
dataset would produce nothing.

## What is here

| | |
|---|---|
| orderbook | 21 daily parquet files, **527,187,384 rows**, 2026-03-06 … 2026-03-26 |
| snapshots | 21 files, full-depth book snapshots, ~41 per token per day |
| trades | 1 file, 254,770 rows |
| metadata | `sports_markets.parquet`, 7,120 markets |
| size | ~4.0 GB orderbook + 0.32 GB snapshots |

Provenance (`run_manifest.json`): derived from a Kaggle dataset
(`marvingozo/polymarket-tick-level-orderbook-dataset`), filtered to sports by
**"conservative keyword fallback (no explicit Sports metadata label found)"**.
The `category` column is empty for all 7,120 rows; classification rests on
keywords in the question text. Treat the sports label as heuristic.

### orderbook schema

`timestamp_received`, `timestamp_created_at` (int64 ms), `market_id`,
`token_id`, `best_bid`, `best_ask`, `mid_price`, `spread`,
`change_price`, `change_size`, `change_side` (BUY/SELL).

This is an **event-level delta stream with top-of-book attached** — not a
level-indexed depth table.

## The good

* **Millisecond timestamps.** Median gap between consecutive updates of the
  same token is **159 ms**; 52.5% of updates arrive under 200 ms. This is
  *finer than the 200 ms grid* used for the MLB work, and it is the resolution
  the ~6-second pre-jump finding actually needs.
* **`mid_price == (best_bid + best_ask)/2` in 100.0% of rows.** Verified, not
  assumed. L1, spread and mid can be used directly with no reconstruction.
* **Full-depth snapshots exist**: median **13 bid / 78 ask levels**, 98% of
  snapshots carry ≥10 ask levels — deeper than the 10-level cap in the MLB
  tensors.
* **21 consecutive calendar days.** This matters more than anything else: the
  MLB study's binding constraint was a single held-out session with 109 test
  positives, which left every architecture comparison unresolved. 21 days
  supports a genuine chronological train/val/test split.
* **24-hour continuous coverage**, so pre-match and in-match are both present.

## The live subset — the only part worth modelling

Filter: ≥3,000 updates/day, daily mid range ≥0.10, median spread ≤5c, median
mid in [0.05, 0.95].

| | |
|---|---|
| **live token-days** | **1,209** |
| **distinct live tokens** | **1,039** |
| rows in the live subset | **62,173,936 (11.8% of all)** |
| median daily mid range | 0.50 (i.e. 50 cents) |
| median spread | 0.026 |
| median updates/day | ~15,000 |

These are **in-play esports and match markets**:

* `LoL: WLGaming Esports vs The Bandits - Game 1 Winner`
* `Counter-Strike: 33 vs Nemiga - Map 1 Winner`
* `Dota 2: PARIVISION vs Tundra Esports (BO3) - PGL Wallachia Group Stage`
* `Will Chelsea reach the UEFA Champions League quarter-finals?`

Structurally the same object as the MLB in-play moneylines: a binary contract
repricing on discrete game events. **1,039 live contracts vs 601 MLB
contracts.**

Movement at second resolution on the live subset: |1-second mid change| p99
**0.015**, p99.9 **0.15**; **0.84% of seconds carry a ≥2c move**. That is real,
frequent, jump-shaped price action.

## The bad — five things you must handle

1. **88% of the data is dead.** Across all 3,114 tokens on a typical day the
   *median* token has a daily mid range of **0.010** and changes its mid on
   1.6% of updates. Median mid is **0.0205** — these are deep longshots
   ("Will X win the Stanley Cup" at 2 cents). Any unfiltered analysis drowns
   in them.

2. **Depth does not reconstruct cleanly.** Applying the delta stream to a
   snapshot (treating `change_size` as the absolute size at `change_price`)
   reproduces the file's own best bid/ask on only **72.8%** of deltas. Either
   the delta semantics differ from that assumption or the stream is
   incomplete — plausibly because the sports filter retained only ~7% of raw
   rows and `raw_coverage` in the manifest is **0.518**. Until this is
   resolved, **use L1/spread/mid (exact) and treat depth as unavailable.**

3. **One token per market.** 3,114 tokens across 3,114 markets, exactly one
   each, and every snapshot carries `side: YES`. The complementary NO leg is
   absent, so the YES/NO mirror check that validated the MLB jump labels
   cannot be run here.

4. **Trades are unusable for microstructure.** 254,770 rows spanning
   **2025-07-17 to 2026-03-26** (eight months) over 1,303 assets — a median of
   **13 trades per asset**. This is not the order flow the Mäkinen feature set
   wants; it will not support trade or market-order intensity features.

5. **No game clock.** The MLB work relied on `data/game_windows.json` for first
   pitch and final out, which was what fixed the volatility-regime problem. No
   equivalent exists here, so in-play windows must be inferred from activity
   (e.g. sustained update rate) and that inference needs its own validation.

Minor: the first five days (03-06 … 03-10) yield only 4–21 live tokens each,
versus 28–145 from 03-11 onward. Coverage ramps up; weight the later days.

## Recommended use

1. Filter to the live subset (`reports/live_subset_by_day.csv` has per-day
   counts). Work with L1, spread and mid only.
2. Re-run Lee–Mykland at **1-second or 5-second bars**, not 1-minute. The
   markets move on a seconds clock and the earlier finding was that the
   precursor lives ~6 seconds out; minute bars cannot see it.
3. Use the 21-day span for a real chronological split — this is the single
   biggest improvement over the MLB dataset.
4. Resolve the delta semantics before attempting any depth feature. Until then
   do not claim depth-based results from this source.
