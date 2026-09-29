# Reproducing this on another machine

> **Note (2026-09-21).** The codebase was tidied: the implementations of the
> closed methods were deleted (recoverable at `40437ab`, catalogued in
> `METHODS_AND_EXPERIMENTS.md`), the unrelated `mlbmodel/` project was removed,
> and with it the orphaned downloads — `kaggle_sports_download/`,
> `kaggle_cross_market_download/`, `kaggle_metadata_full/`,
> `cross_market_universe/` and `data/raw/` (12 GB in total). The Kaggle ones
> are re-downloadable; `data/raw/` was the Elo model's API cache. **The
> recordings were untouched and remain the irreplaceable data** (since
> 2026-09-29 they are all in `polymarket_orderbook/data/live/`).

The repository holds **code and results only — about 13 MB**. None of the data
is in git: the 7.3 GB of recordings are deliberately excluded, and the derived
grids are another 57 GB. This file says exactly what to carry, what to re-download, and
what to rebuild.

Repo: https://github.com/yxxxxxxxx0/plmkt-jump-study

## Where everything lives on the recording machine

Root is `C:\Users\JustinCHENG\Documents\plmkt`.

| path | size | in git? | how to get it back |
|---|---|---|---|
| `polymarket_orderbook/data/live/` | **7.3 GB** | no | **COPY IT — irreplaceable** |
| `polymarket_orderbook/data/jump/` | 57 GB | no | rebuild (below) |
| `polymarket_orderbook/research/*/cache/` | 3.8 GB | no | rebuild (below) |
| `polymarket_orderbook/data/game_windows.json` | 40 KB | **yes** | already cloned |
| `polymarket_orderbook/data/excluded_matches.json` | 5 KB | **yes** | already cloned |
| all `research/**/*.py` and `results/**/*.md` | 13 MB | **yes** | already cloned |

## The only thing you must physically copy: 7.3 GB, in ONE folder

    polymarket_orderbook/data/live/

Every session the recorder has made is in it. The August sessions used to sit
in a second folder at the repo root (`data/live/`); they were moved here on
2026-09-29, so there is no longer a second place to miss. It is your own
recorder's output: nobody can download or regenerate it. Everything else in
this project is derived from it or is a public download.

Contents, so you can check nothing is missed (all xz, compressed losslessly by
`compress_raw.py` after a full bit-for-bit round trip):

    books_2026-08-27.jsonl.jsonl.xz              the doubled extension is real
    books_2026-08-28 .. 08-30.jsonl.xz
    books_2026-09-10 .. 09-13.jsonl.xz
    books_2026-09-17 .. 09-27.jsonl.xz           19 sessions, 6.5 GB in total
    top_of_book_2026-09-10 .. 09-27.csv.xz       15 sessions, 0.9 GB (no CSV in August)
    *.health.json                                per-session recording checks
    sessions.json                                recording manifest

Read the archives in place: `jump_data.py` and the research readers open `.xz`
directly. `decompress_raw.py` restores a `.jsonl` when a tool needs one (the
viewer rebuild, `rebuild_all_slates.py`, still does).

An external drive or `robocopy` is the sane way to move it. If you only want
the minute-bar studies working, the `top_of_book_*.csv.xz` files are enough;
the `books_*.jsonl.xz` files are needed to rebuild the 10-level grid.

## Step by step on the new machine

    # 1. code
    git clone https://github.com/yxxxxxxxx0/plmkt-jump-study.git plmkt
    cd plmkt

    # 2. environment
    python -m venv .venv
    .venv\Scripts\python -m pip install numpy pandas pyarrow scikit-learn ^
        lightgbm xgboost torch matplotlib scipy requests tabulate

    # 3. copy polymarket_orderbook/data/live across by hand (7.3 GB), then:

    # 4. rebuild the 200ms grid -- ~57 GB out, one command per session
    cd polymarket_orderbook
    ..\.venv\Scripts\python jump_data.py --raw data/live/books_2026-09-10.jsonl.xz
    #   ...repeat per session. This writes data/jump/feat_*.parquet and lob_*.npy

    # 4a. scan the raw feeds -- both read data/live directly, and both write
    #     into data/ (gitignored), so a fresh clone must run them
    ..\.venv\Scripts\python continuity_scan.py data/live/books_2026-09-10.jsonl.xz
    #   ...repeat per session, then:
    ..\.venv\Scripts\python continuity_scan.py --exclusions   # -> data/excluded_matches.json
    ..\.venv\Scripts\python crossed_scan.py    # ghost-level stretches -> data/crossed/
    #     (all sessions in parallel, ~10 min). The jump marker and the
    #     cross-market study refuse to run without it: the grid hides a crossed
    #     book behind the last clean one, so only the raw feed can show it.

    # 4b. trim -- REQUIRED, and it is what applies the broken-match rule.
    #     Everything downstream reads feat_*_trimmed.parquet, and the session
    #     lists are discovered from those files, so a session that is built but
    #     not trimmed is silently invisible to every analysis.
    ..\.venv\Scripts\python trim_sessions.py

    # 4c. gate -- do not trust a session that has not passed this
    ..\.venv\Scripts\python audit_datasets.py
    ..\.venv\Scripts\python test_jump_data.py
    ..\.venv\Scripts\python test_match_coverage.py

    # 5. rebuild the research caches
    ..\.venv\Scripts\python research/lee_mykland/build_minute_bars.py
    ..\.venv\Scripts\python research/makinen/build_panel.py
    ..\.venv\Scripts\python research/makinen/label_jumps.py
    ..\.venv\Scripts\python research/makinen/collapse_events.py

    # 6. reproduce the headline results
    ..\.venv\Scripts\python research/makinen/collapse_classify.py   # ROC 0.756
    ..\.venv\Scripts\python research/makinen/collapse_tree.py       # GBM 0.627
    ..\.venv\Scripts\python research/makinen/live_replay.py         # ROC 0.665

    # 7. single upward jumps that break even, one plot per match (~25 min)
    ..\.venv\Scripts\python research/makinen/test_single_jumps.py
    ..\.venv\Scripts\python research/makinen/mark_single_jumps.py
    #   redraw only, from the saved marks:  ... mark_single_jumps.py --from-marks

    # 8. cross-market arbitrage and lead-lag (~8 min)
    ..\.venv\Scripts\python research/makinen/cross_market_timing.py

## For the esports dataset

`polymarket_sports/` (the esports tick data, 4.1 GB) and the pipeline that
built it, `kaggle_sports_pipeline/`, were deleted from disk on 2026-09-29.
Nothing in the MLB work reads them; only `research/sports/` does. To restore:

* the pipeline code: `git checkout archive/kaggle-sports-pipeline -- kaggle_sports_pipeline`
  (it was gitignored, so that branch holds the only copy)
* the reports, audit and run manifest: `git checkout 7936a34 -- polymarket_sports/reports polymarket_sports/metadata`
* the data: run the pipeline in a Kaggle notebook against
  `marvingozo/polymarket-tick-level-orderbook-dataset` (see its README)
* then: `research/sports/build_bars.py` and `research/sports/run_resolution.py`

Read `polymarket_sports/reports/DATA_AUDIT.md` first — only ~12% of that
dataset is usable and the filter matters.

## Where to start reading

1. `polymarket_orderbook/results/PRICE_JUMP_REPORT.md` — what was achieved
2. `polymarket_orderbook/results/RULED_OUT.md` — what is closed, and why
3. `polymarket_orderbook/results/makinen/lee_mykland_method.md` — the detector
4. `polymarket_orderbook/results/makinen/collapse/SUMMARY.md` — the best model

## Two things that will bite you

* **Rebuilding `data/jump` needs the `.xz` books, not the CSVs.** The CSVs carry
  only level 1; the 10-level grid comes from `books_*.jsonl.xz`.
* **`data/jump` is 57 GB and `research/*/cache` is 4 GB.** Have ~70 GB free
  before starting step 4.
