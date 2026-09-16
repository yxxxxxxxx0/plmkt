# Reproducing this on another machine

The repository holds **code and results only — about 13 MB**. None of the data
is in git: 14 GB of it was deliberately excluded, and the derived grids are
another 37 GB. This file says exactly what to carry, what to re-download, and
what to rebuild.

Repo: https://github.com/yxxxxxxxx0/plmkt-jump-study

## Where everything lives on the recording machine

Root is `C:\Users\JustinCHENG\Documents\plmkt`.

| path | size | in git? | how to get it back |
|---|---|---|---|
| `polymarket_orderbook/data/live/` | **12.15 GB** | no | **COPY IT — irreplaceable** |
| `data/live/` (repo ROOT, August sessions) | **2.25 GB** | no | **COPY IT — irreplaceable** |
| `polymarket_orderbook/data/jump/` | 37.22 GB | no | rebuild (below) |
| `polymarket_sports/orderbook/` | 4.00 GB | no | re-download from Kaggle |
| `polymarket_sports/snapshots/` | 0.32 GB | no | re-download from Kaggle |
| `polymarket_sports/trades/` | 1.8 MB | no | re-download from Kaggle |
| `polymarket_orderbook/research/*/cache/` | 3.99 GB | no | rebuild (below) |
| `kaggle_sports_download/` | 4.33 GB | no | re-download (it is the zip) |
| `polymarket_orderbook/data/game_windows.json` | 20 KB | **yes** | already cloned |
| all `research/**/*.py` and `results/**/*.md` | 13 MB | **yes** | already cloned |

## The only thing you must physically copy: 14.40 GB, in TWO folders

The recorder writes to two places, and it is easy to miss the second one:

    polymarket_orderbook/data/live/    12.15 GB   September sessions
    data/live/                          2.25 GB   August sessions (repo ROOT)

Both are your own recorder's output. Neither can be downloaded or regenerated
by anyone. Everything else in this project is derived from them or is a public
download.

Contents of each, so you can check nothing is missed:

    # repo ROOT -- data/live/            (2.25 GB, August sessions)
    books_2026-08-27.jsonl.jsonl.xz      0.31 GB
    books_2026-08-28.jsonl.xz            0.76 GB
    books_2026-08-29.jsonl.xz            0.72 GB
    books_2026-08-30.jsonl.xz            0.46 GB

    # polymarket_orderbook/data/live/    (12.15 GB, September sessions)
    books_2026-09-10..13.jsonl.xz       ~1.46 GB   full 10-level books
    top_of_book_2026-09-10..13.csv      ~6.30 GB   event-level L1 feed
    books_2026-09-16-late2.jsonl         4.10 GB   salvage run, uncompressed
    top_of_book_2026-09-16-late2.csv     0.28 GB
    sessions.json                                  recording manifest

The August archives sit at the root because the recorder has always written
there as well -- the same gap that once left 133 GB visible to `git add -A`
(see .gitignore). It is easy to copy only the September folder and silently
lose four sessions.

An external drive or `robocopy` is the sane way to move it. If you only want
the newer studies working, the four September `top_of_book_*.csv` files are
enough; the `.xz` books are needed only to rebuild the 10-level grid.

## Step by step on the new machine

    # 1. code
    git clone https://github.com/yxxxxxxxx0/plmkt-jump-study.git plmkt
    cd plmkt

    # 2. environment
    python -m venv .venv
    .venv\Scripts\python -m pip install numpy pandas pyarrow scikit-learn ^
        lightgbm xgboost torch matplotlib scipy requests tabulate

    # 3. copy BOTH data/live folders across by hand (14.40 GB total), then:

    # 4. rebuild the 200ms grid -- ~37 GB out, one command per session
    cd polymarket_orderbook
    ..\.venv\Scripts\python jump_data.py --raw ../data/live/books_2026-09-10.jsonl
    #   ...repeat per session. This writes data/jump/feat_*.parquet and lob_*.npy

    # 5. rebuild the research caches
    ..\.venv\Scripts\python research/lee_mykland/build_minute_bars.py
    ..\.venv\Scripts\python research/makinen/build_panel.py
    ..\.venv\Scripts\python research/makinen/label_jumps.py
    ..\.venv\Scripts\python research/makinen/collapse_events.py

    # 6. reproduce the headline results
    ..\.venv\Scripts\python research/makinen/collapse_classify.py   # ROC 0.756
    ..\.venv\Scripts\python research/makinen/collapse_tree.py       # GBM 0.627
    ..\.venv\Scripts\python research/makinen/live_replay.py         # ROC 0.665

## For the esports dataset

Re-download rather than copy — it is public:

* Kaggle: `marvingozo/polymarket-tick-level-orderbook-dataset`
* The extraction that produced `polymarket_sports/` is documented in
  `polymarket_sports/reports/run_manifest.json`
* Then: `research/sports/build_bars.py` and `research/sports/run_resolution.py`

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
* **`data/jump` is 37 GB and `research/*/cache` is 4 GB.** Have ~55 GB free
  before starting step 4.
