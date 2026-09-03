# Polymarket order book recorder & tick-level replay

Records live Polymarket order books for MLB games, reconstructs the exact
tick-level history, and serves a browser viewer that steps through **every
recorded change** — down to microsecond resolution.

Nothing is resampled, downsampled or depth-capped anywhere in the pipeline.
The only data ever discarded is the pregame/postgame tail outside the real
kickoff→final-out window, and that trim is optional.

---

## Pipeline

```
python live_recorder.py                 # 1. record        -> data/live/books.jsonl
python fetch_game_windows.py            # 2. game windows  -> data/game_windows.json
python rebuild_duckdb.py --all          # 3. reconstruct   -> data/uniform/<slug>.json
python build_viewer_chunked.py          # 4. viewer data   -> data/viewer_full/
python build_viewer_multi.py            # 5. viewer page   -> viewer_full.html
python serve_viewer.py                  # 6. serve + open browser
```

Step 3 needs the split-by-game files; `rebuild_duckdb.py` expects
`data/live/by_game/<slug>.jsonl`, produced by:

```
python -c "import rebuild_multi as rm; rm.split_books_jsonl()"
```

Run `fetch_game_windows.py` **after the games finish**. A game still
"In Progress" reports its last completed play as the end time, which silently
truncates the rebuild — one game was cut to 18.8 minutes of a 154-minute game
exactly this way.

---

## 1. Recording — `live_recorder.py`

```
python live_recorder.py                      # all matches in matches.py
python live_recorder.py --duration 120       # short smoke test
python live_recorder.py --core-only          # only moneyline/spread/total/NRFI
python live_recorder.py --live-discover 60   # test against any open markets
```

Records **every order-book market** on the event by default — 15–17 per game
(30–34 tokens), not just the core 9. `--core-only` restores the old behaviour.

- **One websocket per match**, so a drop or reconnect affects only that match.
- **Exchange `timestamp` is the event time**, with local receive time kept
  beside it so delivery lag stays measurable.
- **No disk I/O on the event loop.** Records go to a `queue.Queue`; a daemon
  thread batches them out and fsyncs at most every `--fsync-interval` seconds.
- **One record per snapshot** holding the whole book, not one row per price
  level.

Markets, condition IDs and token IDs are all discovered live from the Gamma
API (`discovery.py`). Edit `matches.py` to change which games are tracked.

Output: `data/live/books.jsonl` (one complete book snapshot per line) and
`data/live/top_of_book.csv`.

---

## 2. Reconstruction — `rebuild_duckdb.py`

Turns each game's JSONL into a run-length-encoded, per-asset order book
history in `data/uniform/<slug>.json`.

DuckDB parses the whole file in one vectorized pass; the bid/ask arrays come
back as Arrow lists that are flattened straight to numpy, and output is
written with orjson's native numpy support. The pure-Python reference
implementation (`rebuild_multi.py`) had not finished one 4.3 GB game in 20+
minutes; this does it in ~140 s.

```
python rebuild_duckdb.py --all
python rebuild_duckdb.py mlb-col-wsh-2026-08-27 --window full
```

- `--window game` (default) trims to kickoff → actual final out.
  `--window full` keeps every recorded tick including pregame.
- `--depth 0` (default) keeps the **complete** book. A cap exists but is
  opt-in and near-useless here — books average ~23 levels/side, so even
  `--depth 30` trims under 5%.

Two details that are easy to get wrong and are enforced in code:

- **Millisecond collisions.** The exchange clock is millisecond-resolution and
  up to 78 distinct snapshots can share one millisecond. They are separated by
  1 µs each, in `seq` order — nothing merged, nothing dropped.
- **Timestamp monotonicity.** The vectorized nudge assumes `ts` never goes
  backwards within an asset. That is *usually* true but not always, so it is
  checked per asset, and any asset that violates it falls back to the exact
  sequential reference loop.

### Verifying it

```
python test_rebuild_parity.py
```

Runs `rebuild_duckdb.py` and `rebuild_multi.py` over the same fixture and
compares every run boundary, every price level, every tick, plus asserts the
streaming writer is byte-identical to serializing in one shot. This exists
because the fast path is a reimplementation, and it has already caught three
real bugs (a `line` field typed as string, a dropped interval in the final-run
edge case, and non-deterministic asset ordering).

---

## 3. Viewer data — `build_viewer_chunked.py`

One match is ~2.6 GB at full fidelity, which no browser will fetch. This
changes the **delivery**, not the data — the same runs, split into three tiers:

| file | contents | size |
|---|---|---|
| `<slug>/ticks.json` | complete tick axis, delta-encoded integer microseconds | ~7 MB |
| `<slug>/<asset>/series.json` | best bid/ask for **every** run, columnar + delta-encoded | ~1.5 MB/token |
| `<slug>/<asset>/cNNNN.json` | full order books for one 5-min chunk | ≤26 MB |

The price chart draws from `series`, so it shows every tick immediately
without loading a single book. The ladder pulls only the chunk under the
playhead. Chunks repeat the run in force at their start boundary so each is
independently sufficient.

```
python build_viewer_chunked.py --chunk-seconds 300
```

---

## 4. Viewer — `viewer_full.html`

```
python build_viewer_multi.py     # template + manifest -> viewer_full.html
python serve_viewer.py           # http://127.0.0.1:8799/viewer_full.html
```

Edit `viewer_full_template.html`, not the built file. An HTTP server is
required — browsers block `fetch()` on `file://` URLs.

Chunks are **LRU-evicted (max 4)** and other matches are released on switch.
This is load-bearing: a 20 MB chunk becomes hundreds of MB of JS objects
(every `[price,size]` level is its own array), and caching them all exhausts
the tab's heap and wedges the renderer.

---

## Data layout

| path | what | size (7-game slate) |
|---|---|---|
| `data/live/books.jsonl` | raw recorder output, append-only | ~43 GB |
| `data/live/by_game/` | the same, split per game (cache) | ~22 GB |
| `data/uniform/` | reconstructed full-fidelity archive | ~13 GB |
| `data/viewer_full/` | chunked viewer tiers | ~13 GB |
| `data/game_windows.json` | kickoff / final-out per game (MLB Stats API) | 4 KB |

Everything under `data/` except `game_windows.json` is gitignored — it is far
past GitHub's limits and is all reproducible from the recorder output.

---

## Lessons this repo encodes

**The first recording (2026-08-26) captured ~6% of the feed and is
unrepairable.** Root cause: `flush()` + `os.fsync()` once per row *on the
event loop*. The busiest second produced 70,926 depth rows; at ~0.36 ms per
fsync that needs ~25 s of disk time to write 1 s of data. The loop froze past
the 20 s keepalive deadline and the sockets were killed repeatedly — 1.89 h of
fsync burn in a 16.8 h run. Compounding it, the exchange's own `timestamp` was
discarded in favour of write time, so surviving rows are stamped when the
backlog drained. On reconnect only the current book is recovered; every
`price_change` during an outage is gone. Order is recoverable; timing is not.
`live_recorder.py` fixes all three causes.

**Two silent corruptions were found in earlier rebuilds**, neither visible
from the viewer: snapshots sharing a truncated millisecond were merged into
one contradictory book (dropping 55% of ticks on one game), and a default
depth cap of 20 truncated books that reach ~90 levels. Any change that makes
the data smaller is suspect by default and must be an explicit, stated choice.

**A useful independent check:** for each two-outcome market, a YES level at
price `p` size `s` implies a NO level at `1-p` size `s`. The two books are
built from separate message streams, so agreement is real evidence the
reconstruction is correct. Expect ~99.7%, not 100% — the residual is genuine
asynchronous updates.

---

## Files

| file | role |
|---|---|
| `matches.py` | which games to track |
| `discovery.py` | Gamma API market/token discovery |
| `live_recorder.py` | the collector |
| `fetch_game_windows.py` | kickoff / final-out times from the MLB Stats API |
| `rebuild_multi.py` | reference reconstruction + JSONL splitting |
| `rebuild_duckdb.py` | fast reconstruction (the one you run) |
| `test_rebuild_parity.py` | proves the two agree exactly |
| `build_match_index.py` | market/token labelling, Gamma metadata, streaming reader |
| `build_viewer_chunked.py` | full-fidelity chunked viewer data |
| `build_viewer_multi.py` | renders the template into `viewer_full.html` |
| `viewer_full_template.html` | the viewer UI (edit this) |
| `serve_viewer.py` | local HTTP server |

## Requirements

```
pip install -r requirements.txt
```

`requests`, `websockets` (recording); `duckdb`, `pyarrow`, `numpy`, `orjson`
(reconstruction); `ijson` (streaming multi-GB JSON).
