# Polymarket in-play order-book study

Can the order book of a live sports prediction market tell you where the price
is about to go, and can you make money from it?

**Short answer: it tells you *when*, reliably, and never *which way*.** Four
months, six recorded sessions, 75 MLB games and eight independent attacks on
the direction problem. The detection works and pays nothing.

Start here:

| document | what it is |
|---|---|
| **`RESEARCH_PROGRESS.md`** | the conclusion, the data inventory, and what is still open |
| **`METHODS_AND_EXPERIMENTS.md`** | every method tried, how it was built, what it measured |
| `REPRODUCE.md` | what to copy, what to re-download, what to rebuild |
| `polymarket_orderbook/results/METHODOLOGY.md` | the surviving detection path, end to end |
| `polymarket_orderbook/results/RULED_OUT.md` | the ledger of closed doors, with figures |

## Layout

```
polymarket_orderbook/
  live_recorder.py        full order books for every MLB contract on the slate
  collect_days.py         nightly driver; waits until 45 min before first pitch
  plan_slate.py           writes matches.py from the MLB schedule
  run_slate_daily.ps1     watchdog tick: is a collector up, and is it alive?
  register_recorder_task.ps1  registers the PolymarketSlateDaily task
  jump_data.py            raw JSONL -> 200ms in-game grid -> features
  trim_sessions.py        in-game trimming; compress_raw.py; verify_recording.py
  check_series_key.py     regression guard for the defect that poisoned pass one

  research/makinen/       the live detection path, and the 2026-09-21 oracle work
  research/jump_prediction/  the LOB-trajectory study
  research/lee_mykland/   jump detection used for external validation
  research/sports/        esports resolution sweep (still open)
  results/                every figure, table and written finding

polymarket_sports/        esports tick data, 1,039 contracts over 21 days
reports/                  early Polymarket strategy screen
```

## What works

- **Collapse detection with no model at all.** Near-touch liquidity below a
  quarter of its own 60-second median. 134,428 events, 26.9% of them real.
- **Real-vs-fake classification at ROC-AUC 0.756** on 24,608 held-out events,
  and 0.665 in honest tick-by-tick replay at 35x real time.
- **Picking the jumps that would be profitable, at 22x lift** and 52.9%
  precision — added 2026-09-21.

Gradient-boosted trees beat CNN, CNN-LSTM and CNN-LSTM-Attention in every
matched comparison. More data helped where more architecture did not.

## What does not

Direction. Eight methods, three with genuine statistical skill, all lose money
— including a CNN that hit a **73.5% directional hit rate and still lost 5.60
ticks per trade.** The model predicts jumps by detecting that liquidity has
gone, and "liquidity has gone" and "trading is expensive" are the same
sentence.

The final measurement: selection is solved and needs 30–55% precision, which is
achieved. Nothing is profitable below **75% directional accuracy** at any
precision. Measured direction skill is ROC-AUC 0.659.

## Running the recorder

```bash
cd polymarket_orderbook
python plan_slate.py --hkt-date 2026-09-22 --write matches.py
python collect_days.py --days 1 --duration-hours 14
```

Or leave the `PolymarketSlateDaily` task alone. It ticks every 20 minutes,
around the clock. Each tick runs `run_slate_daily.ps1`, which finishes in about
two seconds: if a recording is in progress it does nothing, and otherwise it
starts `collect_days.py --auto` **detached** and exits. `--auto` reads the MLB
schedule and picks the slate that is due now, so a tick at any hour restarts
the right night rather than queueing the wrong one.

To see what it is doing:

```powershell
type polymarket_orderbook\logs\watchdog.log          # one line per tick
type polymarket_orderbook\logs\collector_state.json  # pid, slate, phase, heartbeat
type polymarket_orderbook\logs\collector.log         # the collector's own log
```

Re-register the task with `powershell -ExecutionPolicy Bypass -File
polymarket_orderbook\register_recorder_task.ps1`. It proves the task really
executes by watching `watchdog.log` grow, rather than trusting Task Scheduler's
own status fields. **Run it elevated** to get `LogonType=S4U` and an at-startup
trigger; unelevated it registers Interactive, which survives a locked screen
but not a logoff or an unattended reboot, and it says so.

Why it is built this way — the watchdog used to *host* the collector for the
whole 14-hour slate, and the task's repetition had `StopAtDurationEnd`, so Task
Scheduler killed that host every afternoon and left the Python collector running
as an orphan with a dead stdout pipe. Every tick that night saw the orphan and
skipped, and the orphan itself died on its first `print()` when it woke to
record. The 2026-09-22 slate was lost with no error anywhere. The design notes
at the top of `run_slate_daily.ps1` and `register_recorder_task.ps1` carry the
detail.

## A note on trusting results here

Nine data and labelling defects were found over the project, and **every one
produced a convincing false positive first** — a merged spread series that made
a one-line rule score 76%, a jump label that was 37–99% quote-vacuum artefacts,
a feature filter that leaked the future and returned ROC-AUC 1.0000. They are
all catalogued in `METHODS_AND_EXPERIMENTS.md` section 7.

The habits that came out of it: prove any key used to group or join a series is
unique against the raw source; plot the price and compute a model-free oracle
bound before believing a model score; and treat a result concentrated in one
slice as a corruption signature rather than a finding.
