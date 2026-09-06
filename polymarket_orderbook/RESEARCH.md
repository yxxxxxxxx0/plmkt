# Strategy research: what was tested and what survived

Reproduce in this order. Every script writes to `data/analysis/` and `logs/`.

## Pipeline

```
python build_token_map.py              # asset_id -> condition_id + complement
python build_panel.py                  # -> data/panel/<slug>.parquet  (the research panel)
python research_ic.py    --slate ...    # feature scorecard: paper alpha vs executable edge
python backtest.py       --strategy all # event-driven taker backtest, latency-aware
python cross_complement.py --slate ...  # YES/NO complement arb + lead-lag
python cross_market.py     --slate ...  # moneyline/spread/total lead-lag
python maker_adverse.py    --slate ...  # maker half-spread vs adverse selection
python ml_taker.py                      # GBM on the executable taker target
```

## Data integrity findings (check these before trusting any result)

| issue | scale | effect if ignored |
|---|---|---|
| **Crossed books** (ask <= bid) | 0.29% of rows; 0.81% of `spread` markets | fabricates risk-free arbitrage; also *admitted* by any `spread <= X` filter, since crossed rows have negative spread |
| **08-29 slate truncated** | recorder ran 15:26-19:24Z; 5/15 games have in-game data, median 41% coverage | silently trains on pregame quotes |
| **YES/NO are one book** | `bid_YES + ask_NO = 100c` and equal size, 100.0000% of 884,686 simultaneous states | any YES/NO "arbitrage" or "lead-lag" is an artifact |
| **Trade tape resolution** | backfill = seconds, live = milliseconds; 1s holds ~85 book updates | second-resolution understates adverse selection |

## Splits

- **train** 2026-08-27 + 2026-08-28 (22 games)
- **test**  2026-09-04 (15 games, six days later, millisecond trade tape)
- **excluded** 2026-08-29 (truncated recording)

Chronological only. No random splits anywhere.
