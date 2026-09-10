# Mid-frequency sports outcome models on free data

Predicting game winners from pre-game statistics, built entirely on free,
key-less public APIs so the whole thing runs at one request per game-day.

The deep pipeline is MLB (2015-2026, walk-forward tested on 2023-2026). A
lighter generic pipeline covers any league Action Network exposes (NBA, NHL,
NFL, NCAAB).

## Why "mid-frequency"

Nothing here needs a paid feed, a websocket, or a scraper farm:

| Source | Auth | Cost of a daily update | Used for |
|---|---|---|---|
| `statsapi.mlb.com` | none | ~5 requests | schedules, probable starters, team + pitcher box lines |
| `api.actionnetwork.com/web/v1/scoreboard/<league>` | none | 1 request | closing moneylines, results for other leagues |
| `gamma-api.polymarket.com` | none | 1 request | which markets are actually liquid |

Every response is cached under `data/raw/`, so a rebuild is offline and the
APIs are hit once per resource. A full historical backfill is ~1,900 calls;
keeping it current is a handful per day.

## Layout

```
mlbmodel/
  config.py     paths, seasons, Elo hyperparameters
  collect.py    MLB Stats API -> games / team_logs / pitcher_logs
  odds.py       Action Network -> historical closing moneylines
  features.py   Elo, team form, starter form, bullpen, park, context
  ratings.py    schedule-adjusted (Massey/ridge) offence & defence
  baselines.py  log5, Pythagorean log5, negative-binomial run model
  models.py     logistic / GBM / forest / MLP zoo
  advanced.py   Elo-offset GBM, run-margin regression, market blend
  backtest.py   walk-forward driver, metrics, betting simulation
  analyze.py    calibration, closing-line value, edge curves
  tune.py       Elo grid search (pre-test seasons only)
  anysport.py   generic Elo+form pipeline for NBA / NHL / NFL / NCAAB
  run.py        end-to-end entry point
  predict.py    score today's slate, priced against live Polymarket markets
tests/
  test_leakage.py   brute-force checks that no feature sees its own game
  test_pipeline.py  odds maths, baselines, betting sim, calibration
```

## Running it

```powershell
python -m mlbmodel.collect      # ~1,000 cached API calls, once
python -m mlbmodel.odds         # historical closing lines, once
python -m mlbmodel.run --rebuild
python -m mlbmodel.analyze
python tests/test_leakage.py; python tests/test_pipeline.py
python -m mlbmodel.anysport nba
python -m mlbmodel.predict --date 2026-09-09
```

## Result in one line

The models reach 56.6% on MLB and 67.5% on NBA — solid, calibrated, and in
line with the published literature — but the closing line reaches 56.8% and
69.4%, and a model fit with the closing line as a fixed offset cannot improve
on it in any season. There is no edge here. `reports/FINDINGS.md` gives the
numbers and, more usefully, the three controls that killed an apparent +9%
ROI before it could be believed.

## Method

**Leakage is the whole game.** Published MLB papers reporting 90%+ accuracy
are, essentially without exception, computing season-total statistics that
include the game being predicted. Every feature here is built by
`shift(1)` *before* any rolling or expanding window, and `tests/test_leakage.py`
recomputes a random sample of features by brute force to prove it. A
shuffled-label control confirms the feature matrix carries no residual signal.

**Features** (all strictly pre-game):

- *Elo* — margin-of-victory scaled, reverted 30% between seasons, plus a
  variant that shifts the win probability by the listed starter's rolling
  game score. Hyperparameters grid-searched on 2018-2022 only.
- *Team form* — wOBA, runs, K%, BB%, HR% over 15- and 40-game windows and
  season-to-date, built from summed numerators and denominators (not means of
  per-game rates) and regressed toward the league mean by an explicit prior.
- *Schedule-adjusted ratings* — ridge regression of runs scored on
  `offence_i - defence_j + home`, refit every 10 days over a 400-day
  recency-weighted window. Removes strength-of-schedule bias that plain
  rolling averages carry.
- *Starting pitcher* — rolling game score, FIP, K/BF, BB/BF, HR/BF, IP per
  start, days rest, career start count. Relief outings excluded.
- *Bullpen* — relief innings absorbed over the last 5 and 15 games (fatigue)
  and a 40-game relief ERA.
- *Context* — rest days, games in the last 7, series openers, travel,
  day/night, trailing 3-season park factor.

**Models** — three classical baselines (log5, Pythagorean log5, a
negative-binomial run-scoring model), a regularised ML zoo (logistic, elastic
net, LightGBM, XGBoost, random forest, extra trees, MLP, calibrated GBM), and
two structural variants that matter more than any of the above: a LightGBM
that takes the pitcher-aware Elo logit as a fixed offset and only learns the
residual, and a run-margin regressor whose output is mapped through a fitted
logistic link.

**Evaluation** — walk-forward: for test season *S*, train only on seasons
`< S`. Reported per season and pooled: accuracy, log loss, Brier, AUC,
calibration slope, and a betting simulation against the actual closing
moneyline (flat stake and fractional Kelly), plus how the model's probability
compares with the closing line.

See `reports/FINDINGS.md` for results.
