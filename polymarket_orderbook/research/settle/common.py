# Shared loaders for the hold-to-settlement kill tests. Run everything from polymarket_orderbook/.
import glob, json, numpy as np, pandas as pd
IN = 'results/large_jumps/inputs'
FEE = lambda p: 0.05 * p * (1 - p)
TEST_FROM = '2026-09-21'          # fit on sessions before this, report on the rest

def grid():
    g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('research/settle/cache/2026-*.parquet'))])
    g = g.rename(columns={'event_slug': 'slug', 'best_bid': 'bid', 'best_ask': 'ask', 'best_bid_size': 'bsz', 'best_ask_size': 'asz'})
    assert not g.duplicated(['asset_id', 'sec']).any()
    return g.sort_values(['sec']).reset_index(drop=True)

def meta():
    m = pd.read_csv(f'{IN}/yes_meta.csv', dtype=str)
    assert m.yes.is_unique
    return m

def windows():
    w = pd.DataFrame(json.load(open('data/game_windows.json'))).T
    sec = lambda x: (pd.to_datetime(x, format='ISO8601', utc=True) - pd.Timestamp(0, tz='UTC')) // pd.Timedelta('1s')
    w['t0'] = sec(w.start_utc)
    w['t1'] = sec(w.end_utc)
    return w[['t0', 't1']]

def settle(g):
    """YES-token payoff from the last quote recorded: 1 if last bid >= .9, 0 if last ask <= .1, else unknown."""
    q = g[g.bid.notna() | g.ask.notna()].groupby('asset_id').tail(1).set_index('asset_id')
    s = pd.Series(np.nan, q.index)
    s[q.bid >= 0.9] = 1.0
    s[(q.ask <= 0.1) & ~(q.bid >= 0.9)] = 0.0
    return s

def asof(g, keys):
    """keys: DataFrame with asset_id, sec. Returns the quote in force at each (bid, ask, sizes, age in s)."""
    k = keys.sort_values('sec')
    r = pd.merge_asof(k, g[['asset_id', 'sec', 'bid', 'ask', 'bsz', 'asz']].assign(qsec=g.sec), on='sec', by='asset_id')
    r['age'] = r.sec - r.qsec
    return r

def boot(df, col, n=2000, seed=0):
    """Mean of col with a game-clustered 95% bootstrap interval."""
    s = df.groupby('slug')[col].agg(['sum', 'count'])
    r = np.random.default_rng(seed); i = r.integers(0, len(s), (n, len(s)))
    b = s['sum'].values[i].sum(1) / s['count'].values[i].sum(1)
    return df[col].mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
