# 1-second top-of-book grid for YES tokens, one parquet per session: last quote in each second it changed.
#   python research/settle/build.py data/live/top_of_book_2026-09-10.csv.xz
import sys, os, pandas as pd
f = sys.argv[1]; s = os.path.basename(f).split('.')[0].replace('top_of_book_', '')
yes = set(pd.read_csv('results/large_jumps/inputs/yes_meta.csv', dtype=str).yes)
cols = ['ts_recv_ms', 'event_slug', 'market_type', 'line', 'asset_id', 'best_bid', 'best_bid_size', 'best_ask', 'best_ask_size']
out = []
for c in pd.read_csv(f, usecols=cols, dtype={'asset_id': str}, chunksize=2_000_000):
    c = c[c.asset_id.isin(yes)]
    c['sec'] = c.ts_recv_ms // 1000
    out.append(c.groupby(['asset_id', 'sec'], sort=False).tail(1))   # file is in receive order
d = pd.concat(out).groupby(['asset_id', 'sec'], sort=False).tail(1)
assert not d.duplicated(['asset_id', 'sec']).any()
d.drop(columns='ts_recv_ms').to_parquet(f'research/settle/cache/{s}.parquet')
print(s, len(d), d.asset_id.nunique(), d.event_slug.nunique())
