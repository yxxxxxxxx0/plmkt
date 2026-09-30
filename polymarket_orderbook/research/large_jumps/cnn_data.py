# Sequence tensor for every alert: the last 20s (100 x 200ms) of the book before the alert, 42 channels:
# 10 bid px, 10 bid log$, 10 ask px, 10 ask log$ (px in ticks from mid), mid change vs alert (ticks), spread.
# Plus the tabular set (book summary + game state + YES-team-batting) and labels.
import os
import numpy as np, pandas as pd, pyarrow.parquet as pq
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
W, C = 100, 42
A = pd.read_pickle(os.path.join(S, 'alerts_ds.pkl')).reset_index(drop=True)
X = np.lib.format.open_memmap(os.path.join(S, 'cnn_X.npy'), mode='w+', dtype=np.float16, shape=(len(A), W, C))
ok = np.zeros(len(A), bool)
for sess, a in A.groupby('sess'):
    f = pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet', columns=['mid', 'spread_ticks', 'ts', 'series']).to_pandas()
    f = f[f.series.isin(set(a.series))]
    lob = np.load(f'data/jump/lob_{sess}_trimmed.npy', mmap_mode='r')
    for sr, s in f.groupby('series', sort=False):
        s = s.sort_values('ts'); ts = s.ts.to_numpy(); rows = s.index.to_numpy()
        mid = s.mid.to_numpy(); spr = s.spread_ticks.to_numpy()
        aa = a[a.series == sr]
        for ix, t in zip(aa.index, aa.ts):
            i = np.searchsorted(ts, t)
            if i < W - 1 or i >= len(ts) or ts[i] != t or ts[i] - ts[i - W + 1] != (W - 1) * 200: continue
            L = np.asarray(lob[rows[i - W + 1:i + 1]], np.float32).reshape(W, 40)
            X[ix, :, :40] = np.clip(L, -100, 100)
            X[ix, :, 40] = np.clip((mid[i - W + 1:i + 1] - mid[i]) / .01, -100, 100)
            X[ix, :, 41] = np.clip(spr[i - W + 1:i + 1], 0, 100)
            ok[ix] = True
    print(sess, int(ok.sum()), flush=True)
X.flush()

# tabular: game state (causal) for every alert, as in newmodel.py
RE = {0: [.48, .86, 1.10, 1.44, 1.35, 1.78, 1.96, 2.29], 1: [.25, .51, .66, .88, .95, 1.13, 1.35, 1.54],
      2: [.10, .22, .32, .43, .35, .48, .57, .75]}
G = pd.read_csv(os.path.join(IN, 'game_state.csv')).sort_values(['slug', 't'])
meta = pd.read_csv(os.path.join(IN, 'yes_meta.csv'), dtype={'yes': str}).drop_duplicates('yes').set_index('yes')
A['aid'] = A.series.str.split('|').str[-1]; side = A.aid.map(meta.yes_side)
for c in ['g_inning', 'g_outs', 'g_on1', 'g_on2', 'g_on3', 'g_balls', 'g_strikes', 'g_margin', 'g_runs', 'g_re24', 'g_since_event', 'is_top']:
    A[c] = np.nan
for slug, a in A.groupby('slug'):
    g = G[G.slug == slug]
    if not len(g): continue
    j = np.searchsorted(g.t.to_numpy(), a.ts.to_numpy(), side='right') - 1; okj = j >= 0
    r = g.iloc[np.clip(j, 0, None)]
    b = (r.on1 + 2 * r.on2 + 4 * r.on3).to_numpy()
    v = dict(g_inning=r.inning, g_outs=r.outs, g_on1=r.on1, g_on2=r.on2, g_on3=r.on3, g_balls=r.balls, g_strikes=r.strikes,
             g_margin=(r.home - r.away).abs(), g_runs=r.home + r.away, is_top=r.is_top.astype(float))
    for k2, vv in v.items(): A.loc[a.index, k2] = np.where(okj, vv.to_numpy(), np.nan)
    A.loc[a.index, 'g_re24'] = np.where(okj, [RE[min(o, 2)][k] for o, k in zip(r.outs.to_numpy(), b)], np.nan)
    A.loc[a.index, 'g_since_event'] = np.where(okj, (a.ts.to_numpy() - r.t.to_numpy()) / 1000, np.nan)
A['yes_bats'] = np.select([side == 'Over', side == 'away', side == 'home'], [1.0, (A.is_top == 1) * 1.0, (A.is_top == 0) * 1.0], np.nan)
A.loc[A.is_top.isna() & (side != 'Over'), 'yes_bats'] = np.nan
A['seq_ok'] = ok
A.to_pickle(os.path.join(S, 'cnn_meta.pkl'))
print('done', ok.mean())
