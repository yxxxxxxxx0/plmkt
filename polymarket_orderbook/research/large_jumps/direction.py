# Direction of REAL jumps: (1) is the YES team batting?  (2) does the pre-jump drain differ by side?
# Candidates as in xmkt.py (still 5s -> mid moves >=3 ticks within 10s); real = still displaced >=2 ticks
# the same way once the book is quoted 30s later. All drain features end at the level's last slot i.
import os, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
STILL, LOOK, J, LAG = 25, 50, 0.03, 20
meta = pd.read_csv(os.path.join(IN, 'yes_meta.csv'), dtype={'yes': str}).drop_duplicates('yes').set_index('yes')
G = pd.read_csv(os.path.join(IN, 'game_state.csv')).sort_values(['slug', 't'])

rows = []
for p in sorted(glob.glob('data/jump/feat_books_*_trimmed.parquet')):
    sess = p.split('feat_')[1].replace('_trimmed.parquet', '')
    f = pq.read_table(p, columns=['mid', 'spread_ticks', 'ts', 'series']).to_pandas()
    lob = np.load(f'data/jump/lob_{sess}_trimmed.npy', mmap_mode='r')
    for sr, s in f.groupby('series', sort=False):
        s = s.sort_values('ts'); ts = s.ts.to_numpy(); n = len(ts)
        if n < 1000: continue
        spr = s.spread_ticks.to_numpy(); mid = s.mid.to_numpy(); norm = np.median(spr); cut = norm + max(2, norm)
        bid = mid - spr * .005; ask = mid + spr * .005
        contig = np.r_[True, np.diff(ts) == 200]
        rb = pd.Series(bid).rolling(STILL); ra = pd.Series(ask).rolling(STILL)
        still = (((rb.max() - rb.min()) <= .011) & ((ra.max() - ra.min()) <= .011) & (pd.Series(spr).rolling(STILL).max() <= cut)
                 & (pd.Series(contig).rolling(STILL).min() == 1)).to_numpy()
        quoted = np.flatnonzero(spr <= cut)
        ends = np.flatnonzero(still[:-1] & ~still[1:])
        idx = s.index.to_numpy()
        last = -10**9
        for i in ends:
            if i <= last or i < LAG + 5 or i + LOOK >= n or not contig[i - LAG:i + LOOK].all(): continue
            dv = mid[i + 1:i + LOOK + 1] - mid[i]; k = np.flatnonzero(np.abs(dv) >= J - 1e-9)
            if not len(k): continue
            k = k[0]; sg = np.sign(dv[k]); last = i + k
            q = np.searchsorted(quoted, i + 150)
            if q >= len(quoted) or quoted[q] - (i + 150) > 300: continue
            d30 = mid[quoted[q]] - mid[i]
            if not (np.sign(d30) == sg and abs(d30) >= 0.02 - 1e-9): continue          # real only
            L = np.asarray(lob[idx[i - LAG:i + 1]], np.float64)                      # 4s up to the level's end
            db = (np.expm1(L[:, 1]) * (np.abs(L[:, 0]) <= 5)).sum(1); da = (np.expm1(L[:, 3]) * (np.abs(L[:, 2]) <= 5)).sum(1)
            bu, au = np.expm1(L[:, 1, 0]), np.expm1(L[:, 3, 0])
            half = lambda x: (np.flatnonzero(x < 0.5 * max(x[0], 1)) - LAG)[0] * .2 if (x < 0.5 * max(x[0], 1)).any() else 0.0
            rows.append(dict(sess=sess, slug=sr.split('|')[0], series=sr, mtype=sr.split('|')[1], yes=sr.split('|')[-1],
                             ts=int(ts[i]), up=sg > 0, d30=d30, p0=mid[i],
                             bid_ratio=db[-1] / max(db[0], 1), ask_ratio=da[-1] / max(da[0], 1),
                             bid_top_ratio=bu[-1] / max(bu[-6], 1), ask_top_ratio=au[-1] / max(au[-6], 1),
                             bid_half_s=half(db), ask_half_s=half(da),
                             imb_end=(db[-1] - da[-1]) / max(db[-1] + da[-1], 1), imb_start=(db[0] - da[0]) / max(db[0] + da[0], 1)))
    print(sess, len(rows), flush=True)
R = pd.DataFrame(rows)
R['yes_side'] = R.yes.map(meta.yes_side)
# who is batting at the level's end (causal lookup)
R['is_top'] = np.nan
for slug, a in R.groupby('slug'):
    g = G[G.slug == slug]
    if not len(g): continue
    j = np.searchsorted(g.t.to_numpy(), a.ts.to_numpy(), side='right') - 1
    R.loc[a.index, 'is_top'] = np.where(j >= 0, g.is_top.to_numpy()[np.clip(j, 0, None)], np.nan)
R.to_pickle(os.path.join(S, 'direction.pkl'))
