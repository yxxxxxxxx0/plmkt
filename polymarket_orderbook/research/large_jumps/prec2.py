import os
# Precision of the causal depth-drain alert, per market (one recorded token = one market).
# Outcomes: an upward break-even mark on this token (and a large one), and a real jump EITHER way
# (candidate still displaced once the book is quoted 30s later, from xmkt.py).
import numpy as np, pandas as pd, pyarrow.parquet as pq
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
LAG, STILL, H, DEB = 20, 25, 25, 50      # 200ms slots: 4s lookback, 5s still, 5s horizon, 10s debounce
THR = [0.5, 0.25, 0.10]

m = pd.read_csv('results/makinen/single_jumps/marks.csv')
lg = lambda p: np.log(p / (1 - p))
m['large'] = (lg(m.exit_bid.clip(.005, .995)) - lg(m.entry_ask.clip(.005, .995))) >= 0.46
X = pd.read_pickle(os.path.join(SP, 'xmkt.pkl')); X = X[X.real30 == 1]
XS = {k: np.sort(v.ts.to_numpy()) for k, v in X.groupby('series')}


def next_gap(src, n):
    """slots from each slot to the next event in src (10**9 if none)"""
    out = np.full(n, 10**9)
    if len(src):
        ss = np.sort(src); i = np.searchsorted(ss, np.arange(n)); ok = i < len(ss)
        out[ok] = ss[i[ok]] - np.arange(n)[ok]
    return out


def caught(al, js):
    return np.array([((al >= k - H) & (al <= k - 1)).any() for k in js], bool)


out = []
for sess in sorted(m.session.unique()):
    g = m[m.session == sess]
    f = pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet', columns=['mid', 'spread_ticks', 'ts', 'series']).to_pandas()
    lob = np.load(f'data/jump/lob_{sess}_trimmed.npy', mmap_mode='r')
    for sr, s in f.groupby('series', sort=False):
        s = s.sort_values('ts'); ts = s.ts.to_numpy(); n = len(ts)
        if n < 1000: continue
        L = np.asarray(lob[s.index.to_numpy()], np.float64)
        spr = s.spread_ticks.to_numpy(); mid = s.mid.to_numpy(); norm = np.median(spr); cut = norm + max(2, norm)
        bid = mid - spr * .005; ask = mid + spr * .005
        dep = (np.expm1(L[:, 1]) * (np.abs(L[:, 0]) <= 5)).sum(1) + (np.expm1(L[:, 3]) * (np.abs(L[:, 2]) <= 5)).sum(1)
        contig = pd.Series(np.r_[1, np.diff(ts) == 200]).rolling(STILL + LAG).min().to_numpy() == 1
        rb = pd.Series(bid).rolling(STILL); ra = pd.Series(ask).rolling(STILL)
        still = (((rb.max() - rb.min()) <= .011) & ((ra.max() - ra.min()) <= .011)
                 & (pd.Series(spr).rolling(STILL).max() <= cut)).to_numpy() & contig
        r = np.full(n, np.nan); r[LAG:] = dep[LAG:] / np.maximum(dep[:-LAG], 1)
        # mark starts on THIS token: first slot after entry leaving the level
        gm = g[g.series == sr]; js = []
        for x in gm.itertuples():
            e = min(np.searchsorted(ts, x.ts_entry), n - 1); k = e
            while k < n - 1 and abs(bid[k] - bid[e]) <= .011 and abs(ask[k] - ask[e]) <= .011 and spr[k] <= cut: k += 1
            js.append(k)
        js = np.array(js, int); jl = gm.large.to_numpy(bool)
        xs = np.searchsorted(ts, XS.get(sr, np.array([], np.int64))) + 1   # real jump starts after its level's last slot
        nxt, nxtL, nxtR = next_gap(js, n), next_gap(js[jl], n), next_gap(xs, n)
        hit = lambda arr, idx: int(((arr[idx] >= 1) & (arr[idx] <= H)).sum())
        slug = sr.split('|')[0]; hours = n * 0.2 / 3600
        b = np.flatnonzero(still)[::5]
        out.append(dict(kind='base', thr=np.nan, sess=sess, slug=slug, series=sr, n=len(b), hours=hours,
                        hit=hit(nxt, b), hitL=hit(nxtL, b), hitR=hit(nxtR, b)))
        for t in THR:
            al = []; last = -10**9
            for i in np.flatnonzero(still & (r < t)):
                if i - last >= DEB: al.append(i); last = i
            al = np.array(al, int)
            c, cR = caught(al, js), caught(al, xs)
            out.append(dict(kind='alert', thr=t, sess=sess, slug=slug, series=sr, n=len(al), hours=hours,
                            hit=hit(nxt, al), hitL=hit(nxtL, al), hitR=hit(nxtR, al),
                            nj=len(js), caught=int(c.sum()), njL=int(jl.sum()), caughtL=int(c[jl].sum()),
                            nR=len(xs), caughtR=int(cR.sum())))
    print(sess, flush=True)
pd.DataFrame(out).to_pickle(os.path.join(SP, 'prec2.pkl'))
