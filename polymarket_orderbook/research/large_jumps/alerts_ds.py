import os
# One row per depth-drain alert (depth within 5 ticks < 25% of 4s ago, book still 5s), YES token only.
# Features use only the past; outcomes and P&L use the future.
import numpy as np, pandas as pd, pyarrow.parquet as pq
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
LAG, STILL, H, DEB, THR, HOLD = 20, 25, 25, 50, 0.25, 150
FEE = 0.05
fee = lambda p: FEE * p * (1 - p) / 0.01

m = pd.read_csv('results/makinen/single_jumps/marks.csv')
lg = lambda p: np.log(p / (1 - p))
m['large'] = (lg(m.exit_bid.clip(.005, .995)) - lg(m.entry_ask.clip(.005, .995))) >= 0.46
X = pd.read_pickle(os.path.join(SP, 'xmkt.pkl')); X = X[X.real30 == 1]
XS = {k: np.sort(v.ts.to_numpy()) for k, v in X.groupby('series')}


def next_gap(src, n):
    out = np.full(n, 10**9)
    if len(src):
        ss = np.sort(src); i = np.searchsorted(ss, np.arange(n)); ok = i < len(ss)
        out[ok] = ss[i[ok]] - np.arange(n)[ok]
    return out


rows = []
for sess in sorted(m.session.unique()):
    g = m[m.session == sess]
    f = pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet', columns=['mid', 'spread_ticks', 'ts', 'series']).to_pandas()
    lob = np.load(f'data/jump/lob_{sess}_trimmed.npy', mmap_mode='r')
    for sr, s in f.groupby('series', sort=False):
        s = s.sort_values('ts'); ts = s.ts.to_numpy(); n = len(ts)
        if n < 1000: continue
        L = np.asarray(lob[s.index.to_numpy()], np.float64)
        spr = s.spread_ticks.to_numpy(); mid = s.mid.to_numpy(); norm = np.median(spr); cut = norm + max(2, norm)
        bid = np.round(mid - spr * .005, 4); ask = np.round(mid + spr * .005, 4)
        bu, au = np.expm1(L[:, 1, 0]), np.expm1(L[:, 3, 0])
        db = (np.expm1(L[:, 1]) * (np.abs(L[:, 0]) <= 5)).sum(1)
        da = (np.expm1(L[:, 3]) * (np.abs(L[:, 2]) <= 5)).sum(1)
        dep = db + da
        contig = pd.Series(np.r_[1, np.diff(ts) == 200]).rolling(STILL + LAG).min().to_numpy() == 1
        rb = pd.Series(bid).rolling(STILL); ra = pd.Series(ask).rolling(STILL)
        still = (((rb.max() - rb.min()) <= .011) & ((ra.max() - ra.min()) <= .011)
                 & (pd.Series(spr).rolling(STILL).max() <= cut)).to_numpy() & contig
        r = np.full(n, np.nan); r[LAG:] = dep[LAG:] / np.maximum(dep[:-LAG], 1)
        pulled = spr > cut
        cp = np.r_[0, np.cumsum(pulled)]
        upd = np.r_[0, (np.abs(np.diff(L[:, :, 0], axis=0)) > 1e-6).any(1)]
        cu = np.r_[0, np.cumsum(upd)]
        lastpull = pd.Series(np.where(pulled, np.arange(n), np.nan)).ffill().to_numpy()
        # still-run length: slots since the level began
        brk = ~still; runstart = pd.Series(np.where(brk, np.arange(n), np.nan)).ffill().fillna(-1).to_numpy()
        # outcomes
        gm = g[g.series == sr]; js = []
        for x in gm.itertuples():
            e = min(np.searchsorted(ts, x.ts_entry), n - 1); k = e
            while k < n - 1 and abs(bid[k] - bid[e]) <= .011 and abs(ask[k] - ask[e]) <= .011 and spr[k] <= cut: k += 1
            js.append(k)
        js = np.array(js, int); jl = gm.large.to_numpy(bool)
        xs = np.searchsorted(ts, XS.get(sr, np.array([], np.int64))) + 1
        nxt, nxtL, nxtR = next_gap(js, n), next_gap(js[jl], n), next_gap(xs, n)
        quoted = np.flatnonzero(~pulled)
        last = -10**9
        for i in np.flatnonzero(still & (r < THR)):
            if i - last < DEB or i < 600: continue
            last = i
            q = np.searchsorted(quoted, i + HOLD)
            pnl = np.nan
            if q < len(quoted) and quoted[q] - (i + HOLD) <= 300 and au[i] >= 10:
                j = quoted[q]; pnl = (bid[j] - ask[i]) / 0.01 - fee(ask[i]) - fee(bid[j])
            rows.append(dict(
                sess=sess, slug=sr.split('|')[0], series=sr, mtype=sr.split('|')[1], ts=int(ts[i]),
                y_mark=1 <= nxt[i] <= H, y_large=1 <= nxtL[i] <= H, y_real=1 <= nxtR[i] <= H, pnl=pnl,
                f_ratio=r[i], f_ratio_bid=db[i] / max(db[i - LAG], 1), f_ratio_ask=da[i] / max(da[i - LAG], 1),
                f_logdep=np.log1p(dep[i]), f_logdep_before=np.log1p(dep[i - LAG]),
                f_imb_top=(bu[i] - au[i]) / max(bu[i] + au[i], 1), f_logbu=np.log1p(bu[i]), f_logau=np.log1p(au[i]),
                f_spr_rel=spr[i] / norm, f_norm_spr=norm, f_price=ask[i], f_dist50=abs(mid[i] - .5),
                f_upd5=(cu[i + 1] - cu[i - 24]) / 25, f_upd30=(cu[i + 1] - cu[i - 149]) / 150,
                f_pulled30=(cp[i] - cp[i - 150]) / 150, f_pulled120=(cp[i] - cp[i - 600]) / 600,
                f_since_pull=(i - lastpull[i]) * .2 if np.isfinite(lastpull[i]) else 999.,
                f_still_len=(i - runstart[i]) * .2,
                f_absmove60=np.abs(np.diff(mid[i - 300:i + 1])).sum() / 0.01))
    print(sess, len(rows), flush=True)
A = pd.DataFrame(rows)
# cross-market, causal: other markets of this game that alerted in the previous 2s (or this slot)
A['f_others_alerting'] = 0
for slug, gg in A.groupby('slug'):
    t = gg.ts.to_numpy(); s = gg.series.to_numpy(); c = np.zeros(len(gg), int)
    for k in range(len(gg)):
        w = (t >= t[k] - 2000) & (t <= t[k]) & (s != s[k])
        c[k] = len(set(s[w]))
    A.loc[gg.index, 'f_others_alerting'] = c
A.to_pickle(os.path.join(SP, 'alerts_ds.pkl'))
