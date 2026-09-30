# How many marked (upward, YES) jumps can an "alert + trade within 2s" rule catch, entering after latency?
import os, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
FEE = 0.05; fee = lambda p: FEE * p * (1 - p) / 0.01
LAT = [100, 250, 500]

D = pd.read_pickle(os.path.join(S, 'newmodel_ds.pkl'))          # alerts, 09-10 .. 09-27
m = pd.read_csv('results/makinen/single_jumps/marks.csv')
lg = lambda p: np.log(p / (1 - p))
m['large'] = (lg(m.exit_bid.clip(.005, .995)) - lg(m.entry_ask.clip(.005, .995))) >= 0.46
m = m[m.session.isin(D.sess.unique())]

pairs = pd.read_csv(os.path.join(IN, 'token_pairs.csv'), dtype=str); no2yes = dict(zip(pairs.no, pairs.yes))
T = pd.concat([pd.read_csv(f, dtype={'asset_id': str}) for f in glob.glob(os.path.join(IN, 'trades', 'books_2026-09-*.csv'))])
isno = T.asset_id.isin(no2yes).to_numpy()
T['yes'] = np.where(isno, T.asset_id.map(no2yes), T.asset_id); T['p'] = np.where(isno, 1 - T.price, T.price)
T = T.sort_values('ts'); TB = {k: (v.ts.to_numpy(), v.p.to_numpy()) for k, v in T.groupby('yes')}

rows = []
for sess, g in m.groupby('session'):
    f = pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet', columns=['mid', 'spread_ticks', 'ts', 'series']).to_pandas()
    f = f[f.series.isin(set(g.series))]
    Ds = D[D.sess == sess]
    for sr, s in f.groupby('series', sort=False):
        s = s.sort_values('ts'); ts = s.ts.to_numpy(); n = len(ts)
        spr = s.spread_ticks.to_numpy(); mid = s.mid.to_numpy(); norm = np.median(spr); cut = norm + max(2, norm)
        bid = np.round(mid - spr * .005, 4); ask = np.round(mid + spr * .005, 4)
        al = Ds[Ds.series == sr].ts.to_numpy(); aid = sr.split('|')[-1]
        tt, pp = TB.get(aid, (np.array([], np.int64), np.array([])))
        for x in g[g.series == sr].itertuples():
            e = min(np.searchsorted(ts, x.ts_entry), n - 1); k = e
            while k < n - 1 and abs(bid[k] - bid[e]) <= .011 and abs(ask[k] - ask[e]) <= .011 and spr[k] <= cut: k += 1
            t0 = ts[k]                                              # jump start
            r = dict(large=x.large, mark_net=x.net_ticks, exit_bid=x.exit_bid, entry_ask=x.entry_ask)
            a = al[(al >= t0 - 5000) & (al < t0)]
            r['alert'] = len(a) > 0
            if len(a):
                ta = a[0]; ia = np.searchsorted(ts, ta)
                r['alert_lead_s'] = (t0 - ta) / 1000
                i0, i1 = np.searchsorted(tt, ta), np.searchsorted(tt, ta + 2000, side='right')
                for rule, sel in (('any', np.ones(i1 - i0, bool)),
                                  ('through', (pp[i0:i1] >= ask[ia] + .0099) | (pp[i0:i1] <= bid[ia] - .0099))):
                    j = np.flatnonzero(sel)
                    r[f'{rule}_conf'] = len(j) > 0
                    if not len(j): continue
                    tc = tt[i0 + j[0]]; r[f'{rule}_conf_vs_start_s'] = (tc - t0) / 1000
                    for L in LAT:
                        q = min(np.searchsorted(ts, tc + L), n - 1)
                        net = (x.exit_bid - ask[q]) / 0.01 - fee(ask[q]) - fee(x.exit_bid)
                        r[f'{rule}_net_{L}'] = net
                        r[f'{rule}_done_{L}'] = (ask[q] - x.entry_ask) / max(x.exit_bid - x.entry_ask, 1e-9)
                for L in LAT:                                       # no confirmation: enter at the alert itself
                    q = min(np.searchsorted(ts, ta + L), n - 1)
                    r[f'alert_net_{L}'] = (x.exit_bid - ask[q]) / 0.01 - fee(ask[q]) - fee(x.exit_bid)
            rows.append(r)
    print(sess, len(rows), flush=True)
pd.DataFrame(rows).to_pickle(os.path.join(S, 'capture.pkl'))
