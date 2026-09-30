# Combined held-out backtest, YES side only: alert -> model(book + game state + YES-team-batting) -> buy YES.
import os
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
FEE = 0.05; fee = lambda p: FEE * p * (1 - p) / 0.01
LAT_MS, STAND, HS = 100, 5, (50, 150, 300)          # 200ms slots: 1s stand, 10/30/60s holds

D = pd.read_pickle(os.path.join(S, 'newmodel_ds.pkl'))
meta = pd.read_csv(os.path.join(IN, 'yes_meta.csv'), dtype={'yes': str}).drop_duplicates('yes').set_index('yes')
side = D.aid.map(meta.yes_side)
D['yes_bats'] = np.where(side == 'Over', 1.0, np.where(side == 'away', D.g_on1 * 0 + (D.g_inning * 0) + np.nan, np.nan))
# half-inning at the alert (causal) from the state timeline
G = pd.read_csv(os.path.join(IN, 'game_state.csv')).sort_values(['slug', 't'])
top = pd.Series(np.nan, index=D.index)
for slug, a in D.groupby('slug'):
    g = G[G.slug == slug]
    if not len(g): continue
    j = np.searchsorted(g.t.to_numpy(), a.ts.to_numpy(), side='right') - 1
    top[a.index] = np.where(j >= 0, g.is_top.to_numpy()[np.clip(j, 0, None)], np.nan)
D['yes_bats'] = np.select([side == 'Over', side == 'away', side == 'home'], [1.0, (top == 1).astype(float), (top == 0).astype(float)], np.nan)
D.loc[top.isna() & (side != 'Over'), 'yes_bats'] = np.nan

# ---- P&L with latency, per alert ----
cols = {f'pnl{h // 5}': np.full(len(D), np.nan) for h in HS}; cols['pnl_best30'] = np.full(len(D), np.nan)
for sess, ds in D.groupby('sess'):
    f = pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet', columns=['mid', 'spread_ticks', 'ts', 'series']).to_pandas()
    lob = np.load(f'data/jump/lob_{sess}_trimmed.npy', mmap_mode='r')
    f = f[f.series.isin(set(ds.series))]
    for sr, s in f.groupby('series', sort=False):
        s = s.sort_values('ts'); ts = s.ts.to_numpy(); n = len(ts)
        spr = s.spread_ticks.to_numpy(); mid = s.mid.to_numpy(); norm = np.median(spr); cut = norm + max(2, norm)
        bid = np.round(mid - spr * .005, 4); ask = np.round(mid + spr * .005, 4)
        L = lob[s.index.to_numpy()]; au = np.expm1(np.asarray(L[:, 3, 0], np.float64)); bu = np.expm1(np.asarray(L[:, 1, 0], np.float64))
        quoted = np.flatnonzero(spr <= cut)
        a = ds[ds.series == sr]
        for ix, t in zip(a.index, a.ts):
            e = np.searchsorted(ts, t + LAT_MS)
            if e >= n or au[e] < 10: continue
            k = D.index.get_loc(ix); px = ask[e]; sh = 10 / px
            for h in HS:
                q = np.searchsorted(quoted, e + h)
                if q < len(quoted) and quoted[q] - (e + h) <= 300:
                    cols[f'pnl{h // 5}'][k] = (bid[quoted[q]] - px) / .01 - fee(px) - fee(bid[quoted[q]])
            w = bid[e + 1:min(n, e + 151)]; ok = bu[e + 1:min(n, e + 151)] >= sh * w
            ww = np.where(ok, w, -np.inf)
            if len(ww) >= STAND:
                best = np.lib.stride_tricks.sliding_window_view(ww, STAND).min(1).max()
                if np.isfinite(best): cols['pnl_best30'][k] = (best - px) / .01 - fee(px) - fee(best)
for c, v in cols.items(): D[c] = v

# ---- model ----
BOOK = [c for c in D.columns if c.startswith('f_')]; GF = [c for c in D.columns if c.startswith('g_')]
TPc = ['t_n10', 't_usd10', 't_n60', 't_game_n10']
F = BOOK + GF + TPc + ['yes_bats']
tr = D.sess <= 'books_2026-09-20'; te = ~tr; va = D.sess.isin(['books_2026-09-19', 'books_2026-09-20'])
X = D[F].assign(mtype=D.mtype.astype('category'))
mk = lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
# threshold chosen on validation: fit on train minus validation, pick the top-q with best mean 30s P&L
m0 = mk().fit(X[tr & ~va], D.y_mark[tr & ~va]); pv = m0.predict_proba(X[va])[:, 1]
best_q, best_v = None, -1e9
for q in (20, 10, 5, 2, 1, 0.5):
    k = pv >= np.percentile(pv, 100 - q); v = np.nanmean(D.pnl30[va].to_numpy()[k])
    if v > best_v: best_q, best_v = q, v
thr = np.percentile(m0.predict_proba(X[tr])[:, 1], 100 - best_q)   # same model; score-level cut from train distribution
m1 = mk().fit(X[tr], D.y_mark[tr]); p = m1.predict_proba(X[te])[:, 1]
T = D[te].assign(p=p)
print(f'test: {len(T)} alerts, {T.sess.nunique()} sessions | up-mark follows {T.y_mark.mean():.2%} | AUC {roc_auc_score(T.y_mark, p):.3f}')
print(f'validation picked top {best_q}% (val mean 30s P&L {best_v:+.2f}t)')
tot_marks = T.y_mark.sum()
rows = []
for q in (20, 10, 5, 2, 1, 0.5):
    k = T.p >= np.percentile(T.p, 100 - q); x = T[k]
    rows.append(dict(top=f'{q}%', n=len(x), picked_by_val=(q == best_q), precision=x.y_mark.mean(), large=x.y_large.mean(),
                     real_any=x.y_real.mean(), recall=x.y_mark.sum() / tot_marks,
                     pnl10=x.pnl10.mean(), pnl30=x.pnl30.mean(), pnl60=x.pnl60.mean(), best30=x.pnl_best30.mean(),
                     usd30=np.nansum(x.pnl30) * .01 * 10 / x.f_price.mean(), win30=(x.pnl30 > 0).mean()))
R = pd.DataFrame(rows).set_index('top')
print(R.round(3).to_string())
k = T.p >= np.percentile(T.p, 100 - best_q); x = T[k]
print('\nchosen cut, by session (mean 30s P&L, ticks):', x.groupby('sess').pnl30.mean().round(2).to_dict())
print('chosen cut, by market type:', x.groupby('mtype').agg(n=('pnl30', 'size'), prec=('y_mark', 'mean'), pnl30=('pnl30', 'mean')).round(3).to_dict('index'))
g = x.groupby('slug').pnl30.mean().dropna(); rng = np.random.default_rng(0)
bs = [np.nanmean(x[x.slug.isin(rng.choice(g.index, len(g)))].pnl30) for _ in range(300)]
print('chosen cut, 30s P&L game-bootstrap 95%% CI: [%.2f, %.2f]' % tuple(np.percentile(bs, [2.5, 97.5])))
T.to_pickle(os.path.join(S, 'combined_test.pkl'))
for q in (5, 1, 0.5):
    x = T[T.p >= np.percentile(T.p, 100 - q)]
    for nm, msk in (('followed by up-mark', x.y_mark), ('  of which large', x.y_large), ('no up-mark', ~x.y_mark)):
        print(f'top {q}% {nm:22s} n {msk.sum():5d}  30s P&L {x[msk].pnl30.mean():+6.2f}t  best-exit {x[msk].pnl_best30.mean():+6.2f}t')
