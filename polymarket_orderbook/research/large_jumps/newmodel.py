# Which alerts turn into real jumps (either direction)? Book vs +game state vs +trades before vs +trades after.
import os, glob
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs

A = pd.read_pickle(os.path.join(S, 'alerts_ds.pkl'))
A = A[A.sess >= 'books_2026-09-10'].reset_index(drop=True)          # sessions with trades
A['aid'] = A.series.str.split('|').str[-1]
BOOK = [c for c in A.columns if c.startswith('f_')]

# ---------------- game state (causal: latest statsapi state stamped at or before t) ----------------
# Approximate MLB run expectancy for the rest of the inning, by (outs, bases); a direction-free leverage proxy.
RE = {0: [.48, .86, 1.10, 1.44, 1.35, 1.78, 1.96, 2.29], 1: [.25, .51, .66, .88, .95, 1.13, 1.35, 1.54],
      2: [.10, .22, .32, .43, .35, .48, .57, .75]}      # bases idx: empty,1,2,12,3,13,23,123
G = pd.read_csv(os.path.join(IN, 'game_state.csv')).sort_values(['slug', 't'])
GF = ['g_inning', 'g_outs', 'g_on1', 'g_on2', 'g_on3', 'g_balls', 'g_strikes', 'g_pitches_pa', 'g_margin',
      'g_runs', 'g_re24', 'g_risp', 'g_late_close', 'g_since_event', 'g_full']


def add_state(A, lag_ms=0):
    out = pd.DataFrame(index=A.index, columns=GF, dtype=float)
    for slug, a in A.groupby('slug'):
        g = G[G.slug == slug]
        if not len(g): continue
        i = np.searchsorted(g.t.to_numpy(), a.ts.to_numpy() - lag_ms, side='right') - 1
        ok = i >= 0; r = g.iloc[np.clip(i, 0, None)]
        outs = r.outs.clip(upper=2).to_numpy(); b = (r.on1 + 2 * r.on2 + 4 * r.on3).to_numpy()
        bidx = np.array([0, 1, 2, 3, 4, 5, 6, 7])[b]
        v = pd.DataFrame({
            'g_inning': r.inning.to_numpy(), 'g_outs': r.outs.to_numpy(), 'g_on1': r.on1.to_numpy(),
            'g_on2': r.on2.to_numpy(), 'g_on3': r.on3.to_numpy(), 'g_balls': r.balls.to_numpy(),
            'g_strikes': r.strikes.to_numpy(), 'g_pitches_pa': r.pitches_pa.to_numpy(),
            'g_margin': (r.home - r.away).abs().to_numpy(), 'g_runs': (r.home + r.away).to_numpy(),
            'g_re24': [RE[o][k] for o, k in zip(outs, bidx)], 'g_risp': (r.on2 | r.on3).to_numpy(),
            'g_late_close': ((r.inning >= 7) & ((r.home - r.away).abs() <= 1)).to_numpy(),
            'g_since_event': (a.ts.to_numpy() - lag_ms - r.t.to_numpy()) / 1000,
            'g_full': ((r.balls == 3) & (r.strikes == 2)).to_numpy()}, index=a.index).astype(float)
        v[~ok] = np.nan
        out.loc[a.index] = v
    return out.astype(float)


# ---------------- trades, mapped onto the YES token ----------------
pairs = pd.read_csv(os.path.join(IN, 'token_pairs.csv'), dtype=str)
no2yes = dict(zip(pairs.no, pairs.yes)); yesset = set(pairs.yes)
T = pd.concat([pd.read_csv(f, dtype={'asset_id': str}) for f in glob.glob(os.path.join(IN, 'trades', 'books_2026-09-*.csv'))])
isno = T.asset_id.isin(no2yes)
T['yes'] = np.where(isno, T.asset_id.map(no2yes), T.asset_id)
T = T[T.yes.isin(yesset)].copy()
T['p'] = np.where(isno[T.index] if False else T.asset_id.isin(no2yes), 1 - T.price, T.price)   # YES-equivalent price
T['usd'] = T.size * T.price
T = T.sort_values('ts')
TB = {k: (v.ts.to_numpy(), v.p.to_numpy(), v.usd.to_numpy()) for k, v in T.groupby('yes')}
TG = {k: v.ts.to_numpy() for k, v in T.groupby('slug')}


def win(arr_t, t, lo, hi):
    return np.searchsorted(arr_t, t + lo, side='left'), np.searchsorted(arr_t, t + hi, side='right')


def add_trades(A):
    pre = {c: np.zeros(len(A)) for c in ['t_n10', 't_usd10', 't_n60', 't_game_n10']}
    post = {c: np.zeros(len(A)) for c in ['t_n1', 't_usd1', 't_through1', 't_n2', 't_usd2', 't_through2']}
    # book prices at the alert for "trade through the old level"
    bid = A.f_price.to_numpy() - A.f_spr_rel.to_numpy() * A.f_norm_spr.to_numpy() * .01
    ask = A.f_price.to_numpy()
    for k, (aid, slug, t) in enumerate(zip(A.aid, A.slug, A.ts)):
        if aid in TB:
            tt, pp, uu = TB[aid]
            i0, i1 = win(tt, t, -10000, -1); pre['t_n10'][k] = i1 - i0; pre['t_usd10'][k] = uu[i0:i1].sum()
            i0, i1 = win(tt, t, -60000, -1); pre['t_n60'][k] = i1 - i0
            for s, hi in (('1', 1000), ('2', 2000)):
                i0, i1 = win(tt, t, 0, hi)
                post['t_n' + s][k] = i1 - i0; post['t_usd' + s][k] = uu[i0:i1].sum()
                post['t_through' + s][k] = ((pp[i0:i1] >= ask[k] + .0099) | (pp[i0:i1] <= bid[k] - .0099)).sum()
        if slug in TG:
            i0, i1 = win(TG[slug], t, -10000, -1); pre['t_game_n10'][k] = i1 - i0
    return pd.DataFrame(pre, index=A.index), pd.DataFrame(post, index=A.index)


GS = add_state(A); GS10 = add_state(A, 10000)
TP, TQ = add_trades(A)
D = pd.concat([A, GS, TP, TQ], axis=1)
D.to_pickle(os.path.join(S, 'newmodel_ds.pkl'))
print('alerts', len(D), 'games', D.slug.nunique(), '| state found', GS.g_inning.notna().mean().round(3),
      '| any trade within 2s after alert', (TQ.t_n2 > 0).mean().round(3), '| real-jump rate', D.y_real.mean().round(4))

tr = D.sess <= 'books_2026-09-20'; te = ~tr
print('train sessions', D[tr].sess.nunique(), 'alerts', tr.sum(), '| test sessions', D[te].sess.nunique(), 'alerts', te.sum())
TPc, TQc = list(TP.columns), list(TQ.columns)


def fit(cols, y='y_real', data=D):
    X = data[cols].assign(mtype=data.mtype.astype('category'))
    mdl = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    mdl.fit(X[tr], data.loc[tr, y]); p = mdl.predict_proba(X[te])[:, 1]; yt = data.loc[te, y].to_numpy()
    res = dict(auc=roc_auc_score(yt, p), ap=average_precision_score(yt, p), base=yt.mean())
    for q in (10, 5, 1):
        k = p >= np.percentile(p, 100 - q); res[f'top{q}'] = yt[k].mean()
        res[f'rec{q}'] = yt[k].sum() / yt.sum()
    return res, mdl, X.columns


sets = [('book', BOOK), ('book + game', BOOK + GF), ('book + game + trades before', BOOK + GF + TPc),
        ('+ trades 1-2s after (detection)', BOOK + GF + TPc + TQc), ('game only', GF), ('trades after only', TQc)]
rows = []
for nm, cols in sets:
    r, mdl, cn = fit(cols); r['set'] = nm; rows.append(r)
    if nm in ('book + game + trades before', '+ trades 1-2s after (detection)'):
        imp = pd.Series(mdl.booster_.feature_importance('gain'), index=cn); print(nm, 'top gain', (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict())
R = pd.DataFrame(rows).set_index('set')
print(R[['auc', 'ap', 'base', 'top10', 'top5', 'top1', 'rec1']].round(3).to_string())

# leakage check: game state looked up 10s earlier
D10 = D.copy(); D10[GF] = GS10.values
r10, _, _ = fit(BOOK + GF, data=D10)
print('\nleak check  book+game, state 10s older: auc %.3f top1 %.3f (vs %.3f / %.3f)' % (r10['auc'], r10['top1'], R.loc['book + game', 'auc'], R.loc['book + game', 'top1']))

# simple readouts
print('\nreal-jump rate by game-state slice (all alerts):')
for nm, grp in [('re24', pd.qcut(D.g_re24, 4, duplicates='drop')), ('risp', D.g_risp), ('outs', D.g_outs.clip(upper=2)),
                ('late&close', D.g_late_close), ('margin', D.g_margin.clip(upper=5)), ('strikes', D.g_strikes),
                ('trades 2s after', D.t_n2.clip(upper=3)), ('trade through old level 2s', D.t_through2.clip(upper=2))]:
    t = D.groupby(grp).y_real.agg(['size', 'mean']); print(' ', nm, {str(k): (int(v['size']), round(v['mean'], 3)) for k, v in t.iterrows()})
