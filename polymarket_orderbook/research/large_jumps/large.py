# Large jumps only, YES side. Entry: walk the ask for $10 at the alert tick (the "+50ms" book; grid is 200ms).
# Exit: after >=5s, the first time the book has been quoted and still for S seconds -> walk the bid.
import os
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
FEE, STAKE, MINHOLD, TIMEOUT = 0.05, 10.0, 25, 450          # slots: 5s min hold, 90s timeout
STABLE = (10, 15, 25)                                       # 2s, 3s, 5s of stillness
fee = lambda p: FEE * p * (1 - p) / 0.01

D = pd.read_pickle(os.path.join(S, 'combined_test.pkl'))    # (test rows only) -> rebuild full set below
D = pd.read_pickle(os.path.join(S, 'newmodel_ds.pkl'))
meta = pd.read_csv(os.path.join(IN, 'yes_meta.csv'), dtype={'yes': str}).drop_duplicates('yes').set_index('yes')
G = pd.read_csv(os.path.join(IN, 'game_state.csv')).sort_values(['slug', 't'])
side = D.aid.map(meta.yes_side); top = pd.Series(np.nan, index=D.index)
for slug, a in D.groupby('slug'):
    g = G[G.slug == slug]
    if len(g):
        j = np.searchsorted(g.t.to_numpy(), a.ts.to_numpy(), side='right') - 1
        top[a.index] = np.where(j >= 0, g.is_top.to_numpy()[np.clip(j, 0, None)], np.nan)
D['yes_bats'] = np.select([side == 'Over', side == 'away', side == 'home'], [1.0, (top == 1) * 1.0, (top == 0) * 1.0], np.nan)
D.loc[top.isna() & (side != 'Over'), 'yes_bats'] = np.nan


def walk(px, usd_lvls, dollars=None, shares=None):
    """Fill along price levels (ascending for buys, descending for sells). Returns (avg price, filled fraction)."""
    got_sh = got_usd = 0.0
    for p, u in zip(px, usd_lvls):
        if p <= 0 or p >= 1 or u <= 0: continue
        sh = u / p
        take = min(sh, (dollars - got_usd) / p) if dollars is not None else min(sh, shares - got_sh)
        got_sh += take; got_usd += take * p
        if (dollars is not None and got_usd >= dollars - 1e-9) or (shares is not None and got_sh >= shares - 1e-9): break
    need = dollars if dollars is not None else shares
    have = got_usd if dollars is not None else got_sh
    return (got_usd / got_sh if got_sh else np.nan), have / need


out = {f'pnl_s{s // 5}': np.full(len(D), np.nan) for s in STABLE}
out.update(hold_s=np.full(len(D), np.nan), entry_fill=np.full(len(D), np.nan), entry_slip=np.full(len(D), np.nan))
for sess, ds in D.groupby('sess'):
    f = pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet', columns=['mid', 'spread_ticks', 'ts', 'series']).to_pandas()
    lob = np.load(f'data/jump/lob_{sess}_trimmed.npy', mmap_mode='r')
    f = f[f.series.isin(set(ds.series))]
    for sr, s in f.groupby('series', sort=False):
        s = s.sort_values('ts'); ts = s.ts.to_numpy(); n = len(ts)
        spr = s.spread_ticks.to_numpy(); mid = s.mid.to_numpy(); norm = np.median(spr); cut = norm + max(2, norm)
        bid = mid - spr * .005; ask = mid + spr * .005
        L = np.asarray(lob[s.index.to_numpy()], np.float64)
        contig = pd.Series(np.r_[1, np.diff(ts) == 200])
        stab = {}
        for w in STABLE:
            rb, ra = pd.Series(bid).rolling(w), pd.Series(ask).rolling(w)
            stab[w] = np.flatnonzero((((rb.max() - rb.min()) <= .011) & ((ra.max() - ra.min()) <= .011)
                                      & (pd.Series(spr).rolling(w).max() <= cut) & (contig.rolling(w).min() == 1)).to_numpy())
        quoted = np.flatnonzero(spr <= cut)
        a = ds[ds.series == sr]
        for ix, t in zip(a.index, a.ts):
            e = np.searchsorted(ts, t)                       # the alert tick's book ~ what a +50ms order meets
            if e >= n - MINHOLD: continue
            apx = mid[e] + L[e, 2] * .01; ausd = np.expm1(L[e, 3])
            px, fill = walk(apx, ausd, dollars=STAKE)
            k = D.index.get_loc(ix); out['entry_fill'][k] = fill; out['entry_slip'][k] = (px - ask[e]) / .01
            if fill < 0.999: continue                       # the whole book could not fill $10
            sh = STAKE / px
            for w in STABLE:
                q = np.searchsorted(stab[w], e + MINHOLD)
                j = stab[w][q] if q < len(stab[w]) and stab[w][q] <= e + TIMEOUT else None
                if j is None:
                    q2 = np.searchsorted(quoted, e + TIMEOUT)
                    if q2 >= len(quoted): continue
                    j = quoted[q2]
                bpx = mid[j] + L[j, 0] * .01; busd = np.expm1(L[j, 1])
                xp, xf = walk(bpx, busd, shares=sh)
                if xf < 0.999: continue
                out[f'pnl_s{w // 5}'][k] = (xp - px) / .01 - fee(px) - fee(xp)
                if w == 15: out['hold_s'][k] = (j - e) * .2
    print(sess, flush=True)
for c, v in out.items(): D[c] = v
D.to_pickle(os.path.join(S, 'large_ds.pkl'))

BOOK = [c for c in D.columns if c.startswith('f_')]; GF = [c for c in D.columns if c.startswith('g_')]
F = BOOK + GF + ['t_n10', 't_usd10', 't_n60', 't_game_n10', 'yes_bats']
tr = D.sess <= 'books_2026-09-20'; te = ~tr; va = D.sess.isin(['books_2026-09-19', 'books_2026-09-20'])
X = D[F].assign(mtype=D.mtype.astype('category'))
mk = lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
print(f'\nentry: $10 fillable on the book {np.nanmean(D.entry_fill >= .999):.1%} of alerts; slippage past the best ask, median {np.nanmedian(D.entry_slip):.2f}t, 95th pct {np.nanpercentile(D.entry_slip, 95):.2f}t')
m0 = mk().fit(X[tr & ~va], D.y_large[tr & ~va]); pv = m0.predict_proba(X[va])[:, 1]
cands = (10, 5, 2, 1, 0.5, 0.25)
vals = {q: np.nanmean(D.pnl_s3[va].to_numpy()[pv >= np.percentile(pv, 100 - q)]) for q in cands}
best_q = max(vals, key=vals.get)
print('validation mean P&L (3s-stable exit) by cut:', {q: round(v, 2) for q, v in vals.items()}, '-> picked top', best_q, '%')
m1 = mk().fit(X[tr], D.y_large[tr]); p = m1.predict_proba(X[te])[:, 1]; T = D[te].assign(p=p)
print(f'test: {len(T)} alerts | large up-mark follows {T.y_large.mean():.2%} | AUC {roc_auc_score(T.y_large, p):.3f}')
rows = []
for q in cands:
    x = T[T.p >= np.percentile(T.p, 100 - q)]
    rows.append(dict(top=f'{q}%', chosen=q == best_q, n=len(x), prec_large=x.y_large.mean(), prec_any_up=x.y_mark.mean(),
                     recall_large=x.y_large.sum() / T.y_large.sum(), pnl_2s=x.pnl_s2.mean(), pnl_3s=x.pnl_s3.mean(),
                     pnl_5s=x.pnl_s5.mean(), win_3s=(x.pnl_s3 > 0).mean(), hold_med=x.hold_s.median(), usd_3s=np.nansum(x.pnl_s3) * .01 * 10 / x.f_price.mean()))
print(pd.DataFrame(rows).set_index('top').round(3).to_string())
x = T[T.p >= np.percentile(T.p, 100 - best_q)]
for nm, msk in (('large up-mark', x.y_large), ('smaller up-mark', x.y_mark & ~x.y_large), ('no up-mark', ~x.y_mark)):
    print(f'  chosen cut, {nm:16s} n {msk.sum():4d}  P&L (3s-stable) {x[msk].pnl_s3.mean():+6.2f}t  hold {x[msk].hold_s.median():.0f}s')
print('  by session:', x.groupby('sess').pnl_s3.mean().round(2).to_dict())
g = x.slug.unique(); rng = np.random.default_rng(0)
bs = [np.nanmean(x[x.slug.isin(rng.choice(g, len(g)))].pnl_s3) for _ in range(300)]
print('  game-bootstrap 95%% CI: [%.2f, %.2f]' % tuple(np.percentile(bs, [2.5, 97.5])))
print(f'  break-even precision on large at this cut ~ {-x[~x.y_large].pnl_s3.mean() / (x[x.y_large].pnl_s3.mean() - x[~x.y_large].pnl_s3.mean()):.0%}')
