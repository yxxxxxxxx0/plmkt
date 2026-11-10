# Kill tests 1 (decided, not yet settled) and 4 (extra innings vs a state table), plus the follow-ups
# on what tests.py left standing.     python research/settle/tests2.py
import glob, numpy as np, pandas as pd, warnings
from common import *
warnings.filterwarnings('ignore'); pd.set_option('display.width', 250)
c = lambda x: round(100 * x, 2)
G = grid(); W = windows(); S = settle(G)
MT = G.groupby('asset_id')[['slug', 'market_type', 'line']].first().join(W, on='slug').assign(s=S).dropna(subset=['s', 't1'])
pairs = pd.read_csv(f'{IN}/token_pairs.csv', dtype=str); NO2YES = dict(zip(pairs.no, pairs.yes))
st = pd.read_csv(f'{IN}/game_state.csv').sort_values('t'); st['sec'] = (st.t // 1000).astype('int64'); st['tot'] = st.away + st.home
T = pd.concat([pd.read_csv(f, dtype={'asset_id': str}) for f in glob.glob(f'{IN}/trades/*.csv')]).drop_duplicates()
T = T[T.aggressor.notna()]; isno = T.asset_id.isin(NO2YES)
T['yes'] = np.where(isno, T.asset_id.map(NO2YES), T.asset_id)
T['d'] = np.where(T.aggressor == 'buy', 1, -1) * np.where(isno, -1, 1)
T['px'] = np.where(isno, 1 - T.price, T.price); T['sec'] = T.ts // 1000
T = T[T.yes.isin(MT.index)].join(MT[['s', 't0', 't1']], on='yes')
T['test'] = T.slug.str[-10:] >= TEST_FROM

print('=== 1. Decided but not settled: what the winning side costs after the outcome is fixed')
# decision time: an Over that won is decided when the runs pass the line; everything else at the final out
MT['td'] = MT.t1
for a, r in MT[(MT.market_type == 'total') & (MT.s == 1)].iterrows():
    x = st[(st.slug == r.slug) & (st.tot > r.line)]
    MT.loc[a, 'td'] = x.sec.iloc[0] if len(x) else np.nan
D = MT.dropna(subset=['td']); D['mid_game'] = D.td < D.t1 - 60
for lag in (60, 300, 1800):
    q = asof(G, pd.DataFrame({'asset_id': D.index, 'sec': (D.td + lag).astype('int64').values})).set_index('asset_id').join(D[['s', 'slug', 'mid_game', 'market_type']])
    q['cost'] = np.where(q.s == 1, q.ask, 1 - q.bid); q['sz'] = np.where(q.s == 1, q.asz, q.bsz)
    q = q.dropna(subset=['cost']); q = q[q.cost > 0.5]            # a "winner" quoted under 50c is a mislabel or a stub, not an offer
    q['edge'] = 1 - q.cost - FEE(q.cost)
    q['usd'] = np.minimum(q.sz, 100 / q.cost) * q.edge.clip(lower=0)   # most a $100 account could take from this one quote
    for mg, x in q.groupby('mid_game'):
        ok = x[(x.edge > 0) & (x.sz >= 5)]
        print(f'  +{lag:>4}s  {"Over decided mid-game" if mg else "at the final out     "}: {len(x):>4} contracts quoted, {len(ok):>4} with a winner on offer below $1 '
              f'| edge median {c(ok.edge.median()):.2f}c, mean {c(ok.edge.mean()):.2f}c | per contract ${ok.usd.mean():.3f} | per game, best single contract ${ok.groupby("slug").usd.max().mean():.3f} '
              f'| quote age median {ok.age.median():.0f}s')
    if lag == 60: Q60 = q
print('  the biggest edges at +60s (check for stale or stub quotes):')
print(Q60.sort_values('edge', ascending=False).head(8)[['slug', 'market_type', 'cost', 'sz', 'age', 'edge']].to_string())
# did anyone actually trade the decided winner below $1? (taker side, in winner terms)
td = T.join(D.td.rename('td'), on='yes').dropna(subset=['td']); td = td[td.sec > td.td + 30]
td['wpx'] = np.where(td.s == 1, td.px, 1 - td.px); td['buywin'] = np.where(td.s == 1, td.d, -td.d) > 0
td = td[td.wpx > 0.5]
for bw, x in td.groupby('buywin'):
    x = x.assign(e=1 - x.wpx)
    print(f'  prints after decision, taker {"BOUGHT" if bw else "SOLD  "} the winner: {len(x):,} prints in {x.slug.nunique()} games, {x["size"].sum()/x.slug.nunique():,.0f} shares/game, '
          f'discount to $1: median {c(x.e.median()):.2f}c, share-weighted {c((x.e*x["size"]).sum()/x["size"].sum()):.2f}c, total ${(x.e*x["size"]).sum()/x.slug.nunique():.2f}/game')

print('\n=== 2b. Moneyline bought at 98c or more in-game: how often does it lose?')
K = pd.read_pickle('research/settle/cache/calib.pkl'); K = K[(K.market_type == 'moneyline') & ~K.pre]
B = pd.concat([K.assign(p=K.ask, win=K.s), K.assign(p=1 - K.bid, win=1 - K.s)])
for lo, hi in ((.9, .95), (.95, .98), (.98, .995), (.995, 1)):
    x = B[(B.p >= lo) & (B.p < hi)]
    g = x.groupby('slug').win.min()
    print(f'  paid [{lo}, {hi}): {len(x)} samples in {len(g)} games, {int((1-x.win).sum())} losing samples in {int((g == 0).sum())} games; mean price {x.p.mean():.4f}; '
          f'break-even win rate {(x.p + FEE(x.p)).mean():.4f}, observed by game {g.mean():.4f}')

print('\n=== 4. Extra innings against a state table (inning x margin), fit before', TEST_FROM)
pa = st[(st.kind == 'pa_end') & (st.inning >= 6) & (st.inning <= 9)].copy()
pa['xi'] = pa.slug.map(st.groupby('slug').inning.max() > 9) * 1.0
pa['m'] = (pa.away - pa.home).abs().clip(upper=4); pa['test'] = pa.slug.str[-10:] >= TEST_FROM
tab = pa[~pa.test].groupby(['inning', 'm']).xi.mean().rename('fair')
xt = MT[MT.market_type == 'extra_innings'].reset_index()[['asset_id', 'slug', 's']]
e = pa[pa.test].merge(xt, on='slug').join(tab, on=['inning', 'm']).dropna(subset=['fair'])
e['sec'] = e.sec + 10                                            # statsapi stamps the play; give the book 10 s
q = asof(G, e[['asset_id', 'sec']].assign(i=np.arange(len(e)))).sort_values('i')
e = e.assign(bid=q.bid.values, ask=q.ask.values, asz=q.asz.values, bsz=q.bsz.values).dropna(subset=['bid', 'ask'])
print(f'  fit games {pa[~pa.test].slug.nunique()} (base rate {pa[~pa.test].groupby("slug").xi.first().mean():.3f}), test games with the market {e.slug.nunique()}, '
      f'plate appearances {len(e)}, median spread {c((e.ask - e.bid).median()):.0f}c')
for marg in (0.03, 0.06):
    y = e[e.fair - e.ask - FEE(e.ask) > marg].assign(pnl=lambda x: x.s - x.ask - FEE(x.ask))
    n = e[e.bid - e.fair - FEE(e.bid) > marg].assign(pnl=lambda x: x.bid - x.s - FEE(x.bid))
    x = pd.concat([y, n]).groupby(['slug', 'inning']).head(1)    # one entry per game per inning
    if len(x): m, lo, hi = boot(x, 'pnl'); print(f'  edge > {c(marg):.0f}c: {len(x)} trades ({len(y.groupby(["slug","inning"]).head(1))} YES) in {x.slug.nunique()} games, {c(m):+.2f}c/share [{c(lo):+.2f}, {c(hi):+.2f}]')
    else: print(f'  edge > {c(marg):.0f}c: 0 trades')
# hindsight bound: buy the side that wins, at the quote, once per game per inning
h = e.groupby(['slug', 'inning']).head(1)
h['pnl'] = np.where(h.s == 1, 1 - h.ask - FEE(h.ask), h.bid - FEE(h.bid))
print(f'  hindsight (always the right side): {len(h)} trades, {c(h.pnl.mean()):+.2f}c/share')

print('\n=== 3b/6b. Pre-game maker: are there fills, and what is left after the price moves to first pitch?')
pre = T[T.sec < T.t0 - 60].copy()
q = asof(G, pd.DataFrame({'asset_id': pre.yes.values, 'sec': (pre.t0 - 60).astype('int64').values, 'i': np.arange(len(pre))})).sort_values('i')
pre['close'] = ((q.bid + q.ask) / 2).values                      # mid one minute before first pitch
q = asof(G, pd.DataFrame({'asset_id': pre.yes.values, 'sec': pre.sec.values - 1, 'i': np.arange(len(pre))})).sort_values('i')
pre['spr'] = (q.ask - q.bid).values; pre['touch'] = np.where(pre.d.values > 0, q.asz, q.bsz)
pre['mk_close'] = -pre.d * (pre.close - pre.px); pre['mk_set'] = -pre.d * (pre.s - pre.px)
pre['mt'] = np.where(pre.market_type.isin(['moneyline', 'spread', 'total', 'first_inning_run']), pre.market_type, 'other props')
for mt, x in pre.groupby('mt'):
    x = x.dropna(subset=['mk_close'])
    ng = x.slug.nunique(); buy = (x.d > 0).mean()
    print(f'  {mt:17s} {len(x)/ng:5.0f} prints/game, {x["size"].sum()/ng:7,.0f} shares/game ({buy:.0%} taker buys of YES), spread median {c(x.spr.median()):.0f}c, '
          f'shares resting at the touch median {x.touch.median():,.0f}')
    for tn, y in (('fit ', x[~x.test]), ('test', x[x.test])):
        a = boot(y, 'mk_close'); b = boot(y, 'mk_set')
        print(f'      {tn} maker edge at first pitch {c(a[0]):+.2f}c/share [{c(a[1]):+.2f}, {c(a[2]):+.2f}] | held to settlement {c(b[0]):+.2f}c [{c(b[1]):+.2f}, {c(b[2]):+.2f}] | whole market ${ (y.mk_close*y["size"]).sum()/max(y.slug.nunique(),1):.2f}/game')
