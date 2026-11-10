# Kill tests for the seven hold-to-settlement / quiet-time hypotheses.   python research/settle/tests.py
import glob, numpy as np, pandas as pd, warnings
from common import *
warnings.filterwarnings('ignore'); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
c = lambda x: round(100 * x, 2)                      # dollars/share -> cents/share
G = grid(); M = meta().set_index('yes'); W = windows(); S = settle(G)
MT = G.groupby('asset_id')[['slug', 'market_type', 'line']].first()
pairs = pd.read_csv(f'{IN}/token_pairs.csv', dtype=str); NO2YES = dict(zip(pairs.no, pairs.yes))
st = pd.read_csv(f'{IN}/game_state.csv'); st['sec'] = (st.t // 1000).astype('int64')
print(f'grid rows {len(G):,}  tokens {G.asset_id.nunique()}  games {G.slug.nunique()}  settled {S.notna().sum()} (yes {int(S.sum())})  unknown {S.isna().sum()}')

# settlement check: moneyline payoff from last quote vs the final score in the state file
fin = st.sort_values('t').groupby('slug').tail(1).set_index('slug')
ml = MT[MT.market_type == 'moneyline'].join(M.yes_side).join(S.rename('s')).dropna(subset=['s'])
ml = ml.join(fin[['away', 'home']], on='slug').dropna(subset=['away'])
ml = ml[ml.away != ml.home]                           # last state row can predate a walk-off
ml['sc'] = np.where(ml.yes_side == 'away', ml.away > ml.home, ml.home > ml.away) * 1.0
print(f'settlement check, moneyline: {len(ml)} games, last-quote payoff agrees with final score in {(ml.sc == ml.s).mean():.1%}')

# ---- trades, everything in YES-token terms: d=+1 taker bought YES at px, d=-1 taker sold YES at px
T = pd.concat([pd.read_csv(f, dtype={'asset_id': str}) for f in glob.glob(f'{IN}/trades/*.csv')]).drop_duplicates()
T = T[T.aggressor.notna()]
isno = T.asset_id.isin(NO2YES)
T['yes'] = np.where(isno, T.asset_id.map(NO2YES), T.asset_id)
T['d'] = np.where(T.aggressor == 'buy', 1, -1) * np.where(isno, -1, 1)
T['px'] = np.where(isno, 1 - T.price, T.price)
T['fee'] = FEE(T.price); T['sec'] = T.ts // 1000
T = T[T.yes.isin(S.index)].assign(s=lambda x: x.yes.map(S)).dropna(subset=['s'])
T['taker'] = T.d * (T.s - T.px) - T.fee              # $/share to settlement
T['maker'] = -T.d * (T.s - T.px)                     # makers pay no fee
T['test'] = T.slug.str[-10:] >= TEST_FROM
T = T.join(W, on='slug')
print(f'signed prints with a settled contract: {len(T):,} in {T.slug.nunique()} games')

def line(df, col, w=None):
    m, lo, hi = boot(df, col)
    return f'{len(df):>7,} prints  {c(m):+6.2f}c/share  [{c(lo):+.2f}, {c(hi):+.2f}]'

print('\n=== 3. Prop takers held to settlement: what the MAKER side of every print earned (c/share, game-clustered 95% CI)')
for mt, d in T.groupby('market_type'):
    for nm, x in (('fit ', d[~d.test]), ('test', d[d.test])):
        if len(x): print(f'  {mt:18s} {nm} {line(x, "maker")}   per game {len(x)/x.slug.nunique():.0f} prints, median {x["size"].median():.0f} sh')
x = T[(T.market_type != 'moneyline')]
print('  props, by taker price paid (all sessions):')
x['pb'] = pd.cut(np.where(x.d > 0, x.px, 1 - x.px), [0, .1, .25, .4, .6, .75, .9, 1])
for b, d in x.groupby('pb'): print(f'    taker paid {str(b):12s} {line(d, "maker")}')

print('\n=== 2. Calibration: buy at the ask every 10 min, hold to settlement (c/share after fee; price paid bucket)')
rows = []
for a, r in MT.join(W, on='slug').dropna(subset=['t0']).iterrows():
    for t in np.arange(r.t0 - 1800, r.t1, 600): rows.append((a, int(t), t < r.t0))
K = asof(G, pd.DataFrame(rows, columns=['asset_id', 'sec', 'pre'])).dropna(subset=['bid', 'ask'])
K = K[(K.age < 600) & (K.ask > K.bid)].join(MT, on='asset_id').assign(s=lambda x: x.asset_id.map(S)).dropna(subset=['s'])
B = pd.concat([K.assign(side='YES', p=K.ask, pnl=K.s - K.ask - FEE(K.ask)),
               K.assign(side='NO', p=1 - K.bid, pnl=(1 - K.s) - (1 - K.bid) - FEE(K.bid))])
B = B[(B.p > 0) & (B.p < 1)]
B['pb'] = pd.cut(B.p, [0, .05, .1, .25, .4, .6, .75, .9, .95, .98, 1]); B['test'] = B.slug.str[-10:] >= TEST_FROM
B['grp'] = np.where(B.market_type == 'moneyline', 'moneyline', np.where(B.market_type.isin(['spread', 'total']), 'spread/total', 'small props'))
for (grp, pre), d in B.groupby(['grp', 'pre']):
    print(f'  {grp}, {"pre-game" if pre else "in-game"}')
    for b, e in d.groupby('pb'):
        if len(e) < 30: continue
        f, t = e[~e.test], e[e.test]
        print(f'    paid {str(b):13s} fit {line(f, "pnl")} | test {line(t, "pnl") if len(t) else ""}')
K.to_pickle('research/settle/cache/calib.pkl')

print('\n=== 7. Follow large signed prints on the moneyline: enter 1 s later at the ask, hold to settlement')
m = T[(T.market_type == 'moneyline') & (T.sec >= T.t0) & (T.sec <= T.t1)].copy(); m['usd'] = m.price * m['size']
thr = m[~m.test].usd.quantile(.99)
def follow(x):
    q = asof(G, pd.DataFrame({'asset_id': x.yes.values, 'sec': x.sec.values + 1, 'i': np.arange(len(x))})).sort_values('i')
    x = x.assign(bid=q.bid.values, ask=q.ask.values, age=q.age.values).dropna(subset=['bid', 'ask'])
    x = x[(x.ask > x.bid) & (x.ask < 1) & (x.bid > 0)]
    x['pnl'] = np.where(x.d > 0, x.s - x.ask - FEE(x.ask), (1 - x.s) - (1 - x.bid) - FEE(x.bid))
    return x
big = follow(m[m.usd >= thr])
rnd = follow(m[m.usd < thr].sample(min(len(m), 20000), random_state=0))    # ordinary prints, same games, same rule
rr = rnd.assign(pnl=np.where(-rnd.d > 0, rnd.s - rnd.ask - FEE(rnd.ask), (1 - rnd.s) - (1 - rnd.bid) - FEE(rnd.bid)))
print(f'  large = top 1% by dollars on fit sessions (>= ${thr:,.0f})')
for nm, x in (('large, follow', big), ('ordinary prints, follow', rnd), ('ordinary prints, fade', rr)):
    for tn, y in (('fit ', x[~x.test]), ('test', x[x.test])): print(f'  {nm:24s} {tn} {line(y, "pnl")}')

print('\n=== 5/6. Touch maker by game phase (moneyline): every print is a fill for the resting side; value 30 s and 5 min later')
m = T[T.market_type == 'moneyline'].copy()
h = st.sort_values('t'); h['half'] = h.inning * 2 + (~h.is_top)
e = h.groupby(['slug', 'half']).sec.agg(['min', 'max']).reset_index().sort_values(['slug', 'half'])
e['nxt'] = e.groupby('slug')['min'].shift(-1)
e = e.dropna(subset=['nxt']); e['b0'] = e['max'] + 5; e['b1'] = e.nxt - 20     # last out stamped late; stop 20 s before next pitch ends
e = e[e.b1 > e.b0]
print(f'  inning breaks: {len(e)} in {e.slug.nunique()} games, median usable length {(e.b1 - e.b0).median():.0f} s')
m['phase'] = np.where(m.sec < m.t0, 'pre-game', np.where(m.sec > m.t1, 'post-game', 'in play'))
brk = pd.merge_asof(m.sort_values('sec'), e[['slug', 'b0', 'b1']].sort_values('b0').astype({'b0': 'int64'}), left_on='sec', right_on='b0', by='slug')
m = brk; m.loc[(m.phase == 'in play') & (m.sec <= m.b1), 'phase'] = 'inning break'
for hz in (30, 300):
    q = asof(G, pd.DataFrame({'asset_id': m.yes.values, 'sec': m.sec.values + hz, 'i': np.arange(len(m))})).sort_values('i')
    mid = ((q.bid + q.ask) / 2).values
    m[f'mk{hz}'] = -m.d.values * (mid - m.px.values)
    m[f'mx{hz}'] = np.where(m.d.values < 0, q.bid.values - m.px.values - FEE(q.bid.values), m.px.values - q.ask.values - FEE(q.ask.values))
for ph, d in m.groupby('phase'):
    print(f'  {ph:13s} {len(d)/d.slug.nunique():6.0f} prints/game')
    for col, nm in (('mk30', 'at mid +30s '), ('mk300', 'at mid +5min'), ('mx300', 'crossed out +5min'), ('maker', 'held to settlement')):
        for tn, y in (('fit ', d[~d.test]), ('test', d[d.test])):
            y = y.dropna(subset=[col])
            if len(y): print(f'      {nm:18s} {tn} {line(y, col)}')
