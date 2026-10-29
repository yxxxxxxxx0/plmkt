"""Arbitrage inside Polymarket, 3-way football (home / draw / away, exactly one pays $1).
Buy all YES:  profit = 1 - sum(ask) - fees        Buy all NO: profit = sum(bid) - 1 - fees
(NO on outcome i costs 1 - bid_i; three NOs pay $2). Taker fee 0.05*p(1-p) per leg. Quotes must be
< 2 s old. Size = smallest top-level size across the three legs."""
import json, numpy as np, pandas as pd
VIEW = 'data/view'; fee = lambda p: 0.05 * p * (1 - p)
idx = json.load(open(f'{VIEW}/index.json')); W = []
for tag, sess in idx.items():
    for gi in sess['games']:
        d = pd.read_csv(f"{VIEW}/{tag}/aligned/{gi['slug']}.csv.gz")
        keys = sorted({c.split('_')[0] for c in d.columns if c.endswith('_pm_mid')})
        if len(keys) != 3: continue
        g = lambda c: d[c].to_numpy(float)
        fresh = np.all([np.nan_to_num(g(f'{k}_pm_age_ms'), nan=9e9) < 2000 for k in keys], axis=0)
        A = np.array([g(f'{k}_pm_ask1') for k in keys]); Bd = np.array([g(f'{k}_pm_bid1') for k in keys])
        As = np.array([g(f'{k}_pm_ask1_sz') for k in keys]); Bs = np.array([g(f'{k}_pm_bid1_sz') for k in keys])
        for name, edge, sz in (('buy all YES', 1 - A.sum(0) - fee(A).sum(0), As.min(0)),
                               ('buy all NO', Bd.sum(0) - 1 - fee(1 - Bd).sum(0), Bs.min(0))):
            on = (np.nan_to_num(edge, nan=-1) > 0) & fresh; i = 0
            while i < len(on):
                if on[i]:
                    j = i
                    while j < len(on) and on[j]: j += 1
                    W.append(dict(game=gi['slug'], t=int(d['t_ms'].iloc[i]), kind=name, secs=j - i, edge=float(np.nanmax(edge[i:j])), size=float(np.nanmax(sz[i:j]))))
                    i = j
                else: i += 1
        s_ask, s_bid = np.nanmedian(A.sum(0)[fresh]), np.nanmedian(Bd.sum(0)[fresh])
        print(f"{gi['slug']:26s} median sum of YES asks {s_ask:.3f}, of YES bids {s_bid:.3f}")
W = pd.DataFrame(W)
print(f"\n{len(W)} windows across 11 football games")
if len(W):
    print(W.groupby(['kind', pd.cut(W.secs, [0, 1, 2, 5, 1e9], labels=['1 s', '2 s', '3-5 s', '>5 s'])], observed=True).agg(
        n=('secs', 'size'), median_edge_c=('edge', lambda x: round(100 * x.median(), 2)), median_size=('size', 'median')).to_string())
    L = W[W.secs >= 2]; print(f"2 s+ windows: {len(L)}, profit if all taken at top size: ${(L.edge*L['size']).sum():.2f}")
    ev = json.load(open(f'{VIEW}/jump_events.json', encoding='utf-8'))
    near = L.apply(lambda r: any(e['game'] == r.game and abs(e['t'] - r.t) <= 60_000 for e in ev), axis=1)
    print(f"2 s+ windows within 60 s of a 6c jump: {int(near.sum())}/{len(L)}  (profit in those: ${(L.edge*L['size'])[near].sum():.2f})")
