"""Cross-venue arbitrage: buy YES on one venue and NO on the other when the pair costs < $1 after
taker fees (Polymarket 0.05*p(1-p), Kalshi 0.07*p(1-p)); hold both to settlement for $1.
Locked profit per share = bid_A - ask_B - fees (buying NO on A at 1 - bid_A).
Counts windows (consecutive seconds the gap is open), their length, and size at the top level."""
import json, numpy as np, pandas as pd
VIEW = 'data/view'
fee = lambda r, p: r * p * (1 - p)
idx = json.load(open(f'{VIEW}/index.json'))
W = []
for tag, sess in idx.items():
    for gi in sess['games']:
        d = pd.read_csv(f"{VIEW}/{tag}/aligned/{gi['slug']}.csv.gz")
        keys = sorted({c.split('_')[0] for c in d.columns if c.endswith('_pm_mid')})
        for k in keys:
            g = lambda c: d[c].to_numpy(float) if c in d else np.full(len(d), np.nan)
            for A, B in (('pm', 'ks'), ('ks', 'pm')):          # sell YES on A (= buy NO at 1-bid), buy YES on B
                bid, ask = g(f'{k}_{A}_bid1'), g(f'{k}_{B}_ask1')
                fresh = (np.nan_to_num(g(f'{k}_{A}_age_ms'), nan=9e9) < 2000) & (np.nan_to_num(g(f'{k}_{B}_age_ms'), nan=9e9) < 2000)
                sz = np.minimum(g(f'{k}_{A}_bid1_sz'), g(f'{k}_{B}_ask1_sz'))
                fa, fb = {'pm': .05, 'ks': .07}[A], {'pm': .05, 'ks': .07}[B]
                edge = bid - ask - fee(fa, 1 - bid) - fee(fb, ask)
                on = (np.nan_to_num(edge, nan=-1) > 0) & fresh
                i = 0
                while i < len(on):
                    if on[i]:
                        j = i
                        while j < len(on) and on[j]: j += 1
                        W.append(dict(game=gi['slug'], k=k, dir=f'{A}->{B}', secs=j - i, edge=float(np.nanmax(edge[i:j])),
                                      size=float(np.nanmax(sz[i:j])), usd=float(np.nanmax(edge[i:j] * sz[i:j]))))
                        i = j
                    else:
                        i += 1
W = pd.DataFrame(W)
print(f"{len(W)} arbitrage windows across 12 games (gap open after fees, 1 s snapshots)")
if len(W):
    print(W.groupby(pd.cut(W.secs, [0, 1, 2, 5, 30, 1e9], labels=['1 s', '2 s', '3-5 s', '6-30 s', '>30 s']), observed=False).agg(
        windows=('secs', 'size'), median_edge_c=('edge', lambda x: round(100 * x.median(), 1)),
        median_size=('size', 'median'), total_profit_if_all_taken=('usd', 'sum')).to_string())
    print(f"\nedge per share: median {100*W.edge.median():.1f}c, p90 {100*W.edge.quantile(.9):.1f}c")
    L = W[W.secs >= 2]
    print(f"windows lasting 2 s+: {len(L)}, profit if you took every one at full top-level size: ${L.usd.sum():.2f}; "
          f"capped at $100 locked per trade: ${(L.edge * np.minimum(L['size'], 100)).sum():.2f}")
    print(W.groupby('game').secs.size().to_string())
    B2 = W[(W.secs >= 2) & (W.edge >= 0.02)]
    print(f"windows 2 s+ with edge >= 2c: {len(B2)}, median edge {100*B2.edge.median():.1f}c, median size {B2['size'].median():.0f} shares")
