"""Kalshi moves first -> take on Polymarket. Causal alert at second T: Kalshi mid moved >= X over
[T-3, T] while Polymarket mid moved < X/3 the same way. Buy PM at the ask at T+delay (1 s football
taker delay, 0 s MLB), sell at the PM bid H s later, Polymarket taker fee both legs. 30 s debounce."""
import json, os, numpy as np, pandas as pd
VIEW = 'data/view'
fee = lambda p: 0.05 * p * (1 - p)
idx = json.load(open(f'{VIEW}/index.json'))
ev = json.load(open(f'{VIEW}/jump_events.json', encoding='utf-8'))
rows = []
for tag, sess in idx.items():
    for gi in sess['games']:
        d = pd.read_csv(f"{VIEW}/{tag}/aligned/{gi['slug']}.csv.gz"); t = d['t_ms'].to_numpy(np.int64); n = len(t)
        keys = sorted({c.split('_')[0] for c in d.columns if c.endswith('_pm_mid')})
        f = lambda c: pd.Series(d[c].to_numpy(float)).ffill().to_numpy()
        good = [e['t'] for e in ev if e['game'] == gi['slug'] and e['best']['30'] >= 0]
        mlb = gi['slug'].startswith('mlb')
        for X in (0.02, 0.03, 0.04, 0.06):
            for k in keys:
                km, pm = f(f'{k}_ks_mid'), f(f'{k}_pm_mid'); pb, pa = f(f'{k}_pm_bid1'), f(f'{k}_pm_ask1')
                dk = np.full(n, np.nan); dp = np.full(n, np.nan); dk[3:] = km[3:] - km[:-3]; dp[3:] = pm[3:] - pm[:-3]
                cand = np.flatnonzero((np.abs(np.nan_to_num(dk)) >= X) & (np.nan_to_num(dp) * np.sign(np.nan_to_num(dk)) < X / 3))
                last = -999
                for i in cand:
                    if i - last < 30 or i + 40 >= n: continue
                    last = i; sg = np.sign(dk[i]); dl = 0 if mlb else 1; e = i + dl
                    r = dict(X=X, game=gi['slug'], mlb=mlb, good=any(-5_000 <= g - t[i] <= 60_000 for g in good),
                             spread=pa[e] - pb[e])
                    for H in (5, 10, 30):
                        x = e + H
                        pin, pout = (pa[e], pb[x]) if sg > 0 else (1 - pb[e], 1 - pa[x])
                        r[H] = pout - pin - fee(pin) - fee(pout)
                    # did PM follow? PM mid move in Kalshi's direction over the next 10 s
                    r['follow'] = sg * (pm[i + 10] - pm[i])
                    rows.append(r)
D = pd.DataFrame(rows)
print('Alert: Kalshi moved >= X in 3 s, Polymarket not yet (< X/3). Trade Polymarket only.\n')
print('  X   alerts  good jump  PM followed>=X/2  spread at entry | P&L/trade: hold 5s   10s   30s | winners(30s)')
for X, g in D.groupby('X'):
    print(f"{100*X:3.0f}c  {len(g):6d}  {int(g.good.sum()):4d} ({g.good.mean():3.0%})  {np.mean(g.follow >= X/2):10.0%}   {100*g.spread.mean():9.1f}c     |"
          f"   {100*g[5].mean():+5.1f}c {100*g[10].mean():+5.1f}c {100*g[30].mean():+5.1f}c | {int((g[30]>0).sum())}/{len(g)}")
g = D[D.X == 0.04]
for lab, s in (('football (1 s delay)', g[~g.mlb]), ('MLB (no delay)', g[g.mlb])):
    if len(s): print(f"  X=4c {lab:22s} n={len(s):3d}  P&L 10s {100*s[10].mean():+.1f}c  30s {100*s[30].mean():+.1f}c")
s = g[g.good]; print(f"  X=4c, only alerts that became good jumps (hindsight): n={len(s)}  P&L 30s {100*s[30].mean():+.1f}c")
