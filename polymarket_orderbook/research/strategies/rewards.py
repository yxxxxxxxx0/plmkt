# What share of each market's daily liquidity reward would a small two-sided quote earn right now?
# Scoring per the docs: S = ((v - s) / v)^2 per share, Qmin = min of the two sides, pro rata across makers.
#   python research/strategies/rewards.py
import json, numpy as np, pandas as pd, requests
C = 'research/strategies/cache'
d = pd.read_pickle(f'{C}/snapshot.pkl')
E = json.load(open(f'{C}/open_events.json'))
age = {}
for e in E:
    for m in e.get('markets', []):
        t = (json.loads(m.get('clobTokenIds') or '[]') or [None])[0]
        r = [x.get('startDate') for x in (m.get('clobRewards') or [])]
        age[t] = min(r) if r else None
d['rstart'] = d.tok.map(age)
r = d[(d.rew > 0) & (d.minsz <= 50) & d.tok.notna() & (d.maxspr > 0)]
print(f'reward markets with min size <= 50 shares: {len(r)}; reward start date before 2026-11-07: {(r.rstart < "2026-11-07").sum()}')
for nm, pool in (('established (reward running 4+ days)', r[r.rstart < '2026-11-07']), ('new (reward started in the last 2 days)', r[r.rstart >= '2026-11-09'])):
    s = pool.sample(min(600, len(pool)), random_state=0)
    books = {}
    toks = s.tok.tolist()
    for i in range(0, len(toks), 100):
        try:
            for b in requests.post('https://clob.polymarket.com/books', json=[{'token_id': t} for t in toks[i:i + 100]], timeout=60).json(): books[b['asset_id']] = b
        except Exception as ex: print('warn', ex)
    out = []
    for _, m in s.iterrows():
        b = books.get(m.tok)
        if not b: continue
        B = [(float(x['price']), float(x['size'])) for x in b.get('bids', [])]; A = [(float(x['price']), float(x['size'])) for x in b.get('asks', [])]
        if not B or not A: continue
        bb, ba = max(p for p, _ in B), min(p for p, _ in A); mid = (bb + ba) / 2; v = m.maxspr / 100
        S = lambda p: max(0.0, (v - abs(p - mid)) / v) ** 2
        q1 = sum(S(p) * q for p, q in B if q >= m.minsz); q2 = sum(S(p) * q for p, q in A if q >= m.minsz)
        others = min(q1, q2) if not (0.1 <= mid <= 0.9) else max(min(q1, q2), max(q1, q2) / 3)
        n = max(m.minsz, 20); mine = n * 0.25                       # my quote: n shares a side, halfway out in the band
        cap = n * (mid - v / 2) + n * (1 - mid - v / 2)             # dollars to rest both sides
        out.append(dict(q=m.q, rew=m.rew, mid=mid, spread=ba - bb, v=v, others=others, usd_day=m.rew * mine / (mine + others), capital=cap, v24=m.v24, days=m.days, fee=m.fee))
    o = pd.DataFrame(out); o['pct_day'] = 100 * o.usd_day / o.capital
    ok = o[(o.mid >= 0.1) & (o.mid <= 0.9)]
    print(f'\n=== {nm}: {len(o)} sampled with a two-sided book, {len(ok)} with mid 10-90c')
    print(f'  daily pool per market: median ${ok.rew.median():.2f}, mean ${ok.rew.mean():.2f}; capital for one two-sided quote: median ${ok.capital.median():.0f}')
    print(f'  my daily reward per market: 25/50/75/90th pct ${ok.usd_day.quantile(.25):.2f} / ${ok.usd_day.quantile(.5):.2f} / ${ok.usd_day.quantile(.75):.2f} / ${ok.usd_day.quantile(.9):.2f}')
    print(f'  as % of capital per day:    25/50/75/90th pct {ok.pct_day.quantile(.25):.2f}% / {ok.pct_day.quantile(.5):.2f}% / {ok.pct_day.quantile(.75):.2f}% / {ok.pct_day.quantile(.9):.2f}%')
    print(f'  markets where nobody else scores: {(ok.others == 0).mean():.0%}; median spread {100 * ok.spread.median():.0f}c; median 24h volume ${ok.v24.median():.0f}')
    print(ok.sort_values('pct_day', ascending=False).head(8)[['q', 'rew', 'mid', 'spread', 'others', 'usd_day', 'capital', 'v24', 'days']].round(2).to_string(index=False))
    o.to_pickle(f'{C}/rewards_{nm[:3]}.pkl')
