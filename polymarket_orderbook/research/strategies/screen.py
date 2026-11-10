# Strategy screen on public data (see fetch_hist.py and snap.py for the inputs).
#   python research/strategies/screen.py snapshot | closed | crypto
import sys, json, math, collections, datetime as dt
import numpy as np, pandas as pd, requests
C = 'research/strategies/cache'
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
c = lambda x: round(100 * x, 2)

def snapshot():
    E = json.load(open(f'{C}/open_events.json')); now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for e in E:
        for m in e.get('markets', []):
            if m.get('closed') or not m.get('acceptingOrders'): continue
            fs = m.get('feeSchedule') or {}
            r = sum(float(x.get('rewardsDailyRate') or 0) for x in (m.get('clobRewards') or []))
            try: end = (dt.datetime.fromisoformat((m.get('endDate') or '').replace('Z', '+00:00')) - now).total_seconds() / 86400
            except Exception: end = np.nan
            rows.append(dict(event=e.get('title', '')[:50], q=(m.get('question') or '')[:60], fee=m.get('feeType') or 'none', rate=fs.get('rate', 0), rebate=fs.get('rebateRate', 0),
                             rew=r, minsz=m.get('rewardsMinSize') or 0, maxspr=m.get('rewardsMaxSpread') or 0, liq=float(m.get('liquidityClob') or 0),
                             v24=float(m.get('volume24hrClob') or 0), bid=m.get('bestBid'), ask=m.get('bestAsk'), days=end, delay=m.get('secondsDelay') or 0,
                             hold=bool(m.get('holdingRewardsEnabled')), tok=(json.loads(m.get('clobTokenIds') or '[]') or [None])[0], neg=bool(e.get('negRisk'))))
    d = pd.DataFrame(rows); d.to_pickle(f'{C}/snapshot.pkl')
    print(f'{len(E)} events, {len(d)} open markets\n\n--- taker fee rate by fee type (markets, 24h volume $k)')
    print(d.groupby(['fee', 'rate', 'rebate']).agg(n=('q', 'size'), v24k=('v24', lambda x: round(x.sum() / 1e3))).sort_values('v24k', ascending=False).head(15).to_string())

    print('\n--- 1. Liquidity rewards')
    r = d[d.rew > 0]
    print(f'{len(r)} markets pay a daily reward, ${r.rew.sum():,.0f}/day in total; minimum qualifying order (shares): '
          f'{r.minsz.quantile([.1,.5,.9]).round().tolist()} (10/50/90th pct)')
    for cap in (20, 50, 100, 200):
        x = r[r.minsz <= cap]; print(f'   min size <= {cap:>3} shares: {len(x):>5} markets, ${x.rew.sum():>9,.0f}/day')
    s = r[(r.minsz <= 100) & (r.liq > 0)].copy()            # a $100 account can quote ~100 shares a side in one market
    s['pct_day'] = 100 * s.rew / (s.liq + 100)               # my share if I add $100 to the book's resting dollars (proxy: all of it qualifies)
    top = s.sort_values('pct_day', ascending=False).head(25)
    # replace the proxy with the real book: dollars resting within the reward band of the mid
    def band(t, ms):
        try: b = requests.get('https://clob.polymarket.com/book', params={'token_id': t}, timeout=30).json()
        except Exception: return np.nan, np.nan
        B = [(float(x['price']), float(x['size'])) for x in b.get('bids', [])]; A = [(float(x['price']), float(x['size'])) for x in b.get('asks', [])]
        if not B or not A: return np.nan, np.nan
        mid = (max(p for p, _ in B) + min(p for p, _ in A)) / 2; w = ms / 100
        return mid, sum(p * q for p, q in B if p >= mid - w) + sum((1 - p) * q for p, q in A if p <= mid + w)
    bb = [band(t, ms) for t, ms in zip(top.tok, top.maxspr)]
    top['mid'] = [x[0] for x in bb]; top['band_usd'] = [x[1] for x in bb]
    top['usd_day_100'] = 100 * top.rew / (top.band_usd + 100)
    print('   25 best by reward per resting dollar (usd_day_100 = my daily reward for $100 resting in the band, pro rata):')
    print(top[['q', 'rew', 'minsz', 'maxspr', 'mid', 'liq', 'band_usd', 'usd_day_100', 'v24', 'days']].round(2).to_string(index=False))
    ok = top[(top.mid > 0.1) & (top.mid < 0.9)]
    print(f'   of these, mid between 10c and 90c: {len(ok)}; median ${ok.usd_day_100.median():.2f}/day per $100, but payouts under $1 a day are not paid')

    print('\n--- 4. Near-certain, ending within 7 days, no taker delay, not sports')
    n = d[(d.days > 0) & (d.days <= 7) & (d.delay == 0) & ~d.fee.str.contains('sports') & d.ask.notna() & d.bid.notna()].copy()
    n['cost'] = np.where(n.ask >= 0.5, n.ask, 1 - n.bid)     # price of the likelier side
    n = n[(n.cost >= 0.95) & (n.cost < 1)]
    n['edge'] = 1 - n.cost - n.rate * n.cost * (1 - n.cost)
    n['yr'] = n.edge / n.cost * 365 / n.days.clip(lower=0.5)
    for lo, hi in ((.95, .98), (.98, .995), (.995, 1)):
        x = n[(n.cost >= lo) & (n.cost < hi)]
        print(f'   likelier side costs [{lo}, {hi}): {len(x):>5} markets, edge if it wins {c(x.edge.median()):.2f}c, break-even win rate {(1 - x.edge.median()):.4f}')
    print(f'\n--- holding rewards enabled on {d.hold.sum()} markets; no-delay markets {int((d.delay == 0).sum())} of {len(d)}')

def closed():
    d = pd.DataFrame(json.load(open(f'{C}/closed.json'))).sort_values('T').reset_index(drop=True)
    d['test'] = d.index >= len(d) // 2
    print(f'{len(d)} resolved non-sports markets (volume >= $10k), ended {dt.datetime.fromtimestamp(d["T"].min()).date()} to {dt.datetime.fromtimestamp(d["T"].max()).date()}; '
          f'YES won {d.yes.mean():.1%}; fee types {d.fee.value_counts().head(6).to_dict()}')
    for h in (1, 3, 7):
        x = d.dropna(subset=[f'p{h}']); p = x[f'p{h}']
        B = pd.concat([x.assign(side='YES', p=p, win=x.yes), x.assign(side='NO', p=1 - p, win=1 - x.yes)])
        B = B[(B.p > 0.005) & (B.p < 0.999)]
        B['pnl_mid'] = B.win - B.p                             # at the recorded price
        B['pnl'] = B.win - (B.p + 0.01) - 0.05 * B.p * (1 - B.p)   # ASSUMED 1c worse than the recorded price, plus a 0.05 fee
        B['pb'] = pd.cut(B.p, [0, .02, .05, .1, .25, .5, .75, .9, .95, .98, 1])
        print(f'\n  price {h} day(s) before the end: {len(x)} markets')
        for (b, s), e in B.groupby(['pb', 'side']):
            if len(e) < 40: continue
            f, t = e[~e.test], e[e.test]
            se = lambda z: z.pnl_mid.std() / math.sqrt(max(len(z), 1))
            print(f'    {s:3s} at {str(b):12s} n {len(f):>4}/{len(t):>4}  won {f.win.mean():.3f}/{t.win.mean():.3f} vs price {f.p.mean():.3f}/{t.p.mean():.3f} | '
                  f'at price {c(f.pnl_mid.mean()):+6.2f}c (se {c(se(f)):.1f}) / {c(t.pnl_mid.mean()):+6.2f}c (se {c(se(t)):.1f}) | after 1c + fee {c(f.pnl.mean()):+6.2f}c / {c(t.pnl.mean()):+6.2f}c')

def crypto():
    D = json.load(open(f'{C}/crypto.json')); K = {int(k): v for k, v in D['candles'].items()}
    N = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    rows = []
    for m in D['markets']:
        t0 = m['t0']
        if t0 not in K: continue
        s0 = K[t0][0]                                          # Coinbase open of the first minute ~ start price
        for q in m['h']:
            el = q['t'] - t0
            if el < 60 or el > 840: continue
            k = int(el // 60)                                  # minutes fully finished before this price was stamped
            cl = [K.get(t0 + 60 * i) for i in range(k)]
            past = [K.get(t0 - 60 * i) for i in range(1, 61)]
            if any(x is None for x in cl) or sum(x is not None for x in past) < 50: continue
            ret = [math.log(x[1] / x[0]) for x in past if x]
            sig = float(np.std(ret)) or 1e-6                   # per-minute volatility, trailing hour before the window
            s = cl[-1][1]; avg = np.mean([x[1] for x in cl]); rem = 15 - k
            # TWAP so far is locked in; the rest averages around the last price with variance sig^2 * rem / 3
            mu = (k * avg + rem * s) / 15 / s0 - 1
            sd = (rem / 15) * sig * math.sqrt(rem / 3) + 1e-9
            rows.append(dict(t0=t0, k=k, p=q['p'], up=m['up'], fair=N(mu / sd), spot=N(math.log(s / s0) / (sig * math.sqrt(rem)))))
    d = pd.DataFrame(rows); d['test'] = d.t0 >= d.t0.quantile(0.5)
    g = d.groupby('t0').first()
    print(f'{d.t0.nunique()} BTC 15-minute markets, {len(d)} minute samples; Up won {g.up.mean():.1%}')
    br = lambda x: ((x - d.up) ** 2).mean()
    print(f'Brier score (lower is better): market price {br(d.p):.4f} | my TWAP model on 0-60 s old candles {br(d.fair):.4f} | naive spot-vs-start {br(d.spot):.4f}')
    print('taker test: buy the side my TWAP model favours when it disagrees with the market by more than the threshold; pay price + 0.5c + fee 0.07*p*(1-p); hold to settlement')
    for thr in (0.03, 0.06, 0.10, 0.15):
        for nm, x in (('fit ', d[~d.test]), ('test', d[d.test])):
            e = x.fair - x.p
            y = x[e.abs() > thr].copy(); y = y.groupby(['t0', np.sign(y.fair - y.p)]).head(1)   # first signal per market per side
            buy_up = y.fair > y.p; cost = np.where(buy_up, y.p, 1 - y.p)
            y = y[(cost > 0.03) & (cost < 0.97)]; buy_up = y.fair > y.p; cost = np.where(buy_up, y.p, 1 - y.p)
            pnl = np.where(buy_up, y.up, 1 - y.up) - cost - 0.005 - 0.07 * cost * (1 - cost)
            rnd = np.where(buy_up, 1 - y.up, y.up) - (1 - cost) - 0.005 - 0.07 * cost * (1 - cost)   # the opposite side, same moments
            se = pnl.std() / math.sqrt(max(len(pnl), 1))
            print(f'  gap > {c(thr):>2.0f}c {nm}: {len(y):>4} trades in {y.t0.nunique():>3} markets, {c(pnl.mean()):+6.2f}c/share (se {c(se):.1f}) | opposite side {c(rnd.mean()):+6.2f}c | at $100, 20 shares a trade: ${20 * pnl.sum():+.2f}')
    d['gap'] = d.fair - d.p
    print('does the market under-react to the locked-in TWAP? mean (outcome - price) by model-minus-price gap, minutes 5-13:')
    x = d[(d.k >= 5) & (d.k <= 13)]
    print(x.groupby(pd.cut(x.gap, [-1, -.15, -.06, -.02, .02, .06, .15, 1])).apply(lambda z: pd.Series(dict(n=len(z), markets=z.t0.nunique(), price=z.p.mean(), won=z.up.mean(), diff_c=c((z.up - z.p).mean())))).round(3).to_string())

if __name__ == '__main__':
    {'snapshot': snapshot, 'closed': closed, 'crypto': crypto}[sys.argv[1]]()
