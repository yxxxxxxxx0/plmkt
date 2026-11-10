# Two public-data pulls for the strategy screen (nothing is sent but market ids):
#   python research/strategies/fetch_hist.py closed    resolved non-sports markets + price 1/7/30 days before the end
#   python research/strategies/fetch_hist.py crypto    BTC 15-minute up/down markets, 1-minute prices, Coinbase 1-minute candles
import sys, json, time, requests
from concurrent.futures import ThreadPoolExecutor
C = 'research/strategies/cache'
G, CLOB = 'https://gamma-api.polymarket.com', 'https://clob.polymarket.com'
S = requests.Session()

def get(url, **p):
    for i in range(4):
        try:
            r = S.get(url, params=p, timeout=40)
            if r.status_code == 200: return r.json()
        except Exception: pass
        time.sleep(1 + 2 * i)
    return None

def ts(s):
    import datetime as dt
    try: return dt.datetime.fromisoformat(s.replace(' ', 'T').replace('Z', '+00:00').replace('+00', '+00:00').replace('+00:00:00', '+00:00')).timestamp()
    except Exception: return None

def closed():
    ms, cur, n = [], None, 0
    while len(ms) < 5000 and n < 400000:
        kw = dict(closed='true', limit=500, order='closedTime', ascending='false', volume_num_min=10000)
        if cur: kw['after_cursor'] = cur
        r = get(f'{G}/markets/keyset', **kw) or {}
        d, cur = r.get('markets') or [], r.get('next_cursor')
        if not d: break
        n += len(d)
        for m in d:
            ft = m.get('feeType') or ''
            if 'sports' in ft or 'updown' in m.get('slug', '') or 'up-or-down' in m.get('slug', ''): continue
            try: op = [float(x) for x in json.loads(m['outcomePrices'])]; tk = json.loads(m['clobTokenIds'])
            except Exception: continue
            if len(tk) != 2 or max(op) < 0.99: continue             # resolved cleanly to one side
            a, b = ts(m.get('closedTime') or ''), ts(m.get('endDate') or '')
            T = min(x for x in (a, b) if x) if (a or b) else None
            if T: ms.append(dict(slug=m['slug'], tok=tk[0], yes=op[0], T=T, start=ts(m.get('startDate') or '') or 0, vol=m.get('volumeNum'), fee=ft, neg=bool(m.get('negRisk'))))
        print(n, len(ms), flush=True)
        if not cur: break
    def hist(m):
        h = (get(f'{CLOB}/prices-history', market=m['tok'], startTs=int(m['T'] - 8 * 86400), endTs=int(m['T']), fidelity=60) or {}).get('history') or []
        for d in (1, 3, 7):
            t = m['T'] - d * 86400
            x = [q for q in h if q['t'] <= t and q['t'] > t - 6 * 3600]   # a price printed in the 6 h before the sample time
            m[f'p{d}'] = x[-1]['p'] if x and m['start'] < t else None
        return m
    with ThreadPoolExecutor(8) as ex: ms = list(ex.map(hist, ms))
    json.dump(ms, open(f'{C}/closed.json', 'w')); print('saved', len(ms), sum(m['p1'] is not None for m in ms))

def crypto(days=5):
    now = int(time.time()) // 900 * 900 - 1800
    starts = range(now - days * 86400, now, 900)
    def one(t0):
        e = get(f'{G}/events', slug=f'btc-updown-15m-{t0}')
        if not e or not e[0].get('markets'): return None
        m = e[0]['markets'][0]
        try: op = [float(x) for x in json.loads(m['outcomePrices'])]; tk = json.loads(m['clobTokenIds'])
        except Exception: return None
        if max(op) < 0.99: return None
        h = (get(f'{CLOB}/prices-history', market=tk[0], startTs=t0 - 60, endTs=t0 + 900, fidelity=1) or {}).get('history') or []
        return dict(t0=t0, up=op[0], h=h, vol=m.get('volumeNum'))
    with ThreadPoolExecutor(8) as ex: ms = [m for m in ex.map(one, starts) if m]
    cand, t = {}, now - days * 86400 - 7200
    while t < now + 900:                                              # Coinbase: 300 one-minute candles a call
        d = get('https://api.exchange.coinbase.com/products/BTC-USD/candles', granularity=60, start=t, end=t + 299 * 60) or []
        cand.update({int(c[0]): (c[3], c[4]) for c in d}); t += 300 * 60; time.sleep(0.15)   # start -> (open, close)
    json.dump(dict(markets=ms, candles=cand), open(f'{C}/crypto.json', 'w')); print('saved', len(ms), 'markets', len(cand), 'candles')

if __name__ == '__main__':
    S.headers['User-Agent'] = 'research-script'
    {'closed': closed, 'crypto': crypto}[sys.argv[1]]()
