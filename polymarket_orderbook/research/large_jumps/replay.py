# Event-level replay of the large-jump trades: book at alert+50ms, stabilisation exit on a 50ms clock.
import os, sys, json, bisect, subprocess
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
FEE, STAKE, LAT, MINHOLD, STAB, TIMEOUT, STEP = 0.05, 10.0, 50, 5000, 3000, 90000, 50
fee = lambda p: FEE * p * (1 - p) / 0.01

if sys.argv[1] == 'select':
    D = pd.read_pickle(os.path.join(S, 'large_ds.pkl'))
    BOOK = [c for c in D.columns if c.startswith('f_')]; GF = [c for c in D.columns if c.startswith('g_')]
    F = BOOK + GF + ['t_n10', 't_usd10', 't_n60', 't_game_n10', 'yes_bats']
    tr = D.sess <= 'books_2026-09-20'; te = ~tr
    X = D[F].assign(mtype=D.mtype.astype('category'))
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(X[tr], D.y_large[tr])
    T = D[te].assign(p=m.predict_proba(X[te])[:, 1])
    T['top'] = np.where(T.p >= np.percentile(T.p, 99.75), 0.25, np.where(T.p >= np.percentile(T.p, 95), 5.0, np.nan))
    T = T[T.top.notna()]
    T[['sess', 'slug', 'aid', 'ts', 'top', 'y_large', 'y_mark', 'f_norm_spr', 'pnl_s3']].to_pickle(os.path.join(S, 'replay_sel.pkl'))
    print(len(T), T.top.value_counts().to_dict()); sys.exit()

# ---- per-session extraction: python replay.py extract books_2026-09-21 ----
sess = sys.argv[2]
sel = pd.read_pickle(os.path.join(S, 'replay_sel.pkl')); sel = sel[sel.sess == sess]
win = {}
for aid, g in sel.groupby('aid'):
    win[aid] = sorted((t - 2000, t + TIMEOUT + 10000) for t in g.ts)
ev = {aid: [] for aid in win}
proc = subprocess.Popen(['xz', '-dc', f'data/live/{sess}.jsonl.xz'], stdout=subprocess.PIPE, bufsize=1 << 20)
for raw in proc.stdout:
    k = raw.find(b'"asset_id":"')
    if k < 0: continue
    aid = raw[k + 12:k + 12 + 90].split(b'"', 1)[0].decode()
    if aid not in win: continue
    k2 = raw.find(b'"ts":'); ts = int(raw[k2 + 5:raw.find(b',', k2)])
    w = win[aid]; i = bisect.bisect_right(w, (ts, 10**15)) - 1
    if i < 0 or ts > w[i][1]:
        if i + 1 < len(w) and w[i + 1][0] <= ts <= w[i + 1][1]: pass
        else: continue
    d = json.loads(raw)
    if 'bids' not in d or 'asks' not in d: continue
    ev[aid].append((ts, d['bids'], d['asks']))


def walk(levels, dollars=None, shares=None, buy=True):
    lv = sorted(((float(p), float(s)) for p, s in levels), key=lambda z: z[0], reverse=not buy)
    gs = gu = 0.0
    for p, s in lv:
        if s <= 0: continue
        take = min(s, (dollars - gu) / p) if dollars is not None else min(s, shares - gs)
        gs += take; gu += take * p
        if (dollars is not None and gu >= dollars - 1e-9) or (shares is not None and gs >= shares - 1e-9): break
    return (gu / gs if gs else np.nan), (gu / dollars if dollars is not None else gs / shares)


best = lambda lv, hi: (max if hi else min)((float(p) for p, s in lv if float(s) > 0), default=np.nan)
rows = []
for r in sel.itertuples():
    E = sorted(ev.get(r.aid, []), key=lambda z: z[0])
    if not E: continue
    et = np.array([e[0] for e in E])
    k = np.searchsorted(et, r.ts + LAT, side='right') - 1
    if k < 0: continue
    px, fill = walk(E[k][2], dollars=STAKE, buy=True)
    out = dict(sess=sess, slug=r.slug, ts=r.ts, top=r.top, y_large=r.y_large, y_mark=r.y_mark, grid_pnl=r.pnl_s3,
               entry=px, entry_best=best(E[k][2], False), fill=fill, book_age_ms=r.ts + LAT - et[k])
    if fill < .999: rows.append(out); continue
    sh = STAKE / px; t_e = r.ts + LAT; cut = r.f_norm_spr + max(2, r.f_norm_spr)
    grid = np.arange(t_e, t_e + TIMEOUT + STAB, STEP)
    idx = np.searchsorted(et, grid, side='right') - 1; idx = np.clip(idx, 0, None)
    bb = np.array([best(E[i][1], True) for i in idx]); ba = np.array([best(E[i][2], False) for i in idx])
    q = (ba - bb) / .01 <= cut + 1e-6
    n = int(STAB / STEP)
    rb, ra, rq = pd.Series(bb).rolling(n), pd.Series(ba).rolling(n), pd.Series(q.astype(float)).rolling(n)
    stable = ((rb.max() - rb.min() <= .0101) & (ra.max() - ra.min() <= .0101) & (rq.min() == 1)).to_numpy()
    cand = np.flatnonzero(stable & (grid >= t_e + MINHOLD))
    j = cand[0] if len(cand) and grid[cand[0]] <= t_e + TIMEOUT else None
    if j is None:
        qq = np.flatnonzero(q & (grid >= t_e + TIMEOUT)); j = qq[0] if len(qq) else None
    if j is None: rows.append(out); continue
    kx = np.searchsorted(et, grid[j] + LAT, side='right') - 1
    xp, xf = walk(E[kx][1], shares=sh, buy=False)
    out.update(exit=xp, exit_fill=xf, hold_s=(grid[j] - t_e) / 1000,
               pnl=(xp - px) / .01 - fee(px) - fee(xp) if xf >= .999 else np.nan)
    rows.append(out)
pd.DataFrame(rows).to_pickle(os.path.join(S, f'replay_{sess}.pkl'))
print(sess, len(rows), 'events kept', sum(len(v) for v in ev.values()), flush=True)
