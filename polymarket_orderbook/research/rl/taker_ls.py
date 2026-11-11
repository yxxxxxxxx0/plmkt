# Taker RL on large jumps: after a depth-drain alert, learn WHEN to buy YES (or skip) and WHEN to sell.
# Two optimal-stopping problems solved backwards with Longstaff-Schwartz, one LightGBM per decision step:
#   exit:  at each exit step, sell now (value known from the book) or hold (value = regression of what
#          holding actually earned under the later steps' policy);
#   entry: at each 200ms step of the 5s after the alert, buy now, wait, or give up.
# A $10 taker order does not move the book, so every action's payoff is read off the recorded path.
# Positions must be flat by alert + 95s, which makes the exit problem independent of the entry slot.
#
#   python research/rl/taker_ls.py build            per-session episode arrays -> research/rl/cache/
#   python research/rl/taker_ls.py fit [lat_slots]  train <= 09-20 (val 09-19/20 picks the entry margin), test after
#   python research/rl/taker_ls.py jumps [lat_slots] same, but the entry is rewarded for catching a large jump
#   python research/rl/taker_ls.py selftest
import os, sys, glob
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
LJ, C = 'research/large_jumps/cache', 'research/rl/cache'
IN = 'results/large_jumps/inputs'
FEE, STAKE = 0.05, 10.0
K = 25            # entry decisions at slots 0..25 after the alert (5s, the large-jump label horizon)
TX = 475          # flat by slot 475 (95s); forced exit at the first fillable slot after that
NS = TX + 2 + 300 # slots kept per episode: decisions, latency, and up to 60s to find a forced-exit fill
# ponytail: exit decided every 1s for the first 20s, then every 5s (RAM: ~200k episodes); finer if early cuts matter
XS = np.r_[np.arange(5, 100, 5), np.arange(100, TX + 1, 25)]
GRID = ['imb1', 'imb5', 'microprice_dev', 'rv_5', 'rv_25', 'ofi_5', 'dmid_5', 'ofi_25', 'log_bid_usd', 'log_ask_usd']
DYN = ['dmid_a', 'dmid_1s', 'up_a', 'dn_a', 'spr', 'spr_rel', 'cross_buy', 'cross_sell', 'dep_a', 'dep_4s',
       't_n1s', 't_usd1s', 't_n_a', 't_up_a', 't_dn_a'] + GRID
fee = lambda p: FEE * p * (1 - p) / 0.01


def buy_cost(px, usd):
    """Ticks paid (price + fee) to buy $STAKE walking the asks; NaN if the book cannot fill it."""
    usd = np.where((px > 0) & (px < 1), usd, 0.)
    cu = np.cumsum(usd, 1); take = np.clip(STAKE - (cu - usd), 0, usd)
    sh = (take / np.where(px > 0, px, 1)).sum(1); p = STAKE / np.where(sh > 0, sh, np.nan)
    return np.where(cu[:, -1] >= STAKE - 1e-9, p / .01 + fee(p), np.nan)


def sell_value(px, usd, sh0):
    """Ticks received (price - fee) selling sh0 shares walking the bids; NaN if the book cannot take them."""
    sl = np.where((px > 0) & (px < 1), usd / np.where(px > 0, px, 1), 0.)
    cs = np.cumsum(sl, 1); take = np.clip(sh0[:, None] - (cs - sl), 0, sl)
    p = (take * px).sum(1) / sh0
    return np.where(cs[:, -1] >= sh0 - 1e-9, p / .01 - fee(p), np.nan)


def build():
    import pyarrow.parquet as pq
    os.makedirs(C, exist_ok=True)
    D = pd.read_pickle(os.path.join(LJ, 'large_ds.pkl')).reset_index(drop=True)
    pairs = pd.read_csv(os.path.join(IN, 'token_pairs.csv'), dtype=str); no2yes = dict(zip(pairs.no, pairs.yes))
    T = pd.concat([pd.read_csv(f, dtype={'asset_id': str}) for f in glob.glob(os.path.join(IN, 'trades', '*.csv'))])
    isno = T.asset_id.isin(no2yes)
    T['yes'] = np.where(isno, T.asset_id.map(no2yes), T.asset_id); T['p'] = np.where(isno, 1 - T.price, T.price)
    T['usd'] = T['size'] * T.price; T = T.sort_values('ts')
    TB = {k: (v.ts.to_numpy(), v.p.to_numpy(), v.usd.to_numpy()) for k, v in T.groupby('yes')}
    drop = dict(short=0, gap=0)
    for sess, ds in D.groupby('sess'):
        f = pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet', columns=['mid', 'spread_ticks', 'ts', 'series'] + GRID).to_pandas()
        lob = np.load(f'data/jump/lob_{sess}_trimmed.npy', mmap_mode='r')
        f = f[f.series.isin(set(ds.series))]
        rows, B, G, XE, XX = [], [], [], [], []
        for sr, s in f.groupby('series', sort=False):
            s = s.sort_values('ts'); ts = s.ts.to_numpy(); n = len(ts)
            L = np.asarray(lob[s.index.to_numpy()], np.float64)
            mid = s.mid.to_numpy(); spr = s.spread_ticks.to_numpy(); norm = np.median(spr)
            apx, ausd = mid[:, None] + L[:, 2] * .01, np.expm1(L[:, 3])
            bpx, busd = mid[:, None] + L[:, 0] * .01, np.expm1(L[:, 1])
            bc = buy_cost(apx, ausd)
            dep = (busd * (np.abs(L[:, 0]) <= 5)).sum(1) + (ausd * (np.abs(L[:, 2]) <= 5)).sum(1)
            gap = np.r_[0, np.cumsum(np.diff(ts) != 200)]
            gv = s[GRID].to_numpy(np.float64)
            tt, tp, tu = TB.get(sr.split('|')[-1], (np.zeros(0, np.int64), np.zeros(0), np.zeros(0)))
            for ix, t in zip(ds.index[ds.series == sr], ds.ts[ds.series == sr]):
                e = np.searchsorted(ts, t)
                if e < 20 or e + NS > n: drop['short'] += 1; continue
                if gap[e + TX + 2] != gap[e - 20]: drop['gap'] += 1; continue
                w = np.arange(e, e + NS)
                sh0 = np.full(NS, STAKE / (mid[e] + spr[e] * .005))
                B.append(bc[w]); G.append(sell_value(bpx[w], busd[w], sh0)); rows.append(ix)
                a0, b0 = mid[e] + spr[e] * .005, mid[e] - spr[e] * .005
                cm = np.maximum.accumulate(mid[w[:TX + 1]]); cn = np.minimum.accumulate(mid[w[:TX + 1]])

                def feats(sl):
                    i = e + sl
                    j0 = np.searchsorted(tt, t); j1 = np.searchsorted(tt, ts[i], side='right'); jw = np.searchsorted(tt, ts[i] - 1000)
                    nup = np.array([((tp[j0:b] >= a0 + .0099)).sum() for b in j1]); ndn = np.array([((tp[j0:b] <= b0 - .0099)).sum() for b in j1])
                    cus = np.r_[0, np.cumsum(tu)]
                    return np.column_stack([
                        (mid[i] - mid[e]) / .01, (mid[i] - mid[i - 5]) / .01, (cm[sl] - mid[e]) / .01, (mid[e] - cn[sl]) / .01,
                        spr[i], spr[i] / norm, bc[i] - mid[i] / .01, mid[i] / .01 - G[-1][sl], dep[i] / max(dep[e], 1),
                        dep[i] / np.maximum(dep[i - 20], 1), j1 - jw, cus[j1] - cus[jw], j1 - j0, nup, ndn, gv[i]]).astype(np.float32)
                XE.append(feats(np.arange(K + 1))); XX.append(feats(XS))
        if not rows: continue
        np.savez(os.path.join(C, f'ep_{sess}.npz'), idx=np.array(rows), B=np.array(B, np.float32), G=np.array(G, np.float32),
                 XE=np.array(XE), XX=np.array(XX))
        print(sess, len(rows), 'episodes', drop, flush=True)


def load(lat):
    D = pd.read_pickle(os.path.join(LJ, 'large_ds.pkl')).reset_index(drop=True)
    st = [c for c in D.columns if c.startswith(('f_', 'g_'))] + ['t_n10', 't_usd10', 't_n60', 't_game_n10', 'yes_bats']
    D['mt'] = D.mtype.astype('category').cat.codes
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(C, 'ep_*.npz')))]
    idx = np.concatenate([p['idx'] for p in parts])
    Bf, Gf = np.concatenate([p['B'] for p in parts]), np.concatenate([p['G'] for p in parts])
    S = D.loc[idx, st + ['mt']].to_numpy(np.float32)       # alert-time features, shared by every step
    XE, XX = np.concatenate([p['XE'] for p in parts]), np.concatenate([p['XX'] for p in parts])
    Be = Bf[:, np.arange(K + 1) + lat]                       # buy decided at k fills at k + lat
    Gx = Gf[:, XS + lat]                                     # sell decided at XS[j] fills at XS[j] + lat
    tail = Gf[:, TX + lat:]; ok = np.isfinite(tail)          # forced exit: first fillable slot from TX on
    Gterm = np.where(ok.any(1), tail[np.arange(len(tail)), ok.argmax(1)], np.nan)
    first_x = np.searchsorted(XS, np.arange(K + 1) + lat + 1)  # first exit decision after the buy fills
    meta = D.loc[idx, ['sess', 'slug', 'series', 'ts', 'y_large', 'y_mark', 'pnl_s3']].reset_index(drop=True)
    keep = np.isfinite(Gterm)
    return meta[keep].reset_index(drop=True), S[keep], XE[keep], XX[keep], Be[keep], Gx[keep], Gterm[keep], first_x


def reg():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit_exit(S, XX, Gx, Gterm, make=reg):
    """Exit stage, backwards. Returns per-step continuation models and the realized policy value from each step."""
    n, nx = Gx.shape
    cash = Gterm.copy(); Vx = np.empty((n, nx + 1)); Vx[:, nx] = Gterm; mx = [None] * nx
    for j in range(nx - 1, -1, -1):
        X = np.hstack([XX[:, j], S]); mx[j] = make().fit(X, cash); c = mx[j].predict(X)
        stop = np.isfinite(Gx[:, j]) & (Gx[:, j] >= c)
        cash = np.where(stop, Gx[:, j], cash); Vx[:, j] = cash
    return mx, Vx


def fit_entry(S, XE, R, make=reg):
    """Entry stage, backwards: buy at k for reward R[:, k] (NaN = cannot buy), wait, or give up (0)."""
    n = len(R); cin = np.zeros(n); mq, mw = [None] * (K + 1), [None] * (K + 1)
    for k in range(K, -1, -1):
        v = np.isfinite(R[:, k]); X = np.hstack([XE[:, k], S])
        mq[k] = make().fit(X[v], R[v, k]); q = np.where(v, mq[k].predict(X), -np.inf)
        if k < K: mw[k] = make().fit(X, cin); w = mw[k].predict(X)
        else: w = np.zeros(n)
        cin = np.where((q > w) & (q > 0), R[:, k], np.where(w >= 0, cin, 0.))
    return mq, mw


def fit_ls(S, XE, XX, Be, Gx, Gterm, first_x, make=reg):
    """P&L reward: buying at k earns the realized exit-policy value minus the buy cost."""
    mx, Vx = fit_exit(S, XX, Gx, Gterm, make)
    return (mx,) + fit_entry(S, XE, Vx[:, first_x] - Be, make)


def jump_labels(meta, lat):
    """1 where a large up-jump's marked entry falls within 5s after a buy decided at step k would fill."""
    m = pd.read_csv('results/makinen/single_jumps/marks.csv')
    lg = lambda p: np.log(p / (1 - p))
    m = m[(lg(m.exit_bid.clip(.005, .995)) - lg(m.entry_ask.clip(.005, .995))) >= 0.46]   # as alerts_ds.py
    J = {k: np.sort(v.ts_entry.to_numpy()) for k, v in m.groupby('series')}
    Y = np.zeros((len(meta), K + 1), np.float32)
    for sr, g in meta.groupby('series'):
        if sr not in J: continue
        fill = g.ts.to_numpy()[:, None] + (np.arange(K + 1) + lat) * 200
        Y[g.index] = np.searchsorted(J[sr], fill + 5000, side='right') > np.searchsorted(J[sr], fill, side='right')
    return Y


def predict(models, X, S):
    return np.column_stack([m.predict(np.hstack([X[:, j], S])) if m is not None else np.zeros(len(X))
                            for j, m in enumerate(models)])


def act(models, S, XE, XX, Be, Gx, Gterm, first_x, margin=0., tau=None):
    """Run the learned policy forward. Returns (entry slot or -1, P&L ticks or NaN).
    tau: instead of the policy's buy/wait/skip, buy at the first step whose predicted buy value exceeds tau."""
    mx, mq, mw = models
    Q, W, Cx = predict(mq, XE, S), predict(mw, XE, S), predict(mx, XX, S)
    n = len(Be); ent = np.full(n, -1); alive = np.ones(n, bool)
    for k in range(K + 1):
        if tau is None:
            go = alive & np.isfinite(Be[:, k]) & (Q[:, k] > np.maximum(W[:, k], 0) + margin)
            ent[go] = k; alive &= ~go & (W[:, k] > 0)
        else:
            go = alive & np.isfinite(Be[:, k]) & (Q[:, k] > tau); ent[go] = k; alive &= ~go
    pnl = np.full(n, np.nan)
    for i in np.flatnonzero(ent >= 0):
        j0 = first_x[ent[i]]; stop = np.flatnonzero(np.isfinite(Gx[i, j0:]) & (Gx[i, j0:] >= Cx[i, j0:]))
        g = Gx[i, j0 + stop[0]] if len(stop) else Gterm[i]
        pnl[i] = g - Be[i, ent[i]]
    return ent, pnl


def one_at_a_time(meta, ent, lat):
    """Skip an entry while the previous trade in the same market is still open (it held until at most TX)."""
    ok = ent >= 0
    for sr, g in meta[ok].groupby('series'):
        busy = -1
        for i, t in zip(g.index, g.ts):
            if t + ent[i] * 200 < busy: ok[i] = False
            else: busy = t + (TX + lat) * 200
    return ok


def report(T, e, p, lat, tag, Y=None):
    """Y: per-step jump labels; when given, a trade 'became a large jump' only if the jump started after its fill."""
    ok = one_at_a_time(T, e, lat) & np.isfinite(p); x = T[ok].assign(pnl=p[ok], k=e[ok])
    if Y is not None: x['y_large'] = Y[np.flatnonzero(ok), e[ok]].astype(bool)
    if not len(x): print(f'\n{tag}: no trades'); return
    L, NL = x[x.y_large], x[~x.y_large]
    be = -NL.pnl.mean() / (L.pnl.mean() - NL.pnl.mean()) if len(L) and len(NL) else np.nan
    print(f'\n{tag}: {len(x)} trades, {len(L)} became large jumps ({len(L) / len(x):.1%}; break-even ~{be:.0%}) | '
          f'{x.pnl.mean():+.2f}t per trade, total {x.pnl.sum():+.0f}t')
    print(f'  large {L.pnl.mean():+.2f}t | other up-jump {x[x.y_mark & ~x.y_large].pnl.mean():+.2f}t | '
          f'no up-jump {x[~x.y_mark].pnl.mean():+.2f}t | entry slot median {x.k.median():.0f} | '
          f'same alerts, buy at alert + 3s-stable exit: {x.pnl_s3.mean():+.2f}t')
    g = x.slug.unique(); rng = np.random.default_rng(0)
    bs = [x[x.slug.isin(rng.choice(g, len(g)))].pnl.mean() for _ in range(300)]
    print('  game-bootstrap 95%% CI [%.2f, %.2f] | by session %s' % (*np.percentile(bs, [2.5, 97.5]),
          x.groupby('sess').pnl.mean().round(2).to_dict()))


def fit(lat):
    meta, S, XE, XX, Be, Gx, Gterm, fx = load(lat)
    va = meta.sess.isin(['books_2026-09-19', 'books_2026-09-20']).to_numpy(); tr = (meta.sess <= 'books_2026-09-20').to_numpy()
    sub = lambda m: (S[m], XE[m], XX[m], Be[m], Gx[m], Gterm[m], fx)
    print(f'latency {lat * 200} ms | episodes: train {tr.sum()} (val {va.sum()}), test {(~tr).sum()} | '
          f'large jump after alert {meta.y_large.mean():.2%}')
    M0 = fit_ls(*sub(tr & ~va))
    cands = (0, 0.5, 1, 2, 3, 5)
    tot = {}
    for mg in cands:
        e, p = act(M0, *sub(va), margin=mg); tot[mg] = np.nansum(p)
    best = max(tot, key=tot.get)
    print('val total ticks by entry margin:', {m: round(v, 1) for m, v in tot.items()}, '-> margin', best)
    # the buy-value threshold that triggers on q% of validation episodes (from the train-minus-val fit)
    qv = predict(M0[1], XE[va], S[va]).max(1)
    cuts = {q: np.percentile(qv, 100 - q) for q in (10, 5, 2, 1, 0.5, 0.25)}
    print('val: highest predicted buy value per episode, quantiles 50/90/99/99.9/max:',
          np.percentile(qv, [50, 90, 99, 99.9, 100]).round(2))
    M = fit_ls(*sub(tr))
    T = meta[~tr].reset_index(drop=True)
    for mg in cands:
        e, p = act(M, *sub(~tr), margin=mg); report(T, e, p, lat, f'policy, margin {mg}' + (' <- chosen on val' if mg == best else ''))
    qt = predict(M[1], XE[~tr], S[~tr]).max(1)
    print('\ntest: highest predicted buy value per episode, quantiles 50/90/99/99.9/max:',
          np.percentile(qt, [50, 90, 99, 99.9, 100]).round(2))
    for q, tau in cuts.items():
        e, p = act(M, *sub(~tr), tau=tau); report(T, e, p, lat, f'buy when predicted value > {tau:+.2f}t (top {q}% on val)')
    # in-sample, for scale: what the policy claims on the paths it was fitted to
    e, p = act(M, *sub(tr)); print(f'\n(in-sample train, margin 0: {np.isfinite(p).sum()} trades, {np.nanmean(p):+.2f}t per trade)')


def fit_jumps(lat):
    """Entry rewarded for catching a large jump (1 - lam) or not (-lam), instead of P&L; exit and scoring stay in ticks.
    lam is the cost of a trade in jump units: break-even on the classifier's picks was ~34% large."""
    meta, S, XE, XX, Be, Gx, Gterm, fx = load(lat); Y = jump_labels(meta, lat)
    va = meta.sess.isin(['books_2026-09-19', 'books_2026-09-20']).to_numpy(); tr = (meta.sess <= 'books_2026-09-20').to_numpy()
    print(f'latency {lat * 200} ms | episodes: train {tr.sum()} (val {va.sum()}), test {(~tr).sum()} | '
          f'a large jump starts within 5s of a buy at the alert: {Y[:, 0].mean():.2%}, at any step: {Y.max(1).mean():.2%}')
    R = lambda m, lam: np.where(np.isfinite(Be[m]), Y[m] - lam, np.nan)
    lams = (0.01, 0.02, 0.05, 0.1, 0.2, 0.34)
    t0 = tr & ~va; mx0, _ = fit_exit(S[t0], XX[t0], Gx[t0], Gterm[t0])
    vsub = (S[va], XE[va], XX[va], Be[va], Gx[va], Gterm[va], fx)
    tot = {}
    for lam in lams:
        mq, mw = fit_entry(S[t0], XE[t0], R(t0, lam)); e, p = act((mx0, mq, mw), *vsub); tot[lam] = (np.nansum(p), (e >= 0).sum())
    best = max(tot, key=lambda l: tot[l][0])
    print('val (total ticks, trades) by lam:', {l: (round(v[0], 1), v[1]) for l, v in tot.items()}, '-> lam', best)
    qv = predict(fit_entry(S[t0], XE[t0], R(t0, 0.))[0], XE[va], S[va]).max(1)   # P(catch) ranking, lam-free
    cuts = {q: np.percentile(qv, 100 - q) for q in (10, 5, 2, 1, 0.5, 0.25)}
    mx, _ = fit_exit(S[tr], XX[tr], Gx[tr], Gterm[tr])
    tsub = (S[~tr], XE[~tr], XX[~tr], Be[~tr], Gx[~tr], Gterm[~tr], fx)
    T, YT = meta[~tr].reset_index(drop=True), Y[~tr]
    for lam in lams:
        mq, mw = fit_entry(S[tr], XE[tr], R(tr, lam)); e, p = act((mx, mq, mw), *tsub)
        report(T, e, p, lat, f'policy, lam {lam}' + (' <- chosen on val' if lam == best else ''), YT)
    mq0, _ = fit_entry(S[tr], XE[tr], R(tr, 0.))
    print('\ntest: highest predicted P(catch a large jump) per episode, quantiles 50/90/99/99.9/max:',
          np.percentile(predict(mq0, XE[~tr], S[~tr]).max(1), [50, 90, 99, 99.9, 100]).round(3))
    for qq, tau in cuts.items():
        e, p = act((mx, mq0, mq0), *tsub, tau=tau)
        report(T, e, p, lat, f'buy when P(catch) > {tau:.3f} (top {qq}% on val)', YT)


def selftest():
    # Exit: value path known exactly (no noise) -> sell at the peak. Entry: cheapest buy at slot 3.
    class Mean:                                              # regressor = per-path truth, isolates the backward logic
        def fit(self, X, y): self.y = dict(zip(X[:, 0], y)); return self
        def predict(self, X): return np.array([self.y[v] for v in X[:, 0]])
    n, nx = 4, len(XS)
    path = np.arange(n, dtype=float)[:, None]
    XX = np.repeat(path[:, None], nx, 1); XE = np.repeat(path[:, None], K + 1, 1)
    Gx = np.full((n, nx), 100.); Gx[:, 10] = 130.; Gterm = np.full(n, 90.)
    Be = np.full((n, K + 1), 120.); Be[:, 3] = 110.; Be[3] = 200.            # path 3: never worth buying
    fx = np.searchsorted(XS, np.arange(K + 1) + 2); S = np.zeros((n, 0))
    M = fit_ls(S, XE, XX, Be, Gx, Gterm, fx, make=Mean)
    e, p = act(M, S, XE, XX, Be, Gx, Gterm, fx)
    assert list(e) == [3, 3, 3, -1], e
    assert np.allclose(p[:3], 20.) and np.isnan(p[3]), p
    sh = np.array([10 / .5]); px = np.array([[.5, .49]]); usd = np.array([[5., 100.]])
    assert abs(sell_value(px, usd, sh)[0] - ((5 / .5 * .5 + 10 * .49) / 20 / .01 - fee((5 + 4.9) / 20))) < 1e-9
    print('selftest ok')


if __name__ == '__main__':
    lat = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {'build': build, 'selftest': selftest, 'jumps': lambda: fit_jumps(lat)}.get(sys.argv[1], lambda: fit(lat))()
