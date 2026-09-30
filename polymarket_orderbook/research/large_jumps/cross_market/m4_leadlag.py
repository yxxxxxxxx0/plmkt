"""Method 4: does any market reliably move BEFORE another? Three checks, no model to trust.

  a) Granger tests on 1 s series, per game and per ordered pair of markets (A -> B), on
     |price change| (timing) and on oriented price change (direction). Two versions: A's lags 1-5 s,
     and A's lags 3-8 s only -- the second is the TRADABLE one, since a taker's order waits 1-3 s.
  b) Event timing around marked large jumps: for each large jump on market B, when did each sister
     market A first move (a >=2 tick move off a still level) in the 10 s around it.
  c) A bivariate Hawkes process (exponential kernel) fitted per game and pair on move times: how
     strongly a move in A raises the rate of moves in B. Read the CROSS terms only: a move is only
     counted after 5 s of stillness, so a market cannot re-excite itself within 5 s and the self
     terms are pushed towards zero by construction.

Outputs results/large_jumps/cross_market/leadlag_*.csv and a printed summary.

    python research/large_jumps/cross_market/m4_leadlag.py [--smoke]
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, os.path.dirname(__file__))
from common import list_panels, load_panel, bid_ask, smoke_flag, OUT, GRID

ROLE = {0: 'moneyline', 1: 'spread', 2: 'total'}
smoke = smoke_flag(sys.argv)


def lagmat(x, lags):
    return np.column_stack([np.roll(x, l) for l in lags])


def granger(a, b, lags_b, lags_a):
    """F-test: do lags of a improve an OLS forecast of b beyond b's own lags? Returns (p, partial R2)."""
    L = max(max(lags_b), max(lags_a))
    y = b[L:]; Xr = np.c_[np.ones(len(y)), lagmat(b, lags_b)[L:]]; Xu = np.c_[Xr, lagmat(a, lags_a)[L:]]
    if len(y) < 200 or np.std(y) == 0 or np.std(a) == 0: return np.nan, np.nan
    rr = np.sum((y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]) ** 2)
    ru = np.sum((y - Xu @ np.linalg.lstsq(Xu, y, rcond=None)[0]) ** 2)
    q, dfu = len(lags_a), len(y) - Xu.shape[1]
    Fs = ((rr - ru) / q) / (ru / dfu)
    return float(stats.f.sf(Fs, q, dfu)), float((rr - ru) / rr)


def move_onsets(pan, k):
    """Slots where the mid moves >= 2 ticks within 1 s after standing still (range <= 1 c) for 5 s."""
    mid = pd.Series(pan['mid'][k]).ffill().to_numpy()
    r = pd.Series(mid).rolling(25)
    still = ((r.max() - r.min()) <= .0101).to_numpy()
    fwd = np.abs(np.r_[mid[5:] - mid[:-5], np.zeros(5)])
    on = np.flatnonzero(still & (fwd >= .02 - 1e-9))
    if not len(on): return on
    keep = np.r_[True, np.diff(on) > 25]                     # one onset per move
    return on[keep] + 1


def hawkes_nll(params, ta, tb, T):
    mu = np.exp(params[:2]); al = np.exp(params[2:6]).reshape(2, 2); be = np.exp(params[6])
    ev = np.r_[np.c_[ta, np.zeros(len(ta))], np.c_[tb, np.ones(len(tb))]]
    ev = ev[np.argsort(ev[:, 0])]
    R = np.zeros(2); last = 0.0; ll = 0.0
    for t, d in ev:
        R *= np.exp(-be * (t - last)); last = t; d = int(d)
        lam = mu[d] + al[d] @ R
        ll += np.log(max(lam, 1e-12)); R[d] += 1
    comp = mu.sum() * T
    for d, tt in ((0, ta), (1, tb)):
        comp += al[:, d].sum() * np.sum(1 - np.exp(-be * (T - tt))) / be
    return -(ll - comp)


def fit_hawkes(ta, tb, T):
    x0 = np.log([len(ta) / T + 1e-6, len(tb) / T + 1e-6, .1, .1, .1, .1, 1 / 5])
    r = optimize.minimize(hawkes_nll, x0, args=(ta, tb, T), method='L-BFGS-B', options=dict(maxiter=200))
    al = np.exp(r.x[2:6]).reshape(2, 2); be = np.exp(r.x[6])
    return dict(a_on_a=al[0, 0] / be, b_on_a=al[0, 1] / be, a_on_b=al[1, 0] / be, b_on_b=al[1, 1] / be,
                decay_s=1 / be, ok=r.success)


G, E, H = [], [], []
for sess, slug in list_panels(smoke):
    pan = load_panel(sess, slug); meta = pan['meta']; K = len(meta)
    mid = pd.DataFrame(pan['mid'].T).ffill().to_numpy().T[:, ::5]                     # 1 s
    d = np.diff(mid, axis=1) / .01
    d = np.nan_to_num(d) * np.array([m['ori'] for m in meta])[:, None]
    ons = [move_onsets(pan, k) for k in range(K)]
    T = pan['P'].shape[1] * GRID / 1000
    for a in range(K):
        for b in range(K):
            if a == b: continue
            pair = f"{ROLE[meta[a]['role']]}->{ROLE[meta[b]['role']]}"
            row = dict(sess=sess, slug=slug, pair=pair, a=meta[a]['series'], b=meta[b]['series'])
            for nm, x in (('abs', np.abs(d)), ('dir', d)):
                row[f'p_{nm}_1to5'], row[f'r2_{nm}_1to5'] = granger(x[a], x[b], range(1, 6), range(1, 6))
                row[f'p_{nm}_3to8'], row[f'r2_{nm}_3to8'] = granger(x[a], x[b], range(1, 9), range(3, 9))
            G.append(row)
            if a < b and len(ons[a]) > 20 and len(ons[b]) > 20:
                h = fit_hawkes(ons[a] * GRID / 1000, ons[b] * GRID / 1000, T)
                H.append(dict(sess=sess, slug=slug, pair=f"{ROLE[meta[a]['role']]}<->{ROLE[meta[b]['role']]}", **h))
    for b in range(K):
        for s in pan['starts_large'][b]:
            for a in range(K):
                if a == b: continue
                o = ons[a]; w = o[(o >= s - 50) & (o <= s + 50)]
                E.append(dict(sess=sess, slug=slug, pair=f"{ROLE[meta[a]['role']]}->{ROLE[meta[b]['role']]}",
                              lead_s=(s - w.min()) * GRID / 1000 if len(w) else np.nan))
    print(sess, slug, flush=True)

G, E, H = pd.DataFrame(G), pd.DataFrame(E), pd.DataFrame(H)
for nm, df in (('granger', G), ('event_timing', E), ('hawkes', H)):
    df.to_csv(os.path.join(OUT, f'leadlag_{nm}.csv'), index=False)

print('\n(a) GRANGER, share of game-pairs significant at p<0.01, and median partial R^2')
agg = G.groupby('pair').agg(n=('pair', 'size'),
                            abs_1to5=('p_abs_1to5', lambda p: (p < .01).mean()), r2_abs_1to5=('r2_abs_1to5', 'median'),
                            abs_3to8=('p_abs_3to8', lambda p: (p < .01).mean()), r2_abs_3to8=('r2_abs_3to8', 'median'),
                            dir_3to8=('p_dir_3to8', lambda p: (p < .01).mean()), r2_dir_3to8=('r2_dir_3to8', 'median'))
print(agg.round(4).to_string())
print('\n(b) AROUND LARGE JUMPS on market B: when sister market A first moved (lead_s > 0 = A moved first)')
ev = E.groupby('pair').lead_s.agg(n='size', moved=lambda x: x.notna().mean(),
                                  lead_over_2s=lambda x: (x > 2).mean(), lead_0_2s=lambda x: ((x > 0) & (x <= 2)).mean(),
                                  same_or_after=lambda x: (x <= 0).mean())
print(ev.round(3).to_string())
print('\n(c) HAWKES branching ratios (expected extra moves caused by one move), median over games')
print(H.groupby('pair')[['a_on_a', 'b_on_a', 'a_on_b', 'b_on_b', 'decay_s']].median().round(3).to_string())
