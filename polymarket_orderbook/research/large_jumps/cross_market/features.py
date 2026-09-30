"""Hand-built features for one sample (market k, slot t), from the target market and its sister markets.

OWN   the target market's book: spread, pulls, depth drain over 4 s / 10 s, top sizes, oriented price
      change over 2 / 5 / 10 / 30 s, how long the level has stood, distance from 0.50.
CROSS the rest of the game: how many markets are draining or pulled, the deepest drain, which market
      pulled first in the last 10 s and how far ahead of this one, whether the team markets and the
      totals moved together, and how far this market has drifted from its sisters over 5 s and 30 s.
GAME  statsapi game state at t (causal) and whether the YES side is batting.
All windows end at t; nothing looks forward.
"""
import numpy as np, pandas as pd
from common import GRID, game_state, yes_bats, GAME_FEATS, bid_ask, logit


def market_arrays(pan):
    P, mid = pan['P'], pan['mid']
    K, S, _ = P.shape
    A = {}
    dep = np.expm1(P[:, :, 2].astype(np.float32)) + np.expm1(P[:, :, 3].astype(np.float32))
    A['dep'] = dep
    A['pulled'] = P[:, :, 1].astype(np.float32)
    A['cpull'] = np.concatenate([np.zeros((K, 1), np.float32), np.cumsum(A['pulled'], 1)], 1)
    A['spr_rel'] = P[:, :, 0].astype(np.float32)
    A['present'] = P[:, :, 6] > 0
    A['mid'] = pd.DataFrame(mid.T).ffill().to_numpy().T              # carry the last price across holes
    A['ori'] = np.array([m['ori'] for m in pan['meta']], np.float32)
    A['role'] = np.array([m['role'] for m in pan['meta']])
    # slots since bid or ask last changed by more than 1 cent
    since = np.zeros((K, S), np.float32)
    for k in range(K):
        b, a, _ = bid_ask(pan, k)
        b = pd.Series(b).ffill().to_numpy(); a = pd.Series(a).ffill().to_numpy()
        ch = np.r_[True, (np.abs(np.diff(b)) > .011) | (np.abs(np.diff(a)) > .011)]
        last = np.maximum.accumulate(np.where(ch, np.arange(S), 0))
        since[k] = (np.arange(S) - last) * GRID / 1000
    A['since'] = since
    A['pull_idx'] = [np.flatnonzero(A['pulled'][k] > 0) for k in range(K)]
    return A


def _ratio(dep, k, t, lag):
    return dep[k, t] / np.maximum(dep[k, t - lag], 1.0)


def features(pan, smp, A=None, with_cross=True, with_game=True):
    """smp: int array (n, >=2) of (k, t). Returns a float32 DataFrame."""
    A = A or market_arrays(pan)
    k, t = smp[:, 0], smp[:, 1]
    K = A['dep'].shape[0]
    ori, mid = A['ori'], A['mid']
    X = {}
    X['own_spr_rel'] = A['spr_rel'][k, t]
    X['own_pulled'] = A['pulled'][k, t]
    X['own_pulled30'] = (A['cpull'][k, t + 1] - A['cpull'][k, t - 149]) / 150
    X['own_drain4'] = _ratio(A['dep'], k, t, 20); X['own_drain10'] = _ratio(A['dep'], k, t, 50)
    X['own_logdep'] = np.log1p(A['dep'][k, t])
    X['own_logtop_b'] = pan['P'][k, t, 4].astype(np.float32); X['own_logtop_a'] = pan['P'][k, t, 5].astype(np.float32)
    for w, nm in ((10, '2'), (25, '5'), (50, '10'), (150, '30')):
        X[f'own_d{nm}'] = (mid[k, t] - mid[k, t - w]) / .01 * ori[k]
    X['own_since'] = A['since'][k, t]
    X['own_dist50'] = np.abs(mid[k, t] - .5)
    X['own_role'] = A['role'][k].astype(np.float32)
    if with_cross:
        n = len(k)
        dr4 = np.stack([_ratio(A['dep'], np.full(n, j), t, 20) for j in range(K)], 1)            # (n, K)
        pul = A['pulled'][:, t].T
        pul5 = np.stack([(A['cpull'][j, t + 1] - A['cpull'][j, t - 24]) > 0 for j in range(K)], 1)
        pres = A['present'][:, t].T
        other = np.ones((n, K), bool); other[np.arange(n), k] = False; other &= pres
        m_dr = np.where(other, dr4, np.nan)
        X['x_n_markets'] = other.sum(1)
        X['x_n_draining'] = (np.where(other, dr4 < .5, False)).sum(1)
        X['x_min_drain'] = np.nanmin(np.where(other, dr4, np.inf), 1)
        X['x_mean_drain'] = np.nanmean(m_dr, 1)
        X['x_n_pulled'] = (pul.astype(bool) & other).sum(1)
        X['x_n_pulled5'] = (pul5 & other).sum(1)
        # first pull in the last 10 s, per market (slot index or NaN)
        first = np.full((n, K), np.nan)
        for j in range(K):
            pi = A['pull_idx'][j]
            if not len(pi): continue
            i = np.searchsorted(pi, t - 50)
            ok = (i < len(pi)) & (pi[np.minimum(i, len(pi) - 1)] <= t)
            first[ok, j] = pi[np.minimum(i, len(pi) - 1)][ok]
        f_other = np.where(other, first, np.nan)
        earliest = np.nanmin(np.where(np.isnan(f_other), np.inf, f_other), 1)
        earliest = np.where(np.isfinite(earliest), earliest, np.nan)
        own_first = first[np.arange(n), k]
        X['x_first_pull_ago'] = (t - earliest) * GRID / 1000
        X['x_first_vs_own'] = (own_first - earliest) * GRID / 1000                 # >0: another market pulled first
        role_first = np.full(n, np.nan)
        okf = np.isfinite(earliest)
        if okf.any():
            jj = np.nanargmin(np.where(np.isnan(f_other[okf]), np.inf, f_other[okf]), 1)
            role_first[okf] = A['role'][jj]
        X['x_first_role'] = role_first
        # co-movement: oriented 5 s changes of the team markets and of the totals, excluding k
        d5 = np.stack([(mid[j, t] - mid[j, t - 25]) / .01 * ori[j] for j in range(K)], 1)
        team = np.isin(A['role'], [0, 1])[None, :] & other; tot = (A['role'] == 2)[None, :] & other
        X['x_team_d5'] = np.nansum(np.where(team, d5, 0), 1)
        X['x_tot_d5'] = np.nansum(np.where(tot, d5, 0), 1)
        X['x_team_absd5'] = np.nansum(np.where(team, np.abs(d5), 0), 1)
        X['x_tot_absd5'] = np.nansum(np.where(tot, np.abs(d5), 0), 1)
        # drift from sisters (same group) in oriented log-odds, 5 s and 30 s
        for w, nm in ((25, '5'), (150, '30')):
            dl = np.stack([(logit(mid[j, t]) - logit(mid[j, t - w])) * ori[j] for j in range(K)], 1)
            grp = np.isin(A['role'], [0, 1])
            same = np.where(grp[k][:, None], grp[None, :], ~grp[None, :]) & other
            sis = np.nanmedian(np.where(same, dl, np.nan), 1)
            X[f'x_drift{nm}'] = dl[np.arange(n), k] - sis
    if with_game:
        G = game_state(pan['meta'][0]['series'].split('|')[0], (pan['s0'] + t) * GRID)
        for i, c in enumerate(GAME_FEATS): X[c] = G[:, i]
        yb = np.full(len(k), np.nan, np.float32)
        for kk in np.unique(k):
            m = k == kk; yb[m] = yes_bats(pan['meta'][kk], G[m, GAME_FEATS.index('g_is_top')])
        X['g_yes_bats'] = yb
    return pd.DataFrame({c: np.asarray(v, np.float32) for c, v in X.items()})
