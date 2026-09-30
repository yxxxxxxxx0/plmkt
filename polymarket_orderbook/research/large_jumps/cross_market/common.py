"""Shared pieces for the cross-market study: paths, the game panel, samples, features, evaluation.

A GAME PANEL puts every recorded YES market of one game on the shared 200 ms clock:
    P[k, s, f]  float16, k = market, s = slot, f = FEATS below
    mid[k, s]   float32 (NaN where the market has no row)
plus per-market meta (role, line, orientation, normal spread) and the slots where each marked jump starts.

The TARGET is "a large upward YES jump starts on market k more than H_LO_MS and at most H_HI_MS after
t" (default 200 ms .. 5 s). Samples are taken every EVERY_MS (default 200 ms, so a 200 ms lead is not
lost between samples). All three are environment settings, e.g. XM_H_LO_MS=2000 XM_H_HI_MS=8000.

Every method is scored by `report()`: trade at the first sample a market's score clears the cut,
then stay out 10 s on that market; buy the YES ask after each fill delay in DELAYS_MS (default 0.2 s,
1 s, 3 s -- Polymarket reportedly holds marketable orders on live sports markets for 1-3 s), sell the
bid once the book has been quoted and still for 3 s (at least 5 s after entry). Output is the table
Justin asked for: trades, how many became large jumps, the rest, and P&L per trade and break-even
count at each delay.
"""
from __future__ import annotations

import json, os, glob, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))   # polymarket_orderbook/
CACHE = os.path.join(ROOT, 'research', 'large_jumps', 'cache', 'xm')
IN = os.path.join(ROOT, 'results', 'large_jumps', 'inputs')
OUT = os.path.join(ROOT, 'results', 'large_jumps', 'cross_market')
if '--smoke' in sys.argv:                  # smoke runs never write next to real results
    OUT = os.path.join(CACHE, 'smoke_out')
MARKS = os.path.join(ROOT, 'results', 'makinen', 'single_jumps', 'marks.csv')
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)

GRID = 200
_ms = lambda name, default: int(os.environ.get(name, default)) // GRID
EVERY = max(1, _ms('XM_EVERY_MS', 200))       # sample spacing, in slots
H_LO, H_HI = _ms('XM_H_LO_MS', 200), _ms('XM_H_HI_MS', 5000)   # label: jump starts in (t+H_LO, t+H_HI]
DELAYS = [int(d) // GRID for d in os.environ.get('XM_DELAYS_MS', '200,1000,3000').split(',')]
HIST = 100                   # 20 s of history for the sequence models
DEBOUNCE = 50                # 10 s between trades on one market
NEG_SCALE = EVERY / 5        # keeps the number of sampled negatives the same whatever the spacing
MINHOLD, STABLE, TIMEOUT = 25, 15, 450
STAKE, FEE = 10.0, 0.05
TEST_FROM = 'books_2026-09-21'
ROLES = {'moneyline': 0, 'spread': 1, 'total': 2}
K_MAX = 10
FEATS = ['spr_rel', 'pulled', 'logdep_b', 'logdep_a', 'logtop_b', 'logtop_a', 'present']
F = len(FEATS)
LARGE_LOGODDS = 0.46

fee = lambda p: FEE * p * (1 - p) / 0.01
logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))


def sessions():
    return sorted(pd.read_csv(MARKS, usecols=['session']).session.unique())


def is_test(sess):
    return sess >= TEST_FROM


# --------------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------------

def panel_path(sess, slug):
    return os.path.join(CACHE, f'panel_{sess}_{slug}.npz')


def load_panel(sess, slug):
    z = np.load(panel_path(sess, slug), allow_pickle=False)
    meta = json.loads(str(z['meta']))
    return dict(P=z['P'], mid=z['mid'], s0=int(z['s0']), meta=meta,
                starts=[z[f'start_{k}'] for k in range(len(meta))],
                starts_large=[z[f'large_{k}'] for k in range(len(meta))])


def list_panels(smoke=False):
    out = []
    for f in sorted(glob.glob(os.path.join(CACHE, 'panel_*.npz'))):
        b = os.path.basename(f)[len('panel_'):-4]
        sess, slug = b[:len('books_2026-09-21')], b[len('books_2026-09-21') + 1:]
        out.append((sess, slug))
    if smoke:
        keep = sorted({s for s, _ in out})
        keep = [s for s in keep if not is_test(s)][-1:] + [s for s in keep if is_test(s)][:1]
        out = [(s, g) for s, g in out if s in keep][:12]
    return out


def bid_ask(pan, k):
    mid = pan['mid'][k].astype(np.float64)
    spr = pan['P'][k, :, 0].astype(np.float64) * pan['meta'][k]['norm']
    return mid - spr * .005, mid + spr * .005, spr


# --------------------------------------------------------------------------------------------
# samples and labels
# --------------------------------------------------------------------------------------------

def samples(pan, neg_rate=1.0, rng=None):
    """(k, t, y_large, y_mark) for every market and every second with full history and horizon."""
    S = pan['P'].shape[1]
    ts = np.arange(HIST, S - H_HI - 1, EVERY)
    rows = []
    for k in range(len(pan['meta'])):
        pres = pan['P'][k, ts, 6] > 0
        t = ts[pres]
        if not len(t): continue
        def lab(st):
            if not len(st): return np.zeros(len(t), bool)
            i = np.searchsorted(st, t + H_LO + 1)
            return (i < len(st)) & (st[np.minimum(i, len(st) - 1)] <= t + H_HI)
        yl, ym = lab(pan['starts_large'][k]), lab(pan['starts'][k])
        keep = np.ones(len(t), bool)
        if neg_rate < 1.0:
            keep = yl | ym | (rng.random(len(t)) < neg_rate)
        rows.append(np.c_[np.full(keep.sum(), k), t[keep], yl[keep], ym[keep]])
    if not rows: return np.zeros((0, 4), np.int64)
    return np.concatenate(rows).astype(np.int64)


# --------------------------------------------------------------------------------------------
# game state (causal: latest statsapi state stamped at or before the slot's time)
# --------------------------------------------------------------------------------------------

_G = None
RE = {0: [.48, .86, 1.10, 1.44, 1.35, 1.78, 1.96, 2.29], 1: [.25, .51, .66, .88, .95, 1.13, 1.35, 1.54],
      2: [.10, .22, .32, .43, .35, .48, .57, .75]}
GAME_FEATS = ['g_inning', 'g_outs', 'g_on1', 'g_on2', 'g_on3', 'g_balls', 'g_strikes', 'g_margin', 'g_runs',
              'g_re24', 'g_since_event', 'g_is_top']


def game_state(slug, ts_ms):
    global _G
    if _G is None:
        _G = pd.read_csv(os.path.join(IN, 'game_state.csv')).sort_values(['slug', 't'])
    g = _G[_G.slug == slug]
    out = np.full((len(ts_ms), len(GAME_FEATS)), np.nan, np.float32)
    if not len(g): return out
    j = np.searchsorted(g.t.to_numpy(), ts_ms, side='right') - 1; ok = j >= 0
    r = g.iloc[np.clip(j, 0, None)]
    b = (r.on1 + 2 * r.on2 + 4 * r.on3).to_numpy(); o = r.outs.clip(upper=2).to_numpy()
    v = np.c_[r.inning, r.outs, r.on1, r.on2, r.on3, r.balls, r.strikes, (r.home - r.away).abs(), r.home + r.away,
              [RE[a][c] for a, c in zip(o, b)], (ts_ms - r.t.to_numpy()) / 1000, r.is_top.astype(float)]
    out[ok] = v[ok]
    return out


def yes_bats(meta_k, is_top):
    side = meta_k['side']
    if side == 'Over': return np.ones_like(is_top)
    if side == 'Under': return np.zeros_like(is_top)
    return np.where(np.isnan(is_top), np.nan, (is_top == 1) if side == 'away' else (is_top == 0)).astype(np.float32)


# --------------------------------------------------------------------------------------------
# sequence input for the deep models: target market first, then the others in canonical order
# --------------------------------------------------------------------------------------------

SEQ_CH = F + 1               # FEATS (spr_rel..present) + oriented mid change vs t, in ticks


def seq_batch(pan, ks, ts):
    """(B, K_MAX, HIST, SEQ_CH) float32 and (B, K_MAX) role ids (-1 = empty) for samples (k, t)."""
    P, mid, meta = pan['P'], pan['mid'], pan['meta']
    K = len(meta)
    order_all = sorted(range(K), key=lambda j: (meta[j]['role'], meta[j]['line'] or 0, meta[j]['side']))
    ori = np.array([m['ori'] for m in meta], np.float32)
    X = np.zeros((len(ks), K_MAX, HIST, SEQ_CH), np.float32); R = np.full((len(ks), K_MAX), -1, np.int64)
    for b, (k, t) in enumerate(zip(ks, ts)):
        order = [k] + [j for j in order_all if j != k][:K_MAX - 1]
        w = P[order, t - HIST + 1:t + 1, :].astype(np.float32)                 # (n, HIST, F)
        m = mid[order, t - HIST + 1:t + 1]
        d = (m - mid[order, t][:, None]) / .01 * ori[order][:, None]
        X[b, :len(order), :, :F] = np.nan_to_num(w)
        X[b, :len(order), :, F] = np.nan_to_num(np.clip(d, -100, 100))
        R[b, :len(order)] = [meta[j]['role'] for j in order]
    return X, R


# --------------------------------------------------------------------------------------------
# evaluation: debounced trades on the full test grid, priced from the panel's top of book
# --------------------------------------------------------------------------------------------

def price_trade(pan, k, t, delay):
    bid, ask, spr = bid_ask(pan, k)
    meta = pan['meta'][k]; cut = meta['norm'] + max(2, meta['norm'])
    e = t + delay; S = len(bid)
    if e >= S or not np.isfinite(ask[e]): return np.nan
    if np.expm1(float(pan['P'][k, e, 5])) < STAKE: return np.nan                # < $10 at the best ask
    q = np.isfinite(spr) & (spr <= cut)
    lo = e + MINHOLD
    for j in range(lo, min(S, e + TIMEOUT)):
        w = slice(j - STABLE + 1, j + 1)
        if q[w].all() and np.nanmax(bid[w]) - np.nanmin(bid[w]) <= .0101 and np.nanmax(ask[w]) - np.nanmin(ask[w]) <= .0101:
            break
    else:
        nxt = np.flatnonzero(q[e + TIMEOUT:]) if e + TIMEOUT < S else []
        if not len(nxt): return np.nan
        j = e + TIMEOUT + nxt[0]
    return (bid[j] - ask[e]) / .01 - fee(ask[e]) - fee(bid[j])


def report(name, R, cuts=(5, 1, 0.5, 0.25, 0.1), panels=None, days=None):
    """R: DataFrame(sess, slug, k, t, p, y_large, y_mark) over the FULL test grid. Prints and returns the table."""
    R = R.sort_values(['sess', 'slug', 'k', 't']).reset_index(drop=True)
    rows = []
    days = days or R.sess.nunique()
    for q in cuts:
        thr = np.quantile(R.p, 1 - q / 100)
        f = R[R.p >= thr]
        trades = []
        for (sess, slug, k), g in f.groupby(['sess', 'slug', 'k'], sort=False):
            last = -10**9
            for t, yl, ym in zip(g.t, g.y_large, g.y_mark):
                if t - last >= DEBOUNCE:
                    trades.append((sess, slug, k, t, yl, ym)); last = t
        T = pd.DataFrame(trades, columns=['sess', 'slug', 'k', 't', 'y_large', 'y_mark'])
        nl = int(T.y_large.sum()) if len(T) else 0
        row = dict(method=name, cut=f'top {q}%', trades=len(T), trades_per_day=len(T) / days, large=nl,
                   large_pct=nl / max(len(T), 1), rest=len(T) - nl)
        for d in DELAYS:
            tag = f'{d * GRID / 1000:g}s'
            pnl = pd.Series(np.nan, index=T.index)
            if panels is not None and len(T):
                for (sess, slug), g in T.groupby(['sess', 'slug']):
                    pan = panels(sess, slug)
                    pnl[g.index] = [price_trade(pan, k, t, d) for k, t in zip(g.k, g.t)]
            win, loss = pnl[T.y_large == 1].mean(), pnl[T.y_large == 0].mean()
            # share of trades that must be large to break even; undefined unless large trades win and the rest lose
            be = -loss / (win - loss) if np.isfinite(win) and np.isfinite(loss) and win > 0 > loss else np.nan
            row.update({f'pnl_{tag}': pnl.mean(), f'pnl_large_{tag}': win, f'pnl_rest_{tag}': loss,
                        f'breakeven_count_{tag}': int(np.ceil(be * len(T))) if np.isfinite(be) else np.nan})
        rows.append(row)
    out = pd.DataFrame(rows)
    print(f'target: large upward YES jump starting {H_LO * GRID / 1000:g}-{H_HI * GRID / 1000:g} s after the signal; '
          f'P&L at fill delays {", ".join(f"{d * GRID / 1000:g}s" for d in DELAYS)}')
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(out.round(2).to_string(index=False))
    return out


def save_report(name, table):
    p = os.path.join(OUT, 'reports.csv')
    old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    old = old[old.get('method', pd.Series(dtype=str)) != name] if len(old) else old
    # drop columns no remaining row uses (left over from runs with other settings)
    pd.concat([old, table], ignore_index=True).dropna(axis=1, how='all').to_csv(p, index=False)


class PanelCache:
    """Small LRU of loaded panels (each is a few MB)."""
    def __init__(self, n=40):
        self.n, self.d = n, {}

    def __call__(self, sess, slug):
        key = (sess, slug)
        if key not in self.d:
            if len(self.d) >= self.n: self.d.pop(next(iter(self.d)))
            self.d[key] = load_panel(sess, slug)
        return self.d[key]


def smoke_flag(argv):
    return '--smoke' in argv
