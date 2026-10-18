"""LOB state and trajectory features, shared by the event study and models.

Everything here is strictly causal: a feature at row i uses rows <= i only.
The one place that is easy to get wrong is a rolling window, so every rolling
statistic below is computed with pandas' trailing window and then, where a
"change since" is wanted, differenced against a SHIFTED value -- never
centred, never forward-filled from the future.

Representation, per section 2 of the brief:

  price   distance from the current mid, in ticks, not absolute price
  size    log1p of resting dollars at the level
  mask    1 where a level genuinely exists, 0 where the book is shallower
          than K. This matters: the stored tensor encodes a missing level as
          price 0, which after mid-centring becomes a large negative distance
          that a model would read as a real quote far from the touch. The
          mask is derived from size, because a level with zero resting dollars
          did not exist.

The mid itself is kept as a separate optional scalar, since mid-centring is
exactly translation invariance and erases the level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 0.01
GRID_MS = 200
K = 10


# --------------------------------------------------------------------------- #
# decoding the stored tensor
# --------------------------------------------------------------------------- #
def decode(T, mid):
    """(n, 4, K) stored tensor -> price distance in ticks, dollars, mask.

    The tensor channels are [bid_dist, bid_logusd, ask_dist, ask_logusd] with
    distances already expressed in ticks from that row's mid.
    """
    bd = np.asarray(T[:, 0, :], np.float32)
    bl = np.asarray(T[:, 1, :], np.float32)
    ad = np.asarray(T[:, 2, :], np.float32)
    al = np.asarray(T[:, 3, :], np.float32)
    # A level that does not exist was stored with zero size, hence log1p -> 0.
    bmask = (bl > 0).astype(np.float32)
    amask = (al > 0).astype(np.float32)
    busd = np.expm1(bl) * bmask
    ausd = np.expm1(al) * amask
    # zero out the fabricated distance on missing levels so nothing downstream
    # can mistake it for a quote
    bd = bd * bmask
    ad = ad * amask
    return dict(bid_dist=bd, ask_dist=ad, bid_usd=busd, ask_usd=ausd,
                bid_mask=bmask, ask_mask=amask, mid=np.asarray(mid, np.float64))


# --------------------------------------------------------------------------- #
# instantaneous state
# --------------------------------------------------------------------------- #
def _hhi(w):
    """Herfindahl concentration of depth across levels, 1 = all at one level."""
    tot = w.sum(axis=1, keepdims=True)
    p = np.divide(w, np.maximum(tot, 1e-9))
    return (p ** 2).sum(axis=1)


def _entropy(w):
    """Shannon entropy of the depth profile, in nats. 0 = one level only."""
    tot = w.sum(axis=1, keepdims=True)
    p = np.divide(w, np.maximum(tot, 1e-9))
    return -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)


def _slope(dist, usd, mask):
    """Dollars per tick away from the touch: how fast the book thickens.

    Least-squares slope of cumulative dollars against |distance|, computed per
    row over the levels that exist. A flat book has a small slope; a book with
    everything at the touch and nothing behind it has a small slope too, which
    is why concentration is reported alongside it rather than instead of it.
    """
    x = np.abs(dist) * mask
    y = np.cumsum(usd * mask, axis=1)
    n = np.maximum(mask.sum(axis=1), 1)
    mx = (x * mask).sum(axis=1) / n
    my = (y * mask).sum(axis=1) / n
    cov = (((x - mx[:, None]) * (y - my[:, None])) * mask).sum(axis=1)
    var = (((x - mx[:, None]) ** 2) * mask).sum(axis=1)
    return np.divide(cov, np.maximum(var, 1e-9))


def state_features(d, spread_ticks=None):
    """Instantaneous book geometry. One row in, one row out, no time axis."""
    bu, au = d["bid_usd"], d["ask_usd"]
    bm, am = d["bid_mask"], d["ask_mask"]
    bdep = bu.sum(axis=1)
    adep = au.sum(axis=1)
    tot = bdep + adep
    f = {
        "mid": d["mid"],
        "spread_ticks": (spread_ticks if spread_ticks is not None
                         else (d["ask_dist"][:, 0] - d["bid_dist"][:, 0])),
        "l1_bid_usd": bu[:, 0], "l1_ask_usd": au[:, 0],
        "log_bid_depth": np.log1p(bdep), "log_ask_depth": np.log1p(adep),
        "log_total_depth": np.log1p(tot),
        "imbalance": np.divide(bdep - adep, np.maximum(tot, 1e-9)),
        "imbalance_l1": np.divide(bu[:, 0] - au[:, 0],
                                  np.maximum(bu[:, 0] + au[:, 0], 1e-9)),
        "hhi_bid": _hhi(bu), "hhi_ask": _hhi(au),
        "entropy_bid": _entropy(bu), "entropy_ask": _entropy(au),
        "slope_bid": _slope(d["bid_dist"], bu, bm),
        "slope_ask": _slope(d["ask_dist"], au, am),
        "levels_bid": bm.sum(axis=1), "levels_ask": am.sum(axis=1),
        # how far the book reaches, and how much sits close to the touch
        "reach_bid": np.abs(d["bid_dist"] * bm).max(axis=1),
        "reach_ask": np.abs(d["ask_dist"] * am).max(axis=1),
    }
    for w in (2, 5):
        near_b = (bu * (np.abs(d["bid_dist"]) <= w) * bm).sum(axis=1)
        near_a = (au * (np.abs(d["ask_dist"]) <= w) * am).sum(axis=1)
        f[f"log_bid_usd_within_{w}t"] = np.log1p(near_b)
        f[f"log_ask_usd_within_{w}t"] = np.log1p(near_a)
        f[f"near_frac_bid_{w}t"] = np.divide(near_b, np.maximum(bdep, 1e-9))
        f[f"near_frac_ask_{w}t"] = np.divide(near_a, np.maximum(adep, 1e-9))
    return pd.DataFrame(f)


STATE_COLS = None            # filled on first use by state_features caller


# --------------------------------------------------------------------------- #
# trajectory: how the state has been changing
# --------------------------------------------------------------------------- #
TRAJ_BASE = ["log_total_depth", "log_bid_depth", "log_ask_depth", "imbalance",
             "spread_ticks", "hhi_bid", "hhi_ask", "entropy_bid",
             "entropy_ask", "l1_bid_usd", "l1_ask_usd"]


def trajectory_features(F, sid, lags_s=(5, 10, 30), grid_ms=GRID_MS,
                        book_age_ms=None, mid=None):
    """Changes in the state over trailing windows, per series.

    `sid` groups rows into series so a window never spans two markets. All
    windows trail: value_now - value_{t-lag}. Nothing reads forward.
    """
    out = {}
    g = pd.Series(sid)
    for lag_s in lags_s:
        L = int(lag_s * 1000 / grid_ms)
        for c in TRAJ_BASE:
            if c not in F.columns:
                continue
            prev = F[c].groupby(g, sort=False).shift(L)
            out[f"d_{c}_{lag_s}s"] = (F[c] - prev).to_numpy()
        if mid is not None:
            pm = pd.Series(mid).groupby(g, sort=False).shift(L)
            out[f"ret_{lag_s}s"] = ((pd.Series(mid) - pm) / TICK).to_numpy()
            # realised volatility of the mid over the trailing window
            dm = pd.Series(mid).groupby(g, sort=False).diff().fillna(0.0)
            out[f"rv_{lag_s}s"] = (dm.pow(2).groupby(g, sort=False)
                                   .rolling(L, min_periods=2).mean()
                                   .pow(0.5).reset_index(level=0, drop=True)
                                   .to_numpy() / TICK)
        if book_age_ms is not None:
            # update intensity: how many of the trailing slots carried a fresh
            # book. A rising rate is the microstructure "something is
            # happening" signal that a pure snapshot cannot express.
            fresh = pd.Series((np.asarray(book_age_ms) < grid_ms).astype(float))
            out[f"upd_rate_{lag_s}s"] = (fresh.groupby(g, sort=False)
                                         .rolling(L, min_periods=1).mean()
                                         .reset_index(level=0, drop=True)
                                         .to_numpy())
    return pd.DataFrame(out, index=F.index)


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def jump_labels(mid, sid, ts, horizon_s, thresholds, grid_ms=GRID_MS):
    """future_move(t,H) = max_{t < u <= t+H} |mid_u - mid_t|, then thresholded.

    The window EXCLUDES t, so a jump already in progress at t does not label
    itself. `valid` marks rows whose whole forward window is present and
    contiguous in time -- the grid leaves real gaps, and a shift across one
    would silently compare prices minutes apart.
    """
    h = int(horizon_s * 1000 / grid_ms)
    m = pd.Series(np.asarray(mid, np.float64))
    g = pd.Series(np.asarray(sid))
    ts = np.asarray(ts, np.int64)

    # rolling max/min over the NEXT h rows, excluding the current one:
    # shift(-1) first so the window starts at t+1, then take a trailing window
    # of length h and shift it back by h-1.
    fwd = m.groupby(g, sort=False).shift(-1)
    rmax = (fwd.groupby(g, sort=False).rolling(h, min_periods=h).max()
            .reset_index(level=0, drop=True).groupby(g, sort=False).shift(-(h - 1)))
    rmin = (fwd.groupby(g, sort=False).rolling(h, min_periods=h).min()
            .reset_index(level=0, drop=True).groupby(g, sort=False).shift(-(h - 1)))
    up = (rmax - m).to_numpy()
    dn = (m - rmin).to_numpy()
    move = np.maximum(np.nan_to_num(up, nan=-1), np.nan_to_num(dn, nan=-1))

    # contiguity: row i+h must exist, be the same series, and be exactly h
    # grid steps later
    n = len(m)
    idx = np.arange(n)
    j = np.minimum(idx + h, n - 1)
    same = (np.asarray(sid)[j] == np.asarray(sid)) & (idx + h < n)
    contig = same & (ts[j] - ts == h * grid_ms)
    valid = contig & np.isfinite(up) & np.isfinite(dn)

    out = {"future_move": move, "valid_label": valid}
    for J in thresholds:
        out[f"jump_{J:g}"] = (move >= J - 1e-12) & valid
        # three-class: sign of the larger excursion
        sgn = np.where(up >= dn, 1, -1)
        out[f"dir_{J:g}"] = np.where(~out[f"jump_{J:g}"], 0, sgn)
    return pd.DataFrame(out)
