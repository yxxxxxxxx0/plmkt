"""
Build the research panel: one row per book update, with causal microstructure
features and forward returns at several horizons.

Why this exists
---------------
Every experiment downstream would otherwise re-parse 24 GB of JSONL and spend
~40 s per game in DuckDB. This does that once and writes a compact float32
parquet per game, so a backtest or a model fit becomes seconds instead of
hours. Iteration speed is the whole point; a research loop you cannot run
twenty times is a research loop that will overfit on the first answer.

Two rules the feature code obeys
--------------------------------
1. EVERY feature at row i uses only rows <= i. Rolling windows look backwards
   only. This is enforced structurally (cumulative sums indexed by
   searchsorted on t - W), not by convention, because look-ahead in a
   microstructure panel is invisible in every diagnostic except the PnL.

2. Targets are EXECUTABLE, not paper. `fwd_mid` is recorded because it is the
   standard target and makes results comparable to the literature, but the
   columns that decide whether anything here is tradeable are:

       fwd_long_h  = bid(t+h) - ask(t)     buy now as taker, sell later
       fwd_short_h = bid(t) - ask(t+h)     sell now as taker, buy later

   Both are negative by construction at h=0 by the width of the spread. A
   signal that predicts fwd_mid but not these predicts nothing you can trade:
   on this venue the round trip costs ~2.7c of spread against an average 1c
   swing of ~1.5c, so mid-based accuracy is close to meaningless on its own.

Sizes are shares. Prices and returns are stored in CENTS to keep float32
precision meaningful (a 0.001 tick is 0.1c; float32 has ~7 digits, so cents
are exact enough and dollars are not).

    python build_panel.py --games mlb-tor-kc-2026-09-04
    python build_panel.py --slate 2026-09-04 --limit 5
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import rebuild_duckdb as rd
import rebuild_multi as rm

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "panel")

# Horizons in seconds. 10ms/50ms are omitted deliberately: the exchange clock
# is millisecond-resolution and the median gap between updates on a liquid
# token is tens of ms, so a 10ms horizon would mostly resolve to "the same
# row" and manufacture a fake near-perfect target.
HORIZONS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
LIQUID = ("moneyline", "spread", "total")
NEAR = (1.0, 2.0, 5.0)      # cents from the touch for depth buckets
ROLL = (1.0, 5.0)           # rolling window lengths, seconds
SLIP_SHARES = 500.0         # size used for the book-slope feature


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def roll_sum(vals, times, window):
    """Sum of vals over (t-window, t], causal, vectorized.

    vals[i] is attributed to times[i]. Uses a cumulative sum indexed by
    searchsorted, so no row can see a later row -- the property that matters
    most in this file.
    """
    c = np.concatenate(([0.0], np.cumsum(vals)))
    lo = np.searchsorted(times, times - window, side="left")
    return c[np.arange(times.size) + 1] - c[lo]


TOPK = 24        # levels examined per side; depth within 5c never needs more


def top_levels(off, px, sz, idx):
    """Dense (rows x TOPK) view of the top of one side, NaN-padded.

    The ragged per-row slices are the reason the first version of this file
    took 400 s on a busy game: a Python loop over millions of rows, three
    depth buckets and two sides is tens of millions of interpreter steps.
    Levels are sorted best-first, so everything these features need lives in
    the first TOPK entries, and gathering those into a dense array makes the
    whole computation vectorized.

    Truncation is deliberate and safe for its purpose: books here average ~23
    levels a side, and a level beyond the 24th is further than 5c from the
    touch in every market this panel keeps. It would NOT be safe for anything
    that needs total book size, which is why nothing here computes that.
    """
    starts = off[idx]
    ends = off[idx + 1]
    cols = np.arange(TOPK)
    pos = starts[:, None] + cols[None, :]
    valid = pos < ends[:, None]
    if px.size == 0:
        return (np.full((idx.size, TOPK), np.nan), np.zeros((idx.size, TOPK)))
    # Clip as well as mask: a row with zero levels at the very end of the flat
    # array has start == end == len(px), so the masked-out index is itself out
    # of bounds and the gather raises before `valid` can discard it.
    pos = np.clip(np.where(valid, pos, starts[:, None]), 0, px.size - 1)
    P = np.where(valid, px[pos], np.nan)
    S = np.where(valid, sz[pos], 0.0)
    return P, S


def depth_within(P, S, best, cents, side):
    """Resting size within `cents` of the best price, vectorized."""
    lim = best - cents / 100.0 if side == "bid" else best + cents / 100.0
    m = (P >= lim[:, None]) if side == "bid" else (P <= lim[:, None])
    return np.where(np.isfinite(P) & m, S, 0.0).sum(axis=1)


def slope_cents(P, S, best, shares, side):
    """Cents of concession to consume `shares`, vectorized.

    NaN where the visible top-of-book cannot fill that size -- that is real
    information (a thin book) and must not be zero-filled into "no slippage".
    """
    S = np.where(np.isfinite(P), S, 0.0)
    cum = np.cumsum(S, axis=1)
    before = cum - S
    take = np.clip(shares - before, 0.0, S)
    filled = take.sum(axis=1)
    notion = np.nansum(np.where(take > 0, P * take, 0.0), axis=1)
    ok = filled >= shares - 1e-9
    vwap = np.divide(notion, np.maximum(filled, 1e-9))
    out = np.where(ok, (vwap - best) * 100.0 if side == "ask"
                   else (best - vwap) * 100.0, np.nan)
    return out


def touch_arrays(off, px, sz, idx):
    a, b = off[idx], off[idx + 1]
    has = b > a
    p = np.full(idx.size, np.nan)
    s = np.zeros(idx.size)
    p[has] = px[a[has]]
    s[has] = sz[a[has]]
    n = (b - a).astype(np.float64)
    return p, s, n


def load_trades(slug):
    """Millisecond tape if captured live, else the second-resolution backfill."""
    live = os.path.join(BASE, "data", "trades_live", f"{slug}.jsonl")
    back = os.path.join(BASE, "data", "trades", f"{slug}.jsonl")
    by = {}
    if os.path.exists(live):
        src, path, ms = "live-ms", live, True
    elif os.path.exists(back):
        src, path, ms = "backfill-s", back, False
    else:
        return {}, "none"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            aid = t["asset_id"] if ms else t["asset"]
            ts = float(t["ts"]) / 1000.0 if ms else float(t["timestamp"])
            side = (t.get("side") or "").upper()
            by.setdefault(aid, []).append((ts, float(t["size"]), 1.0 if side == "BUY" else -1.0))
    out = {}
    for aid, rows in by.items():
        rows.sort()
        out[aid] = (np.array([r[0] for r in rows]),
                    np.array([r[1] for r in rows]),
                    np.array([r[2] for r in rows]))
    return out, src


def build_game(slug, out_dir, window_mode="game"):
    windows = rm.load_game_windows()
    w = windows.get(slug, {})
    trades, tsrc = load_trades(slug)
    cols = rd.load_columns(slug)
    asset_ids, ts, bpx, bsz, boff, apx, asz, aoff, meta = cols
    t_lo = w.get("kickoff_ts", -np.inf) if window_mode == "game" else -np.inf
    t_hi = w.get("game_end_ts", np.inf) if window_mode == "game" else np.inf

    frames = []
    for aid, lo, hi in rd.asset_boundaries(asset_ids):
        m = meta.get(aid) or {}
        if (m.get("market_type") or "?") not in LIQUID:
            continue
        t = rd.dedup_nudge(ts[lo:hi])
        keep = (t >= t_lo) & (t <= t_hi)
        if keep.sum() < 500:
            continue
        idx = np.arange(lo, hi)[keep]
        tt = t[keep]
        n = tt.size

        Pb, Sb = top_levels(boff, bpx, bsz, idx)
        Pa, Sa = top_levels(aoff, apx, asz, idx)
        bb, bbs, nb = touch_arrays(boff, bpx, bsz, idx)
        ba, bas, na = touch_arrays(aoff, apx, asz, idx)
        two = np.isfinite(bb) & np.isfinite(ba)
        mid = np.where(two, (bb + ba) / 2.0, np.nan)
        spread = np.where(two, (ba - bb) * 100.0, np.nan)

        d = {"t": tt, "mid_c": mid * 100.0, "spread_c": spread,
             "bb_c": bb * 100.0, "ba_c": ba * 100.0,
             "bb_sz": bbs, "ba_sz": bas, "n_bid": nb, "n_ask": na}

        # --- touch imbalance and microprice --------------------------------
        tot = bbs + bas
        d["imb_touch"] = np.where(tot > 0, (bbs - bas) / np.maximum(tot, 1e-9), 0.0)
        micro = np.where(tot > 0, (bb * bas + ba * bbs) / np.maximum(tot, 1e-9), mid)
        d["micro_c"] = micro * 100.0
        d["micro_less_mid_c"] = (micro - mid) * 100.0

        # --- depth at distance, and imbalance there ------------------------
        for c in NEAR:
            db = depth_within(Pb, Sb, bb, c, "bid")
            da = depth_within(Pa, Sa, ba, c, "ask")
            d[f"dep_bid_{c:g}c"] = db
            d[f"dep_ask_{c:g}c"] = da
            s = db + da
            d[f"imb_{c:g}c"] = np.where(s > 0, (db - da) / np.maximum(s, 1e-9), 0.0)

        d["slope_bid_c"] = slope_cents(Pb, Sb, bb, SLIP_SHARES, "bid")
        d["slope_ask_c"] = slope_cents(Pa, Sa, ba, SLIP_SHARES, "ask")

        # --- liquidity flow at the touch (adds and removals) ---------------
        for nm, p_, s_ in (("bid", bb, bbs), ("ask", ba, bas)):
            same = np.zeros(n, dtype=bool)
            same[1:] = (p_[1:] == p_[:-1]) & np.isfinite(p_[1:])
            dd = np.zeros(n)
            dd[1:] = s_[1:] - s_[:-1]
            wiped = np.zeros(n)
            moved = np.zeros(n, dtype=bool)
            moved[1:] = ~same[1:] & np.isfinite(p_[:-1])
            wiped[1:] = np.where(moved[1:], s_[:-1], 0.0)
            rem = np.where(same & (dd < 0), -dd, 0.0) + wiped
            add = np.where(same & (dd > 0), dd, 0.0)
            for W in ROLL:
                d[f"rem_{nm}_{W:g}s"] = roll_sum(rem, tt, W)
                d[f"add_{nm}_{W:g}s"] = roll_sum(add, tt, W)

        # --- trade flow -----------------------------------------------------
        tr = trades.get(aid)
        for W in ROLL:
            if tr is None:
                d[f"tr_buy_{W:g}s"] = np.zeros(n)
                d[f"tr_sell_{W:g}s"] = np.zeros(n)
                d[f"tr_n_{W:g}s"] = np.zeros(n)
            else:
                ttime, tsz, tsign = tr
                cb = np.concatenate(([0.0], np.cumsum(np.where(tsign > 0, tsz, 0.0))))
                cs = np.concatenate(([0.0], np.cumsum(np.where(tsign < 0, tsz, 0.0))))
                cn = np.concatenate(([0.0], np.cumsum(np.ones(ttime.size))))
                hi_i = np.searchsorted(ttime, tt, side="right")
                lo_i = np.searchsorted(ttime, tt - W, side="right")
                d[f"tr_buy_{W:g}s"] = cb[hi_i] - cb[lo_i]
                d[f"tr_sell_{W:g}s"] = cs[hi_i] - cs[lo_i]
                d[f"tr_n_{W:g}s"] = cn[hi_i] - cn[lo_i]

        # --- momentum and realised volatility ------------------------------
        midc = mid * 100.0
        for W in (0.1, 1.0, 5.0):
            j = np.searchsorted(tt, tt - W, side="left")
            d[f"mom_{W:g}s_c"] = midc - midc[np.maximum(j - 1, 0)]
        dm = np.zeros(n)
        dm[1:] = np.abs(np.diff(midc))
        for W in ROLL:
            d[f"absmove_{W:g}s_c"] = roll_sum(dm, tt, W)
        d["dt_prev"] = np.concatenate(([np.nan], np.diff(tt)))
        d["t_since_start"] = tt - tt[0]

        # --- forward targets ------------------------------------------------
        for h in HORIZONS:
            j = np.searchsorted(tt, tt + h, side="left")
            j = np.minimum(j, n - 1)
            valid = (tt[j] - tt) >= h * 0.5      # drop rows with no real future
            tag = f"{h:g}s".replace(".", "p")
            d[f"fwd_mid_{tag}"] = np.where(valid, midc[j] - midc, np.nan)
            d[f"fwd_long_{tag}"] = np.where(valid, bb[j] * 100.0 - ba * 100.0, np.nan)
            d[f"fwd_short_{tag}"] = np.where(valid, bb * 100.0 - ba[j] * 100.0, np.nan)

        tbl = {k: np.asarray(v, dtype=np.float32) for k, v in d.items() if k != "t"}
        tbl["t"] = tt.astype(np.float64)
        tbl["market_type"] = np.full(n, m.get("market_type"), dtype=object)
        tbl["outcome"] = np.full(n, m.get("outcome"), dtype=object)
        tbl["asset_id"] = np.full(n, aid, dtype=object)
        frames.append(pa.table(tbl))

    if not frames:
        return 0, tsrc
    tab = pa.concat_tables(frames)
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(tab, os.path.join(out_dir, f"{slug}.parquet"), compression="zstd")
    return tab.num_rows, tsrc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--slate", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.games:
        slugs = args.games
    else:
        pat = f"*-{args.slate}.jsonl" if args.slate else "*.jsonl"
        slugs = sorted(os.path.basename(f)[:-6]
                       for f in glob.glob(os.path.join(rm.JSONL_SPLIT_DIR, pat)))
    if not args.force:
        slugs = [s for s in slugs
                 if not os.path.exists(os.path.join(args.out_dir, f"{s}.parquet"))]
    if args.limit:
        slugs = slugs[:args.limit]
    if not slugs:
        log("nothing to do (all built; --force to rebuild)")
        return 0

    log(f"{len(slugs)} game(s) -> {args.out_dir}")
    for slug in slugs:
        t0 = time.time()
        n, src = build_game(slug, args.out_dir)
        sz = os.path.getsize(os.path.join(args.out_dir, f"{slug}.parquet")) / 1e6 if n else 0
        log(f"  {slug}: {n:,} rows, {sz:.0f} MB, trades={src}, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
