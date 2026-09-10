"""Sequential bankroll simulation for the filtered market-making strategy.

Why this is separate from adverse_selection.py
----------------------------------------------
That module reports P&L *per fill*. A Sharpe ratio is a property of a return
series, not of a trade population, and the two are not interchangeable: 124k
fills spread over 250 markets cannot all be taken with $50. Turning per-fill
edge into a return requires the binding constraint -- how much capital, how
many positions at once, and how long each one ties the money up.

So this walks the clock. Events from every market are merged into one time
line; whenever the strategy is flat it takes the next qualifying fill, holds
for the exit horizon, marks out at the mid, and frees the capital. That
produces a genuine P&L path, from which daily returns and a Sharpe follow.

Three honesty knobs, all off by default in the optimistic run and reported
side by side:

  cancel_screen   drop events where the touch snapped back within 5s. Those
                  are reprice cycles, not fills -- 45% of one-tick events,
                  and they carry a fake edge (negative adverse selection).
  min_sweep_mult  require the flow to exceed your own size by this multiple
                  before it fills you. This is the joining perturbation: you
                  add $50 to a $9 level, so only flow bigger than $59 reaches
                  you, and bigger flow is more informed.
  exit_slip       cross a fraction of the spread to get out, instead of
                  assuming you exit at the mid for free.

Usage:
    python maker_sim.py
    python maker_sim.py --capital 50 --concurrent 1 --horizon 30
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ADV = os.path.join(BASE, "data", "adverse")


# --------------------------------------------------------------------------- #
def build_events(files, horizon=30, max_spread=0.011, revert_s=5,
                 imb_min=0.45, max_gap=0.011, queue_max_usd=50.0):
    """Extract candidate fills with timestamps, filters and exit marks."""
    out = []
    for f in files:
        df = pd.read_parquet(f)
        if "is_yes" in df.columns:
            df = df[df.is_yes]
        mf = f.replace(".parquet", "_mtypes.parquet")
        mmap = (dict(zip(*pd.read_parquet(mf)[["idx", "market_type"]].values.T))
                if os.path.exists(mf) else {})

        for asset, g in df.groupby("asset", sort=False):
            g = g.sort_values("ts")
            if len(g) < 20:
                continue
            ts = g.ts.to_numpy()
            bid = g.bid.to_numpy(np.float64); ask = g.ask.to_numpy(np.float64)
            bsz = g.bid_sz.to_numpy(np.float64); asz = g.ask_sz.to_numpy(np.float64)
            b3 = g.bid_d3.to_numpy(np.float64); a3 = g.ask_d3.to_numpy(np.float64)
            mid = (bid + ask) / 2.0
            mtype = mmap.get(int(g.mtype.iloc[0]), "?")

            spread_pre = np.full(len(g), np.inf)
            spread_pre[1:] = (ask - bid)[:-1]
            fut = np.clip(np.searchsorted(ts, ts + horizon * 1000, "left"),
                          0, len(ts) - 1)
            rev = np.clip(np.searchsorted(ts, ts + revert_s * 1000, "left"),
                          0, len(ts) - 1)
            enough = ts[fut] >= ts + horizon * 500

            for side in ("bid", "ask"):
                if side == "bid":
                    m = np.zeros(len(g), bool)
                    m[1:] = bid[1:] < bid[:-1] - 1e-9
                    own_sz, own3, far3 = bsz, b3, a3
                    px_arr, far_px_arr = bid, ask
                else:
                    m = np.zeros(len(g), bool)
                    m[1:] = ask[1:] > ask[:-1] + 1e-9
                    own_sz, own3, far3 = asz, a3, b3
                    px_arr, far_px_arr = ask, bid

                idx = np.where(m & enough & (spread_pre <= max_spread))[0]
                idx = idx[idx > 0]
                if idx.size == 0:
                    continue
                p = idx - 1
                fill_px = px_arr[p]
                gap = np.abs(px_arr[idx] - px_arr[p])
                queue_usd = own_sz[p] * fill_px
                imb = own3[p] / np.maximum(own3[p] + far3[p], 1e-9)
                mid_exit = mid[fut][idx]
                spread_exit = (ask - bid)[fut][idx]

                # did the touch snap back to its old price quickly?
                came_back = np.zeros(len(idx), bool)
                for j, i in enumerate(idx):
                    hi = rev[i]
                    if hi > i:
                        seg = px_arr[i:hi + 1]
                        came_back[j] = (seg.max() >= fill_px[j] - 1e-9
                                        if side == "bid"
                                        else seg.min() <= fill_px[j] + 1e-9)

                out.append(pd.DataFrame(dict(
                    ts=ts[idx], exit_ts=ts[fut][idx], asset=asset,
                    market_type=mtype, side=side, fill_px=fill_px,
                    mid_pre=mid[p], mid_exit=mid_exit, spread_exit=spread_exit,
                    gap=gap, queue_usd=queue_usd, imbalance=imb,
                    reverted=came_back)))
    ev = pd.concat(out, ignore_index=True).sort_values("ts").reset_index(drop=True)
    ev["qualifies"] = ((ev.queue_usd <= queue_max_usd) &
                       (ev.imbalance >= imb_min) & (ev.gap <= max_gap))
    return ev


# --------------------------------------------------------------------------- #
def simulate(ev: pd.DataFrame, capital=50.0, concurrent=1, cancel_screen=False,
             min_sweep_mult=0.0, exit_slip=0.0, market_type=None,
             px_lo=0.10, px_hi=0.90, depth_cap=True):
    """Walk the clock, holding at most `concurrent` positions at a time.

    Two constraints that a naive version gets badly wrong:

    `px_lo`/`px_hi` -- equal-notional sizing into very cheap contracts is a
    leverage artifact. $50 into a $0.02 contract is 2,500 shares, so a
    0.003 per-share move reads as a 15% return. Those markets cannot absorb
    $50 anyway. Restricting to mid-range prices keeps the simulation honest.

    `depth_cap` -- you cannot be filled for more than was resting at the
    touch. Without this the sim happily trades $50 into a level holding $9.
    """
    e = ev[ev.qualifies].copy()
    if market_type:
        e = e[e.market_type == market_type]
    if cancel_screen:
        e = e[~e.reverted]
    e = e[(e.fill_px >= px_lo) & (e.fill_px <= px_hi)]
    per_slot = capital / max(concurrent, 1)
    if min_sweep_mult > 0:
        # the flow must clear your own size on top of the queue ahead of you
        e = e[e.queue_usd >= min_sweep_mult * per_slot]
    if e.empty:
        return None

    free_at = np.zeros(concurrent, dtype=np.int64)
    rows = []
    for r in e.itertuples(index=False):
        slot = int(np.argmin(free_at))
        if free_at[slot] > r.ts:
            continue                       # all slots busy
        stake = min(per_slot, r.queue_usd) if depth_cap else per_slot
        if stake < 1.0:                    # below any sane minimum order
            continue
        shares = stake / max(r.fill_px, 1e-6)
        exit_px = (r.mid_exit - exit_slip * r.spread_exit / 2.0
                   if r.side == "bid"
                   else r.mid_exit + exit_slip * r.spread_exit / 2.0)
        pnl = shares * ((exit_px - r.fill_px) if r.side == "bid"
                        else (r.fill_px - exit_px))
        free_at[slot] = r.exit_ts
        rows.append((r.ts, r.exit_ts, pnl, stake, r.market_type))

    if not rows:
        return None
    t = pd.DataFrame(rows, columns=["ts", "exit_ts", "pnl", "stake", "market_type"])
    t["dt"] = pd.to_datetime(t.exit_ts, unit="ms", utc=True)
    return t


def stats(t: pd.DataFrame, capital=50.0, freq="1D") -> dict:
    """Return series statistics, computed on realised P&L per period."""
    if t is None or t.empty:
        return {}
    per = t.set_index("dt").pnl.resample(freq).sum()
    # a period with no fills is a real zero-return period, keep it
    ret = per / capital
    days = max((t.dt.max() - t.dt.min()).total_seconds() / 86400.0, 1e-9)
    ppy = 365.0 / (pd.Timedelta(freq).total_seconds() / 86400.0)
    sd = ret.std(ddof=1)
    sharpe = (ret.mean() / sd * np.sqrt(ppy)) if sd and len(ret) > 1 else np.nan

    eq = capital + t.pnl.cumsum()
    peak = np.maximum.accumulate(np.maximum(eq, capital))
    dd = float(((peak - eq) / peak).max()) if len(eq) else np.nan

    per_fill_sd = t.pnl.std(ddof=1)
    return dict(
        fills=len(t), days=days, fills_per_day=len(t) / days,
        total_pnl=float(t.pnl.sum()),
        total_return=float(t.pnl.sum() / capital),
        mean_stake=float(t.stake.mean()),
        pnl_per_fill=float(t.pnl.mean()),
        sd_per_fill=float(per_fill_sd),
        sharpe_per_fill=float(t.pnl.mean() / per_fill_sd) if per_fill_sd else np.nan,
        periods=len(ret),
        mean_period_ret=float(ret.mean()),
        sd_period_ret=float(sd) if sd else np.nan,
        sharpe_ann=float(sharpe),
        max_drawdown=dd,
        hit_rate=float((t.pnl > 0).mean()),
    )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=50.0)
    ap.add_argument("--concurrent", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--freq", default="1D")
    a = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(ADV, "touch_*.parquet")))
             if "_assets" not in f and "_mtypes" not in f]
    if not files:
        raise SystemExit("run adverse_selection.py --pass1 first")
    print(f"building events from {len(files)} day file(s)...")
    ev = build_events(files, horizon=a.horizon)
    ev.to_parquet(os.path.join(ADV, "sim_events.parquet"), index=False)
    print(f"  {len(ev):,} candidate fills, {ev.qualifies.sum():,} pass the filters")

    pd.set_option("display.width", 240)
    fmt = lambda v: f"{v:,.4f}"

    scenarios = [
        ("optimistic (exit at mid, no screens)", dict()),
        ("+ cancel screen", dict(cancel_screen=True)),
        ("+ cancel screen, cross half spread to exit",
         dict(cancel_screen=True, exit_slip=0.5)),
        ("+ cancel screen, full spread, flow must clear 1x your size",
         dict(cancel_screen=True, exit_slip=1.0, min_sweep_mult=1.0)),
        ("+ cancel screen, full spread, flow must clear 2x your size",
         dict(cancel_screen=True, exit_slip=1.0, min_sweep_mult=2.0)),
    ]
    rows = []
    for label, kw in scenarios:
        t = simulate(ev, capital=a.capital, concurrent=a.concurrent, **kw)
        s = stats(t, capital=a.capital, freq=a.freq)
        if s:
            s["scenario"] = label
            rows.append(s)
    R = pd.DataFrame(rows)
    cols = ["scenario", "fills", "fills_per_day", "mean_stake", "pnl_per_fill",
            "hit_rate", "total_return", "mean_period_ret", "sd_period_ret",
            "sharpe_ann", "sharpe_per_fill", "max_drawdown"]
    print("\n" + "=" * 100)
    print(f"SEQUENTIAL SIMULATION -- ${a.capital:.0f} capital, "
          f"{a.concurrent} position(s) at a time, {a.horizon}s hold")
    print("=" * 100)
    print(R[cols].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 100)
    print("BY MARKET TYPE (cancel screen + full spread exit)")
    print("=" * 100)
    rows = []
    for m in ("spread", "total", "moneyline"):
        t = simulate(ev, capital=a.capital, concurrent=a.concurrent,
                     cancel_screen=True, exit_slip=1.0, market_type=m)
        s = stats(t, capital=a.capital, freq=a.freq)
        if s:
            s["market_type"] = m
            rows.append(s)
    if rows:
        R2 = pd.DataFrame(rows)
        print(R2[["market_type"] + cols[1:]].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 100)
    print("SENSITIVITY TO CONCURRENT POSITIONS (cancel screen + full spread exit)")
    print("=" * 100)
    rows = []
    for c in (1, 2, 5, 10):
        t = simulate(ev, capital=a.capital, concurrent=c,
                     cancel_screen=True, exit_slip=1.0)
        s = stats(t, capital=a.capital, freq=a.freq)
        if s:
            s["concurrent"] = c
            rows.append(s)
    if rows:
        R3 = pd.DataFrame(rows)
        print(R3[["concurrent"] + cols[1:]].to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
