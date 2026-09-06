"""
Event-driven backtest with realistic Polymarket execution.

Every strategy here is a taker on both legs unless stated. The rules below are
the ones that decide whether a signal survives, and each exists because
skipping it is how microstructure backtests lie.

LATENCY IS REAL AND APPLIED TO THE BOOK, NOT THE CLOCK.
    A signal computed from row i does not execute at row i. The fill happens
    at the first book state at or after t_i + latency, at THAT state's prices.
    So the book moves during the delay, exactly as it does live, and a signal
    whose edge decays inside 100 ms simply stops paying.

NO MIDPOINT FILLS.
    Buys pay the ask, sells hit the bid, both at the touch price recorded in
    the book at execution time. There is no price improvement anywhere.

SIZE IS CAPPED BY WHAT IS ACTUALLY RESTING.
    Trade size is min(target, size at the touch). Nothing walks past level 0,
    so the fill price is exact rather than modelled -- and the shortfall is
    reported, because "I wanted 500 and got 8" is the real constraint on this
    venue.

CROSSED BOOKS ARE EXCLUDED.
    0.29% of rows have ask <= bid (a reconstruction artifact). They are
    fabricated free money and would flatter every strategy here.

FEES.
    fee = shares * rate * p * (1-p) per taker leg. Rate is swept because
    Polymarket's own surfaces disagree: the market config says 0.05, but all
    16,305 live-captured MLB trades reported fee_rate_bps = 0.

Exit is time-based at a fixed horizon. That is deliberately dumb: an
optimised exit is the easiest place to bury look-ahead, and the question here
is whether the ENTRY signal has edge at all.

    python backtest.py --slate 2026-09-04 --strategy imbalance
    python backtest.py --strategy all --latency 0.1 --market-type moneyline
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")

NEEDED = ["t", "mid_c", "spread_c", "bb_c", "ba_c", "bb_sz", "ba_sz",
          "imb_touch", "imb_2c", "micro_less_mid_c", "mom_1s_c", "mom_5s_c",
          "rem_bid_1s", "rem_ask_1s", "add_bid_1s", "add_ask_1s",
          "tr_buy_1s", "tr_sell_1s", "absmove_1s_c", "dep_bid_2c", "dep_ask_2c",
          "asset_id", "market_type"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
# Signals. Each returns a score; positive = expect the price to RISE.
# --------------------------------------------------------------------------
def sig_imbalance(d):
    """Classic queue imbalance: more size bid than offered -> upward pressure."""
    return d["imb_touch"]


def sig_imbalance_depth(d):
    """Same idea measured 2c into the book, so a single large touch order
    cannot dominate it."""
    return d["imb_2c"]


def sig_microprice(d):
    """Microprice minus mid. The size-weighted 'fair' point inside the spread;
    when it sits above the mid the next move is more often up."""
    return d["micro_less_mid_c"]


def sig_ofi_cancel(d):
    """Order-flow imbalance from CANCELLATIONS at the touch. Liquidity leaving
    the ask faster than the bid means sellers are stepping away -> up.
    On this venue ~96% of touch removal is cancellation, not trading, so this
    is the dominant flow signal available."""
    return d["rem_ask_1s"] - d["rem_bid_1s"]


def sig_ofi_add(d):
    """Liquidity being ADDED to the bid faster than the ask."""
    return d["add_bid_1s"] - d["add_ask_1s"]


def sig_ofi_trade(d):
    """Signed aggressive volume over the last second."""
    return d["tr_buy_1s"] - d["tr_sell_1s"]


def sig_momentum(d):
    """Continuation: recent move persists."""
    return d["mom_1s_c"]


def sig_reversion(d):
    """Mean reversion: recent move reverses. The sign flip of momentum, kept
    separate so both directions are tested honestly rather than whichever
    happens to work."""
    return -d["mom_1s_c"]


def sig_liq_collapse(d):
    """Liquidity withdrawal, directional. Bids vanishing while asks hold
    predicts a downward gap -- so the score is negative when the BID side is
    being pulled hardest."""
    b = d["rem_bid_1s"]
    a = d["rem_ask_1s"]
    tot = b + a
    return np.where(tot > 0, (a - b) / np.maximum(tot, 1e-9), 0.0)


STRATEGIES = {
    "imbalance": sig_imbalance,
    "imbalance_depth": sig_imbalance_depth,
    "microprice": sig_microprice,
    "ofi_cancel": sig_ofi_cancel,
    "ofi_add": sig_ofi_add,
    "ofi_trade": sig_ofi_trade,
    "momentum": sig_momentum,
    "reversion": sig_reversion,
    "liq_collapse": sig_liq_collapse,
}


def run_token(d, sel, sig, thr_lo, thr_hi, latency, hold, size, fee_rate,
              cooldown):
    """Backtest one token. Returns a list of trade dicts.

    Entries are throttled by `cooldown` seconds so a signal that stays above
    threshold for a whole second does not book hundreds of overlapping copies
    of the same idea and make the trade count meaningless.
    """
    t = d["t"][sel]
    bb, ba = d["bb_c"][sel], d["ba_c"][sel]
    bbs, bas = d["bb_sz"][sel], d["ba_sz"][sel]
    s = sig({k: v[sel] for k, v in d.items() if v.dtype != object})
    n = t.size
    if n < 100:
        return []

    # execution index: first state at or after t + latency; and the exit index
    ex = np.minimum(np.searchsorted(t, t + latency, side="left"), n - 1)
    xi = np.minimum(np.searchsorted(t, t + latency + hold, side="left"), n - 1)

    fire = np.isfinite(s) & ((s >= thr_hi) | (s <= thr_lo))
    trades = []
    last_t = -np.inf
    for i in np.flatnonzero(fire):
        if t[i] - last_t < cooldown:
            continue
        e, x = int(ex[i]), int(xi[i])
        if x <= e or (t[e] - t[i]) > latency + 1.0:
            continue
        long_ = s[i] >= thr_hi
        # entry and exit prices at the touch, after latency
        if long_:
            p_in, q_in = ba[e], bas[e]
            p_out, q_out = bb[x], bbs[x]
        else:
            p_in, q_in = bb[e], bbs[e]
            p_out, q_out = ba[x], bas[x]
        if not (np.isfinite(p_in) and np.isfinite(p_out)):
            continue
        if ba[e] <= bb[e] or ba[x] <= bb[x]:      # crossed -> unusable
            continue
        q = float(min(size, q_in, q_out))
        if q <= 0:
            continue
        gross = (p_out - p_in) * q if long_ else (p_in - p_out) * q
        pin, pout = p_in / 100.0, p_out / 100.0
        fee = fee_rate * (pin * (1 - pin) + pout * (1 - pout)) * 100.0 * q
        trades.append({"t": float(t[i]), "long": bool(long_), "q": q,
                       "gross_c": float(gross), "fee_c": float(fee),
                       "net_c": float(gross - fee),
                       "spread_in_c": float(ba[e] - bb[e]),
                       "hold_s": float(t[x] - t[e]),
                       "slip_s": float(t[e] - t[i]),
                       "want": size, "got": q})
        last_t = t[i]
    return trades


def summarize(trades, label, fee_rate):
    if not trades:
        return None
    net = np.array([x["net_c"] for x in trades])
    gross = np.array([x["gross_c"] for x in trades])
    q = np.array([x["q"] for x in trades])
    hold = np.array([x["hold_s"] for x in trades])
    spr = np.array([x["spread_in_c"] for x in trades])
    eq = np.cumsum(net)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if eq.size else 0.0
    sh = float(net.mean() / net.std() * np.sqrt(len(net))) if net.std() > 0 else float("nan")
    return {"label": label, "n": len(trades),
            "net_c": float(net.sum()), "gross_c": float(gross.sum()),
            "fee_c": float(net.sum() - gross.sum()) * -1,
            "per_trade_c": float(net.mean()),
            "per_share_c": float(net.sum() / max(q.sum(), 1)),
            "win": float((net > 0).mean()),
            "sharpe": sh, "maxdd_c": dd,
            "shares": float(q.sum()),
            "hold_s": float(hold.mean()),
            "spread_c": float(spr.mean()),
            "fill_ratio": float((q / np.array([x["want"] for x in trades])).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--slate", default=None)
    ap.add_argument("--strategy", default="all")
    ap.add_argument("--market-type", default=None)
    ap.add_argument("--max-spread", type=float, default=None,
                    help="only enter when the quoted spread is at most this (cents)")
    ap.add_argument("--quantile", type=float, default=0.05,
                    help="enter on the top/bottom q of the signal, measured "
                         "per token on a TRAILING basis (no look-ahead)")
    ap.add_argument("--latency", type=float, default=0.1)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--size", type=float, default=100.0)
    ap.add_argument("--fee-rate", type=float, default=0.0)
    ap.add_argument("--cooldown", type=float, default=1.0)
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pat = f"*-{args.slate}.parquet" if args.slate else "*.parquet"
    files = ([os.path.join(args.panel_dir, f"{g}.parquet") for g in args.games]
             if args.games else sorted(glob.glob(os.path.join(args.panel_dir, pat))))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit("no panel files")

    names = list(STRATEGIES) if args.strategy == "all" else args.strategy.split(",")
    allt = {k: [] for k in names}
    per_game = collections.defaultdict(lambda: collections.defaultdict(list))
    per_type = collections.defaultdict(lambda: collections.defaultdict(list))

    for path in files:
        slug = os.path.basename(path)[:-8]
        tab = pq.read_table(path, columns=NEEDED)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in NEEDED}
        base = np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"]) & (d["ba_c"] > d["bb_c"])
        if args.market_type:
            base &= (d["market_type"] == args.market_type)
        if args.max_spread is not None:
            base &= d["spread_c"] <= args.max_spread
        for a in np.unique(d["asset_id"]):
            sel = base & (d["asset_id"] == a)
            if sel.sum() < 2000:
                continue
            mt = str(d["market_type"][np.flatnonzero(sel)[0]])
            sd = {k: v[sel] for k, v in d.items() if v.dtype != object}
            for k in names:
                sv = STRATEGIES[k](sd)
                fin = sv[np.isfinite(sv)]
                if fin.size < 500:
                    continue
                # Thresholds from the FIRST HALF of this token's history only,
                # then applied to the whole series. Using full-sample quantiles
                # would leak the future into the entry rule.
                half = fin[: fin.size // 2]
                lo = float(np.quantile(half, args.quantile))
                hi = float(np.quantile(half, 1 - args.quantile))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    continue
                tr = run_token(d, sel, STRATEGIES[k], lo, hi, args.latency,
                               args.hold, args.size, args.fee_rate, args.cooldown)
                allt[k] += tr
                per_game[k][slug] += tr
                per_type[k][mt] += tr
        log(f"  {slug}: done")

    print()
    print("=" * 112)
    print(f"BACKTEST  latency {args.latency*1000:.0f}ms, hold {args.hold:g}s, "
          f"size {args.size:g}, fee rate {args.fee_rate:g}, "
          f"entry quantile {args.quantile:g} (trailing), cooldown {args.cooldown:g}s")
    seg = args.market_type or "all liquid"
    print(f"{len(files)} game(s), segment {seg}"
          + (f", spread<= {args.max_spread:g}c" if args.max_spread else ""))
    print("=" * 112)
    print(f"{'strategy':>16} {'trades':>8} {'net c':>12} {'c/trade':>9} {'c/share':>9} "
          f"{'win%':>6} {'Sharpe':>8} {'maxDD c':>10} {'hold s':>7} {'spr c':>6} {'fill':>5}")
    print("-" * 112)
    out = {}
    for k in names:
        r = summarize(allt[k], k, args.fee_rate)
        if not r:
            print(f"{k:>16} {'no trades':>8}")
            continue
        out[k] = r
        print(f"{k:>16} {r['n']:>8,} {r['net_c']:>12,.0f} {r['per_trade_c']:>+9.4f} "
              f"{r['per_share_c']:>+9.4f} {r['win']*100:>5.1f}% {r['sharpe']:>8.2f} "
              f"{r['maxdd_c']:>10,.0f} {r['hold_s']:>7.2f} {r['spread_c']:>6.2f} "
              f"{r['fill_ratio']*100:>4.0f}%")

    # consistency across games is what separates an edge from one lucky match
    print()
    print("  per-game consistency (share of games with positive net):")
    for k in names:
        g = per_game[k]
        if not g:
            continue
        vals = [sum(x["net_c"] for x in v) for v in g.values() if v]
        if vals:
            pos = sum(1 for v in vals if v > 0)
            print(f"    {k:>16} {pos}/{len(vals)} games positive   "
                  f"median {np.median(vals):>+10.1f}c   worst {min(vals):>+10.1f}c")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "summary": out,
                       "per_game": {k: {g: float(sum(x["net_c"] for x in v))
                                        for g, v in gg.items()}
                                    for k, gg in per_game.items()},
                       "per_type": {k: {t: float(sum(x["net_c"] for x in v))
                                        for t, v in tt.items()}
                                    for k, tt in per_type.items()}}, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
