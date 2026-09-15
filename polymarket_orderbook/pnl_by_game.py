"""P&L broken down per game, for a saved directional signal.

Pooled numbers hide the thing that actually decides whether a signal is real.
The earlier diagnostics found 93.5% of taker P&L coming from 10 of 49 matches
with only 15 of 49 profitable -- a pooled figure that looked like an edge was
really a handful of games carrying everything. So every P&L here is reported
per match, with the concentration summarised explicitly.

Two views, because they answer different questions:

  per-opportunity   every eligible row is an independent decision; P&L is the
                    mean over all of them. Comparable across matches of
                    different lengths, but assumes you could take every
                    signal, which overlapping 5s holds do not allow.
  walk-the-clock    one position at a time per match, honouring hold and
                    cooldown, so overlapping signals collapse into the single
                    trade you could actually have taken. This is the honest
                    count and it is always far smaller.

Costs are the verified schedule: spread crossed on entry AND exit, plus
feeRate*p*(1-p) per taker leg (sports 0.05).

    python pnl_by_game.py --model cnn_direction
    python pnl_by_game.py --model cnn_direction --split test
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from jump_model import frame_offsets
from jump_split import JD, TICK, load
from taker_signal import prepare, taker_pnl, touch_backing


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_direction")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--min-backing", type=float, default=200.0)
    ap.add_argument("--max-spread", type=float, default=2.0)
    ap.add_argument("--category", default="sports")
    ap.add_argument("--hold-s", type=float, default=5.0)
    ap.add_argument("--cooldown-s", type=float, default=5.0)
    ap.add_argument("--stake", type=float, default=5.0)
    ap.add_argument("--use-trimmed", action="store_true", default=True)
    ap.add_argument("--max-rows-per-session", type=int, default=900_000)
    a = ap.parse_args()
    pd.set_option("display.width", 240)

    ep = os.path.join(JD, f"takeredge_{a.model}.npy")
    ip = os.path.join(JD, f"takeridx_{a.model}.npy")
    if not (os.path.exists(ep) and os.path.exists(ip)):
        raise SystemExit(f"no saved signal for {a.model}; run taker_signal.py")
    e = np.load(ep)
    idx = np.load(ip)

    tau = a.tau
    if tau is None:
        t = pd.read_csv(os.path.join(JD, "taker_comparison.csv"))
        r = t[t.model == a.model]
        tau = float(r.iloc[0].tau) if len(r) else 0.9
    log(f"signal '{a.model}': {len(idx):,} scored rows, tau {tau:.2f}")

    _, _, _, lookback = frame_offsets(24, 24, 8)
    F, T, tr, va, te = load(["moneyline", "total", "spread"],
                            lookback=lookback, log=log, max_book_age_s=5.0,
                            split="group", use_trimmed=a.use_trimmed,
                            max_rows_per_session=a.max_rows_per_session)
    F, h = prepare(F, a.hold_s, a.category, a.max_spread)
    if idx.max() >= len(F):
        raise SystemExit(
            "saved index does not match this dataset -- the signal was scored "
            "against a different load configuration. Re-run taker_signal.py.")

    slug = F.series.astype(str).str.split("|").str[0].to_numpy()
    signed = F.signed_ticks.to_numpy(np.float64)
    cost = F.cost_taker.to_numpy(np.float64)
    ts = F.ts.to_numpy(np.int64)

    pnl, side, traded = taker_pnl(e, F, idx, tau)
    g_slug = slug[idx]
    g_signed = signed[idx]
    g_ts = ts[idx]
    g_cost = cost[idx]

    hold_ms = int(a.hold_s * 1000)
    cool_ms = int(a.cooldown_s * 1000)

    rows = []
    for s in sorted(set(g_slug)):
        m = g_slug == s
        if m.sum() < 50:
            continue
        sd, td, pl = side[m], traded[m], pnl[m]
        sg, tsm, cm = g_signed[m], g_ts[m], g_cost[m]
        mv = sg != 0
        hit = (((sd > 0) == (sg > 0))[td & mv].mean()
               if (td & mv).any() else np.nan)

        # walk the clock within this match: one position at a time
        order = np.argsort(tsm, kind="stable")
        busy_until = -1
        wc_n = 0
        wc_pnl_ticks = 0.0
        for j in order:
            if not td[j] or tsm[j] < busy_until:
                continue
            wc_pnl_ticks += sd[j] * sg[j] - cm[j]
            wc_n += 1
            busy_until = tsm[j] + hold_ms + cool_ms

        rows.append(dict(
            match=s, n_rows=int(m.sum()),
            signals=int(td.sum()), signal_frac=float(td.mean()),
            hit=hit,
            pnl_per_opp=float(pl.mean()),
            wc_trades=wc_n,
            wc_pnl_ticks=wc_pnl_ticks,
            wc_pnl_usd=wc_pnl_ticks * TICK * (a.stake / 0.42),
            wc_per_trade=(wc_pnl_ticks / wc_n) if wc_n else np.nan))

    R = pd.DataFrame(rows).sort_values("wc_pnl_ticks", ascending=False)
    print(f"\n{'='*132}")
    print(f"P&L PER GAME  (walk-the-clock: one position at a time, "
          f"{a.hold_s:.0f}s hold + {a.cooldown_s:.0f}s cooldown)")
    print(f"  wc_pnl_usd assumes ~${a.stake:.0f} per trade on a ~$0.42 contract")
    print(f"{'='*132}")
    print(R.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    tot = R.wc_pnl_ticks.sum()
    prof = (R.wc_pnl_ticks > 0).sum()
    print(f"\n  {prof} of {len(R)} games profitable "
          f"({prof/max(len(R),1):.0%})")
    print(f"  total walk-the-clock P&L: {tot:,.1f} ticks "
          f"({R.wc_trades.sum():,} trades)")
    if tot != 0:
        top = R.wc_pnl_ticks.head(3).sum()
        print(f"  top 3 games = {top/tot:>6.1%} of total P&L")
    print(f"  per-opportunity mean across games: "
          f"{R.pnl_per_opp.mean():+.4f} ticks")
    print(f"\n  If a small number of games carry the P&L, the pooled figure is "
          f"not an edge -- it is those games.")

    out = os.path.join(JD, f"pnl_by_game_{a.model}.csv")
    R.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
