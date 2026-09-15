"""How much capital does this actually need, once games run concurrently?

pnl_by_game.py walks the clock INDEPENDENTLY per game: one position at a time
within each match. That answers "what did this match earn" but understates the
bankroll, because an MLB slate runs ~15 games at once -- so the positions in
different matches overlap in time and each needs its own money.

This measures the real requirement:

  peak concurrency   the most positions open simultaneously across ALL
                     matches, which is the capital you must have funded
  per-session peak   the same within each recording session, since matches
                     in different sessions never overlap
  capital efficiency total P&L against peak capital, which is the number that
                     actually matters -- P&L per dollar you had to commit,
                     not per dollar per trade

Stake is per trade and capital at risk equals that stake: on a binary venue a
long can lose its entry price and a short is a long in the complementary
token, so either way the most a position loses is what was put in.

    python capital_required.py --model cnn_direction --stake 5
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
    ap.add_argument("--stake", type=float, default=5.0)
    ap.add_argument("--hold-s", type=float, default=5.0)
    ap.add_argument("--cooldown-s", type=float, default=5.0)
    ap.add_argument("--min-backing", type=float, default=200.0)
    ap.add_argument("--max-spread", type=float, default=2.0)
    ap.add_argument("--category", default="sports")
    ap.add_argument("--max-rows-per-session", type=int, default=900_000)
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    e = np.load(os.path.join(JD, f"takeredge_{a.model}.npy"))
    idx = np.load(os.path.join(JD, f"takeridx_{a.model}.npy"))
    tau = a.tau
    if tau is None:
        t = pd.read_csv(os.path.join(JD, "taker_comparison.csv"))
        r = t[t.model == a.model]
        tau = float(r.iloc[0].tau) if len(r) else 0.9

    _, _, _, lookback = frame_offsets(24, 24, 8)
    F, T, tr, va, te = load(["moneyline", "total", "spread"],
                            lookback=lookback, log=log, max_book_age_s=5.0,
                            split="group", use_trimmed=True,
                            max_rows_per_session=a.max_rows_per_session)
    F, h = prepare(F, a.hold_s, a.category, a.max_spread)

    slug = F.series.astype(str).str.split("|").str[0].to_numpy()
    signed = F.signed_ticks.to_numpy(np.float64)
    cost = F.cost_taker.to_numpy(np.float64)
    mid = F.mid.to_numpy(np.float64)
    ts = F.ts.to_numpy(np.int64)

    pnl_t, side, traded = taker_pnl(e, F, idx, tau)
    g_slug, g_ts = slug[idx], ts[idx]
    g_signed, g_cost, g_mid = signed[idx], cost[idx], mid[idx]

    hold_ms = int(a.hold_s * 1000)
    cool_ms = int(a.cooldown_s * 1000)

    # rebuild the same per-match walk-the-clock ledger, keeping timestamps
    trades = []
    for s in sorted(set(g_slug)):
        m = g_slug == s
        if m.sum() < 50:
            continue
        sd, td = side[m], traded[m]
        sg, tsm, cm, mdm = g_signed[m], g_ts[m], g_cost[m], g_mid[m]
        order = np.argsort(tsm, kind="stable")
        busy = -1
        for j in order:
            if not td[j] or tsm[j] < busy:
                continue
            ticks = sd[j] * sg[j] - cm[j]
            trades.append(dict(match=s, t_in=int(tsm[j]),
                               t_out=int(tsm[j]) + hold_ms,
                               ticks=ticks, price=mdm[j]))
            busy = tsm[j] + hold_ms + cool_ms
    if not trades:
        raise SystemExit("no trades")
    Tr = pd.DataFrame(trades)
    # shares bought with a fixed dollar stake at that contract's price
    Tr["shares"] = a.stake / Tr.price.clip(0.01, 0.99)
    Tr["usd"] = Tr.ticks * TICK * Tr.shares
    Tr["session"] = Tr.match.str.extract(r"(\d{4}-\d{2}-\d{2})$")[0]

    # ---- concurrency: sweep open/close events ---------------------------- #
    def peak_concurrency(df):
        ev = ([(t, +1) for t in df.t_in] + [(t, -1) for t in df.t_out])
        ev.sort()
        cur = peak = 0
        for _, d in ev:
            cur += d
            peak = max(peak, cur)
        return peak

    print(f"\n{'='*104}")
    print(f"CAPITAL REQUIRED   stake ${a.stake:.2f}/trade, "
          f"{a.hold_s:.0f}s hold + {a.cooldown_s:.0f}s cooldown")
    print(f"{'='*104}")
    print(f"  {len(Tr):,} trades across {Tr.match.nunique()} matches, "
          f"{Tr.session.nunique()} session(s)")
    print(f"  total P&L  {Tr.usd.sum():+,.2f} USD "
          f"({Tr.ticks.sum():+,.0f} ticks)")
    print(f"  mean shares per trade {Tr.shares.mean():.1f} "
          f"(at a mean price of ${Tr.price.mean():.3f})")

    print(f"\n  peak simultaneous positions:")
    print(f"  {'session':>14} {'matches':>8} {'trades':>8} {'peak':>6} "
          f"{'capital':>10} {'P&L':>12} {'return':>9}")
    rows = []
    for sess, g in Tr.groupby("session"):
        pk = peak_concurrency(g)
        cap = pk * a.stake
        rows.append(dict(session=sess, matches=g.match.nunique(),
                         trades=len(g), peak=pk, capital=cap,
                         pnl=g.usd.sum(),
                         ret=(g.usd.sum() / cap if cap else np.nan)))
        print(f"  {sess:>14} {g.match.nunique():>8} {len(g):>8,} {pk:>6} "
              f"${cap:>9,.0f} {g.usd.sum():>+11,.2f} "
              f"{g.usd.sum()/cap if cap else 0:>8.1%}")
    allpk = peak_concurrency(Tr)
    allcap = allpk * a.stake
    print(f"  {'ALL':>14} {Tr.match.nunique():>8} {len(Tr):>8,} {allpk:>6} "
          f"${allcap:>9,.0f} {Tr.usd.sum():>+11,.2f} "
          f"{Tr.usd.sum()/allcap if allcap else 0:>8.1%}")

    print(f"\n  Sessions never overlap each other, so the bankroll you must "
          f"fund is the WORST SINGLE SESSION's peak, not the sum:")
    worst = max(rows, key=lambda r: r["capital"])
    print(f"    ${worst['capital']:,.0f}  ({worst['peak']} concurrent "
          f"positions on {worst['session']})")
    print(f"  Total P&L across everything: {Tr.usd.sum():+,.2f} USD")
    print(f"  Return on that bankroll:     "
          f"{Tr.usd.sum()/worst['capital']:+.1%}")

    # ---- per match, so the concentration stays visible -------------------- #
    per = Tr.groupby("match").agg(trades=("usd", "size"),
                                  pnl=("usd", "sum")).sort_values(
                                      "pnl", ascending=False)
    per["capital_if_alone"] = a.stake      # 1 position at a time per match
    per["return_vs_stake"] = per.pnl / a.stake
    print(f"\n  per match (each needs only ${a.stake:.0f} if traded alone, "
          f"since it holds one position at a time):")
    print(per.to_string(float_format=lambda v: f"{v:,.2f}"))
    print(f"\n  {(per.pnl > 0).sum()} of {len(per)} matches profitable")

    out = os.path.join(JD, f"capital_{a.model}.csv")
    Tr.to_csv(out, index=False)
    print(f"\nwrote per-trade ledger -> {out}")


if __name__ == "__main__":
    main()
