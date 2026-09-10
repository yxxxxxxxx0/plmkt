"""Walk-the-clock simulator: what does the signal earn with ONE position at a time?

Every per-opportunity figure in this study so far shares a defect. A 5s label
on a 200ms grid means one price move is the label for up to 25 consecutive
rows, and the model flags all 25 -- so a "trade frac of 0.167" counts the same
move up to 25 times. You would hold one position through it. This is the trap
reports/FINDINGS.md documents in `bet_sim`'s kelly_growth of +504%: compounding
bets one at a time when they actually overlap.

`maker_sim.py` solved the same problem for the maker side by walking the clock.
This does it for the taker side:

  * merge every series into one time line, ordered by exchange timestamp
  * when flat, take the next row whose signal clears tau
  * hold `hold_s`, then exit at the executable price and pay both taker fees
  * stay flat for `cooldown_s` after exiting, so one event cannot be re-entered
  * bankroll is finite: stake is capped by capital, and capital compounds

That converts a per-fill average into a real return series, which is the only
form in which a Sharpe or a drawdown means anything.

Costs use the verified schedule (polymarket_fees.py): entry crosses the
spread and pays feeRate * p * (1-p) per share, exit does the same. Sports
feeRate is 0.05.

What this cannot fix, and the reason to read any positive result narrowly:

  * TIMING. 35% of the 5s move lands within 200ms of the signal, 71% within
    1s. `--entry-delay-s` models acting late; at the delay this venue's own
    delivery lag implies (4.6s dispersion) there is nothing left to capture.
  * GENERALISATION. 93.5% of the earlier P&L came from 10 of 49 series and
    only 15 of 49 were profitable, with a +0.112 train-test AUC gap. A
    simulation on the same session inherits all of that, which is why
    `--session` exists: point it at a session the model never trained on.

    python simulate_taker.py --model cnn_direction --hold-s 5
    python simulate_taker.py --entry-delay-s 1.0 --hold-s 5
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from jump_model import frame_offsets
from jump_split import GRID_MS, JD, TICK, load, subsample
from polymarket_fees import taker_fee_per_share
from taker_signal import prepare, touch_backing


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
def simulate(F, order, edge, tau, hold_rows, cooldown_rows, entry_delay_rows,
             capital, max_stake_frac, category, fee_on=True, flat_stake=None):
    """One position at a time along a single merged time line.

    `order` indexes rows sorted by timestamp; `edge` is aligned to it.
    Returns a trade ledger and the equity curve.
    """
    ts = F.ts.to_numpy(np.int64)
    mid = F.mid.to_numpy(np.float64)
    sp = F.spread_ticks.to_numpy(np.float64)
    sid = F.sid.to_numpy()
    pos_in_series = F.pos.to_numpy(np.int64)
    n_series_rows = F.groupby("sid", sort=False)["pos"].transform("max").to_numpy()

    cash = float(capital)
    busy_until_ts = -1
    trades = []
    equity = [(int(ts[order[0]]), cash)]

    for r in order:
        if ts[r] < busy_until_ts:
            continue
        e = edge[r]
        side = 1.0 if e > tau else (-1.0 if e < -tau else 0.0)
        if side == 0.0:
            continue

        # entry happens `entry_delay_rows` after the signal, exit `hold_rows`
        # after entry; both must stay inside the same series on the grid
        j_in = r + entry_delay_rows
        j_out = j_in + hold_rows
        if pos_in_series[r] + entry_delay_rows + hold_rows > n_series_rows[r]:
            continue
        if sid[j_out] != sid[r]:
            continue
        # the grid can contain gaps (jump_data caps forward-fill), so require
        # exact spacing rather than assuming row arithmetic is time arithmetic
        if ts[j_out] - ts[r] != (entry_delay_rows + hold_rows) * GRID_MS:
            continue

        p_in, p_out = mid[j_in], mid[j_out]
        # cross the spread in and out
        px_in = p_in + side * sp[j_in] / 2.0 * TICK
        px_out = p_out - side * sp[j_out] / 2.0 * TICK
        if not (0.0 < px_in < 1.0) or not (0.0 < px_out < 1.0):
            continue

        # Sizing. On a binary venue there is no shorting: selling YES is
        # buying NO at (1 - p). So the cash a position ties up, and the most
        # it can lose, is the entry price of the side actually bought --
        # px_in for a long, (1 - px_in) for a short.
        #
        # Using px_in for both (the first version of this) levers shorts at
        # low prices by 1/(1-p): a short entered at 0.05 risks 0.95 per share
        # while being charged 0.05, which drove simulated cash to -$585,952
        # on $50 of capital. Bounded correctly, a trade can never lose more
        # than its stake.
        entry_px = px_in if side > 0 else (1.0 - px_in)
        if entry_px <= 0:
            continue
        stake = flat_stake if flat_stake else cash * max_stake_frac
        stake = min(stake, cash)
        if stake <= 0:
            continue
        shares = stake / entry_px
        fee = 0.0
        if fee_on:
            fee = shares * (taker_fee_per_share(px_in, category)
                            + taker_fee_per_share(px_out, category))
        pnl = shares * side * (px_out - px_in) - fee
        cash += pnl
        trades.append(dict(
            ts_signal=int(ts[r]), ts_in=int(ts[j_in]), ts_out=int(ts[j_out]),
            sid=int(sid[r]), side=side, px_in=px_in, px_out=px_out,
            shares=shares, stake=stake, fee=fee, pnl=pnl, cash=cash,
            move_ticks=(p_out - p_in) / TICK,
            edge=float(e)))
        equity.append((int(ts[j_out]), cash))
        busy_until_ts = ts[j_out] + cooldown_rows * GRID_MS
        if cash <= 0:
            log("  ruined: capital exhausted")
            break

    return pd.DataFrame(trades), pd.DataFrame(equity, columns=["ts", "cash"])


def summarise(tr, eq, capital, label):
    if len(tr) == 0:
        log(f"  {label}: no trades taken")
        return dict(label=label, n_trades=0)
    span_h = (tr.ts_out.max() - tr.ts_signal.min()) / 3.6e6
    wins = (tr.pnl > 0).mean()
    ret = eq.cash.iloc[-1] / capital - 1.0
    peak = eq.cash.cummax()
    dd = float(((eq.cash - peak) / peak).min())
    # per-trade returns for a crude Sharpe; not annualised, stated per trade
    r = tr.pnl / tr.stake.replace(0, np.nan)
    out = dict(
        label=label, n_trades=int(len(tr)), hours=round(span_h, 2),
        trades_per_hour=round(len(tr) / max(span_h, 1e-9), 2),
        hit_rate=round(float(wins), 4),
        mean_pnl=round(float(tr.pnl.mean()), 4),
        total_pnl=round(float(tr.pnl.sum()), 2),
        final_cash=round(float(eq.cash.iloc[-1]), 2),
        total_return=round(float(ret), 4),
        max_drawdown=round(dd, 4),
        mean_stake=round(float(tr.stake.mean()), 2),
        fees_paid=round(float(tr.fee.sum()), 2),
        mean_move_ticks=round(float(tr.move_ticks.abs().mean()), 2),
        sharpe_per_trade=(round(float(r.mean() / r.std()), 3)
                          if r.std() and np.isfinite(r.std()) else None),
        n_series=int(tr.sid.nunique()))
    log(f"  {label}: {out['n_trades']:,} trades over {out['hours']}h "
        f"({out['trades_per_hour']}/h), hit {out['hit_rate']:.3f}, "
        f"P&L ${out['total_pnl']:,.2f} on ${capital:,.0f} "
        f"({out['total_return']:+.1%}), maxDD {out['max_drawdown']:.1%}, "
        f"fees ${out['fees_paid']:,.2f}")
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_direction")
    ap.add_argument("--session-file", default=None,
                    help="load ONLY this feat_*.parquet (substring). Required "
                         "when the saved indices came from a single-session "
                         "score, because row indices are relative to whatever "
                         "was loaded -- mixing sessions silently misaligns them")
    ap.add_argument("--session", default=None,
                    help="further restrict to series whose slug contains this")
    ap.add_argument("--capital", type=float, default=50.0)
    ap.add_argument("--flat-stake", type=float, default=5.0,
                    help="fixed dollars per trade (default). Flat staking is "
                         "the honest way to read an edge -- compounding all-in "
                         "over 1,500 overlapping trades produces numbers that "
                         "describe the compounding, not the signal. Set 0 to "
                         "use --max-stake-frac instead")
    ap.add_argument("--max-stake-frac", type=float, default=0.10,
                    help="fraction of capital per trade, used only when "
                         "--flat-stake 0")
    ap.add_argument("--hold-s", type=float, default=5.0)
    ap.add_argument("--cooldown-s", type=float, default=5.0,
                    help="stay flat this long after exiting, so a single "
                         "move cannot be re-entered 25 times")
    ap.add_argument("--entry-delay-s", type=float, default=0.0,
                    help="act this long after the signal fires. 35%% of the "
                         "move lands within 200ms, so this is the parameter "
                         "that decides feasibility")
    ap.add_argument("--tau", type=float, default=None,
                    help="default: the value chosen on validation")
    ap.add_argument("--min-backing", type=float, default=200.0)
    ap.add_argument("--max-spread", type=float, default=2.0)
    ap.add_argument("--category", default="sports")
    ap.add_argument("--market-types", nargs="*",
                    default=["moneyline", "total", "spread"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pd.set_option("display.width", 240)

    OFF, STR, _, lookback = frame_offsets(24, 24, 8)
    F, T, tr_i, va_i, te_i = load(a.market_types, lookback=lookback, log=log,
                                  max_book_age_s=5.0,
                                  sessions=a.session_file)
    F, h = prepare(F, 5.0, a.category, a.max_spread)

    ep = os.path.join(JD, f"takeredge_{a.model}.npy")
    ip = os.path.join(JD, f"takeridx_{a.model}.npy")
    if not (os.path.exists(ep) and os.path.exists(ip)):
        raise SystemExit(f"no saved signal for {a.model}; run taker_signal.py")
    e_saved = np.load(ep)
    idx_saved = np.load(ip)
    if idx_saved.max() >= len(F):
        raise SystemExit(
            f"saved index reaches row {idx_saved.max():,} but only "
            f"{len(F):,} rows are loaded -- the indices came from a different "
            f"set of sessions. Pass --session-file to match how they were "
            f"scored.")

    tau = a.tau
    if tau is None:
        t = pd.read_csv(os.path.join(JD, "taker_comparison.csv"))
        r = t[t.model == a.model]
        tau = float(r.iloc[0].tau) if len(r) else 0.90
        if not len(r):
            log(f"  '{a.model}' not in taker_comparison.csv; tau defaults to "
                f"{tau:.2f} -- pass --tau to be explicit")
    log(f"\nsignal '{a.model}': {len(idx_saved):,} scored rows, tau {tau:.2f}")

    # scatter the saved edges into a full-length array; rows without a score
    # are simply never traded
    edge = np.full(len(F), np.nan)
    edge[idx_saved] = e_saved
    scored = ~np.isnan(edge)

    if a.session:
        keep = F.series.astype(str).str.contains(a.session, regex=False).to_numpy()
        scored &= keep
        log(f"  session filter '{a.session}': {scored.sum():,} scored rows")

    bk = touch_backing(T)
    scored &= (bk >= a.min_backing) & F.tight.to_numpy()
    log(f"  after backing >= ${a.min_backing:,.0f} and tight books: "
        f"{scored.sum():,} candidate rows")
    if scored.sum() == 0:
        raise SystemExit("no candidate rows -- check --session / filters")

    order = np.where(scored)[0]
    order = order[np.argsort(F.ts.to_numpy()[order], kind="stable")]
    hold = int(round(a.hold_s * 1000 / GRID_MS))
    cool = int(round(a.cooldown_s * 1000 / GRID_MS))
    log(f"  merged time line: {len(order):,} rows, "
        f"{pd.to_datetime(F.ts.to_numpy()[order[0]], unit='ms')} -> "
        f"{pd.to_datetime(F.ts.to_numpy()[order[-1]], unit='ms')}")

    rows = []
    log(f"\n{'='*112}\nWALK-THE-CLOCK, one position at a time\n{'='*112}")
    for delay_s in sorted({0.0, a.entry_delay_s, 0.2, 1.0, 2.0}):
        d = int(round(delay_s * 1000 / GRID_MS))
        trd, eq = simulate(F, order, edge, tau, hold, cool, d, a.capital,
                           a.max_stake_frac, a.category,
                           flat_stake=(a.flat_stake or None))
        r = summarise(trd, eq, a.capital, f"entry delay {delay_s:.1f}s")
        r["entry_delay_s"] = delay_s
        rows.append(r)
        if abs(delay_s - a.entry_delay_s) < 1e-9 and len(trd):
            trd.to_csv(os.path.join(JD, f"sim_trades_{a.model}.csv"),
                       index=False)
            eq.to_csv(os.path.join(JD, f"sim_equity_{a.model}.csv"),
                      index=False)

    R = pd.DataFrame(rows)
    print(f"\n{'='*112}")
    print("SUMMARY -- entry delay is the feasibility axis")
    print(f"{'='*112}")
    cols = ["entry_delay_s", "n_trades", "trades_per_hour", "hit_rate",
            "mean_pnl", "total_pnl", "total_return", "max_drawdown",
            "sharpe_per_trade", "fees_paid", "mean_move_ticks", "n_series"]
    print(R[[c for c in cols if c in R.columns]].to_string(index=False))

    # per-series concentration on the realised ledger
    sp = os.path.join(JD, f"sim_trades_{a.model}.csv")
    if os.path.exists(sp):
        t = pd.read_csv(sp)
        if len(t):
            agg = t.groupby("sid").pnl.agg(["size", "sum"]).sort_values(
                "sum", ascending=False)
            tot = agg["sum"].sum()
            print(f"\nper-series concentration of realised P&L "
                  f"({len(agg)} series, total ${tot:,.2f}):")
            for k in (1, 3, 5, 10):
                if len(agg) >= k:
                    print(f"  top {k:>2} = {agg['sum'].head(k).sum()/tot:>7.1%}"
                          if tot else "")
            print(f"  {(agg['sum'] > 0).sum()} of {len(agg)} series profitable")

    with open(os.path.join(JD, f"sim_summary_{a.model}.json"), "w") as fh:
        json.dump(dict(model=a.model, tau=tau, config=vars(a),
                       results=rows, ts=time.strftime("%Y-%m-%dT%H:%M:%S")),
                  fh, indent=2, default=str)
    print(f"\nledger -> data/jump/sim_trades_{a.model}.csv, "
          f"summary -> data/jump/sim_summary_{a.model}.json")


if __name__ == "__main__":
    main()
