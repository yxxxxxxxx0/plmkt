"""
Mark the jumps that break even when BOTH ends are priced at the touch you
actually cross -- never at the mid.

Justin: "price it using bid ask when i enter and exit. do not use mid price."

One thing has to be said first, because it changes what the fix is. The
existing pricing in oracle_jump_scan.py is ALREADY a bid/ask round trip, and
not approximately. The feature grid stores mid and spread, and by construction
mid = (bid + ask) / 2, so

    ask = mid + spread/2        bid = mid - spread/2

are the touches exactly. Its `gross - (spread_in + spread_out)/2` therefore
reproduces "buy at the ask, sell at the bid" to machine precision -- checked,
max absolute difference 7e-15 over all 82,120 jumps. Charging half a spread at
each end IS crossing the spread at each end.

What genuinely still runs on the mid is the CHOICE OF INSTANT, and that is
where mid pricing flatters the result:

    the detector    a move is called a jump when the MID travels 2 ticks. In a
                    ten-tick-wide book a two-tick mid move is not a two-tick
                    tradeable move; it may not be tradeable at all.
    the exit        the exit is the mid's peak. The mid peaks when the book is
                    thinnest, so the instant that looks best on the mid is
                    routinely one of the worst to actually trade against.

So this reprices everything on the executable series and lets the exit be
chosen on the executable series too:

    long   buy at ask(t), sell at bid(t+k)   net = bid(t+k) - ask(t) - fees
    short  sell at bid(t), buy at ask(t+k)   net = bid(t) - ask(t+k) - fees

with k searched over the same 10s window the detector uses, and the better of
the two directions taken. It is still an oracle -- it gets the direction right
and picks the best instant inside the window -- but every price in it is one
you could have hit.

The candidate set is deliberately the SAME durable-jump rule as
oracle_jump_scan.py, so the before/after comparison isolates the pricing change
rather than mixing it with a different detector.

Outputs, under results/makinen/oracle_jumps/
  breakeven_exec.csv            every qualifying jump
  breakeven_exec_summary.csv
  breakeven_exec_marked.png     the price graph with them marked
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

TICK = 0.01
GRID_MS = 200
J_TICKS = 2.0
WIN_S = 10.0
HOLD_S = 3.0
DEBOUNCE_S = 30.0
SPORTS_FEE_RATE = 0.05

SESSIONS = ["books_2026-08-28", "books_2026-08-30", "books_2026-09-10",
            "books_2026-09-11", "books_2026-09-12", "books_2026-09-13",
            "books_2026-09-18", "books_2026-09-19", "books_2026-09-20"]

SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
LINE = "#8d9299"
BAND = "#dcdedf"
GOOD = "#2a78d6"
BAD = "#eb6834"
GRID_C = "#e9e8e4"


def fee_ticks(price):
    p = np.clip(np.asarray(price, dtype=np.float64), 0.0, 1.0)
    return SPORTS_FEE_RATE * p * (1.0 - p) / TICK


def run_ids(sid, ts):
    brk = np.empty(len(ts), dtype=bool)
    brk[0] = True
    brk[1:] = (sid[1:] != sid[:-1]) | (np.diff(ts) != GRID_MS)
    return np.cumsum(brk)


def scan_session(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["mid", "spread_ticks", "ts", "series",
                                        "sid", "valid"])
    df = df.sort_values(["sid", "ts"]).reset_index(drop=True)
    mid = df.mid.to_numpy(np.float64)
    spr = df.spread_ticks.to_numpy(np.float64)
    ts = df.ts.to_numpy(np.int64)
    sid = df.sid.to_numpy()
    ok = df.valid.to_numpy(bool) & np.isfinite(mid) & np.isfinite(spr)
    rid = run_ids(sid, ts)
    n = len(df)

    # The touches. Exact: mid is (bid+ask)/2 and spread is ask-bid.
    ask = mid + spr * TICK / 2.0
    bid = mid - spr * TICK / 2.0

    W = int(WIN_S * 1000 / GRID_MS)
    H = int(HOLD_S * 1000 / GRID_MS)
    idx = np.arange(n)

    # --- detector: unchanged, on the mid, so the candidate set matches -----
    fmax = np.full(n, -np.inf); fmin = np.full(n, np.inf)
    amax = np.full(n, -1, np.int64); amin = np.full(n, -1, np.int64)
    # --- executable: best reachable exit price in each direction -----------
    best_bid = np.full(n, -np.inf)      # for a long: highest bid we can sell into
    best_ask = np.full(n, np.inf)       # for a short: lowest ask we can buy back at
    ibid = np.full(n, -1, np.int64); iask = np.full(n, -1, np.int64)

    for k in range(1, W + 1):
        nxt = idx + k
        good = np.zeros(n, dtype=bool)
        good[: n - k] = (rid[k:] == rid[: n - k]) & ok[k:]

        cm = np.where(good, np.concatenate([mid[k:], np.full(k, np.nan)]), np.nan)
        hi = good & (cm > fmax); fmax = np.where(hi, cm, fmax); amax = np.where(hi, nxt, amax)
        lo = good & (cm < fmin); fmin = np.where(lo, cm, fmin); amin = np.where(lo, nxt, amin)

        cb = np.where(good, np.concatenate([bid[k:], np.full(k, np.nan)]), np.nan)
        hb = good & (cb > best_bid); best_bid = np.where(hb, cb, best_bid); ibid = np.where(hb, nxt, ibid)

        ca = np.where(good, np.concatenate([ask[k:], np.full(k, np.nan)]), np.nan)
        la = good & (ca < best_ask); best_ask = np.where(la, ca, best_ask); iask = np.where(la, nxt, iask)

    up = fmax - mid
    dn = mid - fmin
    take_up = up >= dn
    peak_move = np.where(take_up, up, dn)
    peak_idx = np.where(take_up, amax, amin)

    have = ok & (peak_idx >= 0) & np.isfinite(peak_move)
    hold_idx = np.where(have, np.minimum(peak_idx + H, n - 1), 0)
    have &= (rid[hold_idx] == rid) & ok[hold_idx]
    is_jump = have & (peak_move / TICK >= J_TICKS) & \
              (np.abs(mid[hold_idx] - mid) / TICK >= J_TICKS)

    D = int(DEBOUNCE_S * 1000 / GRID_MS)
    keep, last_sid, last_pos = [], -1, -(10 ** 9)
    for t in np.flatnonzero(is_jump):
        if sid[t] != last_sid or t - last_pos >= D:
            keep.append(t); last_sid, last_pos = sid[t], t
    keep = np.array(keep, dtype=np.int64)
    if len(keep) == 0:
        return pd.DataFrame()

    # --- executable P&L, both ends at the touch ----------------------------
    e_ask, e_bid = ask[keep], bid[keep]
    x_bid, x_ask = best_bid[keep], best_ask[keep]

    long_net = (x_bid - e_ask) / TICK - fee_ticks(e_ask) - fee_ticks(x_bid)
    short_net = (e_bid - x_ask) / TICK - fee_ticks(e_bid) - fee_ticks(x_ask)
    go_long = long_net >= short_net
    net = np.where(go_long, long_net, short_net)
    exit_i = np.where(go_long, ibid[keep], iask[keep])

    ser = df["series"].to_numpy()[keep]
    parts = pd.Series(ser).str.split("|", expand=True)
    return pd.DataFrame({
        "session": path.stem.replace("feat_", "").replace("_trimmed", ""),
        "series": ser, "slug": parts[0].to_numpy(),
        "market_type": parts[1].to_numpy(),
        "ts": ts[keep], "ts_exit": ts[np.clip(exit_i, 0, n - 1)],
        "side": np.where(go_long, "long", "short"),
        "entry_mid": mid[keep], "entry_spread": spr[keep],
        "entry_px": np.where(go_long, e_ask, e_bid),
        "exit_px": np.where(go_long, x_bid, x_ask),
        "exit_spread": spr[np.clip(exit_i, 0, n - 1)],
        "hold_s": (ts[np.clip(exit_i, 0, n - 1)] - ts[keep]) / 1000.0,
        "mid_move_ticks": peak_move[keep] / TICK,
        "exec_net_ticks": net,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=SESSIONS)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    ap.add_argument("--panels", type=int, default=6)
    args = ap.parse_args()
    outdir = Path(args.outdir)

    frames = []
    for s in args.sessions:
        p = ROOT / "data" / "jump" / f"feat_{s}_trimmed.parquet"
        if not p.exists():
            print(f"  [skip] {s}: no trimmed features")
            continue
        d = scan_session(p)
        print(f"  {s}: {len(d):,} durable jumps")
        frames.append(d)
    allj = pd.concat(frames, ignore_index=True)
    win = allj[allj.exec_net_ticks > 0].copy()

    print()
    print("=== MARKED, priced at the touch on entry AND exit ===")
    print(f"  durable jumps scanned        : {len(allj):,}")
    print(f"  break even on executable px  : {len(win):,}  ({len(win)/len(allj):.2%})")
    print(f"  total                        : {win.exec_net_ticks.sum():,.0f} ticks")
    print(f"  mean / median each           : {win.exec_net_ticks.mean():.2f} / "
          f"{win.exec_net_ticks.median():.2f} ticks")
    print(f"  median entry spread          : {win.entry_spread.median():.1f} ticks")
    print(f"  median hold                  : {win.hold_s.median():.1f}s")
    print(f"  long / short                 : {(win.side=='long').sum():,} / "
          f"{(win.side=='short').sum():,}")
    print()
    print(win.groupby("market_type").agg(
        n=("exec_net_ticks", "size"), ticks=("exec_net_ticks", "sum"),
        mean=("exec_net_ticks", "mean")).round(2).to_string())

    lose = allj[allj.exec_net_ticks <= 0]
    mw, ml = win.exec_net_ticks.mean(), lose.exec_net_ticks.mean()
    need = -ml / (mw - ml)
    print()
    print(f"  payer {mw:+.2f}t, non-payer {ml:+.2f}t, base rate {len(win)/len(allj):.2%}")
    print(f"  --> a selector needs {need:.1%} precision just to break even")

    win.sort_values("exec_net_ticks", ascending=False).to_csv(
        outdir / "breakeven_exec.csv", index=False)
    pd.DataFrame([{
        "n_scanned": len(allj), "n_break_even": len(win),
        "share": len(win) / len(allj), "total_ticks": win.exec_net_ticks.sum(),
        "mean_ticks": mw, "median_ticks": win.exec_net_ticks.median(),
        "mean_non_payer": ml, "precision_needed": need,
        "median_entry_spread": win.entry_spread.median(),
        "median_hold_s": win.hold_s.median(),
    }]).to_csv(outdir / "breakeven_exec_summary.csv", index=False)

    draw(win, allj, outdir / "breakeven_exec_marked.png", args.panels)


def draw(win, allj, path, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = win.series.value_counts().head(panels).index.tolist()
    ncol = 2
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.5, 3.3 * nrow), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, ser in zip(axes, picks):
        sess = win.loc[win.series == ser, "session"].iloc[0]
        src = ROOT / "data" / "jump" / f"feat_{sess}_trimmed.parquet"
        df = pd.read_parquet(src, columns=["mid", "spread_ticks", "ts", "series", "valid"])
        df = df[(df.series == ser) & df.valid].sort_values("ts")
        t0 = df.ts.min()
        mins = (df.ts - t0) / 60000.0
        half = df.spread_ticks.to_numpy() * TICK / 2.0

        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID_C, lw=0.8)
        ax.set_axisbelow(True)

        # The tradeable band, not just the mid: everything between bid and ask
        # is cost, and it is what decides whether a jump is markable at all.
        ax.fill_between(mins, df.mid.to_numpy() - half, df.mid.to_numpy() + half,
                        color=BAND, lw=0, zorder=1, label="bid-ask spread")
        ax.plot(mins, df.mid.values, color=LINE, lw=0.9, zorder=2, label="mid")

        other = allj[(allj.series == ser) & (allj.exec_net_ticks <= 0)]
        ax.scatter((other.ts - t0) / 60000.0, other.entry_px, s=13, color=BAD,
                   alpha=0.30, linewidths=0, zorder=3,
                   label="jump that loses at the touch")
        w = win[win.series == ser]
        ax.scatter((w.ts - t0) / 60000.0, w.entry_px, s=44, color=GOOD,
                   edgecolors=SURFACE, linewidths=1.2, zorder=4,
                   label="breaks even at the touch")

        ax.set_title(f"{w.slug.iloc[0]}  {w.market_type.iloc[0]}   -   "
                     f"{len(w)} of {len(other)+len(w)} jumps pay, "
                     f"{w.exec_net_ticks.sum():.0f} ticks",
                     color=INK, fontsize=10, loc="left")
        ax.set_xlabel("minutes into the recording", color=INK2, fontsize=8.5)
        ax.set_ylabel("price", color=INK2, fontsize=8.5)

    for ax in axes[len(picks):]:
        ax.set_visible(False)

    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle("Marked at the touch: buy the ask, sell the bid, both ends",
                 color=INK, fontsize=12.5, x=0.006, y=0.995, ha="left")
    fig.text(0.006, 0.968, f"{len(win):,} of {len(allj):,} durable jumps break even "
             f"({len(win)/len(allj):.2%}); the entry dot sits on the price you would "
             f"actually pay, not the mid",
             color=INK2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.955])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
