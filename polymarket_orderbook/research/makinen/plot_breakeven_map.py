"""Mark every collapse on the price, split by whether it would have paid.

Three classes, and the split is the point:

  TRADEABLE   durable move >= 2 x spread at the collapse instant. The move
              covers a round trip, so with the right direction this one pays.
  TOO SMALL   a genuine durable move (>= 2 ticks) that does NOT cover the
              round trip. Real, but not worth trading.
  NO MOVE     the book emptied and the price did not durably move.

Cost is taken as 2 x the spread AT the collapse instant, which the latency
study supports: 91% of tight collapses still show a spread <= 2 ticks 200 ms
later, so the quoted spread is the one you would actually pay.

    python plot_breakeven_map.py --n-matches 6
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
JD = os.path.join(ROOT, "data", "jump")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen", "breakeven_map")
HKT_MS = 8 * 3600 * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-matches", type=int, default=6)
    ap.add_argument("--session", default=None)
    ap.add_argument("--min-move", type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    E = pd.read_parquet(os.path.join(CACHE, "collapse_events.parquet"))
    E["cost"] = 2 * E.spread_ticks * 0.01
    E["real"] = (E.durable_move >= a.min_move).astype(int)
    E["tradeable"] = ((E.durable_move >= E.cost) & (E.real == 1)).astype(int)
    E["game"] = E.series.astype(str).str.split("|").str[0]
    sess = a.session or sorted(E.session.unique())[-1]
    Es = E[E.session == sess]

    # pick contracts from DIFFERENT games with the most tradeable events
    rank = (Es.groupby(["game", "sid"]).tradeable.sum()
            .reset_index().sort_values("tradeable", ascending=False))
    picks, seen = [], set()
    for _, r in rank.iterrows():
        if r.game in seen:
            continue
        seen.add(r.game)
        picks.append((r.game, int(r.sid)))
        if len(picks) >= a.n_matches:
            break

    F = pq.read_table(os.path.join(JD, "feat_%s_trimmed.parquet" % sess),
                      columns=["ts", "sid", "series", "mid", "valid"]).to_pandas()
    F = F[F.valid.to_numpy(bool)]

    ncol = 2
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    tot = dict(tradeable=0, small=0, none=0)

    for ax, (game, sid) in zip(axes, picks):
        g = F[F.sid == sid].sort_values("ts")
        if len(g) < 50:
            ax.set_visible(False)
            continue
        t = pd.to_datetime(g.ts.to_numpy() + HKT_MS, unit="ms")
        ax.plot(t, g.mid, lw=0.9, color="#333333", zorder=1, label="mid price")

        ev = Es[Es.sid == sid].copy()
        ev["t"] = pd.to_datetime(ev.ts.to_numpy() + HKT_MS, unit="ms")
        mid_at = dict(zip(g.ts.to_numpy(), g.mid.to_numpy()))
        ev["mid_at"] = ev.ts.map(mid_at)
        ev = ev[ev.mid_at.notna()]

        none_ = ev[ev.real == 0]
        small = ev[(ev.real == 1) & (ev.tradeable == 0)]
        trade = ev[ev.tradeable == 1]
        tot["none"] += len(none_); tot["small"] += len(small)
        tot["tradeable"] += len(trade)

        ax.scatter(none_.t, none_.mid_at, marker="x", s=14, color="#bbbbbb",
                   linewidth=.7, zorder=2, label="no move (%d)" % len(none_))
        ax.scatter(small.t, small.mid_at, marker="o", s=26, color="#ff9f1c",
                   edgecolor="none", alpha=.85, zorder=3,
                   label="real but TOO SMALL (%d)" % len(small))
        up = trade[trade.signed_peak > 0]
        dn = trade[trade.signed_peak <= 0]
        ax.scatter(up.t, up.mid_at, marker="^", s=130, color="#2a9d8f",
                   edgecolor="black", linewidth=.6, zorder=5,
                   label="TRADEABLE up (%d)" % len(up))
        ax.scatter(dn.t, dn.mid_at, marker="v", s=130, color="#e63946",
                   edgecolor="black", linewidth=.6, zorder=5,
                   label="TRADEABLE down (%d)" % len(dn))

        lbl = str(g.series.iloc[0]).split("|")
        ax.set_title("%s  |  %s %s" % (lbl[0], lbl[1], lbl[2]), fontsize=9)
        ax.set_ylabel("mid"); ax.grid(alpha=.25)
        ax.legend(fontsize=6.5, loc="best", ncol=2)
    for ax in axes[len(picks):]:
        ax.set_visible(False)

    fig.suptitle(
        "Collapse events by whether they would PAY   |   session %s\n"
        "TRADEABLE = durable move >= 2 x spread at the collapse instant   "
        "(%d tradeable, %d real-but-too-small, %d no move across these matches)"
        % (sess, tot["tradeable"], tot["small"], tot["none"]), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    f = os.path.join(RES, "breakeven_map_%s.png" % sess)
    fig.savefig(f, dpi=125)
    plt.close(fig)

    print("session %s" % sess)
    print("  matches plotted: %d" % len(picks))
    for k, v in tot.items():
        print("  %-10s %d" % (k, v))
    print("\nsession-wide:")
    print("  collapses {:,} | real {:,} ({:.1%}) | TRADEABLE {:,} ({:.1%})"
          .format(len(Es), int(Es.real.sum()), Es.real.mean(),
                  int(Es.tradeable.sum()), Es.tradeable.mean()))
    Es.groupby("game").agg(collapses=("real", "size"), real=("real", "sum"),
                           tradeable=("tradeable", "sum")).to_csv(
        os.path.join(RES, "by_game_%s.csv" % sess))
    print("wrote", f)


if __name__ == "__main__":
    sys.exit(main())
