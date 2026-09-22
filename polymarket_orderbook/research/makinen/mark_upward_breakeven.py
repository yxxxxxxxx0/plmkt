"""Mark every UPWARD jump that breaks even, one picture per game.

Justin asked for the upward break-even jumps marked and drawn. "Upward" and
"breaks even" both need a definition that survives contact with this data, and
the two chosen here are the strict ones:

  upward       the trade that paid was a LONG -- bought at the ask, sold at a
               higher bid later. Every one of the marks satisfies
               exit_px > entry_px by construction, so no judgement is involved.
               Note this is NOT the same as the detector's own up/down call:
               38 of the marks are jumps whose LARGER mid excursion was
               downward, and whose upward side is nevertheless the one that
               cleared its costs at the touch. The executable definition wins
               here, because it is the one you could have traded.

  breaks even  exec_net_ticks > 0 from mark_breakeven_exec.py: entry and exit
               both at a price that actually stood in the book, exit chosen by
               maximising the executable price within 10s, sports fee charged
               on both legs.

Both legs of a market are separate series (the key carries asset_id), so a
home-token rise and the matching away-token fall are two rows for one real
event. They are kept separate deliberately: each is its own instrument with
its own book and its own spread, and merging them would assume a complement
map this project does not store.

WHAT STOPS TWO CLOSE JUMPS BECOMING ONE
---------------------------------------
Detection is deliberately over-eager: every 200ms slot is tested independently,
so one real 8-tick move produces hundreds of consecutive candidates. The
suppression is what turns that into events -- walk the candidates in time
order, take one, and ignore every later candidate on the SAME series for
DEBOUNCE_S = 30s (150 slots). Because a jump's measurement window is 10s of
peak search plus a 3s persistence check = 13s, and 13 < 30, two kept jumps on
one contract can never share a single grid slot of forward path. No stretch of
price can contribute to two marks.

The rule keeps the EARLIEST candidate, not the largest, and it discards a
genuinely separate second move arriving inside 30s. Both errors run toward
undercounting, never toward marking one move twice.

Outputs, under results/makinen/upward_breakeven/
  <slug>.png                    one per game, a panel per marked contract
  upward_breakeven_marks.csv    every mark, with its game, contract and P&L
  upward_breakeven_by_game.csv  the per-game totals in the figure titles

    python research/makinen/mark_upward_breakeven.py
    python research/makinen/mark_upward_breakeven.py --games mlb-phi-laa-2026-08-28
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from match_filter import load_windows  # noqa: E402

TICK = 0.01
STAKE = 10.0

# The repo's chart palette. Checked against the six colour checks rather than
# eyeballed: normal-vision dE 33.6 (floor 15), deuteranopia 31.6, protanopia
# 24.5, tritanopia 43.8 (target 8), and every ink >= 3:1 on the surface.
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
LINE = "#8d9299"
BAND = "#dcdedf"
GOOD = "#2a78d6"
GRID_C = "#e9e8e4"


def fee_per_share(price):
    p = np.clip(np.asarray(price, dtype=np.float64), 0.0, 1.0)
    return 0.05 * p * (1.0 - p)


def marks(outdir: Path) -> pd.DataFrame:
    """The upward break-even set, from the touch-priced marking."""
    a = pd.read_parquet(outdir / "breakeven_exec_all.parquet")
    m = a[(a.exec_net_ticks > 0) & (a.side == "long")].copy()
    assert (m.exit_px > m.entry_px).all(), "a long mark that did not rise"
    # shares are stake / the price actually paid, which for a long is the ask
    m["shares"] = STAKE / m.entry_px
    m["pnl_usd"] = m.shares * (m.exit_px - m.entry_px) - m.shares * (
        fee_per_share(m.entry_px) + fee_per_share(m.exit_px))
    return m.sort_values(["slug", "series", "ts"]).reset_index(drop=True)


def contract_label(series: str) -> str:
    """slug|market_type|line|asset_id -> something a human can read."""
    parts = str(series).split("|")
    mt = parts[1] if len(parts) > 1 else "?"
    line = parts[2] if len(parts) > 2 else ""
    tok = parts[3][-6:] if len(parts) > 3 and parts[3] else ""
    lab = mt if not line or line == "None" else mt + " " + line
    return lab + "  ..." + tok


def draw_game(slug, gm, feat, window, outpath):
    """One figure per game: a panel per contract that carries a mark."""
    series_ids = list(dict.fromkeys(gm.series))
    n = len(series_ids)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.5 * n + 1.4), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor(SURFACE)

    t0 = window[0] if window else int(feat.ts.min())

    for ax, sid in zip(axes, series_ids):
        f = feat[feat.series == sid]
        mk = gm[gm.series == sid]
        x = (f.ts.to_numpy() - t0) / 60000.0
        mid = f.mid.to_numpy()
        half = f.spread_ticks.to_numpy() * TICK / 2.0

        ax.set_facecolor(SURFACE)
        # the executable band: everything between the touches is cost
        ax.fill_between(x, mid - half, mid + half, color=BAND, linewidth=0,
                        zorder=1, label="bid-ask")
        ax.plot(x, mid, color=LINE, linewidth=1.0, zorder=2, label="mid")

        xe = (mk.ts.to_numpy() - t0) / 60000.0
        xx = (mk.ts_exit.to_numpy() - t0) / 60000.0
        # A mark lasts a median 9s on an axis three hours wide, so the
        # entry->exit segment is sub-pixel here. The vertical rule is what
        # makes it findable; the segment is still drawn so a zoom is honest.
        first = True
        for a, b, ep, xp in zip(xe, xx, mk.entry_px, mk.exit_px):
            ax.axvline(a, color=GOOD, alpha=0.22, linewidth=0.8, zorder=3)
            ax.plot([a, b], [ep, xp], color=GOOD, linewidth=2.0,
                    solid_capstyle="round", zorder=4,
                    label="upward break-even jump" if first else None)
            ax.plot([a, b], [ep, xp], "o", color=GOOD, markersize=5.0,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)
            first = False

        # one direct label per panel -- the biggest mark. More than one
        # collides, because the marks cluster where the book is moving.
        r = mk.nlargest(1, "exec_net_ticks").iloc[0]
        ax.annotate("+%.1ft" % r.exec_net_ticks,
                    ((r.ts_exit - t0) / 60000.0, r.exit_px),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=7.5, color=INK2, zorder=6, annotation_clip=True,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE,
                              edgecolor="none", alpha=0.85))

        ax.set_title("%s   -   %d mark(s), %.1f ticks"
                     % (contract_label(sid), len(mk), mk.exec_net_ticks.sum()),
                     fontsize=9, color=INK, loc="left", pad=4)
        ax.grid(True, color=GRID_C, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID_C)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_ylabel("price", fontsize=8, color=INK2)

    axes[-1].set_xlabel("minutes from first pitch", fontsize=8, color=INK2)
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False, ncol=3)

    # dollars must be escaped or matplotlib reads the pair as mathtext and
    # renders "$4.17 at $10" as an italic "4.17at10"
    fig.suptitle("%s   -   %d upward break-even jump(s), %.1f ticks, "
                 "\\$%.2f at \\$%.0f a trade"
                 % (slug, len(gm), gm.exec_net_ticks.sum(),
                    gm.pnl_usd.sum(), STAKE),
                 fontsize=11, color=INK, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(outpath, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",
                    default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    ap.add_argument("--figdir",
                    default=str(ROOT / "results" / "makinen" / "upward_breakeven"))
    ap.add_argument("--games", nargs="*", default=None,
                    help="only these slugs (default: every game with a mark)")
    args = ap.parse_args()
    outdir, figdir = Path(args.outdir), Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    m = marks(outdir)
    if args.games:
        m = m[m.slug.isin(args.games)]
    windows = load_windows()

    print("upward break-even jumps : %d" % len(m))
    print("  across                : %d games, %d contracts, %d sessions"
          % (m.slug.nunique(), m.series.nunique(), m.session.nunique()))
    print("  ticks                 : %.1f" % m.exec_net_ticks.sum())
    print("  P&L at $%.0f/trade      : $%.2f" % (STAKE, m.pnl_usd.sum()))
    print("  by market type        : %s" % m.market_type.value_counts().to_dict())
    print()

    m.to_csv(figdir / "upward_breakeven_marks.csv", index=False)
    by_game = (m.groupby(["session", "slug"])
                 .agg(marks=("series", "size"), contracts=("series", "nunique"),
                      ticks=("exec_net_ticks", "sum"), pnl_usd=("pnl_usd", "sum"))
                 .reset_index().sort_values("ticks", ascending=False))
    by_game.to_csv(figdir / "upward_breakeven_by_game.csv", index=False)

    made = 0
    for sess, sm in m.groupby("session"):
        src = ROOT / "data" / "jump" / ("feat_%s_trimmed.parquet" % sess)
        feat = pd.read_parquet(src, columns=["series", "ts", "mid", "spread_ticks"])
        feat = feat[feat.series.isin(set(sm.series))]
        for slug, gm in sm.groupby("slug"):
            f = feat[feat.series.isin(set(gm.series))]
            draw_game(slug, gm, f, windows.get(slug), figdir / ("%s.png" % slug))
            made += 1
            print("  %s: %d mark(s) over %d contract(s), %.1ft"
                  % (slug, len(gm), gm.series.nunique(), gm.exec_net_ticks.sum()))
        del feat

    print("\nwrote %d figure(s) + 2 csv(s) to %s" % (made, figdir))


if __name__ == "__main__":
    main()
