"""
Zoomed examples of the jumps that break even after fees.

mark_breakeven_jumps.py answers "which ones pay" and draws them as dots on a
three-hour price line. At that zoom a qualifying jump is a dot, and the thing
Justin actually wants to judge -- whether these moments look like anything you
could recognise before they happen -- is invisible. This draws one panel per
jump, about 90 seconds wide, with the book's spread shown as a band so the cost
side is visible rather than asserted.

Examples are picked by PERCENTILE of net ticks, not by size. The marked set is
severely top-heavy: the best 100 of 935 jumps carry half the total ticks, and
the median jump nets 0.87 ticks. A panel of the nine biggest would be a
different and much more flattering picture than the set actually is, so the
nine panels here are spread from the 99.9th percentile down to the 0.5th and
each is labelled with where it sits.

The pricing is not recomputed -- every number comes from the same events.parquet
that mark_breakeven_jumps.py reads, so the two agree by construction.

Stated once, as in the sibling script: this set is chosen with hindsight. A jump
is in it because of what the price did next.

Output: results/makinen/oracle_jumps/breakeven_examples.png
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

# Reference palette, light mode, on the #fcfcfb surface this repo's figures use.
# One accent hue only: entry and exit are told apart by shape and by a direct
# label, never by colour alone.
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
LINE = "#6f7378"
BAND = "#dcdedf"
ACCENT = "#2a78d6"
GRID = "#e9e8e4"

# Where in the net-ticks distribution to sample. Deliberately spread.
PERCENTILES = [99.9, 99.0, 95.0, 75.0, 50.0, 25.0, 5.0, 1.0, 0.5]


def pick_examples(win: pd.DataFrame, pcts: list[float]) -> pd.DataFrame:
    """One jump per requested percentile of net ticks, no repeats."""
    chosen, used = [], set()
    for p in pcts:
        target = np.percentile(win.net, p)
        order = (win.net - target).abs().sort_values().index
        for idx in order:
            if idx not in used:
                used.add(idx)
                row = win.loc[idx].copy()
                row["pct"] = p
                chosen.append(row)
                break
    return pd.DataFrame(chosen)


def series_window(session: str, series: str, t0: int, t1: int,
                  pad_ms: int) -> pd.DataFrame:
    src = ROOT / "data" / "jump" / f"feat_{session}_trimmed.parquet"
    df = pd.read_parquet(src, columns=["mid", "spread_ticks", "ts", "series",
                                       "valid", "stale"])
    d = df[df.series == series].sort_values("ts")
    return d[(d.ts >= t0 - pad_ms) & (d.ts <= t1 + pad_ms)]


def draw(ex: pd.DataFrame, path: Path, pad_s: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ex)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16.5, 4.3 * nrow),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    pad_ms = int(pad_s * 1000)

    for ax, (_, r) in zip(axes, ex.iterrows()):
        d = series_window(r.session, r.series, int(r.ts), int(r.ts_peak), pad_ms)
        secs = (d.ts - r.ts) / 1000.0
        half = d.spread_ticks.values * TICK / 2.0

        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

        # The book, not just the price: everything between these two lines is
        # cost. Where the band is fat the trade is not takeable at the mid.
        ax.fill_between(secs, d.mid.values - half, d.mid.values + half,
                        color=BAND, lw=0, zorder=1, label="bid-ask spread")
        ax.plot(secs, d.mid.values, color=LINE, lw=1.4, zorder=2, label="mid")

        # The hold, entry to exit.
        hold = (r.ts_peak - r.ts) / 1000.0
        ax.axvspan(0, hold, color=ACCENT, alpha=0.07, lw=0, zorder=0)
        ax.plot([0], [r.entry_mid], marker="o", ms=9, color=ACCENT,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=5)
        ax.plot([hold], [r.peak_mid], marker="D", ms=8.5, color=ACCENT,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=5)

        # Headroom so the entry/exit labels never sit on the frame or the data.
        band_lo = float(np.nanmin(d.mid.values - half))
        band_hi = float(np.nanmax(d.mid.values + half))
        span = max(band_hi - band_lo, 0.01)
        ax.set_ylim(band_lo - span * 0.18, band_hi + span * 0.18)

        ax.annotate("entry", (0, r.entry_mid), textcoords="offset points",
                    xytext=(0, -19), ha="center", fontsize=8.5, color=ACCENT,
                    fontweight="bold")
        ax.annotate("exit", (hold, r.peak_mid), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=8.5, color=ACCENT,
                    fontweight="bold")

        # All the arithmetic lives in the title block, outside the axes: inside
        # it there is no corner that is reliably free of the price line, the
        # spread band and the two markers across nine different panels.
        resolving = (r.peak_mid > 0.95) or (r.peak_mid < 0.05)
        tag = "    [market resolving]" if resolving else ""
        ax.set_title(
            f"p{r.pct:g} of the marked set  -  nets {r.net:.2f} ticks{tag}\n"
            f"{r.slug}  {r.market_type}\n"
            f"move {r.gross_peak:.1f}t  -  spread {r.cost_peak:.1f}t  "
            f"-  fee {r.fee_peak:.1f}t  =  {r.net:.2f}t\n"
            f"entry spread {r.entry_spread:.1f}t, exit {r.peak_spread:.1f}t, "
            f"held {hold:.1f}s",
            color=INK2, fontsize=8.6, loc="left", pad=8, linespacing=1.65)

        ax.set_xlabel("seconds from entry", color=INK2, fontsize=8.5)
        ax.set_ylabel("price", color=INK2, fontsize=8.5)
        ax.axhline(r.entry_mid, color=MUTED, lw=0.7, ls=(0, (4, 4)), zorder=1)

    for ax in axes[n:]:
        ax.set_visible(False)

    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle("What a break-even jump actually looks like, across the whole "
                 "range of the marked set", color=INK, fontsize=13.5, x=0.006,
                 y=0.992, ha="left")
    fig.text(0.006, 0.971, "shaded band is the bid-ask spread - the cost side of "
             "the trade; the blue span is the hold, entry to exit. "
             "935 of 82,120 jumps qualify; the median one nets 0.87 ticks.",
             color=INK2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=[0, 0.028, 1, 0.962])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    ap.add_argument("--pad-seconds", type=float, default=40.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    win = pd.read_csv(outdir / "breakeven_jumps.csv")
    ex = pick_examples(win, PERCENTILES)

    print("examples chosen (by percentile of net ticks):")
    print(ex[["pct", "slug", "market_type", "entry_mid", "peak_mid",
              "gross_peak", "cost_peak", "fee_peak", "net",
              "lead_to_peak_s"]].round(2).to_string(index=False))
    ex.to_csv(outdir / "breakeven_examples.csv", index=False)

    draw(ex, outdir / "breakeven_examples.png", args.pad_seconds)


if __name__ == "__main__":
    main()
