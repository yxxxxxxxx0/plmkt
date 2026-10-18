"""
Separate jumps the PRICE caused from jumps the SPREAD caused.

The mid is (bid + ask) / 2, so it has two ways of moving and only one of them
is a price change:

    both touches move the same way      the whole book repriced.   A price move.
    one touch moves, or they separate   the spread opened or shut.  Not a price
                                        move -- the mid slid because a quote was
                                        pulled, and nobody traded anywhere near
                                        the new mid.

This distinction is not cosmetic on this dataset. audit_label.py already found
the jump label "heavily contaminated by spread-driven mid noise", with
prevalence running 0.42 -> 0.98 across spread buckets -- i.e. in wide books
almost everything looks like a jump. Anything built on the unfiltered label is
partly modelling quote withdrawal.

The decomposition is exact, not a heuristic. Writing d for the change from jump
onset to peak:

    d_mid = (d_bid + d_ask) / 2          identically
    d_spread = d_ask - d_bid

so the part of the mid move that both touches share is the price component, and
the rest is the spread opening or closing:

    common = sign * min(|d_bid|, |d_ask|)   when d_bid and d_ask agree in sign
           = 0                              when they do not
    price_share = common / d_mid            in [0, 1] when the signs agree

price_share = 1 means the book moved rigidly -- bid and ask shifted by the same
amount, the spread is unchanged, and the mid moved because the price did.
price_share = 0 means one touch did not move at all, so half the mid move is
pure spread and the trade is against a quote that was never there.

Everything is read from oracle_jump_scan.py's events.parquet, which carries
entry/peak mid and spread, so the touches are reconstructed exactly and no
pricing is recomputed.

Outputs, under results/makinen/oracle_jumps/
  jump_cause.csv              every jump with its decomposition and class
  jump_cause_summary.csv       counts and ticks by class
  price_driven_breakeven.csv   the break-even set, price-driven only
  jump_cause.png               the split, and what it does to the marked set

    python classify_jump_cause.py --min-price-share 0.8
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

SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
GRID = "#e9e8e4"
PRICE_C = "#2a78d6"    # reference palette slot 1
SPREAD_C = "#eb6834"   # slot 2
MUTED = "#b9b8b3"


def decompose(ev: pd.DataFrame) -> pd.DataFrame:
    """Split each jump's mid move into a price part and a spread part."""
    d = ev.copy()
    # Touches, exactly: the mid and the spread are both recorded.
    entry_bid = d.entry_mid - d.entry_spread * TICK / 2.0
    entry_ask = d.entry_mid + d.entry_spread * TICK / 2.0
    peak_bid = d.peak_mid - d.peak_spread * TICK / 2.0
    peak_ask = d.peak_mid + d.peak_spread * TICK / 2.0

    d["d_bid"] = (peak_bid - entry_bid) / TICK
    d["d_ask"] = (peak_ask - entry_ask) / TICK
    d["d_mid"] = (d.d_bid + d.d_ask) / 2.0
    d["d_spread"] = d.d_ask - d.d_bid

    same_sign = np.sign(d.d_bid) == np.sign(d.d_ask)
    common = np.where(same_sign & (np.sign(d.d_bid) != 0),
                      np.sign(d.d_bid) * np.minimum(d.d_bid.abs(), d.d_ask.abs()),
                      0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(d.d_mid.abs() > 1e-9, common / d.d_mid, np.nan)
    d["common_ticks"] = common
    d["price_share"] = np.clip(share, 0.0, 1.0)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    ap.add_argument("--min-price-share", type=float, default=0.8,
                    help="a jump counts as price-driven when at least this "
                         "fraction of the mid move is both touches together")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    ev = pd.read_parquet(outdir / "events.parquet")
    j = ev[ev.is_jump & np.isfinite(ev.pnl_peak)].copy()
    j = decompose(j)

    thr = args.min_price_share
    j["cause"] = np.where(j.price_share >= thr, "price", "spread")
    j["rigid"] = j.price_share >= 0.999      # spread literally unchanged
    j["pays"] = j.pnl_peak > 0

    print(f"{len(j):,} durable jumps, split at price_share >= {thr}")
    print()
    g = j.groupby("cause").agg(
        n=("cause", "size"),
        share=("cause", lambda s: len(s) / len(j)),
        median_move=("gross_peak", "median"),
        median_entry_spread=("entry_spread", "median"),
        median_d_spread=("d_spread", "median"),
        n_pays=("pays", "sum"),
    )
    g["pay_rate"] = g.n_pays / g.n
    print(g.round(3).to_string())

    print()
    print("  the spread-driven ones are exactly the wide-book ones:")
    j["spread_bucket"] = pd.cut(j.entry_spread, [-0.01, 1, 2, 5, 10, 1e9],
                                labels=["<=1t", "1-2t", "2-5t", "5-10t", ">10t"])
    pv = j.pivot_table(index="spread_bucket", columns="cause", values="ts",
                       aggfunc="size", observed=True).fillna(0).astype(int)
    pv["price_%"] = (100 * pv.get("price", 0) / pv.sum(axis=1)).round(1)
    print(pv.to_string())

    # --- what it does to the marked set -----------------------------------
    win = j[j.pays].copy()
    wp = win[win.cause == "price"]
    print()
    print("=== the break-even set, before and after ===")
    print(f"  all break-even jumps      : {len(win):>5,}   {win.pnl_peak.sum():>8.0f} ticks")
    print(f"  price-driven only         : {len(wp):>5,}   {wp.pnl_peak.sum():>8.0f} ticks "
          f"({len(wp)/len(win):.0%} of them, {wp.pnl_peak.sum()/win.pnl_peak.sum():.0%} of the ticks)")
    print(f"  rigid book (spread fixed) : {int(win.rigid.sum()):>5,}   "
          f"{win[win.rigid].pnl_peak.sum():>8.0f} ticks")

    # The bar a selector faces, recomputed within the price-driven population.
    jp = j[j.cause == "price"]
    mw = jp[jp.pays].pnl_peak.mean()
    ml = jp[~jp.pays].pnl_peak.mean()
    base = jp.pays.mean()
    need = -ml / (mw - ml)
    print()
    print(f"  within price-driven jumps: base rate {base:.2%}, payer {mw:+.2f}t, "
          f"non-payer {ml:+.2f}t")
    print(f"  --> a selector needs {need:.1%} precision (vs 77.3% on the unfiltered set)")

    cols = ["session", "slug", "market_type", "series", "ts", "entry_mid", "peak_mid",
            "entry_spread", "peak_spread", "d_bid", "d_ask", "d_mid", "d_spread",
            "price_share", "cause", "rigid", "gross_peak", "pnl_peak", "pays"]
    j[cols].to_csv(outdir / "jump_cause.csv", index=False)
    g.to_csv(outdir / "jump_cause_summary.csv")
    wp[cols].sort_values("pnl_peak", ascending=False).to_csv(
        outdir / "price_driven_breakeven.csv", index=False)
    pd.DataFrame({
        "population": ["all jumps", "price-driven jumps"],
        "n": [len(j), len(jp)],
        "base_rate": [j.pays.mean(), base],
        "mean_payer_ticks": [j[j.pays].pnl_peak.mean(), mw],
        "mean_non_payer_ticks": [j[~j.pays].pnl_peak.mean(), ml],
        "precision_needed": [
            -j[~j.pays].pnl_peak.mean() /
            (j[j.pays].pnl_peak.mean() - j[~j.pays].pnl_peak.mean()), need],
    }).to_csv(outdir / "jump_cause_precision_bar.csv", index=False)

    draw(j, outdir / "jump_cause.png", thr)


def draw(j: pd.DataFrame, path: Path, thr: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), facecolor=SURFACE)

    def style(ax):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # 1. where the mid move comes from. Log y: 63% of the mass sits in the
    #    single bar at zero and would flatten everything else to nothing.
    ax = axes[0]; style(ax)
    ax.hist(j.price_share.dropna(), bins=50, color=MUTED, lw=0)
    ax.set_yscale("log")
    ax.axvline(thr, color=INK, lw=1.3, ls=(0, (4, 3)))
    zero = (j.price_share <= 1e-9).mean()
    ax.annotate(f"{zero:.0%} of all jumps sit here:\none touch never moved at all",
                (0.02, ax.get_ylim()[1] * 0.35), fontsize=9, color=INK2, va="top")
    ax.annotate(f"price-driven\n(>= {thr:g})", (thr + 0.02, ax.get_ylim()[1] * 0.006),
                fontsize=9, color=INK, va="bottom", fontweight="bold")
    ax.set_title("How much of each jump's mid move is\nboth touches moving together",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel("price share of the mid move", color=INK2, fontsize=9)
    ax.set_ylabel("jumps (log scale)", color=INK2, fontsize=9)

    # 2. the split by entry spread
    ax = axes[1]; style(ax)
    pv = j.pivot_table(index="spread_bucket", columns="cause", values="ts",
                       aggfunc="size", observed=True).fillna(0)
    frac = pv.div(pv.sum(axis=1), axis=0)
    pr = frac.get("price", pd.Series([0.0] * len(frac), index=frac.index))
    x = np.arange(len(frac))
    ax.bar(x, pr.values, color=PRICE_C, width=0.6, label="price moved")
    ax.bar(x, 1 - pr.values, bottom=pr.values, color=SPREAD_C, width=0.6,
           label="spread moved")
    ax.set_xticks(x); ax.set_xticklabels(frac.index, fontsize=9)
    ax.set_ylim(0, 1.16)
    for i, v in enumerate(pr.values):
        ax.text(i, v + 0.025, f"{v:.0%}", ha="center", va="bottom", color=PRICE_C,
                fontsize=9.5, fontweight="bold")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2,
                    bbox_to_anchor=(1.0, 1.02))
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title("The wider the book, the less of a 'jump'\nis actually the price",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel("spread at entry", color=INK2, fontsize=9)
    ax.set_ylabel("share of jumps", color=INK2, fontsize=9)

    # 3. the punchline: what filtering does to the base rate
    ax = axes[2]; style(ax)
    rows = []
    for cause, lab in (("price", "price moved"), ("spread", "spread moved")):
        sub = j[j.cause == cause]
        rows.append((lab, sub.pays.mean(), int(sub.pays.sum()), len(sub)))
    x = np.arange(2)
    ax.bar(x, [r[1] for r in rows], width=0.5, color=[PRICE_C, SPREAD_C])
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylim(0, max(r[1] for r in rows) * 1.34)
    for i, (lab, rate, k, n) in enumerate(rows):
        ax.text(i, rate + max(r[1] for r in rows) * 0.035,
                f"{rate:.1%}\n{k:,} of {n:,}", ha="center", va="bottom",
                fontsize=9.5, color=INK, linespacing=1.5)
    ax.set_title("Share of jumps that break even after fees,\nby what actually moved",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_ylabel("break-even rate", color=INK2, fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    fig.suptitle("A mid can move because the price moved, or because a quote was "
                 "pulled - and only one of those is tradable",
                 color=INK, fontsize=13, x=0.006, y=0.985, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.915])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
