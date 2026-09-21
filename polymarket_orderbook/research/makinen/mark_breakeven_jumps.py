"""
Mark ONLY the jumps that break even after fees.

Justin's correction: the previous pass averaged over every jump, most of which
lose. That is not the question. The question is the subset that nets positive
AFTER the Polymarket taker fee -- nothing in the marked set is allowed to be a
loss.

A jump qualifies when, entering at its onset and exiting at the peak of the
move, the trade nets strictly positive:

    net = |move| - (spread_in + spread_out)/2 - fee_in - fee_out  >  0

Everything is taken from `oracle_jump_scan.py`'s events.parquet, so the pricing
is identical -- this only selects and draws.

Stated plainly, and only once: the marked set is chosen with hindsight. A jump
is in it because of what the price and the book did afterwards. It is what a
perfect trader would have taken, which is the point of looking at it -- the
question the picture answers is whether those moments look like anything you
could recognise at the time.

Outputs, under results/makinen/oracle_jumps/
  breakeven_jumps.csv          every qualifying jump, one row each
  breakeven_summary.csv        the set's own statistics
  breakeven_by_game.csv        how they fall across games and contracts
  breakeven_marked.png         the price graph with them marked
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

GRID_MS = 200
TICK = 0.01


def load_qualifying(outdir: Path, criterion: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_parquet(outdir / "events.parquet")
    ev = ev[ev.is_jump & np.isfinite(ev.pnl_peak) & np.isfinite(ev.pnl_best)].copy()
    col = {"peak": "pnl_peak", "best": "pnl_best"}[criterion]
    ev["net"] = ev[col]
    return ev, ev[ev.net > 0].copy()


def contract_hours(sessions: list[str]) -> float:
    """In-game contract-hours behind the scan, for a rate per contract-hour."""
    total = 0.0
    for s in sessions:
        path = ROOT / "data" / "jump" / f"feat_{s}_trimmed.parquet"
        if not path.exists():
            continue
        n = pd.read_parquet(path, columns=["valid"]).valid.sum()
        total += n * GRID_MS / 1000.0 / 3600.0
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    ap.add_argument("--criterion", choices=["peak", "best"], default="peak",
                    help="peak: exit at the move's peak. best: any exit within 300s.")
    ap.add_argument("--panels", type=int, default=6)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    allj, win = load_qualifying(outdir, args.criterion)

    cols = ["session", "slug", "market_type", "series", "ts", "ts_peak", "lead_to_peak_s",
            "entry_mid", "peak_mid", "entry_spread", "peak_spread",
            "gross_peak", "cost_peak", "fee_peak", "net"]
    win.sort_values("net", ascending=False)[cols].to_csv(
        outdir / "breakeven_jumps.csv", index=False)

    hours = contract_hours(sorted(allj.session.unique()))
    summary = pd.DataFrame({
        "criterion": [args.criterion],
        "n_jumps_scanned": [len(allj)],
        "n_break_even": [len(win)],
        "share_of_jumps": [len(win) / len(allj)],
        "total_ticks": [win.net.sum()],
        "mean_ticks_each": [win.net.mean()],
        "median_ticks_each": [win.net.median()],
        "min_ticks_each": [win.net.min()],
        "p90_ticks_each": [win.net.quantile(0.90)],
        "in_game_contract_hours": [hours],
        "per_contract_hour": [len(win) / hours if hours else np.nan],
        "distinct_contracts": [win.series.nunique()],
        "distinct_games": [win.slug.nunique()],
        "median_move": [win.gross_peak.median()],
        "median_entry_spread": [win.entry_spread.median()],
        "median_exit_spread": [win.peak_spread.median()],
        "median_fee": [win.fee_peak.median()],
        "median_lead_to_peak_s": [win.lead_to_peak_s.median()],
    })
    summary.to_csv(outdir / "breakeven_summary.csv", index=False)

    print(f"MARKED SET: jumps that net > 0 after fees (exit = {args.criterion})")
    print(summary.T.to_string(header=False))

    print()
    print("  by market type")
    print(win.groupby("market_type").agg(
        n=("net", "size"), ticks=("net", "sum"), mean=("net", "mean"),
        entry_spread=("entry_spread", "median"), move=("gross_peak", "median"),
    ).round(2).to_string())

    print()
    print("  by session")
    per = win.groupby("session").agg(n=("net", "size"), ticks=("net", "sum")).round(1)
    per["n_games"] = win.groupby("session").slug.nunique()
    print(per.to_string())

    by_game = win.groupby(["session", "slug"]).agg(
        n=("net", "size"), ticks=("net", "sum"), best=("net", "max")).sort_values(
        "ticks", ascending=False)
    by_game.to_csv(outdir / "breakeven_by_game.csv")
    print()
    print("  top 10 games")
    print(by_game.head(10).round(2).to_string())

    # What a $100 stake would have made, if you took every one and nothing else.
    stake = 100.0
    shares = stake / win.entry_mid.clip(lower=0.01)
    dollars = (shares * win.net * TICK).sum()
    print()
    print(f"  taking all {len(win)} and nothing else, $100 a trade: "
          f"${dollars:,.0f} over {len(win.session.unique())} sessions")

    # The bar any selector has to clear. Mixing payers and non-payers at
    # precision p gives EV = p*mean(payer) + (1-p)*mean(non-payer); setting that
    # to zero gives the precision needed to break even at all.
    lose = allj[allj.net <= 0]
    mw, ml = win.net.mean(), lose.net.mean()
    need = -ml / (mw - ml)
    print()
    print(f"  a payer is worth {mw:+.2f}t, a non-payer {ml:+.2f}t, base rate "
          f"{len(win)/len(allj):.2%}")
    print(f"  --> ANY selector needs {need:.1%} precision just to break even")
    for pr in (0.10, 0.25, 0.50, 0.90):
        print(f"       at {pr:.0%} precision: {pr * mw + (1 - pr) * ml:+.2f} t/trade")
    pd.DataFrame({
        "mean_payer_ticks": [mw],
        "mean_non_payer_ticks": [ml],
        "base_rate": [len(win) / len(allj)],
        "precision_needed_to_break_even": [need],
    }).to_csv(outdir / "breakeven_precision_bar.csv", index=False)

    draw(win, allj, outdir / "breakeven_marked.png", args.panels)


def draw(win: pd.DataFrame, allj: pd.DataFrame, path: Path, panels: int) -> None:
    """The price graph of the contracts holding the most qualifying jumps, with
    only those jumps marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE = "#fcfcfb"
    INK, INK2 = "#0b0b0b", "#52514e"
    LINE = "#8d9299"
    GOOD, BAD = "#0ca30c", "#d03b3b"

    picks = win.series.value_counts().head(panels).index.tolist()
    ncol = 2
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, ser in zip(axes, picks):
        sess = win.loc[win.series == ser, "session"].iloc[0]
        src = ROOT / "data" / "jump" / f"feat_{sess}_trimmed.parquet"
        df = pd.read_parquet(src, columns=["mid", "ts", "series", "valid"])
        df = df[(df.series == ser) & df.valid].sort_values("ts")
        t0 = df.ts.min()
        mins = (df.ts - t0) / 60000.0

        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color="#e9e8e4", lw=0.8)
        ax.set_axisbelow(True)

        ax.plot(mins, df.mid.values, color=LINE, lw=0.9)

        other = allj[(allj.series == ser) & (allj.net <= 0)]
        ax.scatter((other.ts - t0) / 60000.0, other.entry_mid, s=14, color=BAD,
                   alpha=0.30, linewidths=0, zorder=3, label="jump that loses")

        w = win[win.series == ser]
        ax.scatter((w.ts - t0) / 60000.0, w.entry_mid, s=46, color=GOOD,
                   edgecolors=SURFACE, linewidths=1.2, zorder=4,
                   label="breaks even after fees")

        slug, mtype = w.slug.iloc[0], w.market_type.iloc[0]
        ax.set_title(f"{slug}  {mtype}   -   {len(w)} of {len(other) + len(w)} jumps pay, "
                     f"{w.net.sum():.0f} ticks", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("minutes into the recording", color=INK2, fontsize=8.5)
        ax.set_ylabel("mid", color=INK2, fontsize=8.5)

    for ax in axes[len(picks):]:
        ax.set_visible(False)

    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle("Only the jumps that clear the spread and the fee are marked",
                 color=INK, fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
