"""
When a run scores, when does the book move -- and could you have been early?

This is the sharp version of "use the match feed for direction". The aggregate
lag statistic in align_mlb_plays.py says jumps do not cluster after plays, but
that pools 920 plays of which 836 are routine. A strikeout changes a win
probability by very little; a run changes it a lot. If the feed is ever going
to give direction, it is on the 84 scoring plays, so those are measured on
their own here.

The measurement is an event study on absolute mid movement, bucketed in 5s
bins around each scoring play's endTime from statsapi. Absolute movement, not
signed, because the point is TIMING rather than side: the recording does not
label which token is the home team, but it does not need to. Where the
movement spikes relative to t=0 answers the only question that matters --

    spike BEFORE t=0   the market knew first; the feed is a lagging record of
                       something already priced, and there is nothing to trade.
    spike AFTER t=0    the feed led the book by that gap, and the gap is the
                       trade.

The null is the same curve computed around random in-game times, which is what
"no reaction" looks like given that the book is moving constantly anyway.

    python scoring_play_event_study.py --session books_2026-09-11

Outputs, under results/makinen/mlb_plays/
  scoring_event_study_<session>.csv
  scoring_event_study_<session>.png
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

SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
GRID = "#e9e8e4"
REAL_C = "#2a78d6"
NULL_C = "#eb6834"

BIN_S = 5.0
HALF_WINDOW_S = 150.0


def curve(mid_df: pd.DataFrame, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean |mid change| per 5s bucket, relative to each anchor time."""
    edges = np.arange(-HALF_WINDOW_S, HALF_WINDOW_S + BIN_S, BIN_S)
    centres = (edges[:-1] + edges[1:]) / 2.0
    sums = np.zeros(len(centres))
    counts = np.zeros(len(centres))

    ts = mid_df.ts.to_numpy(dtype=float)
    mid = mid_df.mid.to_numpy(dtype=float)
    dmid = np.abs(np.diff(mid, prepend=mid[:1])) / 0.01  # ticks

    for a in anchors:
        lo, hi = a - HALF_WINDOW_S * 1000, a + HALF_WINDOW_S * 1000
        i0, i1 = np.searchsorted(ts, [lo, hi])
        if i1 <= i0:
            continue
        rel = (ts[i0:i1] - a) / 1000.0
        b = np.digitize(rel, edges) - 1
        ok = (b >= 0) & (b < len(centres))
        np.add.at(sums, b[ok], dmid[i0:i1][ok])
        np.add.at(counts, b[ok], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return centres, np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="books_2026-09-11")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "mlb_plays"))
    ap.add_argument("--market-type", default="moneyline")
    ap.add_argument("--anchor", choices=["end_ms", "start_ms"], default="end_ms",
                    help="endTime is when the play FINISHED; startTime is when "
                         "it began. The difference decides whether the book is "
                         "ahead of the event or merely ahead of the paperwork")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    plays = pd.read_csv(outdir / f"plays_{args.session}.csv")
    scoring = plays[plays.is_scoring & plays.end_ms.notna() & plays.start_ms.notna()]
    print(f"{len(plays):,} plays, {len(scoring)} scoring")

    feat = ROOT / "data" / "jump" / f"feat_{args.session}_trimmed.parquet"
    df = pd.read_parquet(feat, columns=["mid", "ts", "series", "valid"])
    df = df[df.valid]
    df["slug"] = df.series.str.split("|").str[0]
    df["mtype"] = df.series.str.split("|").str[1]
    df = df[df.mtype == args.market_type]
    print(f"{len(df):,} valid {args.market_type} rows across "
          f"{df.slug.nunique()} games")

    rng = np.random.default_rng(0)
    real_sums = np.zeros(int(2 * HALF_WINDOW_S / BIN_S))
    real_n = np.zeros_like(real_sums)
    null_sums = np.zeros_like(real_sums)
    null_n = np.zeros_like(real_sums)
    used = 0

    for slug, g in df.groupby("slug"):
        anch = scoring.loc[scoring.slug == slug, args.anchor].to_numpy(dtype=float)
        if len(anch) == 0:
            continue
        for ser, s in g.groupby("series"):
            s = s.sort_values("ts")
            if len(s) < 50:
                continue
            c, y = curve(s, anch)
            nulls = rng.uniform(s.ts.min(), s.ts.max(), size=len(anch))
            _, yn = curve(s, nulls)
            real_sums += np.nan_to_num(y); real_n += np.isfinite(y)
            null_sums += np.nan_to_num(yn); null_n += np.isfinite(yn)
        used += len(anch)

    centres = np.arange(-HALF_WINDOW_S, HALF_WINDOW_S, BIN_S) + BIN_S / 2
    real = real_sums / np.maximum(real_n, 1)
    null = null_sums / np.maximum(null_n, 1)

    out = pd.DataFrame({"sec_from_play_end": centres,
                        "mean_abs_tick_move": real,
                        "null_mean_abs_tick_move": null,
                        "ratio": real / np.maximum(null, 1e-12)})
    out.to_csv(outdir / f"scoring_event_study_{args.session}_{args.anchor}.csv", index=False)

    pre = (centres < 0)
    post = (centres >= 0)
    print()
    print(f"=== absolute mid movement around {used} scoring-play anchors ===")
    print(f"  mean |move| per 5s bucket, BEFORE the play ends : {real[pre].mean():.4f} ticks")
    print(f"  ...AFTER                                        : {real[post].mean():.4f} ticks")
    print(f"  null (random in-game times)                     : {null.mean():.4f} ticks")
    print(f"  post/null ratio                                 : {real[post].mean()/max(null.mean(),1e-12):.2f}x")
    k = int(np.nanargmax(real))
    print(f"  peak bucket                                     : {centres[k]:+.0f}s "
          f"({real[k]:.4f} ticks, {real[k]/max(null[k],1e-12):.2f}x null)")
    print()
    print("  bucket-by-bucket around zero:")
    for i in range(len(centres)):
        if -40 <= centres[i] <= 60:
            print(f"    {centres[i]:+6.0f}s  real {real[i]:.4f}  null {null[i]:.4f}  "
                  f"{real[i]/max(null[i],1e-12):5.2f}x")

    draw(centres, real, null, outdir / f"scoring_event_study_{args.session}_{args.anchor}.png",
         f"{args.session} anchored on {args.anchor}", used, args.market_type)


def draw(centres, real, null, path, session, n_anchor, mtype):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d3")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    ax.axvspan(-HALF_WINDOW_S, 0, color="#f0efec", lw=0, zorder=0)
    ax.plot(centres, real, color=REAL_C, lw=2.2, label="around a scoring play")
    ax.plot(centres, null, color=NULL_C, lw=2.0, ls=(0, (5, 3)),
            label="around random in-game times (null)")
    ax.axvline(0, color=INK, lw=1.2)
    ax.annotate("play ends\n(statsapi endTime)", (0, ax.get_ylim()[1] * 0.94),
                xytext=(8, 0), textcoords="offset points", fontsize=9,
                color=INK, va="top")
    ax.annotate("before", (-HALF_WINDOW_S * 0.8, ax.get_ylim()[1] * 0.06),
                fontsize=9.5, color=INK2)

    ax.set_xlabel("seconds from the end of the scoring play", color=INK2, fontsize=9.5)
    ax.set_ylabel("mean |mid change| per 5s bucket (ticks)", color=INK2, fontsize=9.5)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle(f"Does the book move when a run scores, and is the feed early?",
                 color=INK, fontsize=13, x=0.006, y=0.985, ha="left")
    fig.text(0.006, 0.925, f"{session}  -  {n_anchor} scoring plays  -  {mtype} tokens",
             color=INK2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
