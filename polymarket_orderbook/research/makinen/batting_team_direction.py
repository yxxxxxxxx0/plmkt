"""
Does knowing WHOSE TURN IT IS TO BAT tell you which way a jump goes?

Justin's correction, and it is a different hypothesis from the scoring-play
latency test in scoring_play_event_study.py. That test asked whether the feed
reports a run before the book moves -- it does not, the book peaks 8s before
statsapi's endTime. This asks nothing about news. Which half-inning it is, is a
slowly-varying STATE known minutes in advance, so there is no latency race at
all: at the instant a jump structure is detected you already know who is
batting, for free.

The mechanism that would make it work is real and specific to baseball. Only
the batting team can score. So the batting team's win probability should drift
gently DOWN as outs accumulate and jump UP when a run comes in -- a compensated
jump process. A martingale forces the unconditional drift to zero; it does not
forbid the direction of the LARGE moves from being skewed, as long as the small
moves lean the other way often enough to pay for it. That is precisely the
regime a jump detector selects on.

No token-to-team mapping is needed, and that matters because none is stored
locally. Two framings, neither of which requires knowing which token is the
home side:

  ASSOCIATION   per token, compare P(jump is up | bottom of the inning) with
                P(jump is up | top). Under the null these are equal. Under the
                hypothesis they differ, with opposite sign for the two tokens
                of a market. The aggregate statistic is mean |difference|,
                tested against a permutation null that shuffles the half-inning
                labels within each game -- which destroys the relationship
                while preserving how many jumps and how many innings there
                were.

  PREDICTION    learn each token's orientation ("this one jumps up when the
                home side bats") on the FIRST half of the game, apply it to the
                SECOND. Leakage-free, and it yields an actual directional
                accuracy comparable to the ~75% bar in SELECTOR.md.

Half-inning at time t is taken from the most recent play at or before t in
statsapi's feed. Jumps more than --max-gap seconds after the last play are
dropped: that is an inning break or a pitching change, where "whose turn" is
genuinely ambiguous.

    python batting_team_direction.py --session books_2026-09-11

Outputs, under results/makinen/batting/
  batting_association_<session>.csv
  batting_prediction_<session>.csv
  batting_team_direction_<session>.png
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
GRID_C = "#e9e8e4"
REAL_C = "#2a78d6"
NULL_C = "#eb6834"


def attach_half(ev: pd.DataFrame, plays: pd.DataFrame, max_gap_s: float) -> pd.DataFrame:
    """Label each jump with the half-inning in force when it happened."""
    out = []
    for slug, g in ev.groupby("slug"):
        pl = plays[(plays.slug == slug) & plays.start_ms.notna()].sort_values("start_ms")
        if pl.empty:
            continue
        pt = pl.start_ms.to_numpy(dtype=float)
        top = pl.is_top.to_numpy()
        inn = pl.inning.to_numpy()
        jt = g.ts.to_numpy(dtype=float)
        idx = np.searchsorted(pt, jt, side="right") - 1
        ok = idx >= 0
        g = g[ok].copy()
        idx = idx[ok]
        g["is_top"] = top[idx]
        g["inning"] = inn[idx]
        g["gap_s"] = (g.ts.to_numpy(dtype=float) - pt[idx]) / 1000.0
        out.append(g[g.gap_s <= max_gap_s])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def association(d: pd.DataFrame) -> pd.DataFrame:
    """Per token: P(up | bottom) - P(up | top), and the counts behind it."""
    rows = []
    for (slug, ser), g in d.groupby(["slug", "series"]):
        bot = g[~g.is_top]
        top = g[g.is_top]
        if len(bot) < 20 or len(top) < 20:
            continue
        p_bot = (bot.signed > 0).mean()
        p_top = (top.signed > 0).mean()
        rows.append(dict(slug=slug, series=ser, market_type=g.market_type.iloc[0],
                         n_bottom=len(bot), n_top=len(top),
                         p_up_bottom=p_bot, p_up_top=p_top, diff=p_bot - p_top))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="books_2026-09-11")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "batting"))
    ap.add_argument("--plays", default=None)
    ap.add_argument("--max-gap", type=float, default=180.0)
    ap.add_argument("--market-types", nargs="*", default=["moneyline", "spread"],
                    help="team-specific markets only. A total is not a team, so "
                         "whose turn it is cannot orient it")
    ap.add_argument("--price-driven-only", action="store_true")
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plays_path = Path(args.plays) if args.plays else (
        ROOT / "results" / "makinen" / "mlb_plays" / f"plays_{args.session}.csv")
    plays = pd.read_csv(plays_path)

    ev = pd.read_parquet(ROOT / "results" / "makinen" / "oracle_jumps" / "events.parquet")
    ev = ev[(ev.session == args.session) & ev.is_jump].copy()
    if args.price_driven_only:
        cz = pd.read_csv(ROOT / "results" / "makinen" / "oracle_jumps" / "jump_cause.csv",
                         usecols=["session", "slug", "series", "ts", "cause"])
        ev = ev.merge(cz, on=["session", "slug", "series", "ts"], how="inner")
        ev = ev[ev.cause == "price"]
    ev = ev[ev.market_type.isin(args.market_types)]
    print(f"{args.session}: {len(ev):,} jumps in {args.market_types}")

    d = attach_half(ev, plays, args.max_gap)
    print(f"{len(d):,} of them sit within {args.max_gap:.0f}s of a play "
          f"({len(d)/max(len(ev),1):.0%})")
    print(f"  top of the inning: {int((d.is_top).sum()):,}   "
          f"bottom: {int((~d.is_top).sum()):,}")

    # ---- 1. association ---------------------------------------------------
    assoc = association(d)
    if assoc.empty:
        raise SystemExit("not enough jumps per token to measure anything")
    obs = assoc["diff"].abs().mean()
    print()
    print("=== ASSOCIATION: P(up | bottom) - P(up | top), per token ===")
    print(f"  {len(assoc)} tokens with >=20 jumps in each half-inning")
    print(f"  mean |difference|            : {obs:.4f}")
    print(f"  mean signed difference       : {assoc['diff'].mean():+.4f}")

    rng = np.random.default_rng(0)
    null = np.empty(args.n_perm)
    for i in range(args.n_perm):
        dd = d.copy()
        dd["is_top"] = dd.groupby("slug").is_top.transform(
            lambda s: rng.permutation(s.to_numpy()))
        a = association(dd)
        null[i] = a["diff"].abs().mean() if not a.empty else np.nan
    p = float(np.mean(null >= obs))
    print(f"  permutation null (n={args.n_perm}) : "
          f"{np.nanmean(null):.4f} +/- {np.nanstd(null):.4f}")
    print(f"  p-value                      : {p:.4f}")
    assoc.to_csv(outdir / f"batting_association_{args.session}.csv", index=False)

    # ---- 2. prediction, fitted on the first half of each game -------------
    print()
    print("=== PREDICTION: orientation fitted on each game's first half ===")
    rows = []
    for (slug, ser), g in d.groupby(["slug", "series"]):
        g = g.sort_values("ts")
        split = g.ts.min() + (g.ts.max() - g.ts.min()) / 2
        tr, te = g[g.ts < split], g[g.ts >= split]
        if len(tr) < 20 or len(te) < 10:
            continue
        # does this token go up when the home side (bottom) bats?
        up_bot = (tr.loc[~tr.is_top, "signed"] > 0).mean() if (~tr.is_top).any() else 0.5
        up_top = (tr.loc[tr.is_top, "signed"] > 0).mean() if (tr.is_top).any() else 0.5
        pred = np.where(te.is_top,
                        np.where(up_top >= 0.5, 1, -1),
                        np.where(up_bot >= 0.5, 1, -1))
        rows.append(pd.DataFrame(dict(slug=slug, series=ser,
                                      market_type=te.market_type.to_numpy(),
                                      hit=(pred == te.signed.to_numpy()))))
    pred = pd.concat(rows, ignore_index=True)
    acc = pred.hit.mean()
    n = len(pred)
    se = np.sqrt(acc * (1 - acc) / n)
    print(f"  held-out jumps : {n:,}")
    print(f"  accuracy       : {acc:.1%}  95% CI [{acc-1.96*se:.1%}, {acc+1.96*se:.1%}]")
    print(f"  coin flip 50%  |  profitable from ~75% (SELECTOR.md)")
    print()
    print(pred.groupby("market_type").hit.agg(["size", "mean"]).round(3).to_string())
    pred.to_csv(outdir / f"batting_prediction_{args.session}.csv", index=False)

    draw(assoc, null, obs, acc, n, outdir / f"batting_team_direction_{args.session}.png",
         args.session, p)


def draw(assoc, null, obs, acc, n, path, session, pval):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), facecolor=SURFACE)

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID_C, lw=0.8)
        ax.set_axisbelow(True)

    ax = axes[0]; style(ax)
    ax.scatter(assoc.p_up_top, assoc.p_up_bottom, s=34, color=REAL_C,
               edgecolors=SURFACE, linewidths=0.8)
    lim = [min(assoc.p_up_top.min(), assoc.p_up_bottom.min()) - 0.05,
           max(assoc.p_up_top.max(), assoc.p_up_bottom.max()) + 0.05]
    ax.plot(lim, lim, color=INK2, lw=1.1, ls=(0, (4, 3)))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("If whose turn it is mattered, these\nwould separate off the diagonal",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel("P(jump up) when the away side bats", color=INK2, fontsize=9)
    ax.set_ylabel("P(jump up) when the home side bats", color=INK2, fontsize=9)

    ax = axes[1]; style(ax)
    ax.hist(null, bins=40, color=NULL_C, lw=0, alpha=0.85,
            label="permutation null")
    ax.axvline(obs, color=REAL_C, lw=2.4, label="observed")
    ax.set_title(f"Association vs a shuffled null\np = {pval:.3f}",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel("mean |P(up|bottom) - P(up|top)|", color=INK2, fontsize=9)
    ax.set_ylabel("permutations", color=INK2, fontsize=9)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)

    ax = axes[2]; style(ax)
    ax.bar([0], [acc], width=0.45, color=REAL_C)
    ax.axhline(0.5, color=INK2, lw=1.1, ls=(0, (4, 3)))
    ax.axhline(0.75, color=NULL_C, lw=1.8)
    ax.annotate("coin flip", (0.42, 0.505), fontsize=8.5, color=INK2, ha="right")
    ax.annotate("profitable from here", (0.42, 0.757), fontsize=8.5,
                color=NULL_C, ha="right", fontweight="bold")
    ax.text(0, acc + 0.012, f"{acc:.1%}\nn={n:,}", ha="center", va="bottom",
            fontsize=10, color=INK, linespacing=1.4)
    ax.set_xticks([0]); ax.set_xticklabels(["batting-team rule"], fontsize=9.5)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0.4, 0.82)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Held-out directional accuracy\n(orientation fitted on the first half)",
                 color=INK, fontsize=10.5, loc="left", pad=8)

    fig.suptitle("Does knowing who is batting tell you which way a jump goes?",
                 color=INK, fontsize=13, x=0.006, y=0.985, ha="left")
    fig.text(0.006, 0.925, session, color=INK2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
