"""
At the instant a jump starts, do the OTHER markets on the same game tell you
which way it goes?

Justin's question, and it is the first direction idea in this project that needs
no outside feed. Polymarket lists ~8 correlated contracts per game -- moneyline,
spread and total, two tokens each -- and they are mechanically linked: runs move
the total and the moneyline together, and the two sides of any market are
complements. So if one contract reprices a moment before another, the direction
of the slow one is readable from the fast one.

Design, and the reason for each constraint:

  the label        sign of the jump's move from onset to peak (events.parquet
                   `signed`, verified identical to sign(peak_mid - entry_mid)).

  the feature      for every OTHER series Y on the same game, the signed move
                   of Y over [t - L, t], where mid_Y(t) is the last quote at or
                   BEFORE t. Nothing after the jump onset is read. This is
                   exactly what a live system can see at the moment it detects
                   the jump.

  the sign map     Y's move has to be oriented onto X before it can vote --
                   "total went up" only implies a direction for a moneyline
                   once you know the historical sign between them. That is
                   estimated as sign(corr) of their 1s returns, computed on the
                   FIRST HALF of the game, and applied only to jumps in the
                   SECOND half. Estimating it on the whole game would leak: the
                   correlation would be partly fitted on the very move being
                   predicted. Halving is cruder than an expanding window but it
                   cannot leak, which matters more here -- this project has
                   already produced one ROC-AUC 1.0000 from a feature that saw
                   the future.

  the vote         sum over Y of corr_XY * move_Y, predicted direction is its
                   sign. Deliberately a sum of oriented moves and not a fitted
                   model: if a trained classifier beats it that is worth
                   knowing, but the raw lead-lag has to exist first.

  TWO CONTROLS     and they are the whole result, so they run by default.
                   (1) the jumping market's OWN prior move, same rule. If that
                   calls direction as well as the sisters do, the cross-market
                   vote is an echo rather than information.
                   (2) the same rule at RANDOM in-game times. If fading the
                   last L seconds predicts the next 10s just as well anywhere,
                   the effect is not about jumps at all.

The bar is not 50%. SELECTOR.md: nothing is profitable below ~75% directional
accuracy at any selection precision, so 55% is interesting evidence and still
not a trade.

Reported as an UPPER BOUND on what the idea can do, for two reasons: the jump
onset is used as the detection time, whereas a live detector fires some
milliseconds later, and fills are assumed at the touch.

    python cross_market_direction.py --session books_2026-09-11

Outputs, under results/makinen/cross_market/
  direction_by_lookback_<session>.csv
  cross_market_direction_<session>.png
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

GRID_MS = 1000.0
TICK = 0.01
LOOKBACKS = [1, 2, 5, 10, 30, 60]

SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
GRID_C = "#e9e8e4"
ACC_C = "#2a78d6"
BAR_C = "#eb6834"


def game_grid(g: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """1s forward-filled mid grid for every series on one game."""
    series = sorted(g.series.unique())
    t0 = float(g.ts.min())
    t1 = float(g.ts.max())
    times = np.arange(t0, t1 + GRID_MS, GRID_MS)
    M = np.full((len(times), len(series)), np.nan)
    for k, s in enumerate(series):
        d = g[g.series == s].sort_values("ts")
        ts = d.ts.to_numpy(dtype=float)
        mid = d.mid.to_numpy(dtype=float)
        idx = np.searchsorted(ts, times, side="right") - 1
        ok = idx >= 0
        M[ok, k] = mid[idx[ok]]
    return times, series, M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="books_2026-09-11")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "cross_market"))
    ap.add_argument("--price-driven-only", action="store_true")
    ap.add_argument("--corr-horizon", type=float, default=10.0,
                    help="seconds of return used to estimate the sign map. At "
                         "1s these contracts are nearly uncorrelated and the "
                         "sign is noise; the economic link needs a longer bar")
    ap.add_argument("--min-abs-corr", type=float, default=0.10,
                    help="ignore pairs weaker than this. sign() of a 0.01 "
                         "correlation is a coin flip that then gets multiplied "
                         "by a large move")
    ap.add_argument("--weight", choices=["corr", "sign"], default="corr",
                    help="corr: weight each vote by rho, so weak pairs barely "
                         "count. sign: the crude +/-1 version")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ev = pd.read_parquet(ROOT / "results" / "makinen" / "oracle_jumps" / "events.parquet")
    ev = ev[(ev.session == args.session) & ev.is_jump].copy()
    if args.price_driven_only:
        cz = pd.read_csv(ROOT / "results" / "makinen" / "oracle_jumps" / "jump_cause.csv",
                         usecols=["session", "slug", "series", "ts", "cause"])
        ev = ev.merge(cz, on=["session", "slug", "series", "ts"], how="inner")
        ev = ev[ev.cause == "price"]
    print(f"{args.session}: {len(ev):,} jumps"
          f"{' (price-driven only)' if args.price_driven_only else ''}")

    feat = ROOT / "data" / "jump" / f"feat_{args.session}_trimmed.parquet"
    df = pd.read_parquet(feat, columns=["mid", "ts", "series", "valid"])
    df = df[df.valid].copy()
    df["slug"] = df.series.str.split("|").str[0]

    rows = []
    for slug, g in df.groupby("slug"):
        jr = ev[ev.slug == slug]
        if jr.empty or g.series.nunique() < 2:
            continue
        times, series, M = game_grid(g)
        sidx = {s: i for i, s in enumerate(series)}

        # Sign map from the FIRST half only, on --corr-horizon returns.
        half = len(times) // 2
        step = max(1, int(args.corr_horizon * 1000 / GRID_MS))
        ret = M[:half][step:] - M[:half][:-step]
        with np.errstate(invalid="ignore"):
            C = pd.DataFrame(ret).corr().to_numpy()
        C = np.nan_to_num(C)
        C[np.abs(C) < args.min_abs_corr] = 0.0
        S = np.sign(C) if args.weight == "sign" else C
        np.fill_diagonal(S, 0.0)

        # Evaluate only jumps in the SECOND half.
        t_split = times[half]
        jr = jr[jr.ts >= t_split]
        if jr.empty:
            continue
        jt = jr.ts.to_numpy(dtype=float)
        ji = np.searchsorted(times, jt, side="right") - 1
        keep = ji >= 0
        jr, jt, ji = jr[keep], jt[keep], ji[keep]
        if len(jr) == 0:
            continue
        xcol = np.array([sidx.get(s, -1) for s in jr.series])
        good = xcol >= 0
        jr, ji, xcol = jr[good], ji[good], xcol[good]

        for L in LOOKBACKS:
            back = np.maximum(ji - int(L * 1000 / GRID_MS), 0)
            moves = (M[ji] - M[back]) / TICK          # (n_jumps, n_series)
            moves = np.nan_to_num(moves)
            # Orient every other series onto the jumping one, then sum.
            orient = S[xcol]                          # (n_jumps, n_series)
            orient[np.arange(len(xcol)), xcol] = 0.0  # never vote with itself
            vote = np.einsum("ij,ij->i", orient, moves)
            rows.append(pd.DataFrame(dict(
                slug=slug, lookback_s=L, vote=vote,
                signed=jr.signed.to_numpy(),
                market_type=jr.market_type.to_numpy(),
                gross=jr.gross_peak.to_numpy())))

    res = pd.concat(rows, ignore_index=True)
    res["pred"] = np.sign(res.vote)
    res["hit"] = (res.pred == res.signed)
    decided = res[res.pred != 0]

    print()
    print("=== directional accuracy from the other markets on the same game ===")
    out = []
    for L, g in decided.groupby("lookback_s"):
        acc = g.hit.mean()
        n = len(g)
        se = np.sqrt(acc * (1 - acc) / n)
        cov = n / len(res[res.lookback_s == L])
        out.append(dict(lookback_s=L, n=n, coverage=cov, accuracy=acc,
                        lo=acc - 1.96 * se, hi=acc + 1.96 * se))
        print(f"  {L:>3}s lookback: {acc:.1%}  95% CI [{acc-1.96*se:.1%}, {acc+1.96*se:.1%}]"
              f"   n={n:,}  ({cov:.0%} of jumps get a non-zero vote)")
    summ = pd.DataFrame(out)
    summ.to_csv(outdir / f"direction_by_lookback_{args.session}.csv", index=False)

    best = summ.loc[summ.accuracy.idxmax()]
    print()
    print(f"  best: {best.accuracy:.1%} at {best.lookback_s:.0f}s")
    print(f"  coin flip 50%   |   profitable from ~75% (SELECTOR.md)")

    print()
    print("  by market type, at the best lookback:")
    b = decided[decided.lookback_s == best.lookback_s]
    print(b.groupby("market_type").hit.agg(["size", "mean"]).round(3).to_string())

    print()
    print("  does a bigger vote mean a better call?")
    b = b.copy()
    b["absvote"] = b.vote.abs()
    b["q"] = pd.qcut(b.absvote, 5, labels=False, duplicates="drop")
    print(b.groupby("q").agg(n=("hit", "size"), accuracy=("hit", "mean"),
                             median_abs_vote=("absvote", "median")).round(3).to_string())

    controls(df, ev, outdir, args)

    draw(summ, b, outdir / f"cross_market_direction_{args.session}.png",
         args.session, args.price_driven_only)


def controls(df, ev, outdir, args):
    """The two tests that decide whether the cross-market number means anything."""
    rng = np.random.default_rng(0)
    own_rows, rnd_p, rnd_f = [], {L: [] for L in LOOKBACKS}, {L: [] for L in LOOKBACKS}

    for slug, g in df.groupby("slug"):
        series = sorted(g.series.unique())
        sidx = {s: i for i, s in enumerate(series)}
        times, _, M = game_grid(g)
        half = len(times) // 2

        jr = ev[(ev.slug == slug) & (ev.ts >= times[half])]
        if not jr.empty:
            ji = np.searchsorted(times, jr.ts.to_numpy(float), side="right") - 1
            keep = ji >= 0
            jr2, ji = jr[keep], ji[keep]
            xc = np.array([sidx.get(s, -1) for s in jr2.series])
            ok = xc >= 0
            jr2, ji, xc = jr2[ok], ji[ok], xc[ok]
            for L in LOOKBACKS:
                back = np.maximum(ji - int(L), 0)
                own = np.nan_to_num((M[ji, xc] - M[back, xc]) / TICK)
                own_rows.append(pd.DataFrame(dict(lookback_s=L, own=own,
                                                  signed=jr2.signed.to_numpy())))

        lo, hi = half + 70, len(times) - 70
        if hi > lo:
            for k in range(M.shape[1]):
                col = M[:, k]
                a = rng.integers(lo, hi, size=min(600, hi - lo))
                for L in LOOKBACKS:
                    pr = (col[a] - col[a - int(L)]) / TICK
                    fw = (col[a + 10] - col[a]) / TICK      # next 10s
                    m = np.isfinite(pr) & np.isfinite(fw) & (pr != 0) & (fw != 0)
                    rnd_p[L].append(np.sign(pr[m])); rnd_f[L].append(np.sign(fw[m]))

    own = pd.concat(own_rows, ignore_index=True)
    own = own[own.own != 0]
    out = []
    print()
    print("=== CONTROL 1: the jumping market's OWN prior move ===")
    for L, g in own.groupby("lookback_s"):
        mom = (np.sign(g.own) == g.signed).mean()
        out.append(dict(control="own_market", lookback_s=L, n=len(g),
                        momentum=mom, reversal=1 - mom))
        print(f"  {L:>3}s: reversal {1-mom:.1%}   n={len(g):,}")
    print()
    print("=== CONTROL 2: the same rule at RANDOM in-game times ===")
    for L in LOOKBACKS:
        p = np.concatenate(rnd_p[L]); f = np.concatenate(rnd_f[L])
        mom = (p == f).mean()
        out.append(dict(control="random_times", lookback_s=L, n=len(p),
                        momentum=mom, reversal=1 - mom))
        print(f"  {L:>3}s: reversal {1-mom:.1%}   n={len(p):,}")
    pd.DataFrame(out).to_csv(
        outdir / f"direction_controls_{args.session}.csv", index=False)
    print()
    print("  If control 2 matches or beats control 1, the reversal is a property")
    print("  of the quoted mid everywhere -- bid-ask bounce -- and not a signal.")


def draw(summ, best, path, session, price_only):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), facecolor=SURFACE)

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
    x = np.arange(len(summ))
    ax.bar(x, summ.accuracy, width=0.55, color=ACC_C)
    ax.errorbar(x, summ.accuracy, yerr=[summ.accuracy - summ.lo, summ.hi - summ.accuracy],
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
    ax.axhline(0.5, color=INK2, lw=1.1, ls=(0, (4, 3)))
    ax.axhline(0.75, color=BAR_C, lw=1.6)
    ax.annotate("coin flip", (len(summ) - 0.4, 0.505), fontsize=8.5, color=INK2, ha="right")
    ax.annotate("profitable from here (~75%)", (len(summ) - 0.4, 0.757),
                fontsize=8.5, color=BAR_C, ha="right", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:.0f}s" for v in summ.lookback_s], fontsize=9)
    ax.set_ylim(0.4, 0.82)
    for i, v in enumerate(summ.accuracy):
        ax.text(i, v + 0.012, f"{v:.1%}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Direction called by the other markets on the same game",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("lookback used to read the other markets", color=INK2, fontsize=9)
    ax.set_ylabel("directional accuracy", color=INK2, fontsize=9)

    ax = axes[1]; style(ax)
    q = best.groupby("q").hit.agg(["size", "mean"]).reset_index()
    ax.bar(q["q"], q["mean"], width=0.55, color=ACC_C)
    ax.axhline(0.5, color=INK2, lw=1.1, ls=(0, (4, 3)))
    ax.axhline(0.75, color=BAR_C, lw=1.6)
    ax.set_xticks(q["q"]); ax.set_xticklabels(
        ["weakest", "", "middle", "", "strongest"], fontsize=9)
    ax.set_ylim(0.4, 0.82)
    for i, r in q.iterrows():
        ax.text(r["q"], r["mean"] + 0.012, f"{r['mean']:.1%}\nn={int(r['size']):,}",
                ha="center", va="bottom", fontsize=8.5, color=INK, linespacing=1.4)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Does a stronger cross-market signal call it better?",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("strength of the oriented cross-market move", color=INK2, fontsize=9)
    ax.set_ylabel("directional accuracy", color=INK2, fontsize=9)

    tag = "price-driven jumps only" if price_only else "all durable jumps"
    fig.suptitle("Can the sister markets tell you which way a jump goes?",
                 color=INK, fontsize=13, x=0.006, y=0.985, ha="left")
    fig.text(0.006, 0.925, f"{session}  -  {tag}  -  sign map fitted on the first "
             f"half of each game, evaluated on the second",
             color=INK2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
