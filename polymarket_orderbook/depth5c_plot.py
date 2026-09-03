"""
Per-match depth diagram, restricted to liquidity within 5 cents of the best.

Three panels:

  1. Level-by-level depth   resting volume q against distance d from the best
                            price, d = BB-p on the bid, d = p-BA on the ask.
                            Tick is 1 cent, so d falls in 6 buckets, 0..5c.
  2. Cumulative depth       Q_bid(x) = sum of q over p in [BB-x, BB]
                            Q_ask(x) = sum of q over p in [BA, BA+x]
                            for 0 <= x <= 0.05.
  3. Total 5c depth vs time V_bid,5c(t) and V_ask,5c(t).

Panels 1 and 2 are cross-sectional: they describe the SHAPE of the book, which
depends on t. Picking one instant would be arbitrary, so both are aggregated
over the whole match on a 1-second grid -- the line is the median across
samples and the band is the inter-quartile range. Median rather than mean
because resting size is heavily right-skewed; a single 300k-share wall would
otherwise dominate the average and misrepresent the typical book.

Samples where the relevant side is empty are excluded from the shape panels
(there is no distance-from-best when there is no best) but still count as zero
in panel 3, where an empty book is a real measurement rather than a gap.

Usage:
    python depth5c_plot.py --match mlb-ari-sf-2026-08-27
    python depth5c_plot.py --all
    python depth5c_plot.py --all --market total --outcome NO
"""
import argparse
import bisect
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import orjson

BASE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(BASE, "data", "viewer_full")
OUT = os.path.join(BASE, "data", "plots")

TICK = 0.01
MAXD = 0.05
NB = int(round(MAXD / TICK)) + 1          # 6 buckets: 0,1,2,3,4,5 cents
DIST = np.arange(NB) * TICK
CENT = "¢"


def load_runs(slug, asset_id):
    """Runs for one token. Chunks repeat boundary-straddling runs, so dedupe."""
    adir = os.path.join(VIEWER, slug, asset_id)
    seen, out = set(), []
    for path in sorted(glob.glob(os.path.join(adir, "c*.json"))):
        with open(path, "rb") as f:
            for r in orjson.loads(f.read()):
                t = r["t_start"]
                if t in seen:
                    continue
                seen.add(t)
                out.append(r)
    out.sort(key=lambda r: r["t_start"])
    return out


def bucketise(levels, best, is_bid):
    """Volume per 1c bucket within 5c of best. Levels beyond 5c are dropped."""
    v = np.zeros(NB)
    for p, q in levels:
        d = (best - p) if is_bid else (p - best)
        if -1e-9 <= d <= MAXD + 1e-9:
            v[int(round(d / TICK))] += q
    return v


def build(match, market_type, outcome, outdir):
    slug = match["event_slug"]
    mkts = match["market_types"].get(market_type)
    if not mkts:
        return None, "no " + market_type + " market"
    tokens = mkts[0]["tokens"]
    tok = next((t for t in tokens if t["outcome"] == outcome), tokens[0])

    runs = load_runs(slug, tok["asset_id"])
    if not runs:
        return None, "no runs"
    ts = [r["t_start"] for r in runs]

    grid = np.arange(match["start_ts"], match["end_ts"], 1.0)
    bid_rows, ask_rows = [], []
    vb5 = np.zeros(len(grid))
    va5 = np.zeros(len(grid))

    for i, t in enumerate(grid):
        j = bisect.bisect_right(ts, t) - 1
        if j < 0:
            continue
        r = runs[j]
        if r["best_bid"] is not None and r["bids"]:
            b = bucketise(r["bids"], r["best_bid"], True)
            bid_rows.append(b)
            vb5[i] = b.sum()
        if r["best_ask"] is not None and r["asks"]:
            a = bucketise(r["asks"], r["best_ask"], False)
            ask_rows.append(a)
            va5[i] = a.sum()

    if not bid_rows and not ask_rows:
        return None, "book never populated"
    B = np.array(bid_rows) if bid_rows else np.zeros((1, NB))
    A = np.array(ask_rows) if ask_rows else np.zeros((1, NB))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [1, 1, 1.15],
                                          "hspace": 0.33})
    fig.suptitle(match["match_name"] + "   -   " + str(match["date"]) +
                 "   -   " + market_type + " / " + tok["label"] +
                 "   -   depth within 5" + CENT,
                 fontsize=12, fontweight="bold", y=0.965)

    # 1. level by level
    ax = axes[0]
    w = 0.38
    xs = np.arange(NB)
    ax.bar(xs - w / 2, np.median(B, 0), w, color="#1a7f37", label="bid")
    ax.bar(xs + w / 2, np.median(A, 0), w, color="#cf222e", label="ask")
    ax.vlines(xs - w / 2, np.percentile(B, 25, 0), np.percentile(B, 75, 0),
              color="#0b3d1c", lw=1.4)
    ax.vlines(xs + w / 2, np.percentile(A, 25, 0), np.percentile(A, 75, 0),
              color="#7a1119", lw=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(round(d * 100))) + CENT for d in DIST])
    ax.set_xlabel("distance from best price,  d")
    ax.set_ylabel("resting volume  q")
    ax.set_title("1. Level-by-level depth   (median, whisker = IQR)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # 2. cumulative
    ax = axes[1]
    for M, c, lab in ((B, "#1a7f37", "$Q^{bid}(x)$"),
                      (A, "#cf222e", "$Q^{ask}(x)$")):
        C = np.cumsum(M, axis=1)
        ax.plot(DIST, np.median(C, 0), color=c, lw=1.8, marker="o", ms=4, label=lab)
        ax.fill_between(DIST, np.percentile(C, 25, 0), np.percentile(C, 75, 0),
                        color=c, alpha=0.15, lw=0)
    ax.set_xlabel("x   (distance from best price)")
    ax.set_ylabel("cumulative volume  Q(x)")
    ax.set_title("2. Cumulative depth curve,  0 <= x <= 0.05   (median, shaded = IQR)",
                 fontsize=10, loc="left")
    ax.set_xlim(0, MAXD)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    # 3. total 5c depth over time
    ax = axes[2]
    x = mdates.date2num(np.array(grid, dtype="datetime64[s]")
                        .astype("datetime64[ms]").tolist())
    ax.plot(x, vb5, lw=0.6, color="#1a7f37", label="$V_{bid,5c}(t)$")
    ax.plot(x, va5, lw=0.6, color="#cf222e", label="$V_{ask,5c}(t)$")
    ax.set_ylabel("volume within 5" + CENT)
    ax.set_xlabel("time (UTC)")
    ax.set_title("3. Total 5" + CENT + " depth over time", fontsize=10, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    fig.text(0.995, 0.004,
             "1 s grid - " + format(len(grid), ",") + " samples - tick 1" + CENT +
             " - panels 1 and 2 aggregated over the match",
             ha="right", fontsize=8, color="#666")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, slug + "__" + market_type + "_" +
                        tok["outcome"] + "__depth5c.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path, format(len(runs), ",") + " runs, " + format(len(grid), ",") + " samples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--market", default="moneyline")
    ap.add_argument("--outcome", default="YES")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    with open(os.path.join(VIEWER, "index.json"), "rb") as f:
        matches = orjson.loads(f.read())["matches"]
    if args.match:
        matches = [m for m in matches if m["event_slug"] == args.match]
        if not matches:
            raise SystemExit("no such match: " + args.match)
    elif not args.all:
        raise SystemExit("pass --match <slug> or --all")

    for i, m in enumerate(matches, 1):
        path, note = build(m, args.market, args.outcome, args.out)
        name = os.path.basename(path) if path else "SKIP"
        print("[" + str(i) + "/" + str(len(matches)) + "] " +
              m["event_slug"].ljust(26) + " " + name + "  (" + note + ")", flush=True)


if __name__ == "__main__":
    main()
