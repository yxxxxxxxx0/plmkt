"""
Does the MLB play-by-play feed tell you which way a jump goes -- and in time?

Justin's question: pull the live match data, see whose turn it is, and use that
for direction. Direction is the binding constraint (SELECTOR.md section 7): the
selector works, the side does not, and nothing is profitable below ~75%
directional accuracy.

There are two different claims hiding in "use the game state", and they have
opposite prospects:

  STATE     "the home team is batting with two on" -- the market already knows
            this. The price IS the market's estimate given the state, so state
            alone should carry no direction. Testing it is still worth one
            measurement, because that is an assumption until measured.

  TIMING    "a run just scored" -- worth everything, but only if the feed says
            so BEFORE the book reprices. This is a latency race, and it is
            decidable from the recordings: statsapi's feed/live carries
            millisecond startTime/endTime per play and works on finished games,
            so the play clock and the book clock can be put on the same axis.

So the decisive measurement is the SIGN OF THE LAG, not any model score. If
jumps land before the play appears in the feed, the feed is a reaction and
there is nothing to trade. If they land after, the gap is the edge and its size
is the whole story.

The null matters as much as the signal. Jumps are frequent and plays are
frequent, so some alignment happens by coincidence; every lag statistic here is
reported against the same statistic computed on times drawn uniformly from the
same game, which is what "no relationship" actually looks like.

    python align_mlb_plays.py --session books_2026-09-11

Outputs, under results/makinen/mlb_plays/
  plays_<session>.csv     every play, with epoch ms, half-inning and outcome
  lag_<session>.csv       per-jump lag to the nearest preceding play
  lag_summary_<session>.csv
  play_alignment_<session>.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

SCHED = "https://statsapi.mlb.com/api/v1/schedule"
FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
GRID = "#e9e8e4"
REAL_C = "#2a78d6"     # reference palette slot 1
NULL_C = "#eb6834"     # slot 2


def iso_to_ms(s: str) -> float:
    if not s:
        return np.nan
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000.0


def slug_to_gamepk(slugs: list[str], us_date: str) -> dict[str, int]:
    """Map each recorded slug to its gamePk via the abbreviations plan_slate uses."""
    import plan_slate as ps

    games = ps.schedule(us_date)
    by_pair = {}
    for g in games:
        by_pair[frozenset((g["away_abbr"].lower(), g["home_abbr"].lower()))] = g["gamePk"]
    out = {}
    for s in slugs:
        parts = s.split("-")
        # mlb-<away>-<home>-<YYYY>-<MM>-<DD>
        pair = frozenset((parts[1], parts[2]))
        pk = by_pair.get(pair)
        if pk:
            out[s] = pk
    return out


def fetch_plays(slug: str, pk: int) -> pd.DataFrame:
    f = requests.get(FEED.format(pk=pk), timeout=90).json()
    rows = []
    for p in f["liveData"]["plays"]["allPlays"]:
        ab, res = p["about"], p["result"]
        rows.append(dict(
            slug=slug, gamePk=pk,
            inning=ab.get("inning"),
            half=ab.get("halfInning"),
            is_top=ab.get("isTopInning"),
            start_ms=iso_to_ms(ab.get("startTime")),
            end_ms=iso_to_ms(ab.get("endTime")),
            is_scoring=bool(ab.get("isScoringPlay")),
            has_out=bool(ab.get("hasOut")),
            event=res.get("event"),
            away_score=res.get("awayScore"),
            home_score=res.get("homeScore"),
            description=(res.get("description") or "")[:120],
        ))
    return pd.DataFrame(rows)


def lag_to_previous(jump_ts: np.ndarray, play_ts: np.ndarray) -> np.ndarray:
    """Seconds from the most recent play at or before each jump. NaN if none."""
    if len(play_ts) == 0:
        return np.full(len(jump_ts), np.nan)
    order = np.sort(play_ts)
    idx = np.searchsorted(order, jump_ts, side="right") - 1
    out = np.where(idx >= 0, (jump_ts - order[np.clip(idx, 0, None)]) / 1000.0, np.nan)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="books_2026-09-11")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "mlb_plays"))
    ap.add_argument("--window", type=float, default=120.0,
                    help="seconds after a play to count as 'following' it")
    ap.add_argument("--price-driven-only", action="store_true",
                    help="keep only jumps classify_jump_cause.py calls real "
                         "price moves. 95.3%% of jumps are the spread moving, "
                         "and a quote being pulled has no reason to follow a "
                         "play, so the unfiltered population dilutes whatever "
                         "relationship exists")
    ap.add_argument("--scoring-only", action="store_true",
                    help="align against scoring plays only -- the ones with an "
                         "unambiguous reason to move a win probability")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ev = pd.read_parquet(ROOT / "results" / "makinen" / "oracle_jumps" / "events.parquet")
    ev = ev[(ev.session == args.session) & ev.is_jump].copy()
    if ev.empty:
        raise SystemExit(f"no jumps for {args.session}")

    if args.price_driven_only:
        cause_path = ROOT / "results" / "makinen" / "oracle_jumps" / "jump_cause.csv"
        cz = pd.read_csv(cause_path, usecols=["session", "slug", "series", "ts", "cause"])
        before = len(ev)
        ev = ev.merge(cz, on=["session", "slug", "series", "ts"], how="inner")
        ev = ev[ev.cause == "price"]
        print(f"price-driven filter: {before:,} -> {len(ev):,} jumps")
        if ev.empty:
            raise SystemExit("no price-driven jumps left")
    us_date = args.session.replace("books_", "")
    slugs = sorted(ev.slug.unique())
    print(f"{args.session}: {len(ev):,} jumps across {len(slugs)} games")

    pks = slug_to_gamepk(slugs, us_date)
    print(f"matched {len(pks)} of {len(slugs)} slugs to a gamePk")
    missing = [s for s in slugs if s not in pks]
    if missing:
        print(f"  unmatched: {missing}")

    frames = []
    for slug, pk in pks.items():
        try:
            frames.append(fetch_plays(slug, pk))
            time.sleep(0.4)  # be polite to statsapi
        except Exception as e:
            print(f"  [warn] {slug} (pk {pk}): {type(e).__name__}: {e}")
    plays = pd.concat(frames, ignore_index=True)
    plays.to_csv(outdir / f"plays_{args.session}.csv", index=False)
    print(f"{len(plays):,} plays fetched; {int(plays.is_scoring.sum())} scoring")

    # --- the decisive measurement: where does a jump sit relative to a play? --
    rows = []
    rng = np.random.default_rng(0)
    for slug, g in ev.groupby("slug"):
        pl = plays[plays.slug == slug]
        if args.scoring_only:
            pl = pl[pl.is_scoring]
        if pl.empty:
            continue
        jt = g.ts.to_numpy(dtype=float)
        pe = pl.end_ms.dropna().to_numpy(dtype=float)
        # The null: same count of times, uniform over the same recorded span.
        # Anything that survives this is not "plays and jumps are both common".
        nt = rng.uniform(jt.min(), jt.max(), size=len(jt))
        rows.append(pd.DataFrame(dict(
            slug=slug, ts=jt, signed=g.signed.to_numpy(),
            market_type=g.market_type.to_numpy(),
            lag_s=lag_to_previous(jt, pe),
            null_lag_s=lag_to_previous(nt, pe))))
    lag = pd.concat(rows, ignore_index=True)
    lag.to_csv(outdir / f"lag_{args.session}.csv", index=False)

    w = args.window
    real_in = (lag.lag_s >= 0) & (lag.lag_s <= w)
    null_in = (lag.null_lag_s >= 0) & (lag.null_lag_s <= w)
    print()
    print(f"=== do jumps follow plays? (window {w:.0f}s) ===")
    print(f"  jumps within {w:.0f}s after a play : {real_in.mean():.1%}")
    print(f"  the same for random times         : {null_in.mean():.1%}")
    print(f"  lift                              : {real_in.mean()/max(null_in.mean(),1e-9):.2f}x")
    print()
    print("  lag to previous play, seconds:")
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"    jumps  {[round(v,1) for v in lag.lag_s.quantile(q)]}")
    print(f"    random {[round(v,1) for v in lag.null_lag_s.quantile(q)]}")

    summary = pd.DataFrame({
        "session": [args.session],
        "n_jumps": [len(lag)],
        "n_plays": [len(plays)],
        "share_within_window": [real_in.mean()],
        "share_within_window_null": [null_in.mean()],
        "lift": [real_in.mean() / max(null_in.mean(), 1e-9)],
        "median_lag_s": [lag.lag_s.median()],
        "median_lag_s_null": [lag.null_lag_s.median()],
    })
    summary.to_csv(outdir / f"lag_summary_{args.session}.csv", index=False)

    draw(lag, plays, outdir / f"play_alignment_{args.session}.png", args.session, w)


def draw(lag, plays, path, session, w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=SURFACE)

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    ax = axes[0]; style(ax)
    bins = np.arange(0, 301, 5)
    ax.hist(lag.lag_s.dropna().clip(0, 300), bins=bins, color=REAL_C, alpha=0.85,
            lw=0, label="jumps")
    ax.hist(lag.null_lag_s.dropna().clip(0, 300), bins=bins, color=NULL_C,
            alpha=0.55, lw=0, label="random times (null)")
    ax.set_title("Time from the previous play to a jump", color=INK,
                 fontsize=11, loc="left", pad=8)
    ax.set_xlabel("seconds since the play ended", color=INK2, fontsize=9)
    ax.set_ylabel("count", color=INK2, fontsize=9)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)

    ax = axes[1]; style(ax)
    ws = [5, 10, 15, 30, 60, 120, 300]
    real = [((lag.lag_s >= 0) & (lag.lag_s <= x)).mean() for x in ws]
    null = [((lag.null_lag_s >= 0) & (lag.null_lag_s <= x)).mean() for x in ws]
    x = np.arange(len(ws))
    ax.bar(x - 0.19, real, width=0.36, color=REAL_C, label="jumps")
    ax.bar(x + 0.19, null, width=0.36, color=NULL_C, label="random times (null)")
    ax.set_xticks(x); ax.set_xticklabels([f"{v}s" for v in ws], fontsize=9)
    for i, (r, n) in enumerate(zip(real, null)):
        ax.text(i, max(r, n) + 0.015, f"{r/max(n,1e-9):.2f}x", ha="center",
                va="bottom", fontsize=8.5, color=INK)
    ax.set_ylim(0, max(max(real), max(null)) * 1.25)
    ax.set_title("Share of jumps falling within N seconds of a play",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.set_ylabel("share", color=INK2, fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle(f"Do price jumps follow MLB plays?   {session}   "
                 f"{len(lag):,} jumps, {len(plays):,} plays",
                 color=INK, fontsize=12.5, x=0.006, y=0.985, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
