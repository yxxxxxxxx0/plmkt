"""Two whole games, every market, jumps marked, against the real play-by-play.

This is the visual verification step. A detected jump is only credible if it
lines up with something that actually happened in the baseball game, so this
plots, for one match at a time:

  * the 1-minute mid of each of the game's main markets (moneyline, run line,
    total), stacked on a shared clock,
  * every Lee-Mykland jump marked on the price it was detected at, and
  * the actual scoring plays fetched from the MLB Stats API.

Two independent checks fall out of the layout. Markets on the same game should
jump in the SAME minute, because one run moves all of them; and those minutes
should coincide with a run actually crossing the plate. Neither check is used
to select or filter jumps -- they are only drawn, so a jump that lines up with
nothing is still shown.

    python plot_match_examples.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.csv as pacsv  # noqa: E402
import requests  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LIVE = os.path.join(ROOT, "data", "live")
RES = os.path.join(ROOT, "results", "lee_mykland")
OUT = os.path.join(RES, "match_examples")
HKT_MS = 8 * 3600 * 1000
FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def market_questions(session, slug):
    """asset_id -> human market question, read from the raw feed for one game."""
    p = os.path.join(LIVE, "top_of_book_%s.csv" % session)
    if not os.path.exists(p):
        return {}
    t = pacsv.read_csv(
        p, read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            include_columns=["event_slug", "asset_id", "market_question",
                             "outcome"],
            column_types={"asset_id": pa.string()})).to_pandas()
    t = t[t.event_slug == slug]
    t = t.drop_duplicates("asset_id")
    return dict(zip(t.asset_id, t.market_question + "  [" + t.outcome + "]"))


def scoring_plays(pk):
    """(timestamp_ms, description, score) for each run-scoring play."""
    try:
        j = requests.get(FEED.format(pk=pk), timeout=30).json()
    except Exception as e:
        log("  play-by-play fetch failed: %s" % e)
        return []
    out = []
    for p in j.get("liveData", {}).get("plays", {}).get("allPlays", []):
        if not p.get("about", {}).get("isScoringPlay"):
            continue
        end = p.get("about", {}).get("endTime") or p.get("about", {}).get("startTime")
        if not end:
            continue
        ms = pd.Timestamp(end).value // 10**6
        r = p.get("result", {})
        out.append((ms, r.get("description", "")[:70],
                    "%s-%s" % (r.get("awayScore"), r.get("homeScore"))))
    return out


def pick_series(D, slug):
    """One representative asset per market type: the most-quoted YES leg."""
    g = D[(D.event_slug == slug) & (D.outcome == "YES")]
    out = []
    for mt in ("moneyline", "spread", "total"):
        s = g[g.market_type == mt]
        if not len(s):
            continue
        # dropna=False matters: moneyline has line = NaN, and a default
        # groupby silently drops it -- which is why the moneyline panel was
        # missing from the first version of this figure.
        best = (s.groupby(["asset_id", "line"], dropna=False).n_events.sum()
                .sort_values(ascending=False))
        if not len(best):
            continue
        aid, line = best.index[0]
        out.append((mt, line, aid))
    return out


def plot_match(D, slug, pk, qmap, plays, path):
    sel = pick_series(D, slug)
    if not sel:
        return False
    fig, axes = plt.subplots(len(sel), 1, figsize=(15, 3.3 * len(sel)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    n_j = 0
    for ax, (mt, line, aid) in zip(axes, sel):
        s = D[D.asset_id == aid].sort_values("minute")
        has = np.isfinite(s.mid.to_numpy())
        ax.plot(s.ts_hkt[has], s.mid[has], lw=1.1, color="#1f77b4", zorder=2,
                label="1-min mid")
        up = s[s.jump_direction == 1]
        dn = s[s.jump_direction == -1]
        n_j += len(up) + len(dn)
        ax.scatter(up.ts_hkt, up.mid, marker="^", s=95, color="#2ca02c",
                   edgecolor="black", linewidth=0.5, zorder=4,
                   label="positive jump (%d)" % len(up))
        ax.scatter(dn.ts_hkt, dn.mid, marker="v", s=95, color="#d62728",
                   edgecolor="black", linewidth=0.5, zorder=4,
                   label="negative jump (%d)" % len(dn))
        for ms, desc, score in plays:
            ax.axvline(pd.Timestamp(ms + HKT_MS, unit="ms"), color="#777777",
                       ls=":", lw=1.0, zorder=1)
        q = qmap.get(aid, "%s %s" % (mt, line))
        ax.set_title(q, fontsize=9, loc="left")
        ax.set_ylabel("mid")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
    # annotate the scoring plays once, along the top
    if plays:
        a0 = axes[0]
        for ms, desc, score in plays:
            a0.annotate(score, xy=(pd.Timestamp(ms + HKT_MS, unit="ms"), 1.0),
                        xytext=(0, 4), textcoords="offset points",
                        fontsize=7, color="#333333", ha="center")
    axes[-1].set_xlabel("time (HKT).  Dotted grey lines = actual run-scoring "
                        "plays from the MLB Stats API")
    fig.suptitle("%s   |   Lee-Mykland jumps vs the real game   "
                 "(K=30, alpha=0.01, arithmetic returns, in-game only)   "
                 "|   %d jumps shown, %d scoring plays"
                 % (slug, n_j, len(plays)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.autofmt_xdate()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*",
                    default=["mlb-cws-stl-2026-09-12", "mlb-phi-atl-2026-09-11"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    D = pd.read_csv(os.path.join(RES, "lee_mykland_jumps.csv"),
                    dtype={"asset_id": str})
    D["ts_hkt"] = pd.to_datetime(D.ts_hkt)
    W = json.load(open(os.path.join(ROOT, "data", "game_windows.json"),
                       encoding="utf-8"))

    for slug in a.slugs:
        sub = D[D.event_slug == slug]
        if not len(sub):
            log("no rows for %s" % slug)
            continue
        session = sub.session.iloc[0]
        pk = W.get(slug, {}).get("gamePk")
        log("%s (session %s, gamePk %s): %d assets, %d jumps"
            % (slug, session, pk, sub.asset_id.nunique(), int(sub.jump.sum())))
        qmap = market_questions(session, slug)
        plays = scoring_plays(pk) if pk else []
        log("  %d run-scoring plays" % len(plays))
        p = os.path.join(OUT, "%s.png" % slug)
        if plot_match(D, slug, pk, qmap, plays, p):
            log("  wrote %s" % p)


if __name__ == "__main__":
    sys.exit(main())
