"""
Does a game-state win-probability model disagree with the market?

The premise of a forecasting strategy is that you can price a game better than
Polymarket can. The bar for that is high and now measured: on 65 independent
MLB games the market priced 29.6% where the outcome ran 28.8%. It is
calibrated. So the question is not "can I be accurate" but "can I be MORE
accurate than a market that is already right on average".

The cheapest way to find out is to compare the market against the crudest
possible baseline: a win-probability model that knows only the inning and the
score difference. That model uses no scouting, no pitcher quality, no lineup,
no weather -- nothing the market does not also have, and much less. Two
possible outcomes, both informative:

  * the market tracks the baseline closely -> it is at least as good as a
    state model, and beating it requires genuine private edge, not better
    arithmetic on public state.
  * the market deviates systematically from the baseline -> there is a
    behavioural pattern (over-reaction to a run just scored, say) that a
    disciplined model could trade.

Method: join the MLB play-by-play (data/plays, timestamped to the in-play
pitch) to the recorded moneyline book, so each play gives a game state and the
price the market was showing at that moment. The empirical win rate for each
state comes from the same games, which is circular for calibration but fine
for the comparison being made here -- both series are scored against the same
outcomes.

Home team is parsed from the slug: mlb-{away}-{home}-{date}.

    python winprob_vs_market.py
"""
import collections
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")
PLAYS = os.path.join(BASE, "data", "plays")
TOKMAP = os.path.join(BASE, "data", "token_map.json")

DIFF_BINS = [-99, -4, -2, -1, 0, 1, 2, 4, 99]      # home minus away
INN_BINS = [1, 4, 6, 7, 8, 9, 99]


def iso(s):
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def bin_of(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return len(edges) - 2


def main():
    tokens = {}
    if os.path.exists(TOKMAP):
        with open(TOKMAP, "r", encoding="utf-8") as f:
            tokens = json.load(f)["tokens"]

    # state[(inn_bin, diff_bin)] -> [n, home_wins, sum_market_p_home]
    state = collections.defaultdict(lambda: np.zeros(3))
    # DISTINCT GAMES per cell -- the honest sample size. Many plays come from
    # one game, so a cell with n=152 plays may rest on a dozen independent
    # results. Counting plays as observations is the same error that made 58
    # correlated crypto markets look like a favourite-longshot bias.
    cell_games = collections.defaultdict(set)
    n_games = n_obs = 0

    for path in sorted(glob.glob(os.path.join(PANEL, "*.parquet"))):
        slug = os.path.basename(path)[:-8]
        pf = os.path.join(PLAYS, f"{slug}.json")
        if not os.path.exists(pf):
            continue
        parts = slug.split("-")
        if len(parts) < 5:
            continue
        away, home = parts[1], parts[2]

        with open(pf, "r", encoding="utf-8") as f:
            plays = json.load(f)["plays"]
        if not plays:
            continue

        cols = ["t", "mid_c", "bb_c", "ba_c", "asset_id", "market_type"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        ok = (np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"])
              & (d["ba_c"] > d["bb_c"]) & (d["market_type"] == "moneyline"))
        if ok.sum() < 5000:
            continue

        # pick the moneyline token whose outcome names the HOME side
        chosen = None
        for aid in np.unique(d["asset_id"][ok]):
            info = tokens.get(aid) or {}
            name = (info.get("outcome_name") or "").lower()
            q = (info.get("question") or "").lower()
            # question is "Away vs. Home"; the home team is the later name
            if not name or " vs" not in q:
                continue
            tail = q.split(" vs")[-1]
            if name and name.split()[-1] in tail:
                chosen = aid
                break
        if chosen is None:
            continue

        sel = ok & (d["asset_id"] == chosen)
        t = d["t"][sel]
        mid = d["mid_c"][sel]
        if t.size < 2000:
            continue

        # final result from the last scoring-bearing play
        last = [p for p in plays if p.get("awayScore") is not None]
        if not last:
            continue
        fa, fh = last[-1]["awayScore"], last[-1]["homeScore"]
        if fa == fh:
            continue
        home_won = 1 if fh > fa else 0
        n_games += 1

        for p in plays:
            ts = iso(p.get("anchor_utc"))
            if ts is None or p.get("awayScore") is None:
                continue
            if ts < t[0] or ts > t[-1]:
                continue
            k = int(np.searchsorted(t, ts, side="right")) - 1
            if k < 0:
                continue
            diff = p["homeScore"] - p["awayScore"]
            inn = p.get("inning") or 1
            b = (bin_of(inn, INN_BINS), bin_of(diff, DIFF_BINS))
            rec = state[b]
            rec[0] += 1
            rec[1] += home_won
            rec[2] += mid[k] / 100.0
            cell_games[b].add(slug)
            n_obs += 1

    print()
    print("=" * 96)
    print("MARKET PRICE vs A MINIMAL GAME-STATE MODEL")
    print(f"{n_games} games, {n_obs:,} play-level observations")
    print("The state model knows ONLY the inning and the score difference.")
    print("If the market matches it, the market is at least as good as public state.")
    print("=" * 96)
    print(f"  {'inning':>10} {'home lead':>11} {'plays':>7} {'GAMES':>6} "
          f"{'market P(home)':>16} {'actual':>9} {'gap':>8}")
    print("  " + "-" * 76)
    gaps = []
    for b in sorted(state):
        r = state[b]
        if r[0] < 40:
            continue
        n, wins, psum = r
        mk = psum / n
        ac = wins / n
        gap = ac - mk
        gaps.append((abs(gap), gap, n))
        ib, db = b
        inn = f"{INN_BINS[ib]}-{INN_BINS[ib+1]-1}" if INN_BINS[ib + 1] < 99 else f"{INN_BINS[ib]}+"
        lo, hi = DIFF_BINS[db], DIFF_BINS[db + 1]
        lead = f"{lo:+d}..{hi-1:+d}" if abs(lo) < 90 and abs(hi) < 90 else (
            f"<={hi-1:+d}" if lo < -90 else f">={lo:+d}")
        ng = len(cell_games[b])
        print(f"  {inn:>10} {lead:>11} {int(n):>7,} {ng:>6} {mk*100:>15.1f}% "
              f"{ac*100:>8.1f}% {gap*100:>+7.1f}")
    if gaps:
        w = sum(g[2] for g in gaps)
        mae = sum(g[0] * g[2] for g in gaps) / w
        bias = sum(g[1] * g[2] for g in gaps) / w
        print()
        print(f"  weighted mean |gap| {mae*100:.2f} pp   signed bias {bias*100:+.2f} pp")
        print("  A small |gap| means the market already prices game state correctly,")
        print("  and a state-only model has nothing to add.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
