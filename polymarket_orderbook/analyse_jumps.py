"""
Are price jumps preceded by makers pulling quotes -- and which jumps are
simply a run scoring?

Three questions, one pass over the book:

 1. CANCELLATION NEAR JUMPS. For every swing, how fast was liquidity leaving
    the touch in the seconds before it, relative to that token's own baseline?
    If makers step away before the move, intensity > 1 and there is something
    to watch in real time.

 2. AT THE TOUCH, NOT THE WHOLE BOOK. Removal is measured only at level 0 of
    each side, because that is the only price a resting order competes at.
    Deep 0.01/0.99 stubs churn constantly and can never trade; including them
    (as an earlier whole-book measurement did) inflates everything and hides
    the signal.

 3. SCORING VS EVERYTHING ELSE. Each jump is matched against the timestamped
    in-play pitch of every scoring play (fetch_plays.py). A jump inside the
    window after a run scored has a known cause. The rest are either slower
    information or noise -- and the difference between those two groups is
    the whole question of whether a jump is worth trading.

Removal is counted two ways, kept separate because they mean different things
to a resting order:
    same-price  size shrinking while the best price holds -- a fill or a
                cancel ahead of you in the queue
    wiped       the best price moving away entirely -- everything at that
                price is gone

Trades are subtracted where the tape allows, but note the tape's timestamps
are SECONDS while the book is microseconds, so this is a statistical
correction, not per-event attribution. It barely matters: trades are ~4% of
touch removal on moneylines and under 0.2% elsewhere.

    python analyse_jumps.py --slate 2026-08-27
    python analyse_jumps.py --games mlb-ari-sf-2026-08-27 --pre-window 5
"""
import argparse
import collections
import datetime as dt
import glob
import json
import os
import sys
import time

import numpy as np

import rebuild_duckdb as rd
import rebuild_multi as rm
from sim_jump_capture import best_price_column, zigzag_up_legs

BASE = os.path.dirname(os.path.abspath(__file__))
THETAS_C = (1.0, 2.0, 3.0, 5.0, 10.0)
LIQUID = ("moneyline", "spread", "total")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def iso_to_unix(s):
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_scoring(slug):
    p = os.path.join(BASE, "data", "plays", f"{slug}.json")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for pl in d["plays"]:
        if not pl["isScoringPlay"]:
            continue
        t = iso_to_unix(pl.get("anchor_utc"))
        if t:
            out.append(t)
    return sorted(out)


def load_trade_times(slug):
    """asset_id -> (sorted trade times in unix seconds, sizes), and the source.

    Prefers the live millisecond tape when it exists. The two sources are NOT
    equivalent and the difference decides what the numbers can be used for:

        data/trades_live/  captured by live_recorder, MILLISECOND stamps --
                           can be aligned to the book per event
        data/trades/       backfilled from data-api, SECOND stamps -- a second
                           holds a median of ~85 book updates here, so it only
                           supports aggregate rates, never per-event matching

    Falling back silently would make a millisecond-grade result look identical
    to a second-grade one, so the source is returned and reported.
    """
    live = os.path.join(BASE, "data", "trades_live", f"{slug}.jsonl")
    back = os.path.join(BASE, "data", "trades", f"{slug}.jsonl")
    by = collections.defaultdict(list)

    if os.path.exists(live):
        src = "live-ms"
        with open(live, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                by[t["asset_id"]].append((float(t["ts"]) / 1000.0, float(t["size"])))
    elif os.path.exists(back):
        src = "backfill-s"
        with open(back, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                by[t["asset"]].append((float(t["timestamp"]), float(t["size"])))
    else:
        return {}, "none"

    out = {}
    for aid, rows in by.items():
        rows.sort()
        out[aid] = (np.array([r[0] for r in rows]),
                    np.array([r[1] for r in rows]))
    return out, src


def touch_series(off, px, sz, idx):
    a, b = off[idx], off[idx + 1]
    has = b > a
    p = np.full(idx.size, np.nan)
    s = np.zeros(idx.size)
    p[has] = px[a[has]]
    s[has] = sz[a[has]]
    return p, s


def removal_streams(boff, bpx, bsz, aoff, apx, asz, idx):
    """Per-transition shares removed at the touch, split bid/ask.

    Index i of the result corresponds to the transition INTO snapshot i+1, so
    it lines up with times[1:].
    """
    out = {}
    for name, (off, px, sz) in (("bid", (boff, bpx, bsz)),
                                ("ask", (aoff, apx, asz))):
        p, s = touch_series(off, px, sz, idx)
        same = (p[1:] == p[:-1]) & ~np.isnan(p[1:])
        d = s[1:] - s[:-1]
        same_px = np.where(same & (d < 0), -d, 0.0)
        wiped = np.where(~same & ~np.isnan(p[:-1]), s[:-1], 0.0)
        out[name] = same_px + wiped
    return out


def windowed(cum, times, t0, t1):
    """Sum of a per-transition stream over (t0, t1], via its cumulative sum."""
    i0 = np.searchsorted(times, t0, side="left")
    i1 = np.searchsorted(times, t1, side="right")
    if i1 <= i0:
        return 0.0
    return float(cum[i1 - 1] - (cum[i0 - 1] if i0 > 0 else 0.0))


def analyse_game(slug, pre_window, match_window, agg, lags, cover,
                 max_spread=0.05, split_dir=None):
    windows = rm.load_game_windows()
    w = windows.get(slug, {})
    scoring = np.array(load_scoring(slug))
    trades, trade_src = load_trade_times(slug)

    cols = rd.load_columns(slug, split_dir=split_dir or rm.JSONL_SPLIT_DIR)
    asset_ids, ts, bpx, bsz, boff, apx, asz, aoff, meta = cols
    t_lo = w.get("kickoff_ts", -np.inf)
    t_hi = w.get("game_end_ts", np.inf)

    log(f"    trade tape: {trade_src}")
    n_tok = 0
    for aid, lo, hi in rd.asset_boundaries(asset_ids):
        mt = (meta.get(aid) or {}).get("market_type") or "?"
        if mt not in LIQUID:
            continue
        t = rd.dedup_nudge(ts[lo:hi])
        keep = (t >= t_lo) & (t <= t_hi)
        if keep.sum() < 100:
            continue
        idx = np.arange(lo, hi)[keep]
        times = t[keep]
        n_tok += 1

        bb = best_price_column(lo, hi, boff, bpx)[keep]
        ba = best_price_column(lo, hi, aoff, apx)[keep]
        two = ~(np.isnan(bb) | np.isnan(ba))
        mid = np.where(two, (bb + ba) / 2.0, np.nan)

        rem = removal_streams(boff, bpx, bsz, aoff, apx, asz, idx)
        tt = times[1:]
        cum = {k: np.cumsum(v) for k, v in rem.items()}
        span = float(times[-1] - times[0]) or 1.0
        base = {k: float(v[-1]) / span for k, v in cum.items()}   # shares/sec
        if base["bid"] <= 0 or base["ask"] <= 0:
            continue

        tr_t, tr_s = trades.get(aid, (np.array([]), np.array([])))
        tr_cum = np.cumsum(tr_s) if tr_s.size else np.array([])

        # How much of THIS token's window sits inside a scoring window.
        # Without it, "12% of jumps are scoring-driven" is uninterpretable --
        # the comparison is against how much of the clock those windows cover,
        # not against zero. Intervals are merged so overlapping scoring plays
        # are not double counted.
        if scoring.size:
            lo_t, hi_t = float(times[0]), float(times[-1])
            merged = []
            for st in scoring:
                a0, a1 = max(st, lo_t), min(st + match_window, hi_t)
                if a1 <= a0:
                    continue
                if merged and a0 <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], a1)
                else:
                    merged.append([a0, a1])
            cover["covered"] += sum(b - a for a, b in merged)
        cover["total"] += float(times[-1] - times[0])

        for theta_c in THETAS_C:
            for i, j in zigzag_up_legs(mid, theta_c / 100.0):
                if not (two[i] and two[j]):
                    continue
                # Only swings that could actually be traded. Without this the
                # averages are dominated by stub-quote books whose "mid" is
                # meaningless and whose spread no one could cross.
                if (ba[i] - bb[i] > max_spread) or (ba[j] - bb[j] > max_spread):
                    continue
                t0 = float(times[i])
                # Liquidity leaving the touch in the seconds BEFORE the move.
                pre = {k: windowed(cum[k], tt, t0 - pre_window, t0)
                       for k in ("bid", "ask")}
                rate = {k: pre[k] / pre_window for k in pre}
                inten = {k: rate[k] / base[k] for k in pre}

                traded_pre = 0.0
                if tr_t.size:
                    traded_pre = windowed(tr_cum, tr_t, t0 - pre_window, t0)

                lag = None
                if scoring.size:
                    k = np.searchsorted(scoring, t0, side="right") - 1
                    if k >= 0:
                        d = t0 - scoring[k]
                        if 0 <= d <= match_window:
                            lag = float(d)

                grp = "scoring" if lag is not None else "other"
                a = agg[(theta_c, grp)]
                a["n"] += 1
                a["move"] += float(mid[j] - mid[i])
                a["int_bid"] += inten["bid"]
                a["int_ask"] += inten["ask"]
                a["int_max"] += max(inten["bid"], inten["ask"])
                a["asym"] += abs(inten["bid"] - inten["ask"])
                a["traded_pre"] += traded_pre
                a["spread"] += float(ba[i] - bb[i])
                if lag is not None:
                    lags[theta_c].append(lag)
    del cols
    return n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--slate", default=None)
    ap.add_argument("--pre-window", type=float, default=5.0,
                    help="seconds before the swing over which to measure "
                         "liquidity removal")
    ap.add_argument("--match-window", type=float, default=60.0,
                    help="a swing starting within this many seconds after the "
                         "in-play pitch of a scoring play is called scoring-driven")
    ap.add_argument("--max-spread", type=float, default=0.05,
                    help="skip swings whose quoted spread at either end exceeds "
                         "this -- keeps stub-quote books out of the averages")
    ap.add_argument("--split-dir", default=None)
    ap.add_argument("--json", default=None,
                    help="write this run's totals here instead of printing a "
                         "report -- lets a slate be processed in batches")
    ap.add_argument("--merge-json", nargs="*", default=None,
                    help="print one combined report from --json files and exit. "
                         "Totals are additive, so batching changes nothing about "
                         "the result; it only survives a run being interrupted.")
    args = ap.parse_args()

    if args.merge_json:
        agg = collections.defaultdict(lambda: collections.defaultdict(float))
        lags = collections.defaultdict(list)
        cover = collections.defaultdict(float)
        games, ntok = [], 0
        for path in args.merge_json:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            games += d["games"]
            ntok += d["n_tokens"]
            for k, v in d["agg"].items():
                th, grp = k.split("|")
                for kk, vv in v.items():
                    agg[(float(th), grp)][kk] += vv
            for th, v in d["lags"].items():
                lags[float(th)] += v
            for k, v in d["cover"].items():
                cover[k] += v
        report(agg, lags, cover, len(games), ntok,
               d["pre_window"], d["match_window"], d["max_spread"])
        return 0

    if args.games:
        slugs = args.games
    else:
        pat = f"*-{args.slate}.jsonl" if args.slate else "*.jsonl"
        slugs = sorted(os.path.basename(f)[:-6] for f in
                       glob.glob(os.path.join(args.split_dir or rm.JSONL_SPLIT_DIR, pat)))
    if not slugs:
        raise SystemExit("no games found")

    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    lags = collections.defaultdict(list)
    cover = collections.defaultdict(float)
    log(f"{len(slugs)} game(s), pre-window {args.pre_window}s, "
        f"scoring match window {args.match_window}s")
    ntok = 0
    for slug in slugs:
        t0 = time.time()
        ntok += analyse_game(slug, args.pre_window, args.match_window,
                             agg, lags, cover, args.max_spread, args.split_dir)
        log(f"  {slug}: {time.time()-t0:.0f}s")

    if args.json:
        out = {"games": slugs, "n_tokens": ntok, "pre_window": args.pre_window,
               "match_window": args.match_window, "max_spread": args.max_spread,
               "agg": {f"{th}|{grp}": dict(v) for (th, grp), v in agg.items()},
               "lags": {str(th): v for th, v in lags.items()},
               "cover": dict(cover)}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f)
        log(f"wrote {args.json} ({len(slugs)} game(s), {ntok} tokens)")
        return 0

    report(agg, lags, cover, len(slugs), ntok, args.pre_window,
           args.match_window, args.max_spread)
    return 0


def report(agg, lags, cover, n_games, ntok, pre_window, match_window, max_spread):
    print()
    print("=" * 92)
    print("CANCELLATION AT THE TOUCH BEFORE A JUMP, AND WHETHER A RUN SCORED")
    print(f"{n_games} game(s), {ntok} liquid tokens; intensity = removal rate in the "
          f"{pre_window:g}s before")
    print("the swing, divided by that token's own average removal rate (1.0 = normal)")
    print("=" * 92)
    print(f"{'swing':>7} {'cause':>9} {'jumps':>8} {'avg move':>10} {'bid int':>9} "
          f"{'ask int':>9} {'max int':>9} {'asym':>8} {'spread':>8}")
    print("-" * 92)
    for th in THETAS_C:
        for grp in ("scoring", "other"):
            a = agg[(th, grp)]
            n = a["n"]
            if not n:
                continue
            print(f"{th:>6.1f}c {grp:>9} {int(n):>8,} "
                  f"{a['move']/n*100:>9.2f}c {a['int_bid']/n:>9.2f} "
                  f"{a['int_ask']/n:>9.2f} {a['int_max']/n:>9.2f} "
                  f"{a['asym']/n:>8.2f} {a['spread']/n*100:>7.2f}c")

    exp = (cover["covered"] / cover["total"] * 100) if cover["total"] else 0.0
    print()
    print("### how much of each swing bucket is explained by a run scoring")
    print(f"scoring windows cover {exp:.1f}% of in-game clock, so that is the share "
          f"expected by chance.")
    print(f"enrichment = observed / {exp:.1f}%; 1.0x means runs explain nothing.")
    print(f"{'swing':>7} {'scoring':>9} {'other':>9} {'% scoring':>11} {'enrichment':>12}")
    print("-" * 54)
    for th in THETAS_C:
        s = agg[(th, "scoring")]["n"]
        o = agg[(th, "other")]["n"]
        if s + o:
            pct = s / (s + o) * 100
            print(f"{th:>6.1f}c {int(s):>9,} {int(o):>9,} {pct:>10.1f}% "
                  f"{pct/exp if exp else float('nan'):>11.2f}x")

    print()
    print("### lag from the in-play pitch to the start of the swing (seconds)")
    print(f"{'swing':>7} {'n':>7} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7}")
    print("-" * 52)
    for th in THETAS_C:
        v = np.array(lags[th])
        if v.size < 5:
            continue
        q = np.percentile(v, [10, 25, 50, 75, 90])
        print(f"{th:>6.1f}c {v.size:>7,} " + " ".join(f"{x:>6.1f}s" for x in q))


if __name__ == "__main__":
    sys.exit(main())
