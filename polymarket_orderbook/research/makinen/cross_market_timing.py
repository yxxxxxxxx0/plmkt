"""Cross-market arbitrage and lead-lag between moneyline, spread and total.

Two questions from Justin, on the same eight contracts per game:

  1. ARBITRAGE. Are the markets ever inconsistent enough to lock in a profit?
  2. LEAD-LAG. When a game event reprices the markets, which moves first, by
     how long, and is the gap tradeable?

TOKENS. Every recorded token is the YES side of its own market (checked on the
raw `outcome` field): the moneyline is "team H wins", each total is "Over L",
each spread is "team T wins by more than |line|". On Polymarket's CLOB the NO
side is the mirror of the YES book, so for any event E written as a token T or
its negation:

    ask(T)     = ask_T          bid(T)     = bid_T
    ask(not T) = 1 - bid_T      bid(not T) = 1 - ask_T

and the size on the mirrored side is the size on the opposite side of T.

ARBITRAGE. If event A implies event B (A is a subset of B), then P(A) <= P(B)
must hold. It is violated in an executable way when bid(A) > ask(B): buy B at
its ask, sell A at its bid (i.e. buy "not A"). The pair pays at least $1 per
share at settlement in every outcome, and costs ask(B) + 1 - bid(A) < $1, so
the profit is locked in and needs no exit -- held to resolution. Taker fee on
both legs (5% * p * (1-p) per share). The implications used, all exact for a
baseball game (no ties):

    Over 9.5  => Over 8.5  => Over 7.5                       (and 9.5 => 7.5)
    T by 3+   => T by 2+   => T wins          for both teams, where
                                             "away wins" = not(moneyline)
    T by 2+   => not(U by 2+)                 for the two teams T, U
    T by 2+   => not(U wins)                  (same as T by 2+ => T wins)

A violation only counts if it STANDS for STAND_S (a 200ms flicker cannot be
filled on both legs), holds at least $STAKE on both legs, does not touch a
recording silence, and does not touch a stretch in which either leg's raw
recorded book was crossed. The first run found a 7.6-cent "arbitrage" on
ari-sf 08-29 (Over 8.5 bid 0.62 above Over 7.5 ask 0.52 for three minutes);
the 0.52 was a ghost -- the raw book had the bid trading straight through it
-- and crossed_scan.py exists because of it.

LEAD-LAG. Each contract's price path is cut into quoted levels exactly as
mark_single_jumps.py does. A REPRICING is a move of at least MOVE_C cents
between consecutive quoted levels. Its time is the instant the executable
price crossed HALFWAY to the new level: the bid for an up-move, the ask for a
down-move. That instant ignores quote pulls, which move the far side of the
book without repricing anything. Repricings in different contracts of one
game within CLUSTER_S of each other are one game event; within an event,
every contract's time is measured against the moneyline's.

Clock: the grid carries the exchange's book timestamp, so these lags are
when each market repriced AT THE EXCHANGE, not when the change reached us.

Outputs, under results/makinen/cross_market/
  arb_episodes.csv         every standing, sized violation, one row per episode
  arb_summary.csv          by relation
  repricings.parquet       every repricing in every contract
  event_lags.csv           per game event, each contract's lag vs the moneyline
  lag_summary.csv          by market type
  follow_trade.csv         buy the laggard after the leader moves: does it pay?
  lead_lag.png

    python research/makinen/cross_market_timing.py
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mark_single_jumps as M  # noqa: E402

TICK = 0.01
GRID_MS = 200
STAND_S = 1.0
STAKE = 10.0
MOVE_C = 0.03          # a repricing: at least 3 cents between quoted levels
CLUSTER_S = 30.0       # repricings this close on one game are one event
LATENCIES_S = (0.2, 0.5, 1.0, 2.0)

OUT = ROOT / "results" / "makinen" / "cross_market"
CACHE = HERE / "cache" / "token_meta.parquet"


def fee(p):
    p = np.clip(p, 0.0, 1.0)
    return 0.05 * p * (1.0 - p)


# --------------------------------------------------------------------------
# token metadata, from the raw recordings
# --------------------------------------------------------------------------

def token_meta(sessions) -> pd.DataFrame:
    """asset_id -> outcome label, read from each recording's opening records."""
    have = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["asset_id"])
    known = set(have.asset_id)
    rows = []
    for sess in sessions:
        grid = ROOT / "data" / "jump" / f"feat_{sess}_trimmed.parquet"
        ser = pd.read_parquet(grid, columns=["series"]).series.unique()
        want = {s.split("|")[3] for s in ser} - known
        if not want:
            continue
        raw = ROOT / "data" / "live" / f"{sess}.jsonl.xz"
        if not raw.exists():
            raw = ROOT / "data" / "live" / f"{sess}.jsonl.jsonl.xz"
        with lzma.open(raw, "rb") as f:
            for line in f:
                r = orjson.loads(line)
                a = r.get("asset_id")
                if a in want:
                    rows.append(dict(asset_id=a, outcome=r.get("outcome"),
                                     outcome_name=r.get("outcome_name"),
                                     question=r.get("market_question"),
                                     condition_id=r.get("condition_id")))
                    want.discard(a)
                    if not want:
                        break
        if want:
            print(f"  {sess}: {len(want)} token(s) with no label in the raw file")
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("asset_id")
    CACHE.parent.mkdir(exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out


# --------------------------------------------------------------------------
# arbitrage
# --------------------------------------------------------------------------

def game_events(g: pd.DataFrame, meta: dict) -> dict:
    """Name every event this game's tokens can express, as (series, negated)."""
    ev = {}
    home = None
    for sr in g.series.unique():
        _, mt, line, aid = sr.split("|")
        m = meta.get(aid)
        if m is None or m["outcome"] != "YES":
            continue
        name = m["outcome_name"]
        if mt == "moneyline":
            home = name
            ev[("win", name)] = (sr, False)
        elif mt == "total" and name == "Over":
            ev[("over", float(line))] = (sr, False)
        elif mt == "spread":
            ev[("by", name, abs(float(line)))] = (sr, False)
    if home is not None:
        teams = {k[1] for k in ev if k[0] == "by"} | {home}
        for t in teams - {home}:
            ev[("win", t)] = (ev[("win", home)][0], True)
    return ev


def implications(ev: dict):
    """(A, B, label) with A implying B."""
    out = []
    overs = sorted(k[1] for k in ev if k[0] == "over")
    for i, hi in enumerate(overs):
        for lo in overs[:i]:
            out.append((("over", hi), ("over", lo), f"over {hi} <= over {lo}"))
    teams = {k[1] for k in ev if k[0] in ("by", "win")}
    for t in teams:
        lines = sorted(k[2] for k in ev if k[0] == "by" and k[1] == t)
        for i, big in enumerate(lines):
            for small in lines[:i]:
                out.append((("by", t, big), ("by", t, small), f"team by {big + .5:.0f}+ <= by {small + .5:.0f}+"))
        if ("win", t) in ev:
            for ln in lines:
                out.append((("by", t, ln), ("win", t), f"team by {ln + .5:.0f}+ <= team wins"))
    tl = sorted(teams)
    if len(tl) == 2:
        a, b = tl
        for la in [k[2] for k in ev if k[0] == "by" and k[1] == a]:
            for lb in [k[2] for k in ev if k[0] == "by" and k[1] == b]:
                out.append((("by", a, la), ("not_by", b, lb), "both teams can't cover: sum <= 1"))
    return out


def arb_game(g: pd.DataFrame, meta: dict, gaps: np.ndarray, suspect: dict) -> list[dict]:
    ev = game_events(g, meta)
    if not ev:
        return []
    wide = {}
    for sr, s in g.groupby("series"):
        wide[sr] = s.set_index("ts")[["bid", "ask", "bid_usd", "ask_usd"]]
    ts_all = np.unique(g.ts.to_numpy())
    idx = pd.Index(ts_all)

    def side(event, want):
        """(price, size in shares) to BUY (want='ask') or SELL (want='bid') the event."""
        if event[0] == "not_by":
            sr, neg = ev[("by", event[1], event[2])]
            neg = not neg
        else:
            sr, neg = ev[event]
        b = wide[sr].reindex(idx)
        bid, ask = b.bid.to_numpy(), b.ask.to_numpy()
        bsh = b.bid_usd.to_numpy() / np.maximum(bid, 1e-9)       # shares at the bid
        ash = b.ask_usd.to_numpy() / np.maximum(ask, 1e-9)
        if not neg:
            return (ask, ash) if want == "ask" else (bid, bsh)
        # not T: buying it hits T's bid; selling it lifts T's ask
        return (1 - bid, bsh) if want == "ask" else (1 - ask, ash)

    stand = int(round(STAND_S * 1000 / GRID_MS))
    out = []
    for A, B, label in implications(ev):
        if A[0] != "not_by" and A not in ev:
            continue
        if B[0] != "not_by" and B not in ev:
            continue
        bid_a, sz_a = side(A, "bid")
        ask_b, sz_b = side(B, "ask")
        edge = bid_a - ask_b - fee(bid_a) - fee(ask_b)          # $ per share, locked
        shares = STAKE / np.maximum(ask_b + 1 - bid_a, 1e-9)
        ok = np.isfinite(edge) & (edge > 0) & (sz_a >= shares) & (sz_b >= shares)
        if not ok.any():
            continue
        # contiguous runs on the 200ms grid
        brk = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for r0, r1 in zip(brk[::2], brk[1::2] - 1):
            if r1 - r0 + 1 < stand or ts_all[r1] - ts_all[r0] != (r1 - r0) * GRID_MS:
                continue
            t0, t1 = int(ts_all[r0]), int(ts_all[r1])
            if len(gaps) and np.any((gaps[:, 0] < t1) & (gaps[:, 1] > t0 - 5000)):
                continue
            # either leg's recorded book crossed nearby: a ghost level, not a price
            legs = [ev[("by", E[1], E[2])][0] if E[0] == "not_by" else ev[E][0] for E in (A, B)]
            if any(len(w := suspect.get(sr.split("|")[3], ())) and
                   np.any((w[:, 0] < t1) & (w[:, 1] > t0)) for sr in legs):
                continue
            k = r0 + int(np.argmax(edge[r0:r1 + 1]))
            out.append(dict(relation=label, A=str(A), B=str(B), ts_start=t0, ts_end=t1,
                            secs=(t1 - t0) / 1000 + GRID_MS / 1000,
                            best_edge_c=100 * edge[k], sell_A=bid_a[k], buy_B=ask_b[k],
                            usd_at_stake=edge[k] * STAKE / (ask_b[k] + 1 - bid_a[k]),
                            size_shares=float(min(sz_a[k], sz_b[k]))))
    return out


# --------------------------------------------------------------------------
# repricings and lead-lag
# --------------------------------------------------------------------------

def repricings(s: pd.DataFrame) -> list[dict]:
    ts = s.ts.to_numpy(np.int64)
    bid = s.bid.to_numpy(np.float64)
    ask = s.ask.to_numpy(np.float64)
    spr = s.spread_ticks.to_numpy(np.float64)
    if len(ts) < 100:
        return []
    normal = float(np.median(spr))
    cut = normal + max(2.0, normal)
    P = [(i0, i1) for i0, i1 in M.plateaus(ts, bid, ask) if np.median(spr[i0:i1 + 1]) <= cut]
    out = []
    for (a0, a1), (b0, b1) in zip(P[:-1], P[1:]):
        if ts[b0] - ts[a1] != (b0 - a1) * GRID_MS or (ts[b0] - ts[a1]) > 60_000:
            continue
        m_a = np.median((bid[a0:a1 + 1] + ask[a0:a1 + 1]) / 2)
        m_b = np.median((bid[b0:b1 + 1] + ask[b0:b1 + 1]) / 2)
        d = m_b - m_a
        if abs(d) < MOVE_C - 1e-9:
            continue
        seg = slice(a1 + 1, b0 + 1)
        if d > 0:       # up: the bid reprices; halfway from the old bid to the new
            lvl = bid[a1] + d / 2
            hit = np.flatnonzero(bid[seg] >= lvl - 1e-9)
        else:           # down: the ask reprices
            lvl = ask[a1] + d / 2
            hit = np.flatnonzero(ask[seg] <= lvl + 1e-9)
        if not len(hit):
            continue
        h = a1 + 1 + int(hit[0])
        pull = np.flatnonzero(spr[seg] > cut)
        out.append(dict(series=s.series.iat[0], ts_leave=int(ts[a1]) + GRID_MS,
                        ts_half=int(ts[h]), ts_arrive=int(ts[b0]),
                        ts_pull=int(ts[a1 + 1 + pull[0]]) if len(pull) else np.nan,
                        move_c=100 * d, pre_bid=bid[a1], pre_ask=ask[a1],
                        post_bid=float(np.median(bid[b0:b1 + 1])),
                        post_ask=float(np.median(ask[b0:b1 + 1])),
                        b0_i=b0, a1_i=a1))
    return out


def cluster_events(rp: pd.DataFrame) -> pd.DataFrame:
    """Group repricings of one game within CLUSTER_S into events."""
    rp = rp.sort_values(["slug", "ts_half"]).copy()
    new = (rp.slug != rp.slug.shift()) | (rp.ts_half.diff() > CLUSTER_S * 1000)
    rp["event"] = np.cumsum(new)
    return rp


def follow_trade(rp: pd.DataFrame, by_series: dict, normal_spread: dict) -> list[dict]:
    """After the moneyline reprices, buy (or sell) a contract that has not.

    At t = moneyline's halfway instant + latency, trade the laggard in the
    direction of ITS coming move at the price then standing, and exit at its
    new level's standing price (post_bid for a buy, post_ask for a sell).
    Only laggards whose own halfway instant is still in the future at t
    count: if it has already repriced there is nothing to follow.
    """
    out = []
    for ev, e in rp.groupby("event"):
        ml = e[e.market_type == "moneyline"]
        if len(ml) != 1:
            continue
        t_ml = int(ml.ts_half.iat[0])
        for r in e[e.market_type != "moneyline"].itertuples():
            s = by_series[r.series]
            tsv = s.ts.to_numpy()
            normal = normal_spread[r.series]
            for lat in LATENCIES_S:
                t = t_ml + int(lat * 1000)
                if r.ts_half <= t:
                    out.append(dict(event=ev, market_type=r.market_type, latency_s=lat,
                                    lagging=False, net_c=np.nan))
                    continue
                i = np.searchsorted(tsv, t)
                if i >= len(tsv):
                    continue
                if r.move_c > 0:
                    px = s.ask.iat[i]
                    net = r.post_bid - px - fee(px) - fee(r.post_bid)
                else:
                    px = s.bid.iat[i]
                    net = px - r.post_ask - fee(px) - fee(r.post_ask)
                spr_t = s.spread_ticks.iat[i]
                size_t = (s.ask_usd.iat[i] if r.move_c > 0 else s.bid_usd.iat[i])
                out.append(dict(event=ev, market_type=r.market_type, latency_s=lat,
                                lagging=True, net_c=100 * net, move_c=r.move_c,
                                lead_s=(r.ts_half - t_ml) / 1000,
                                spread_at_t=spr_t, normal_spread=normal,
                                # still showing an ordinary quote with size:
                                # the stale-quote case a follower would need
                                quoted=bool(spr_t <= normal + 1e-6 and size_t >= STAKE)))
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=M.SESSIONS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta_df = token_meta(args.sessions)
    meta = {r.asset_id: r._asdict() for r in meta_df.itertuples()}
    assert (meta_df.outcome == "YES").all(), "expected only YES tokens in the grid"

    arbs, reps, follows = [], [], []
    for sess in args.sessions:
        df = M.load_session(sess)
        gaps = M.load_gaps(sess)
        suspect = M.suspect_windows(sess)
        rp_s = []
        by_series = {}
        for slug, g in df.groupby(df.series.str.split("|").str[0]):
            gz = gaps.get(slug, np.empty((0, 2), np.int64))
            for a in arb_game(g, meta, gz, suspect):
                a.update(session=sess, slug=slug)
                arbs.append(a)
            for sr, s in g.groupby("series"):
                by_series[sr] = s.reset_index(drop=True)
                for r in repricings(by_series[sr]):
                    r.update(session=sess, slug=slug)
                    rp_s.append(r)
        rp = pd.DataFrame(rp_s)
        if len(rp):
            rp["market_type"] = rp.series.str.split("|").str[1]
            # drop repricings that touch a recording silence or a crossed book
            keep = []
            none = np.empty((0, 2), np.int64)
            for r in rp.itertuples():
                gz = gaps.get(r.slug, none)
                sz = suspect.get(r.series.split("|")[3], none)
                bad = (len(gz) and np.any((gz[:, 0] < r.ts_arrive) & (gz[:, 1] > r.ts_leave - 5000))) or                       (len(sz) and np.any((sz[:, 0] < r.ts_arrive) & (sz[:, 1] > r.ts_leave - 5000)))
                keep.append(not bad)
            rp = rp[np.array(keep)]
            rp = cluster_events(rp)
            rp["event"] = rp.event.astype(str) + "@" + sess
            normal = {k: float(np.median(v.spread_ticks)) for k, v in by_series.items()}
            follows += follow_trade(rp, by_series, normal)
            reps.append(rp)
        n_a = sum(1 for a in arbs if a["session"] == sess)
        print(f"{sess}: {len(rp)} repricings, {n_a} standing arbitrage episode(s)", flush=True)
        del df, by_series

    arb = pd.DataFrame(arbs)
    arb.to_csv(OUT / "arb_episodes.csv", index=False)
    if len(arb):
        s = (arb.groupby("relation")
             .agg(episodes=("secs", "size"), games=("slug", "nunique"),
                  median_secs=("secs", "median"), median_edge_c=("best_edge_c", "median"),
                  max_edge_c=("best_edge_c", "max"), usd_at_10=("usd_at_stake", "sum"))
             .sort_values("episodes", ascending=False))
        s.to_csv(OUT / "arb_summary.csv")
        print("\nARBITRAGE (standing >= 1s, >= $10 both legs, net of fees):")
        print(s.round(3).to_string())
    else:
        print("\nARBITRAGE: no standing, sized violation anywhere")

    rp = pd.concat(reps, ignore_index=True)
    rp.to_parquet(OUT / "repricings.parquet", index=False)
    ml = rp[rp.market_type == "moneyline"]
    ml = ml[~ml.event.duplicated(keep=False)].set_index("event")   # one ML repricing per event
    lag = rp[rp.market_type != "moneyline"].join(
        ml[["ts_half", "ts_pull", "move_c"]].rename(columns=lambda c: "ml_" + c), on="event", how="inner")
    lag["lag_half_s"] = (lag.ts_half - lag.ml_ts_half) / 1000
    lag["lag_pull_s"] = (lag.ts_pull - lag.ml_ts_pull) / 1000
    lag.to_csv(OUT / "event_lags.csv", index=False)

    def q(x):
        return pd.Series({"n": len(x), "median_s": x.median(), "p25_s": x.quantile(.25),
                          "p75_s": x.quantile(.75), "share_before_ml": (x < 0).mean(),
                          "share_same_slot": (x == 0).mean(), "share_after_ml": (x > 0).mean()})
    ls = lag.groupby("market_type").lag_half_s.apply(q).unstack()
    lp = lag.dropna(subset=["lag_pull_s"]).groupby("market_type").lag_pull_s.apply(q).unstack()
    ls.to_csv(OUT / "lag_summary.csv")
    print("\nLAG of each market's repricing vs the moneyline's (seconds; + = after):")
    print(ls.round(2).to_string())
    print("\nLAG of the makers' quote pull vs the moneyline's pull:")
    print(lp.round(2).to_string())

    fo = pd.DataFrame(follows)
    fo.to_csv(OUT / "follow_trade.csv", index=False)
    if len(fo):
        fs = (fo.groupby(["latency_s", "market_type"])
              .apply(lambda x: pd.Series({
                  "laggard_events": int(x.lagging.sum()), "all_events": len(x),
                  "still_lagging": x.lagging.mean(),
                  "median_net_c": x.net_c.median(),
                  "share_paying": (x.net_c > 0).sum() / max(1, x.lagging.sum()),
                  "mean_net_c": x.net_c.mean()}), include_groups=False))
        fs.to_csv(OUT / "follow_summary.csv")
        print("\nFOLLOW THE MONEYLINE: trade the laggard at ML-halfway + latency, exit at its new level:")
        print(fs.round(3).to_string())
        lagg = fo[fo.lagging]
        fq = (lagg.groupby(["latency_s", "market_type", "quoted"])
              .net_c.agg(n="size", median_net_c="median", mean_net_c="mean",
                         share_paying=lambda x: (x > 0).mean()))
        fq.to_csv(OUT / "follow_by_quote.csv")
        print("\n...split by whether the laggard was STILL QUOTED normally (a stale quote to hit):")
        print(fq.round(3).to_string())
    draw(lag, fo)


def draw(lag, fo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), facecolor=M.SURFACE)
    cols = {"spread": "#2a78d6", "total": "#eb6834"}
    bins = np.arange(-10, 10.2, 0.2)
    for mt, c in cols.items():
        x = lag[lag.market_type == mt].lag_half_s.clip(-10, 10)
        ax[0].hist(x, bins=bins, histtype="step", lw=1.8, color=c,
                   label=f"{mt}: median {lag[lag.market_type == mt].lag_half_s.median():+.1f}s (n={len(x)})")
    ax[0].axvline(0, color=M.INK2, lw=0.8)
    ax[0].set_xlabel("seconds after the moneyline repriced (halfway point, exchange clock)")
    ax[0].set_ylabel("repricings")
    ax[0].set_title("When does each market reprice, relative to the moneyline?", loc="left")
    ax[0].legend(frameon=False)
    if len(fo):
        f = fo[fo.lagging]
        for mt, c in cols.items():
            g = f[f.market_type == mt].groupby("latency_s").net_c.median()
            ax[1].plot(g.index, g.values, marker="o", color=c, lw=2, label=mt)
        ax[1].axhline(0, color=M.INK2, lw=0.8)
        ax[1].set_xlabel("seconds after the moneyline repriced that you trade the laggard")
        ax[1].set_ylabel("median net cents per share")
        ax[1].set_title("Following the moneyline into a slower market", loc="left")
        ax[1].legend(frameon=False)
    for a in ax:
        a.set_facecolor(M.SURFACE)
        a.grid(color=M.GRID_C, lw=0.6)
        for sp in a.spines.values():
            sp.set_color(M.GRID_C)
    fig.tight_layout()
    fig.savefig(OUT / "lead_lag.png", dpi=150, facecolor=M.SURFACE)


if __name__ == "__main__":
    main()
