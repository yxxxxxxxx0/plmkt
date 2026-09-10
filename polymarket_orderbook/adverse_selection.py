"""Where is the mid 30s after the touch gets taken? -- realised maker P&L.

The question this answers
------------------------
A market maker resting at the best bid gets filled when someone sells into
it. If the mid then falls, the maker bought just before the price dropped:
that is adverse selection, and it is the cost that decides whether making
markets is profitable at all.

    maker edge per share  =  half_spread  -  adverse_selection

`half_spread` is what you earn if the mid does not move. `adverse_selection`
is how far the mid runs away from you after the fill. If the second exceeds
the first, no amount of capital makes market making work, so this must be
measured before any position sizing question is worth asking.

What the data can and cannot say
--------------------------------
live_recorder.py records `seed`, `book` and `price_change` events only --
line 324 of that file explicitly drops `last_trade_price`. So there are no
trade prints here, and a size decrease at the touch is ambiguous: it may be
a fill or a cancel. Two mitigations:

  * events are stratified by how strongly they imply a trade. A sweep that
    consumes a whole level and walks the touch is almost certainly
    aggressive flow; a partial size decrease at an unchanged price may be a
    cancel.
  * both strata are reported. Cancels dilute the estimate toward zero rather
    than biasing it negative, so a clearly negative result in the strong
    stratum is meaningful, and the weak stratum is a lower bound on severity.

Pass 1 streams the (very large) JSONL into a compact touch series, keeping
only updates where the touch actually changed. Pass 2 does the event study
on that much smaller table.

Usage:
    python adverse_selection.py --pass1 ../data/live/books_2026-08-30.jsonl
    python adverse_selection.py --pass2
    python adverse_selection.py --pass2 --horizon 30 60 120
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import orjson
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "adverse")
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# Pass 1: JSONL -> compact touch series
# --------------------------------------------------------------------------- #
# `outcome` matters more than it looks: every market is recorded as BOTH its
# YES and NO token, and the two are exact mirrors (YES bid = 1 - NO ask). So a
# bid-side event on YES is the very same trade as an ask-side event on NO.
# Without keeping the outcome we cannot drop the mirror, and every event is
# counted twice -- which leaves the means unbiased but shrinks the standard
# errors by sqrt(2) and makes the bid/ask split meaningless.
TOUCH_COLS = ["ts", "asset", "mtype", "bid", "bid_sz", "ask", "ask_sz",
              "bid_d3", "ask_d3", "et", "is_yes"]


def pass1(path: str, out_parquet: str, chunk_rows: int = 4_000_000,
          progress_every: int = 2_000_000):
    """Stream one books.jsonl into a compact per-touch table."""
    assets: dict[str, int] = {}
    mtypes: dict[str, int] = {}
    ets = {"seed": 0, "book": 1, "price_change": 2}
    last: dict[int, tuple] = {}

    buf = {c: [] for c in TOUCH_COLS}
    parts = []
    n_lines = n_kept = 0
    t0 = time.time()

    def flush():
        if not buf["ts"]:
            return
        df = pd.DataFrame({
            "ts": np.array(buf["ts"], dtype=np.int64),
            "asset": np.array(buf["asset"], dtype=np.int32),
            "mtype": np.array(buf["mtype"], dtype=np.int8),
            "bid": np.array(buf["bid"], dtype=np.float32),
            "bid_sz": np.array(buf["bid_sz"], dtype=np.float32),
            "ask": np.array(buf["ask"], dtype=np.float32),
            "ask_sz": np.array(buf["ask_sz"], dtype=np.float32),
            "bid_d3": np.array(buf["bid_d3"], dtype=np.float32),
            "ask_d3": np.array(buf["ask_d3"], dtype=np.float32),
            "et": np.array(buf["et"], dtype=np.int8),
            "is_yes": np.array(buf["is_yes"], dtype=bool),
        })
        p = f"{out_parquet}.part{len(parts)}"
        df.to_parquet(p, index=False)
        parts.append(p)
        for c in buf:
            buf[c].clear()

    with open(path, "rb") as fh:
        for line in fh:
            n_lines += 1
            try:
                r = orjson.loads(line)
            except Exception:
                continue
            bids, asks = r.get("bids"), r.get("asks")
            if not bids or not asks:
                continue

            aid = r["asset_id"]
            ai = assets.get(aid)
            if ai is None:
                ai = assets[aid] = len(assets)
            mt = r.get("market_type") or "?"
            mi = mtypes.get(mt)
            if mi is None:
                mi = mtypes[mt] = len(mtypes)

            b0, bs0 = bids[0][0], bids[0][1]
            a0, as0 = asks[0][0], asks[0][1]
            # crossed/locked books are data artifacts, not tradeable states
            if a0 <= b0:
                continue
            bd3 = sum(l[1] for l in bids[:3])
            ad3 = sum(l[1] for l in asks[:3])

            key = (b0, bs0, a0, as0)
            if last.get(ai) == key:
                continue                       # touch unchanged -> skip
            last[ai] = key

            buf["ts"].append(r["ts"]); buf["asset"].append(ai)
            buf["mtype"].append(mi)
            buf["bid"].append(b0); buf["bid_sz"].append(bs0)
            buf["ask"].append(a0); buf["ask_sz"].append(as0)
            buf["bid_d3"].append(bd3); buf["ask_d3"].append(ad3)
            buf["et"].append(ets.get(r.get("et"), 3))
            buf["is_yes"].append(r.get("outcome") == "YES")
            n_kept += 1

            if len(buf["ts"]) >= chunk_rows:
                flush()
            if n_lines % progress_every == 0:
                log(f"  {n_lines:,} lines, {n_kept:,} touch updates, "
                    f"{n_lines/(time.time()-t0):,.0f} lines/s")

    flush()
    log(f"  concatenating {len(parts)} parts")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out_parquet, index=False)
    for p in parts:
        os.remove(p)

    inv_a = {v: k for k, v in assets.items()}
    inv_m = {v: k for k, v in mtypes.items()}
    pd.DataFrame({"idx": list(inv_a), "asset_id": [inv_a[i] for i in inv_a]}) \
        .to_parquet(out_parquet.replace(".parquet", "_assets.parquet"), index=False)
    pd.DataFrame({"idx": list(inv_m), "market_type": [inv_m[i] for i in inv_m]}) \
        .to_parquet(out_parquet.replace(".parquet", "_mtypes.parquet"), index=False)

    log(f"  {n_lines:,} lines -> {len(df):,} touch updates, "
        f"{len(assets)} assets, {time.time()-t0:.0f}s")
    return df


# --------------------------------------------------------------------------- #
# Pass 2: event study on the touch series
# --------------------------------------------------------------------------- #
def detect_and_measure(df: pd.DataFrame, horizons=(30,), min_sz: float = 5.0,
                       max_spread: float = 0.011, yes_only: bool = True):
    """For each inferred fill at the touch, mark the mid `h` seconds later.

    Two strata:
      sweep  -- the best price moved against the resting side, i.e. the level
                was consumed. Strong evidence of aggressive flow.
      shrink -- size at an unchanged best price fell. Could be a cancel.

    `max_spread` is the filter that makes this meaningful. Only ~20% of
    recorded touch states are within one tick; the rest are wide books where
    a level disappearing is a market maker repricing, not a trade. Measuring
    across all of them produces a fake +0.06/share "edge" that is really the
    uncapturable half of an 11-cent spread. Market making happens at the
    tight touch, so that is where this must be measured.

    `yes_only` drops the mirror token so each real trade counts once.
    """
    if yes_only and "is_yes" in df.columns:
        df = df[df.is_yes]
    out = []
    for asset, g in df.groupby("asset", sort=False):
        g = g.sort_values("ts")
        if len(g) < 20:
            continue
        ts = g.ts.to_numpy()
        bid = g.bid.to_numpy(dtype=np.float64)
        ask = g.ask.to_numpy(dtype=np.float64)
        bsz = g.bid_sz.to_numpy(dtype=np.float64)
        asz = g.ask_sz.to_numpy(dtype=np.float64)
        mid = (bid + ask) / 2.0
        mtype = int(g.mtype.iloc[0])

        # ---- bid-side fills (maker BUYS at the bid) ----
        bid_fell = np.zeros(len(g), bool)
        bid_shrank = np.zeros(len(g), bool)
        bid_fell[1:] = bid[1:] < bid[:-1] - 1e-9
        bid_shrank[1:] = (np.abs(bid[1:] - bid[:-1]) < 1e-9) & (bsz[1:] < bsz[:-1] - min_sz)

        # ---- ask-side fills (maker SELLS at the ask) ----
        ask_rose = np.zeros(len(g), bool)
        ask_shrank = np.zeros(len(g), bool)
        ask_rose[1:] = ask[1:] > ask[:-1] + 1e-9
        ask_shrank[1:] = (np.abs(ask[1:] - ask[:-1]) < 1e-9) & (asz[1:] < asz[:-1] - min_sz)

        # the book must have been tight *before* the event for a resting
        # quote at the touch to be a realistic market-making position
        spread_pre = np.full(len(g), np.inf)
        spread_pre[1:] = (ask - bid)[:-1]
        tight = spread_pre <= max_spread

        for h in horizons:
            fut = np.searchsorted(ts, ts + h * 1000, side="left")
            fut = np.clip(fut, 0, len(ts) - 1)
            valid = (ts[fut] >= ts + h * 1000 * 0.5) & tight
            mid_fut = mid[fut]

            for side, strong, weak in (("bid", bid_fell, bid_shrank),
                                       ("ask", ask_rose, ask_shrank)):
                for stratum, m in (("sweep", strong), ("shrink", weak)):
                    idx = np.where(m & valid)[0]
                    idx = idx[idx > 0]
                    if idx.size == 0:
                        continue
                    # fill price is the touch BEFORE the event
                    fill_px = (bid[idx - 1] if side == "bid" else ask[idx - 1])
                    mid_pre = mid[idx - 1]
                    # size that was resting and got taken -- this is what a
                    # small order has to get through to be filled at all
                    consumed = (bsz[idx - 1] if side == "bid" else asz[idx - 1])
                    # Book shape BEFORE the fill, which is all a maker can
                    # condition on when deciding whether to quote.
                    #   own_depth   size at your own touch (queue ahead of you)
                    #   support     levels 2-3 behind your touch: what stops
                    #               the price gapping once your level is gone
                    #   far_depth   size on the opposite side
                    own_d3 = (g.bid_d3.to_numpy(dtype=np.float64) if side == "bid"
                              else g.ask_d3.to_numpy(dtype=np.float64))[idx - 1]
                    far_d3 = (g.ask_d3.to_numpy(dtype=np.float64) if side == "bid"
                              else g.bid_d3.to_numpy(dtype=np.float64))[idx - 1]
                    support = np.maximum(own_d3 - consumed, 0.0)
                    far_px = (ask[idx - 1] if side == "bid" else bid[idx - 1])
                    half_spread = (mid_pre - fill_px) if side == "bid" \
                        else (fill_px - mid_pre)
                    # maker P&L: bought at bid -> gains if the mid rises
                    pnl = (mid_fut[idx] - fill_px) if side == "bid" \
                        else (fill_px - mid_fut[idx])
                    # signed mid drift: negative means the mid ran away
                    drift = (mid_fut[idx] - mid_pre) if side == "bid" \
                        else (mid_pre - mid_fut[idx])
                    out.append(pd.DataFrame(dict(
                        asset=asset, mtype=mtype, side=side, stratum=stratum,
                        horizon=h, fill_px=fill_px, mid_pre=mid_pre,
                        mid_fut=mid_fut[idx], half_spread=half_spread,
                        pnl=pnl, drift=drift, adverse=half_spread - pnl,
                        consumed=consumed, consumed_usd=consumed * fill_px,
                        support_usd=support * fill_px,
                        far_usd=far_d3 * far_px,
                        imbalance=own_d3 / np.maximum(own_d3 + far_d3, 1e-9),
                        # how far the touch gapped: a 1-tick move means the
                        # sweep barely cleared the level, several ticks means
                        # it walked the book and would have eaten a joining
                        # order too
                        gap=(np.abs((bid[idx] - bid[idx - 1]) if side == "bid"
                                    else (ask[idx] - ask[idx - 1]))))))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize(ev: pd.DataFrame, mtypes: dict | None = None) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame()
    g = ev.groupby(["horizon", "stratum", "side"], observed=True)
    s = g.agg(n=("pnl", "size"),
              half_spread=("half_spread", "mean"),
              adverse=("adverse", "mean"),
              mean_pnl=("pnl", "mean"),
              median_pnl=("pnl", "median"),
              sd_pnl=("pnl", "std"),
              mean_px=("fill_px", "mean")).reset_index()
    s["se_pnl"] = s.sd_pnl / np.sqrt(s.n)
    s["t_stat"] = s.mean_pnl / s.se_pnl
    # return per dollar at risk, which is what a bankroll actually earns
    s["roi_pct"] = s.mean_pnl / s.mean_px * 100
    return s.drop(columns=["sd_pnl"])


def queue_position(ev: pd.DataFrame, mtype_map: dict | None = None,
                   thresholds=(0, 500, 1000, 2000, 5000)) -> pd.DataFrame:
    """P&L as a function of how much size the sweep consumed.

    This is the analysis that decides whether a SMALL bankroll can make
    markets. A resting order fills only once the size ahead of it in the
    queue is gone, so a $50 order behind $5,000 of depth is filled only by
    sweeps larger than $5,000. The average sweep is therefore the wrong
    benchmark -- the right one is the tail it can actually reach.

    If P&L falls as consumed size rises, small size is selectively filled on
    the most informed trades, and queue position (i.e. capital) is what
    separates a profitable maker from an unprofitable one.
    """
    e = ev[ev.stratum == "sweep"].copy()
    if e.empty or "consumed_usd" not in e.columns:
        return pd.DataFrame()
    rows = []
    for th in thresholds:
        g = e[e.consumed_usd >= th]
        if len(g) < 30:
            continue
        se = g.pnl.std() / np.sqrt(len(g))
        rows.append(dict(
            min_sweep_usd=th, n=len(g),
            half_spread=g.half_spread.mean(), adverse=g.adverse.mean(),
            mean_pnl=g.pnl.mean(), se=se, t=g.pnl.mean() / se if se else np.nan,
            roi_pct=g.pnl.mean() / g.fill_px.mean() * 100))
    return pd.DataFrame(rows)


def queue_buckets(ev: pd.DataFrame) -> pd.DataFrame:
    e = ev[ev.stratum == "sweep"].copy()
    if e.empty or "consumed_usd" not in e.columns:
        return pd.DataFrame()
    e["bucket"] = pd.cut(e.consumed_usd, [-1, 50, 200, 1000, 5000, 20000, 1e12],
                         labels=["<$50", "$50-200", "$200-1k", "$1k-5k",
                                 "$5k-20k", ">$20k"])
    s = e.groupby("bucket", observed=True).agg(
        n=("pnl", "size"), med_consumed=("consumed_usd", "median"),
        half_spread=("half_spread", "mean"), adverse=("adverse", "mean"),
        mean_pnl=("pnl", "mean"), sd=("pnl", "std"),
        mean_px=("fill_px", "mean")).reset_index()
    s["se"] = s.sd / np.sqrt(s.n)
    s["t"] = s.mean_pnl / s.se
    s["roi_pct"] = s.mean_pnl / s.mean_px * 100
    return s.drop(columns=["sd"])


def book_shape(ev: pd.DataFrame, col: str, edges, labels) -> pd.DataFrame:
    """P&L conditioned on one pre-fill book-shape variable.

    A maker can only condition on what the book looked like *before* being
    filled, so these are the only usable filters: own-touch depth, the
    support behind it, the far side, and the imbalance between them.
    """
    e = ev[ev.stratum == "sweep"].copy()
    if e.empty or col not in e.columns:
        return pd.DataFrame()
    e["bucket"] = pd.cut(e[col], edges, labels=labels)
    s = e.groupby("bucket", observed=True).agg(
        n=("pnl", "size"), med=(col, "median"),
        half_spread=("half_spread", "mean"), adverse=("adverse", "mean"),
        mean_pnl=("pnl", "mean"), sd=("pnl", "std"),
        mean_px=("fill_px", "mean")).reset_index()
    s["se"] = s.sd / np.sqrt(s.n)
    s["t"] = s.mean_pnl / s.se
    s["roi_pct"] = s.mean_pnl / s.mean_px * 100
    return s.drop(columns=["sd"]).rename(columns={"bucket": col})


def small_maker_regime(ev: pd.DataFrame, capital: float = 50.0,
                       support_thresholds=(0, 100, 500, 2000)) -> pd.DataFrame:
    """Is there a filter that is BOTH profitable AND reachable with `capital`?

    Two conditions have to hold at once, and they pull in opposite
    directions:

      reachable   the queue ahead must be small enough that `capital` is
                  actually near the front, i.e. own-touch depth <= capital.
      profitable  net kept after adverse selection must be positive.

    Thin books satisfy the first and are usually claimed to violate the
    second ("do not quote into a fragile book"). This tests whether that
    trade-off has a sweet spot, splitting thin-touch fills by how much
    support sits behind the touch -- the thing that stops the price gapping
    once your level is taken.
    """
    e = ev[ev.stratum == "sweep"].copy()
    if e.empty or "support_usd" not in e.columns:
        return pd.DataFrame()
    reachable = e[e.consumed_usd <= capital]
    rows = []
    for lo in support_thresholds:
        g = reachable[reachable.support_usd >= lo]
        if len(g) < 30:
            continue
        se = g.pnl.std() / np.sqrt(len(g))
        rows.append(dict(
            min_support_usd=lo, n=len(g),
            med_touch_usd=g.consumed_usd.median(),
            med_support_usd=g.support_usd.median(),
            half_spread=g.half_spread.mean(), adverse=g.adverse.mean(),
            mean_pnl=g.pnl.mean(), se=se,
            t=g.pnl.mean() / se if se else np.nan,
            roi_pct=g.pnl.mean() / g.fill_px.mean() * 100))
    return pd.DataFrame(rows)


def combined_filter(ev: pd.DataFrame, capital: float = 50.0,
                    imb_min: float = 0.45, max_gap: float = 0.011,
                    market_type: str | None = None) -> pd.DataFrame:
    """Stack the filters that the conditional tables say actually matter.

    Three conditions, each measured above:
      1. queue ahead <= capital        -- so the position is reachable
      2. imbalance >= imb_min          -- your side thick vs the far side,
                                          which is where adverse selection
                                          roughly halves
      3. gap <= max_gap                -- the sweep only cleared one tick, so
                                          it was small flow rather than a
                                          walk through the book

    Condition 3 is the important honesty check. The reachable-queue result is
    measured on books where only ~$9 was resting. Joining with $50 makes that
    level $59, so the fills a joining order would actually get are the sweeps
    big enough to clear $59 -- not the ones that cleared $9. Filtering to
    one-tick gaps keeps only events where the flow was genuinely small.
    """
    e = ev[ev.stratum == "sweep"].copy()
    if e.empty or "gap" not in e.columns:
        return pd.DataFrame()
    if market_type and "market_type" in e.columns:
        e = e[e.market_type == market_type]
    steps = [
        ("all sweeps", e),
        ("+ queue ahead <= $%d" % capital, None),
        ("+ imbalance >= %.2f" % imb_min, None),
        ("+ gap <= 1 tick", None),
    ]
    cur = e
    rows = []
    for i, (label, _) in enumerate(steps):
        if i == 1:
            cur = cur[cur.consumed_usd <= capital]
        elif i == 2:
            cur = cur[cur.imbalance >= imb_min]
        elif i == 3:
            cur = cur[cur.gap <= max_gap]
        if len(cur) < 20:
            rows.append(dict(filter=label, n=len(cur)))
            continue
        se = cur.pnl.std() / np.sqrt(len(cur))
        rows.append(dict(
            filter=label, n=len(cur),
            half_spread=cur.half_spread.mean(), adverse=cur.adverse.mean(),
            mean_pnl=cur.pnl.mean(), se=se,
            t=cur.pnl.mean() / se if se else np.nan,
            roi_pct=cur.pnl.mean() / cur.fill_px.mean() * 100,
            med_px=cur.fill_px.median()))
    return pd.DataFrame(rows)


def summarize_by_type(ev: pd.DataFrame) -> pd.DataFrame:
    """Group on the resolved market-type NAME, not the per-file index."""
    if ev.empty or "market_type" not in ev.columns:
        return pd.DataFrame()
    e = ev[ev.stratum == "sweep"]
    s = e.groupby(["market_type", "horizon"], observed=True).agg(
        n=("pnl", "size"), half_spread=("half_spread", "mean"),
        adverse=("adverse", "mean"), mean_pnl=("pnl", "mean"),
        sd=("pnl", "std"), mean_px=("fill_px", "mean")).reset_index()
    s["se"] = s.sd / np.sqrt(s.n)
    s["t"] = s.mean_pnl / s.se
    s["roi_pct"] = s.mean_pnl / s.mean_px * 100
    return s.drop(columns=["sd"]).sort_values("n", ascending=False)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1", nargs="*", default=None,
                    help="books.jsonl file(s) to compact")
    ap.add_argument("--pass2", action="store_true")
    ap.add_argument("--horizon", type=int, nargs="*", default=[10, 30, 60])
    ap.add_argument("--min-size", type=float, default=5.0,
                    help="minimum share decrease to count as a shrink event")
    ap.add_argument("--max-spread", type=float, default=0.011,
                    help="only measure events where the pre-event spread was "
                         "this tight (default one tick)")
    ap.add_argument("--all-spreads", action="store_true",
                    help="disable the tight-book filter (diagnostic only)")
    a = ap.parse_args()

    if a.pass1:
        for src in a.pass1:
            name = os.path.basename(src).replace(".jsonl", "").replace(" ", "_")
            dst = os.path.join(OUT, f"touch_{name}.parquet")
            if os.path.exists(dst):
                log(f"skip {name} (already compacted)")
                continue
            log(f"=== pass1 {src} ===")
            pass1(src, dst)

    if a.pass2:
        files = sorted(glob.glob(os.path.join(OUT, "touch_*.parquet")))
        files = [f for f in files if "_assets" not in f and "_mtypes" not in f]
        if not files:
            raise SystemExit("no compacted touch files; run --pass1 first")
        # Each day file numbers market types independently, so the index ->
        # name maps collide when merged. Resolve names per file instead and
        # carry the string through, otherwise the by-type table is mislabelled.
        per_file_map = {}
        for f in files:
            mf = f.replace(".parquet", "_mtypes.parquet")
            per_file_map[f] = (dict(zip(*pd.read_parquet(mf)
                                        [["idx", "market_type"]].values.T))
                               if os.path.exists(mf) else {})

        max_spread = np.inf if a.all_spreads else a.max_spread
        frames = []
        for f in files:
            log(f"=== pass2 {os.path.basename(f)} ===")
            df = pd.read_parquet(f)
            log(f"  {len(df):,} touch updates, {df.asset.nunique()} assets")
            ev = detect_and_measure(df, horizons=a.horizon, min_sz=a.min_size,
                                    max_spread=max_spread)
            log(f"  {len(ev):,} events")
            if not ev.empty:
                ev["market_type"] = ev.mtype.map(per_file_map[f])
                ev["src"] = os.path.basename(f)
            frames.append(ev)
        ev = pd.concat(frames, ignore_index=True)
        ev.to_parquet(os.path.join(OUT, "events.parquet"), index=False)

        pd.set_option("display.width", 220)
        fmt = lambda v: f"{v:,.5f}"
        print("\n" + "=" * 96)
        print("MAKER P&L AFTER AN INFERRED FILL AT THE TOUCH")
        print(f"  tight-book filter: pre-event spread <= {max_spread}")
        print("  half_spread = what you earn if the mid does not move")
        print("  adverse     = how far the mid ran away from you (>0 = against you)")
        print("  pnl         = half_spread - adverse  (what you actually keep)")
        print("=" * 96)
        print(summarize(ev).to_string(index=False, float_format=fmt))

        print("\n" + "=" * 96)
        print("BY MARKET TYPE (sweep stratum only -- strongest fill evidence)")
        print("=" * 96)
        print(summarize_by_type(ev).to_string(index=False, float_format=fmt))

        print("\n" + "=" * 96)
        print("ADVERSE SELECTION vs TRADE SIZE  (sweep stratum)")
        print("  larger sweeps = more informed flow = what a back-of-queue order gets")
        print("=" * 96)
        print(queue_buckets(ev).to_string(index=False, float_format=fmt))

        print("\n" + "=" * 96)
        print("WHAT A SMALL ORDER ACTUALLY EARNS")
        print("  a $50 order behind $X of depth fills only on sweeps larger than $X")
        print("=" * 96)
        print(queue_position(ev).to_string(index=False, float_format=fmt))

        if "market_type" in ev.columns:
            print("\n  moneyline only:")
            print(queue_position(ev[ev.market_type == "moneyline"])
                  .to_string(index=False, float_format=fmt))

        print("\n" + "=" * 96)
        print("BOOK SHAPE BEFORE THE FILL -- the only thing a maker can filter on")
        print("=" * 96)
        E = [-1, 50, 200, 1000, 5000, 1e12]
        L = ["<$50", "$50-200", "$200-1k", "$1k-5k", ">$5k"]
        for col in ("support_usd", "far_usd"):
            t = book_shape(ev, col, E, L)
            if not t.empty:
                print(f"\n  by {col} (depth behind your touch / on the far side):")
                print(t.to_string(index=False, float_format=fmt))
        t = book_shape(ev, "imbalance", [-.01, .3, .45, .55, .7, 1.01],
                       ["<0.30", "0.30-0.45", "0.45-0.55", "0.55-0.70", ">0.70"])
        if not t.empty:
            print("\n  by imbalance (your side's share of top-3 depth):")
            print(t.to_string(index=False, float_format=fmt))

        print("\n" + "=" * 96)
        print("IS THERE A REGIME THAT IS BOTH PROFITABLE AND REACHABLE WITH $50?")
        print("  restricted to fills where the queue ahead was <= $50,")
        print("  then split by how much support sat behind the touch")
        print("=" * 96)
        print(small_maker_regime(ev).to_string(index=False, float_format=fmt))
        if "market_type" in ev.columns:
            for m in ("moneyline", "spread", "total"):
                t = small_maker_regime(ev[ev.market_type == m])
                if not t.empty:
                    print(f"\n  {m}:")
                    print(t.to_string(index=False, float_format=fmt))

        print("\n" + "=" * 96)
        print("STACKING THE FILTERS  (each row adds one condition)")
        print("=" * 96)
        print(combined_filter(ev).to_string(index=False, float_format=fmt))
        if "market_type" in ev.columns:
            for m in ("spread", "total", "moneyline"):
                t = combined_filter(ev, market_type=m)
                if not t.empty:
                    print(f"\n  {m}:")
                    print(t.to_string(index=False, float_format=fmt))

        print("\n" + "=" * 96)
        print("HOW FAR THE TOUCH GAPPED  (1 tick = small flow, more = a walk)")
        print("=" * 96)
        print(book_shape(ev, "gap", [-1e-9, 0.011, 0.021, 0.051, 1.0],
                         ["1 tick", "2 ticks", "3-5 ticks", ">5 ticks"])
              .to_string(index=False, float_format=fmt))
        log(f"wrote {os.path.join(OUT, 'events.parquet')}")


if __name__ == "__main__":
    main()
