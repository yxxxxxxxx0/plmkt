"""
Do different MLB markets on the same game lead or lag each other?

YES/NO is not a candidate: those two books are one book mirrored exactly
(bid_YES + ask_NO = 100c with identical size, 100.0000% of simultaneous
states), so nothing can lead anything there. The real candidates are markets
whose books are genuinely independent but whose prices are driven by the same
underlying game state:

    moneyline   P(team wins)
    spread      P(team wins by more than the line)
    total       P(combined runs over the line)

A run scoring must move all of them. If one reprices first, the others are
briefly stale and predictable.

Method
------
Every market is reduced to a single signed series so that different contracts
are comparable: the mid of the token whose outcome is the HOME-favouring side,
in cents. Changes are then cross-correlated at a grid of lags on a common
event-time axis, forward-filled (never interpolated).

Two guards that decide whether the answer means anything:

  * crossed books excluded -- 0.29% of rows have ask <= bid, which is a
    reconstruction artifact and fabricates spurious co-movement.
  * a lag peak is only reported as a lead if it beats lag 0 by more than the
    bootstrap noise band. With millions of rows almost any lag is
    "significant" by a t-test, so the comparison that matters is the SHAPE of
    the curve, not its distance from zero.

Even a real lead is only tradeable if the laggard's spread is smaller than the
predicted move, so the mean spread of each market type is printed alongside.

    python cross_market.py --slate 2026-09-04
"""
import argparse
import collections
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")
TOKMAP = os.path.join(BASE, "data", "token_map.json")
LAGS_MS = (-2000, -1000, -500, -250, -100, -50, 0, 50, 100, 250, 500, 1000, 2000)
TYPES = ("moneyline", "spread", "total")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ffill_onto(src_t, src_v, grid):
    i = np.searchsorted(src_t, grid, side="right") - 1
    out = np.full(grid.size, np.nan)
    ok = i >= 0
    out[ok] = src_v[i[ok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--slate", default=None)
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--min-rows", type=int, default=20000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with open(TOKMAP, "r", encoding="utf-8") as f:
        tokens = json.load(f)["tokens"]

    pat = f"*-{args.slate}.parquet" if args.slate else "*.parquet"
    files = ([os.path.join(args.panel_dir, f"{g}.parquet") for g in args.games]
             if args.games else sorted(glob.glob(os.path.join(args.panel_dir, pat))))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit("no panel files")

    corr = collections.defaultdict(list)
    spreads = collections.defaultdict(list)
    n_game = 0

    for path in files:
        slug = os.path.basename(path)[:-8]
        cols = ["t", "bb_c", "ba_c", "spread_c", "asset_id", "market_type"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        aid = d["asset_id"]

        # One representative series per market type: the busiest token of that
        # type, restricted to outcome_index 0 so the sign convention is stable
        # across contracts.
        series = {}
        for mt in TYPES:
            best, best_n = None, 0
            for a in np.unique(aid[d["market_type"] == mt]):
                info = tokens.get(a) or {}
                if info.get("outcome") not in ("YES", None):
                    continue
                idx = np.flatnonzero(aid == a)
                if idx.size > best_n:
                    best, best_n = idx, idx.size
            if best is not None and best_n >= args.min_rows:
                bb, ba = d["bb_c"][best], d["ba_c"][best]
                good = np.isfinite(bb) & np.isfinite(ba) & (ba > bb)
                if good.sum() >= args.min_rows:
                    series[mt] = (d["t"][best][good], ((bb + ba) / 2.0)[good])
                    spreads[mt].append(float(np.nanmean(d["spread_c"][best][good])))
        if len(series) < 2:
            continue
        n_game += 1

        for x, y in itertools.combinations(sorted(series), 2):
            tx, vx = series[x]
            ty, vy = series[y]
            grid = np.union1d(tx, ty)
            if grid.size < args.min_rows:
                continue
            gx = ffill_onto(tx, vx, grid)
            gy = ffill_onto(ty, vy, grid)
            dx = np.diff(gx)
            dy = np.diff(gy)
            gt = grid[1:]
            m = np.isfinite(dx) & np.isfinite(dy)
            if m.sum() < args.min_rows or dx[m].std() == 0 or dy[m].std() == 0:
                continue
            for L in LAGS_MS:
                # y shifted forward by L: positive L asks whether x's move now
                # is echoed by y LATER, i.e. x leads y.
                ysh = ffill_onto(gt[m], dy[m], gt[m] + L / 1000.0)
                ok = np.isfinite(ysh)
                if ok.sum() < 1000:
                    continue
                a_, b_ = dx[m][ok], ysh[ok]
                if a_.std() > 0 and b_.std() > 0:
                    corr[(x, y, L)].append(float(np.corrcoef(a_, b_)[0, 1]))
        log(f"  {slug}: {n_game} game(s) with >=2 usable market types")

    print()
    print("=" * 88)
    print("CROSS-MARKET LEAD-LAG  (same game, genuinely independent books)")
    print(f"{n_game} game(s). positive lag = the SECOND market echoes later, "
          f"i.e. the first leads.")
    print("=" * 88)
    print(f"  mean quoted spread: " + "  ".join(
        f"{k}={np.mean(v):.2f}c" for k, v in sorted(spreads.items()) if v))
    print()
    for x, y in itertools.combinations(TYPES, 2):
        rows = [(L, corr.get((x, y, L))) for L in LAGS_MS]
        rows = [(L, v) for L, v in rows if v]
        if not rows:
            continue
        print(f"  {x} -> {y}   ({len(rows[0][1])} games)")
        base = dict(rows).get(0)
        b0 = np.mean(base) if base else float("nan")
        for L, v in rows:
            mu = np.mean(v)
            se = np.std(v) / max(np.sqrt(len(v)), 1)
            flag = ""
            if L != 0 and np.isfinite(b0) and mu > b0 + 2 * se:
                flag = "  <-- exceeds lag 0"
            print(f"     {L:>6}ms  corr {mu:>8.4f}  +/-{2*se:.4f}{flag}")
        print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"games": n_game,
                       "corr": {f"{x}|{y}|{L}": float(np.mean(v))
                                for (x, y, L), v in corr.items() if v},
                       "spread": {k: float(np.mean(v)) for k, v in spreads.items() if v}},
                      f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
