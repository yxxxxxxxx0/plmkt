"""
Score every microstructure feature against paper AND executable returns.

The distinction is the whole point. A feature can predict the mid beautifully
and still be worthless, because taking the trade costs the spread. So each
feature is reported twice:

    IC_mid    rank correlation with the forward MID change. This is what the
              LOB-prediction literature usually reports, and what makes
              signals look good.
    edge_exec the mean EXECUTABLE return of actually acting on the signal:
              go long the top decile and score fwd_long (buy the ask now,
              sell the bid at t+h); go short the bottom decile and score
              fwd_short. No mid, no midpoint fills, no assumed price
              improvement.

A feature is only interesting if edge_exec > 0. On this venue the round trip
starts ~2.7-9.7c in the hole depending on market type, so almost nothing will
clear it -- and knowing precisely which features fail, and by how much, is the
useful output.

Deciles are computed WITHIN each (game, token) and then pooled, so a token
with a permanently wide spread cannot masquerade as a signal by sitting in one
tail of a globally-ranked feature. Everything is pooled across games rather
than fitted, so there is nothing here to overfit yet -- this is measurement,
not selection.

    python research_ic.py --slate 2026-09-04
    python research_ic.py --games ... --market-type moneyline --horizon 1s
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")

FEATURES = [
    "imb_touch", "micro_less_mid_c", "imb_1c", "imb_2c", "imb_5c",
    "slope_bid_c", "slope_ask_c", "spread_c",
    "rem_bid_1s", "rem_ask_1s", "add_bid_1s", "add_ask_1s",
    "tr_buy_1s", "tr_sell_1s", "tr_n_1s",
    "mom_0.1s_c", "mom_1s_c", "mom_5s_c",
    "absmove_1s_c", "absmove_5s_c",
    "bb_sz", "ba_sz", "n_bid", "n_ask",
]
# Signals that are naturally directional get a derived signed version; the
# raw one-sided quantities are kept too because their level matters as a
# regime marker even when the difference is the tradeable part.
DERIVED = {
    "ofi_rem_1s": lambda d: d["rem_ask_1s"] - d["rem_bid_1s"],
    "ofi_add_1s": lambda d: d["add_bid_1s"] - d["add_ask_1s"],
    "ofi_trade_1s": lambda d: d["tr_buy_1s"] - d["tr_sell_1s"],
    "slope_diff_c": lambda d: d["slope_ask_c"] - d["slope_bid_c"],
    "depth_ratio_2c": lambda d: np.log1p(d["dep_bid_2c"]) - np.log1p(d["dep_ask_2c"]),
}
HORIZ = ("0p1s", "0p25s", "0p5s", "1s", "2s", "5s", "10s")
NDEC = 10


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rank_deciles(x, n=NDEC):
    """Decile label per row, NaN-safe, ties handled by ordinal rank."""
    out = np.full(x.size, -1, dtype=np.int8)
    ok = np.isfinite(x)
    if ok.sum() < n * 20:
        return out
    v = x[ok]
    order = np.argsort(v, kind="stable")
    lab = np.empty(v.size, dtype=np.int8)
    edges = (np.arange(n + 1) * v.size / n).astype(np.int64)
    for k in range(n):
        lab[order[edges[k]:edges[k + 1]]] = k
    out[ok] = lab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--slate", default=None)
    ap.add_argument("--market-type", default=None,
                    help="restrict to one of moneyline/spread/total")
    ap.add_argument("--max-spread", type=float, default=None,
                    help="only rows whose quoted spread is at most this (cents)")
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pat = f"*-{args.slate}.parquet" if args.slate else "*.parquet"
    files = ([os.path.join(args.panel_dir, f"{g}.parquet") for g in args.games]
             if args.games else sorted(glob.glob(os.path.join(args.panel_dir, pat))))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit("no panel files; run build_panel.py first")

    names = FEATURES + list(DERIVED)
    # acc[(feat, horizon)] -> per-decile [n, sum_mid, sum_exec]
    acc = collections.defaultdict(lambda: np.zeros((NDEC, 3)))
    n_rows = 0

    for path in files:
        cols = (set(FEATURES) | {"dep_bid_2c", "dep_ask_2c", "market_type",
                                 "spread_c", "asset_id"}
                | {f"fwd_{k}_{h}" for k in ("mid", "long", "short") for h in HORIZ})
        have = set(pq.ParquetFile(path).schema.names)
        missing = cols - have
        if missing:
            raise SystemExit(f"{os.path.basename(path)} lacks columns: "
                             f"{sorted(missing)[:6]} -- rebuild the panel")
        tab = pq.read_table(path, columns=sorted(cols))
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in tab.column_names}
        for k, fn in DERIVED.items():
            d[k] = fn(d)

        keep = np.ones(d["spread_c"].size, dtype=bool)
        if args.market_type:
            keep &= (d["market_type"] == args.market_type)
        if args.max_spread is not None:
            keep &= np.isfinite(d["spread_c"]) & (d["spread_c"] <= args.max_spread)
        if keep.sum() < 1000:
            continue
        n_rows += int(keep.sum())

        # Deciles within each token, then pooled -- a token that simply sits
        # at a wide spread all game must not become a "signal".
        aid = d["asset_id"][keep]
        for tok in np.unique(aid):
            m = aid == tok
            if m.sum() < 500:
                continue
            sub = {k: v[keep][m] for k, v in d.items() if v.dtype != object}
            for f in names:
                lab = rank_deciles(np.asarray(sub[f], dtype=np.float64))
                if lab.max() < 0:
                    continue
                for h in HORIZ:
                    fm = np.asarray(sub[f"fwd_mid_{h}"], dtype=np.float64)
                    fl = np.asarray(sub[f"fwd_long_{h}"], dtype=np.float64)
                    fs = np.asarray(sub[f"fwd_short_{h}"], dtype=np.float64)
                    a = acc[(f, h)]
                    for k in range(NDEC):
                        sel = (lab == k) & np.isfinite(fm) & np.isfinite(fl)
                        c = int(sel.sum())
                        if not c:
                            continue
                        a[k, 0] += c
                        a[k, 1] += fm[sel].sum()
                        # top decile -> go long, bottom -> go short; the middle
                        # deciles are scored long so the monotonic shape stays
                        # visible, and only the tails are ever traded.
                        a[k, 2] += (fl[sel].sum() if k >= NDEC // 2 else fs[sel].sum())
        log(f"  {os.path.basename(path)[:-8]}: cumulative {n_rows:,} rows")

    print()
    print("=" * 104)
    print("FEATURE SCORECARD -- paper alpha vs executable edge")
    seg = args.market_type or "all liquid"
    sp = f", spread<={args.max_spread:g}c" if args.max_spread is not None else ""
    print(f"{len(files)} game(s), {n_rows:,} rows, segment: {seg}{sp}")
    print("D10-D1 mid  = top-decile minus bottom-decile forward MID move (paper alpha)")
    print("edge D10    = mean fwd_long in the top decile   (buy ask now, sell bid at t+h)")
    print("edge D1     = mean fwd_short in the bottom decile (sell bid now, buy ask at t+h)")
    print("=" * 104)

    rows = []
    for f in names:
        for h in HORIZ:
            a = acc[(f, h)]
            if a[:, 0].sum() < 5000:
                continue
            n = a[:, 0]
            mid = np.divide(a[:, 1], np.maximum(n, 1))
            ex = np.divide(a[:, 2], np.maximum(n, 1))
            rows.append((f, h, mid[-1] - mid[0], ex[-1], ex[0],
                         int(n[-1]), int(n[0])))

    best = sorted(rows, key=lambda r: -max(r[3], r[4]))
    print(f"{'feature':>16} {'horiz':>6} {'D10-D1 mid':>11} {'edge D10':>10} "
          f"{'edge D1':>10} {'n D10':>10} {'n D1':>10}")
    print("-" * 104)
    for r in best[:40]:
        print(f"{r[0]:>16} {r[1]:>6} {r[2]:>+10.4f}c {r[3]:>+9.4f}c "
              f"{r[4]:>+9.4f}c {r[5]:>10,} {r[6]:>10,}")

    pos = [r for r in best if max(r[3], r[4]) > 0]
    print()
    print(f"features with POSITIVE executable edge in a tail decile: {len(pos)} "
          f"of {len(rows)} (feature, horizon) pairs")
    for r in pos[:15]:
        side = "long D10" if r[3] > r[4] else "short D1"
        print(f"    {r[0]:>16} @ {r[1]:<5} {side}  {max(r[3], r[4]):+.4f}c/share")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"segment": seg, "max_spread": args.max_spread,
                       "n_rows": n_rows, "games": len(files),
                       "rows": [{"feature": r[0], "horizon": r[1],
                                 "d10_d1_mid": r[2], "edge_d10": r[3],
                                 "edge_d1": r[4], "n_d10": r[5], "n_d1": r[6]}
                                for r in rows]}, fh, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
