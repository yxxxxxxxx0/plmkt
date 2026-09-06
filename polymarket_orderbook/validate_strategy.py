"""
Walk-forward validation of the market-making strategy, with an explicit
look-ahead audit and a data-continuity gate.

Everything reported earlier was edge PER FILL pooled over a slate, with
quintile boundaries taken from each token's own first half. That is close to
causal but not actually a walk-forward test, and "close to causal" is not a
standard worth trusting money to. This file re-does it under rules that can
be checked line by line.

LOOK-AHEAD AUDIT -- every parameter, and where it comes from
------------------------------------------------------------
  Lee-Mykland sigma      trailing bipower window, strictly before t     OK
  Lee-Mykland threshold  Gumbel critical value, depends only on n       OK
  Hawkes intensity       recursion over jumps strictly before t         OK
  Hawkes decay           FIXED constant, never tuned on test data       OK
  Kalman Q, R            first 20% of each token -- and that warmup
                         region is then EXCLUDED from evaluation, so no
                         row is scored with parameters fitted on itself  OK
  intensity buckets      quantiles from EARLIER GAMES ONLY, never from
                         the game being evaluated                       OK
  sizing weights mu/s^2  estimated on EARLIER GAMES ONLY                 OK
  adverse selection      uses future mid -- it is the TARGET, not an
                         input; no feature touches it                    OK

The one thing that is NOT causal is the fill itself: a fill is taken from the
observed trade tape, so this measures the edge available on flow that really
happened. That is a fill-rate assumption, not a look-ahead bias, and it is
stated rather than hidden -- it is the same open question a live pilot has to
settle.

DATA CONTINUITY GATE
--------------------
A recorded feed is not automatically a continuous one. Rows are dropped when:
  * the book is crossed (ask <= bid) or locked (ask == bid)
  * either side is empty
  * the previous update is more than --max-gap seconds old, which means the
    quote is stale or the socket dropped. A "price move" measured across a
    reconnect never happened at a tradeable price.
  * the row falls inside a detected feed gap, or within --gap-buffer seconds
    after one, since the first quotes after a reconnect are the exchange
    resynchronising rather than the market moving.
  * the row is inside the Kalman warmup region.

    python validate_strategy.py --market-type moneyline
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

from jump_models import hawkes_intensity, lee_mykland

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel")
DECAY = 0.5
NBUCKET = 5
WARMUP = 0.20


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_trades(slug, tape="auto"):
    """Fills for one game, preferring the millisecond tape.

    Returns (fills, source). The source is carried through to every report
    because it changes what the numbers mean: the backfilled tape stamps
    trades to the SECOND, and one second holds a median of ~85 book updates
    here, so the mid at the recorded "fill time" is uncertain. That blurring
    averages away part of the immediate adverse move and therefore
    OVERSTATES maker edge. Live-ms and backfill games must never be pooled
    into a single headline number without saying so.
    """
    live = os.path.join(BASE, "data", "trades_live", f"{slug}.jsonl")
    back = os.path.join(BASE, "data", "trades", f"{slug}.jsonl")
    if tape in ("auto", "live") and os.path.exists(live):
        path, ms, src = live, True, "live-ms"
    elif tape in ("auto", "backfill") and os.path.exists(back):
        path, ms, src = back, False, "backfill-s"
    else:
        return {}, "none"
    rows = collections.defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            aid = t["asset_id"] if ms else t["asset"]
            ts = float(t["ts"]) / 1000.0 if ms else float(t["timestamp"])
            rows[aid].append((ts, float(t["size"]),
                              1.0 if (t.get("side") or "").upper() == "BUY" else -1.0))
    out = {a: (np.array([x[0] for x in v]), np.array([x[1] for x in v]),
               np.array([x[2] for x in v]))
           for a, v in ((a, sorted(v)) for a, v in rows.items())}
    return out, src


def token_fills(d, sel, trades, aid, horizon, max_gap, gap_buffer, quality):
    """Clean, continuous fills for one token, with causal jump intensity.

    Returns (intensity, net_pnl, half_spread, adverse, fill_times) or None.
    """
    t = d["t"][sel]
    mid = d["mid_c"][sel]
    spr = d["spread_c"][sel]
    bb, ba = d["bb_c"][sel], d["ba_c"][sel]
    n = t.size
    if n < 20000 or aid not in trades:
        return None

    # --- continuity: find feed gaps -----------------------------------------
    dt = np.diff(t, prepend=t[0])
    gap = dt > max_gap
    quality["rows"] += n
    quality["gap_rows"] += int(gap.sum())
    # a row is unusable if it IS a gap, or sits within gap_buffer after one
    bad = gap.copy()
    if gap.any():
        gidx = np.flatnonzero(gap)
        for g in gidx:
            end = np.searchsorted(t, t[g] + gap_buffer, side="right")
            bad[g:end] = True
    quality["post_gap_rows"] += int(bad.sum() - gap.sum())

    warm = int(n * WARMUP)
    usable = ~bad
    usable[:warm] = False                       # Kalman warmup excluded
    quality["warmup_rows"] += warm
    if usable.sum() < 5000:
        return None

    L, r, sig = lee_mykland(mid)
    nn = max(n, 2)
    Cn = (2 * np.log(nn)) ** 0.5 / 0.7979 - (
        np.log(np.pi) + np.log(np.log(nn))) / (2 * 0.7979 * (2 * np.log(nn)) ** 0.5)
    Sn = 1.0 / (0.7979 * (2 * np.log(nn)) ** 0.5)
    jump = np.abs(L) > (Cn + Sn * (-np.log(-np.log(0.99))))
    lam = hawkes_intensity(t, jump, DECAY)

    ttime, tsz, tsign = trades[aid]
    k = np.searchsorted(t, ttime, side="right") - 1
    ok = k >= 0
    k = k[ok]
    sgn = tsign[ok]
    ft = ttime[ok]
    keep = usable[k]
    quality["fills_seen"] += int(ok.sum())
    quality["fills_dropped_continuity"] += int((~keep).sum())
    if keep.sum() < 30:
        return None
    k, sgn, ft = k[keep], sgn[keep], ft[keep]

    j = np.minimum(np.searchsorted(t, ft + horizon, side="left"), n - 1)
    real = (t[j] - ft) >= horizon * 0.5
    adv = (mid[j] - mid[k]) * sgn
    half = spr[k] / 2.0
    v = np.isfinite(adv) & np.isfinite(half) & real
    quality["fills_dropped_target"] += int((~v).sum())
    if v.sum() < 30:
        return None
    return lam[k][v], (half - adv)[v], half[v], adv[v], ft[v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", default=None,
                    help="restrict to one slate; omit for every panel game")
    ap.add_argument("--tape", default="auto", choices=("auto", "live", "backfill"))
    ap.add_argument("--market-type", default="moneyline")
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--max-gap", type=float, default=30.0,
                    help="a quote older than this is stale / post-reconnect")
    ap.add_argument("--gap-buffer", type=float, default=5.0,
                    help="also drop this many seconds after a detected gap")
    ap.add_argument("--min-train-games", type=int, default=4)
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pat = f"*-{args.slate}.parquet" if args.slate else "*.parquet"
    files = sorted(glob.glob(os.path.join(args.panel_dir, pat)))
    if len(files) < args.min_train_games + 2:
        raise SystemExit("not enough games")

    quality = collections.Counter()
    per_game = {}
    log(f"loading {len(files)} games...")
    for path in files:
        slug = os.path.basename(path)[:-8]
        trades, tsrc = load_trades(slug, args.tape)
        if not trades:
            continue
        cols = ["t", "mid_c", "spread_c", "bb_c", "ba_c", "asset_id", "market_type"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        # quality gate: two-sided, uncrossed, unlocked
        base = np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"])
        quality["raw_rows"] += base.size
        crossed = base & (d["ba_c"] <= d["bb_c"])
        quality["crossed_rows"] += int(crossed.sum())
        ok = base & (d["ba_c"] > d["bb_c"])
        if args.market_type:
            ok &= (d["market_type"] == args.market_type)

        lam_all, net_all, half_all, adv_all, ft_all = [], [], [], [], []
        for aid in np.unique(d["asset_id"][ok]):
            sel = ok & (d["asset_id"] == aid)
            r = token_fills(d, sel, trades, aid, args.horizon,
                            args.max_gap, args.gap_buffer, quality)
            if r is None:
                continue
            lam_all.append(r[0]); net_all.append(r[1]); half_all.append(r[2])
            adv_all.append(r[3]); ft_all.append(r[4])
        if not lam_all:
            continue
        per_game[slug] = {
            "lam": np.concatenate(lam_all), "net": np.concatenate(net_all),
            "half": np.concatenate(half_all), "adv": np.concatenate(adv_all),
            "t0": float(np.concatenate(ft_all).min()), "tape": tsrc,
        }
        log(f"  {slug} [{tsrc}]: {per_game[slug]['net'].size:,} clean fills")

    print()
    print("=" * 92)
    print("DATA QUALITY GATE")
    print("=" * 92)
    q = quality
    print(f"  panel rows scanned          {q['raw_rows']:>14,}")
    print(f"  crossed books removed       {q['crossed_rows']:>14,}  "
          f"({q['crossed_rows']/max(q['raw_rows'],1)*100:.4f}%)")
    print(f"  rows in liquid tokens       {q['rows']:>14,}")
    print(f"  feed-gap rows (> {args.max_gap:g}s)     {q['gap_rows']:>14,}  "
          f"({q['gap_rows']/max(q['rows'],1)*100:.4f}%)")
    print(f"  post-gap buffer rows        {q['post_gap_rows']:>14,}")
    print(f"  Kalman warmup rows excluded {q['warmup_rows']:>14,}")
    print(f"  fills observed              {q['fills_seen']:>14,}")
    print(f"  fills dropped: continuity   {q['fills_dropped_continuity']:>14,}")
    print(f"  fills dropped: no target    {q['fills_dropped_target']:>14,}")
    kept = sum(v["net"].size for v in per_game.values())
    print(f"  CLEAN FILLS USED            {kept:>14,}  "
          f"({kept/max(q['fills_seen'],1)*100:.1f}% of observed)")

    # ---- walk-forward -----------------------------------------------------
    order = sorted(per_game, key=lambda s: per_game[s]["t0"])
    print()
    print("=" * 92)
    print(f"WALK-FORWARD  ({args.market_type}, horizon {args.horizon:g}s)")
    print("buckets and sizing weights fitted on EARLIER games only, then applied "
          "unchanged forward")
    print("=" * 92)
    print(f"  {'#':>3} {'test game':>28} {'tape':>11} {'fills':>7} "
          f"{'equal-size':>12} {'mu/sd2-sized':>14} {'delta':>10}")
    print("-" * 92)
    rows = []
    for i in range(args.min_train_games, len(order)):
        tr = order[:i]
        te = order[i]
        lam_tr = np.concatenate([per_game[s]["lam"] for s in tr])
        net_tr = np.concatenate([per_game[s]["net"] for s in tr])
        edges = np.quantile(lam_tr, np.linspace(0, 1, NBUCKET + 1)[1:-1])
        b_tr = np.searchsorted(edges, lam_tr)
        wts = np.ones(NBUCKET)
        for bi in range(NBUCKET):
            m = b_tr == bi
            if m.sum() > 50:
                mu = net_tr[m].mean()
                var = max(net_tr[m].var(), 1e-9)
                wts[bi] = max(mu / var, 0.0)
        wts = wts / max(wts.max(), 1e-9)          # relative size, capped at 1

        lam_te = per_game[te]["lam"]
        net_te = per_game[te]["net"]
        b_te = np.searchsorted(edges, lam_te)
        w = wts[np.clip(b_te, 0, NBUCKET - 1)]
        eq = float(net_te.mean())
        sz = float((w * net_te).sum() / max(w.sum(), 1e-9))
        # Comparing per-SHARE edge is not a fair test of a sizing rule: any
        # rule that shrinks the highest-edge bucket lowers per-share edge by
        # construction while also lowering risk. The question is whether the
        # risk fell FASTER than the return, so both schemes are scored by the
        # Sharpe of their realised per-fill PnL series.
        pnl_eq = net_te
        pnl_sz = w * net_te
        def ir(x):
            sd = x.std(ddof=1)
            return float(x.mean() / sd * np.sqrt(x.size)) if sd > 0 else float("nan")
        rows.append({"game": te, "fills": int(net_te.size), "equal": eq,
                     "sized": sz, "weights": wts.tolist(),
                     "tape": per_game[te]["tape"],
                     "slate": te[-10:],
                     "ir_equal": ir(pnl_eq), "ir_sized": ir(pnl_sz),
                     "tot_equal": float(pnl_eq.sum()),
                     "tot_sized": float(pnl_sz.sum()),
                     "cap_equal": float(pnl_eq.size),
                     "cap_sized": float(w.sum())})
        print(f"  {i-args.min_train_games+1:>3} {te:>28} "
              f"{per_game[te]['tape']:>11} {net_te.size:>7,} "
              f"{eq:>+11.4f}c {sz:>+13.4f}c {sz-eq:>+9.4f}c")

    if rows:
        eqs = np.array([r["equal"] for r in rows])
        szs = np.array([r["sized"] for r in rows])
        nn = np.array([r["fills"] for r in rows])
        print("-" * 92)
        print(f"  {'':>3} {'MEAN (fill-weighted)':>28} {nn.sum():>8,} "
              f"{np.average(eqs, weights=nn):>+11.4f}c "
              f"{np.average(szs, weights=nn):>+13.4f}c "
              f"{np.average(szs-eqs, weights=nn):>+9.4f}c")
        print(f"  games positive: equal-size {int((eqs>0).sum())}/{len(eqs)}   "
              f"sized {int((szs>0).sum())}/{len(szs)}")
        se = eqs.std(ddof=1) / np.sqrt(len(eqs)) if len(eqs) > 1 else float("nan")
        print(f"  equal-size mean {eqs.mean():+.4f}c +/- {2*se:.4f}c (2 s.e. "
              f"across games)")
        print(f"  t-stat vs zero: {eqs.mean()/se:.2f}" if se > 0 else "")
        print()
        print("  BY SLATE (chronological) and BY TAPE SOURCE:")
        print(f"    {'group':>16} {'games':>7} {'fills':>9} {'equal-size':>12} "
              f"{'positive':>10}")
        groups = collections.defaultdict(list)
        for r in rows:
            groups[("slate", r["slate"])].append(r)
            groups[("tape", r["tape"])].append(r)
        for key in sorted(groups, key=lambda k: (k[0], k[1])):
            g = groups[key]
            n = sum(x["fills"] for x in g)
            m = np.average([x["equal"] for x in g],
                           weights=[x["fills"] for x in g])
            pos = sum(1 for x in g if x["equal"] > 0)
            print(f"    {key[1]:>16} {len(g):>7} {n:>9,} {m:>+11.4f}c "
                  f"{pos:>7}/{len(g)}")
        print("    (backfill-s games OVERSTATE edge -- second-resolution fill "
              "timestamps blur adverse selection)")
        print()
        ire = np.array([r["ir_equal"] for r in rows])
        irs = np.array([r["ir_sized"] for r in rows])
        print()
        print("  RISK-ADJUSTED comparison (Sharpe of the realised per-fill PnL):")
        print(f"    {'game':>28} {'IR equal':>10} {'IR sized':>10} {'capital':>9}")
        for r in rows:
            print(f"    {r['game']:>28} {r['ir_equal']:>10.2f} {r['ir_sized']:>10.2f} "
                  f"{r['cap_sized']/max(r['cap_equal'],1)*100:>8.0f}%")
        print(f"    {'MEAN':>28} {np.nanmean(ire):>10.2f} {np.nanmean(irs):>10.2f}")
        print(f"    sized better on IR in {int((irs>ire).sum())}/{len(irs)} games; "
              f"capital deployed {np.mean([r['cap_sized']/max(r['cap_equal'],1) for r in rows])*100:.0f}% of equal")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"quality": dict(quality), "walk_forward": rows}, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
