"""
Perfect-foresight jump-capture simulation: if you caught EVERY price jump --
bought at the exact bottom and sold at the exact top -- would the round trip
survive Polymarket's costs?

The premise is deliberately impossible. Entry and exit are chosen with full
knowledge of the future, so nothing here is a strategy that could be run. It
is an upper bound: whatever this says you make is strictly more than any real
signal could make on the same book. If the bound is negative, no amount of
predictive skill fixes it, because the loss is structural.

What is granted (free, perfect)
-------------------------------
  * the timing of every swing, chosen after seeing the whole game
  * instant fills. Verified fair, not assumed: taker_delay_check.py reports
    seconds_delay = 0 for all 52 markets across the games sampled here, so a
    taker order does execute against the book you can see.

What is charged (measured, not modelled)
----------------------------------------
  * THE SPREAD. Entry lifts the ask, exit hits the bid. This is the cost that
    perfect foresight cannot dodge: you know where the price is going, but you
    still have to cross to get on and off.
  * DEPTH. Orders walk the real recorded ladder, level by level, so a size
    bigger than the touch pays the second level and the third.
  * THE TAKER FEE, on both legs, since both legs are taker orders:

        fee = shares * rate * p * (1 - p)          [per fill, per level]

    That is Polymarket's published formula, and the shape of it drives the
    whole result: the fee peaks at p = 0.50 and vanishes toward 0 and 1, so
    the same 2c jump is cheap in the tails and ruinous at a coin flip.
    Makers are never charged; both our legs are takers, so we pay twice.

    `rate` is ambiguous in Polymarket's own sources, so this does not pick
    one -- it reports a sweep:
      0.00  no fee (the pure spread+depth cost; assumption-free floor)
      0.05  the documented sports rate (docs.polymarket.com/trading/fees)
      0.10  what the live API actually returns for these MLB markets
            (GET /fee-rate -> {"base_fee": 1000}, i.e. 1000bps = 0.10)
    Since fee scales linearly in `rate`, the breakeven rate is also reported,
    which settles the question without needing to resolve the discrepancy.

What is excluded, and why it has to be
--------------------------------------
A book quoting 0.01 bid / 0.99 ask has a "mid" of 0.50, and when the stub
quotes twitch, that mid "jumps" tens of cents. Those are not price moves and
no one can trade them: the round trip costs 96c of spread. Before the game
starts most of the exotic books (extra innings, 1st-5-innings totals) look
exactly like this, so counting them would bury the real result under
artifacts of quotes nobody could hit.

So a swing is only tradeable if, at BOTH ends, the book is two-sided, the
spread is at most --max-spread, and at least --min-size shares rest on the
side being hit. Every rejected candidate is counted and reported, so the gate
is visible rather than silent. --window game (the default) also trims to
kickoff -> final out, which removes most of the pregame stub era on its own.

How "every jump" is defined
---------------------------
A zigzag/swing filter on the mid price with threshold `theta`. Each completed
up-swing of at least theta is one trade: buy at the trough, sell at the peak.
This is the standard way to enumerate swings without double counting -- legs
strictly alternate, so no tick is ever sold twice or bought twice.

theta matters and is swept, because it is the honest knob: at theta = 0.1c you
"catch" hundreds of thousands of one-tick wiggles that cannot pay for a
spread; at theta = 5c you catch few, real, large moves. Reporting a single
theta would be choosing the answer.

A third scenario, MAKER, is reported alongside: instead of crossing, you
rest a bid at the trough and an offer at the peak. Makers are never charged a
fee, and you earn the spread rather than paying it, so the same swings look
completely different. Treat it as a ceiling, not a plan -- it assumes both
resting orders fill, which is the one thing a maker cannot choose. In reality
a resting bid fills when someone insists on selling to you, which is
disproportionately when the price is about to keep falling. That adverse
selection is not modelled here and is not small.

Two execution sizes are reported per configuration:
  * 1 share    -- touch price only, no depth cost. The most favourable case
                  that exists; a strict upper bound on the upper bound.
  * --shares N -- walks the real ladder on both legs, and is capped by the
                  size actually resting there, so it never invents liquidity.

Usage:
    python sim_jump_capture.py --games mlb-ari-sf-2026-08-27
    python sim_jump_capture.py --slate 2026-08-27 --shares 500
    python sim_jump_capture.py --slate 2026-08-27 --json out.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

import rebuild_duckdb as rd
import rebuild_multi as rm

BASE = os.path.dirname(os.path.abspath(__file__))

# Swing thresholds in cents. 0.1c is the finest tick any of these markets
# quotes (minimum_tick_size is 0.001 on the O/U books, 0.01 on others).
THETAS_C = (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
FEE_RATES = (0.0, 0.05, 0.10)

# Fee peaks at p=0.5 and dies in the tails, so results are bucketed by entry
# price -- an average over all prices would hide exactly the effect that
# decides whether this is profitable.
PRICE_BUCKETS = ((0.0, 0.05), (0.05, 0.15), (0.15, 0.35),
                 (0.35, 0.65), (0.65, 0.85), (0.85, 0.95), (0.95, 1.0))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def best_price_column(lo, hi, off, px):
    """Best price per snapshot row, vectorized -- NaN where that side is empty.

    Level 0 is the best level: the recorder writes bids descending and asks
    ascending, which is the same assumption build_asset_runs makes (it takes
    bids[0,0] as best_bid) and is checked in verify_level_order below.
    """
    r = np.arange(lo, hi)
    a, b = off[r], off[r + 1]
    has = b > a
    out = np.full(r.size, np.nan)
    out[has] = px[a[has]]
    return out


def verify_level_order(lo, hi, boff, bpx, aoff, apx, sample=400):
    """Confirm level 0 really is the best level, on a sample of rows.

    Cheap insurance against a silent sign error: if the recorder ever wrote a
    book unsorted, every entry and exit price in this simulation would be
    wrong in the flattering direction, and nothing else here would notice.
    """
    rows = np.linspace(lo, hi - 1, min(sample, hi - lo)).astype(np.int64)
    for r in rows:
        b0, b1 = boff[r], boff[r + 1]
        a0, a1 = aoff[r], aoff[r + 1]
        if b1 - b0 > 1 and bpx[b0] < bpx[b0:b1].max():
            return False
        if a1 - a0 > 1 and apx[a0] > apx[a0:a1].min():
            return False
    return True


def zigzag_up_legs(v, theta):
    """Index pairs (i_low, i_high) for every completed up-swing >= theta.

    Standard alternating-extreme swing filter. A leg is emitted only once the
    price has retraced theta from the peak, which is what makes the peak a
    real local maximum rather than whatever the last tick happened to be.
    The final in-progress up-leg is emitted too, since its high is known.
    """
    n = v.size
    if n < 2:
        return []
    # Seed the extremes on the first tick that HAS a two-sided book. Seeding
    # at index 0 unconditionally is wrong and silently so: if v[0] is NaN,
    # every comparison against it is False, i_lo/i_hi never advance, and the
    # token yields zero legs no matter what the price did afterwards. Books
    # that are one-sided at kickoff are common, so that is not a corner case.
    start = -1
    for k in range(n):
        if v[k] == v[k]:
            start = k
            break
    if start < 0 or start == n - 1:
        return []
    legs = []
    i_lo = i_hi = start
    trend = 0
    for i in range(start + 1, n):
        x = v[i]
        if x != x:  # NaN: no two-sided book, cannot trade here
            continue
        if trend == 1:
            if x >= v[i_hi]:
                i_hi = i
            elif v[i_hi] - x >= theta:
                legs.append((i_lo, i_hi))
                trend = -1
                i_lo = i
        elif trend == -1:
            if x <= v[i_lo]:
                i_lo = i
            elif x - v[i_lo] >= theta:
                trend = 1
                i_hi = i
        else:
            if x - v[i_lo] >= theta:
                trend, i_hi = 1, i
            elif v[i_hi] - x >= theta:
                trend, i_lo = -1, i
            else:
                if x < v[i_lo]:
                    i_lo = i
                if x > v[i_hi]:
                    i_hi = i
    if trend == 1 and i_hi > i_lo:
        legs.append((i_lo, i_hi))
    return legs


def walk_ladder(row, off, px, sz, shares):
    """Take `shares` from one side at `row`, level by level.

    Returns (filled, notional, fee_basis) where fee_basis = sum p(1-p)*qty, so
    a fee at any rate is rate*fee_basis -- that is what lets one pass over the
    data answer every rate in the sweep.
    """
    i0, i1 = off[row], off[row + 1]
    if i1 <= i0:
        return 0.0, 0.0, 0.0
    p = px[i0:i1]
    s = sz[i0:i1]
    before = np.concatenate(([0.0], np.cumsum(s)[:-1]))
    take = np.clip(shares - before, 0.0, s)
    filled = float(take.sum())
    if filled <= 0:
        return 0.0, 0.0, 0.0
    return filled, float((p * take).sum()), float((p * (1.0 - p) * take).sum())


def side_total(row, off, sz):
    i0, i1 = off[row], off[row + 1]
    return float(sz[i0:i1].sum()) if i1 > i0 else 0.0


def bucket_of(p):
    for k, (lo, hi) in enumerate(PRICE_BUCKETS):
        if lo <= p < hi or (k == len(PRICE_BUCKETS) - 1 and p == hi):
            return k
    return len(PRICE_BUCKETS) - 1


class Agg:
    """Running totals for one (theta, size) configuration."""

    __slots__ = ("n", "shares", "gross", "basis", "hold", "move",
                 "by_bucket", "by_mtype", "wins", "spread_in")

    def __init__(self):
        self.n = 0
        self.shares = 0.0
        self.gross = 0.0          # proceeds - cost, in dollars
        self.basis = 0.0          # sum p(1-p)*qty over both legs
        self.hold = 0.0           # seconds held, summed
        self.move = 0.0           # sum of move*shares, so move/shares is
                                  # the share-weighted swing per share
        self.spread_in = 0.0      # quoted spread at entry, summed (dollars)
        self.wins = {r: 0 for r in FEE_RATES}
        self.by_bucket = {}
        self.by_mtype = {}

    def add(self, shares, gross, basis, hold, move, bucket, mtype, spread_in=0.0):
        self.n += 1
        self.shares += shares
        self.gross += gross
        self.basis += basis
        self.hold += hold
        self.move += move * shares
        self.spread_in += spread_in
        for r in FEE_RATES:
            if gross - r * basis > 0:
                self.wins[r] += 1
        for key, store in ((bucket, self.by_bucket), (mtype, self.by_mtype)):
            d = store.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0, 0.0])
            d[0] += 1
            d[1] += shares
            d[2] += gross
            d[3] += basis
            d[4] += move * shares
            d[5] += spread_in

    def net(self, rate):
        return self.gross - rate * self.basis

    def breakeven_rate(self):
        """The fee rate at which this configuration makes exactly zero."""
        return self.gross / self.basis if self.basis > 0 else float("nan")


def simulate_game(slug, window, shares_target, agg1, aggN, aggM,
                  split_dir=None, max_spread=0.05, min_size=5.0, rej=None):
    cols = rd.load_columns(slug, split_dir=split_dir or rm.JSONL_SPLIT_DIR)
    asset_ids, ts, bpx, bsz, boff, apx, asz, aoff, meta = cols
    boundaries = rd.asset_boundaries(asset_ids)
    if window:
        t_lo, t_hi = window["kickoff_ts"], window["game_end_ts"]
    else:
        t_lo, t_hi = -np.inf, np.inf

    n_tok = 0
    for aid, lo, hi in boundaries:
        if not verify_level_order(lo, hi, boff, bpx, aoff, apx):
            log(f"    !! {aid[:12]}: book levels not sorted best-first; skipped")
            continue
        t = rd.dedup_nudge(ts[lo:hi])
        keep = (t >= t_lo) & (t <= t_hi)
        if keep.sum() < 2:
            continue
        bb = best_price_column(lo, hi, boff, bpx)
        ba = best_price_column(lo, hi, aoff, apx)
        rows = np.arange(lo, hi)[keep]
        t = t[keep]
        bb = bb[keep]
        ba = ba[keep]

        two_sided = ~(np.isnan(bb) | np.isnan(ba))
        mid = np.where(two_sided, (bb + ba) / 2.0, np.nan)
        mtype = (meta.get(aid) or {}).get("market_type") or "?"
        n_tok += 1

        for theta_c in THETAS_C:
            theta = theta_c / 100.0
            legs = zigzag_up_legs(mid, theta)
            for i, j in legs:
                if not (two_sided[i] and two_sided[j]):
                    rej[theta_c]["one_sided"] += 1
                    continue
                # Tradeability gate at BOTH ends. Checked before anything is
                # counted, so a rejected candidate never reaches the totals.
                if (ba[i] - bb[i] > max_spread) or (ba[j] - bb[j] > max_spread):
                    rej[theta_c]["wide_spread"] += 1
                    continue
                if (side_total(rows[i], aoff, asz) < min_size
                        or side_total(rows[j], boff, bsz) < min_size):
                    rej[theta_c]["thin"] += 1
                    continue
                rej[theta_c]["traded"] += 1
                entry_ask, exit_bid = ba[i], bb[j]
                entry_bid, exit_ask = bb[i], ba[j]
                hold = float(t[j] - t[i])
                move = float(mid[j] - mid[i])
                bkt = bucket_of(float(mid[i]))

                # 1-share leg: touch prices, zero depth cost.
                g1 = exit_bid - entry_ask
                b1 = entry_ask * (1 - entry_ask) + exit_bid * (1 - exit_bid)
                agg1[theta_c].add(1.0, g1, b1, hold, move, bkt, mtype,
                                  float(ba[i] - bb[i]))

                # Maker: buy the bid at the trough, sell the ask at the peak.
                # Earns the spread instead of paying it, and pays no fee at
                # all -- fee_basis is 0 because makers are never charged.
                aggM[theta_c].add(1.0, exit_ask - entry_bid, 0.0, hold, move,
                                  bkt, mtype, float(ba[i] - bb[i]))

                # Sized leg: capped by what is actually resting on both sides,
                # so a thin book yields a small trade, never a phantom fill.
                q = min(shares_target,
                        side_total(rows[i], aoff, asz),
                        side_total(rows[j], boff, bsz))
                if q > 0:
                    fb, cost, basis_b = walk_ladder(rows[i], aoff, apx, asz, q)
                    fs, proceeds, basis_s = walk_ladder(rows[j], boff, bpx, bsz, q)
                    q = min(fb, fs)
                    if q > 0:
                        # Re-walk at the common size so both legs trade the
                        # same quantity -- otherwise the cheaper side would be
                        # sized off a fill the other side could not match.
                        _, cost, basis_b = walk_ladder(rows[i], aoff, apx, asz, q)
                        _, proceeds, basis_s = walk_ladder(rows[j], boff, bpx, bsz, q)
                        aggN[theta_c].add(q, proceeds - cost,
                                          basis_b + basis_s, hold, move, bkt,
                                          mtype, float(ba[i] - bb[i]) * q)
    del cols
    return n_tok


def fmt_c(x):
    return f"{x*100:+.3f}c"


def report(agg1, aggN, aggM, shares_target, n_games, n_tokens, rej,
           max_spread, min_size):
    print()
    print("=" * 84)
    print("PERFECT-FORESIGHT JUMP CAPTURE -- buy the exact bottom, sell the exact top")
    print(f"{n_games} game(s), {n_tokens} token(s); gate: spread <= "
          f"{max_spread*100:.0f}c and >= {min_size:g} shares at both ends")
    print("=" * 84)

    # Cost decomposition. `move` is the mid-to-mid swing perfect foresight
    # captures; `exec` is what crossing the book takes out of it (half-spread at
    # each end, plus depth on the sized run); `fee` is the taker fee on both
    # legs. net = move - exec - fee, so these columns show exactly where a
    # perfect forecast goes to die.
    for label, agg in (("1 SHARE (touch prices, no depth cost)", agg1),
                       (f"{shares_target:g} SHARES (walks the real ladder)", aggN),
                       ("MAKER 1 SHARE (rest both legs; no fee, earns the spread)",
                        aggM)):
        print(f"\n### {label}")
        print(f"{'swing':>7} {'trades':>8} {'move':>9} {'exec':>9} {'fee@.05':>9} "
              f"{'NET@.05':>9} {'NET@.10':>9} {'win@.05':>8} {'win@0':>7} {'BE rate':>8}")
        print("-" * 84)
        for th in THETAS_C:
            a = agg[th]
            if a.n == 0:
                print(f"{th:>6.1f}c {0:>8}")
                continue
            q = a.shares
            move = a.move / q
            gross = a.gross / q
            ex = move - gross
            f05 = 0.05 * a.basis / q
            print(f"{th:>6.1f}c {a.n:>8,} {fmt_c(move):>9} {fmt_c(-ex):>9} "
                  f"{fmt_c(-f05):>9} {fmt_c(a.net(0.05)/q):>9} "
                  f"{fmt_c(a.net(0.10)/q):>9} "
                  f"{a.wins[0.05]/a.n*100:>7.1f}% {a.wins[0.0]/a.n*100:>6.1f}% "
                  f"{a.breakeven_rate():>8.4f}")

        print("\n  dollar totals -- every qualifying jump traded:")
        for th in THETAS_C:
            a = agg[th]
            if a.n == 0:
                continue
            print(f"    {th:>4.1f}c  {a.n:>8,} trades  {a.shares:>13,.0f} shares  "
                  f"gross {a.gross:>12,.0f}  net@.05 {a.net(0.05):>12,.0f}  "
                  f"net@.10 {a.net(0.10):>12,.0f}")

    # What a jump has to clear to pay for itself. Independent of any forecast,
    # so this is the answer that does not depend on the simulation at all.
    a = aggN[1.0]
    if a.n:
        print(f"\n### round-trip cost you must beat -- {shares_target:g} shares, 1c swing")
        print(f"{'entry price':>14} {'trades':>9} {'spread':>9} {'exec':>9} "
              f"{'fee@.05':>9} {'fee@.10':>9} {'need@.05':>9} {'need@.10':>9}")
        print("-" * 84)
        for k, (lo, hi) in enumerate(PRICE_BUCKETS):
            d = a.by_bucket.get(k)
            if not d:
                continue
            n, q, g, b, mv, sp = d
            ex = (mv - g) / q
            f05, f10 = 0.05 * b / q, 0.10 * b / q
            print(f"  {lo:.2f}-{hi:.2f}    {n:>9,} {fmt_c(sp/q):>9} {fmt_c(ex):>9} "
                  f"{fmt_c(f05):>9} {fmt_c(f10):>9} "
                  f"{fmt_c(ex+f05):>9} {fmt_c(ex+f10):>9}")

        print(f"\n### by market type -- {shares_target:g} shares, 1c swing")
        print(f"{'market':>18} {'trades':>9} {'move':>9} {'exec':>9} "
              f"{'NET@.05':>9} {'NET@.10':>9}")
        print("-" * 70)
        for mt, d in sorted(a.by_mtype.items(), key=lambda kv: -kv[1][0]):
            n, q, g, b, mv, sp = d
            print(f"  {mt:>16} {n:>9,} {fmt_c(mv/q):>9} {fmt_c((mv-g)/q):>9} "
                  f"{fmt_c((g-0.05*b)/q):>9} {fmt_c((g-0.10*b)/q):>9}")

    print("\n### candidate swings rejected by the tradeability gate")
    print(f"{'swing':>7} {'traded':>10} {'wide spread':>13} {'thin':>10} "
          f"{'one-sided':>11} {'% tradeable':>12}")
    print("-" * 70)
    for th in THETAS_C:
        r = rej[th]
        tot = r["traded"] + r["wide_spread"] + r["thin"] + r["one_sided"]
        if not tot:
            continue
        print(f"{th:>6.1f}c {r['traded']:>10,} {r['wide_spread']:>13,} "
              f"{r['thin']:>10,} {r['one_sided']:>11,} "
              f"{r['traded']/tot*100:>11.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--slate", default=None,
                    help="US date tag; every by_game file ending in it")
    ap.add_argument("--shares", type=float, default=500.0,
                    help="target size per leg for the depth-aware run")
    ap.add_argument("--window", choices=("game", "full"), default="game")
    ap.add_argument("--split-dir", default=None,
                    help="where <slug>.jsonl lives (default data/live/by_game)")
    ap.add_argument("--max-spread", type=float, default=0.05,
                    help="skip swings whose quoted spread at either end exceeds "
                         "this, in dollars -- guards against stub-quote artifacts")
    ap.add_argument("--min-size", type=float, default=5.0,
                    help="require at least this many shares resting on the side "
                         "being hit, at both ends")
    ap.add_argument("--json", default=None, help="write machine-readable totals here")
    args = ap.parse_args()

    if args.games:
        slugs = args.games
    else:
        pat = f"*-{args.slate}.jsonl" if args.slate else "*.jsonl"
        slugs = sorted(os.path.basename(f)[:-6]
                       for f in glob.glob(os.path.join(
                           args.split_dir or rm.JSONL_SPLIT_DIR, pat)))
    if not slugs:
        raise SystemExit("no games found; pass --games or --slate, and split first")

    windows = rm.load_game_windows() if args.window == "game" else {}
    agg1 = {th: Agg() for th in THETAS_C}
    aggN = {th: Agg() for th in THETAS_C}
    aggM = {th: Agg() for th in THETAS_C}
    rej = {th: {"traded": 0, "wide_spread": 0, "thin": 0, "one_sided": 0}
           for th in THETAS_C}

    log(f"{len(slugs)} game(s), size {args.shares:g} shares/leg, "
        f"swings {', '.join(f'{t}c' for t in THETAS_C)}")
    n_tokens = 0
    for slug in slugs:
        t0 = time.time()
        w = windows.get(slug, {})
        if not w and args.window == "game":
            log(f"  {slug}: no game window -- using full recorded span")
        n = simulate_game(slug, w, args.shares, agg1, aggN, aggM,
                          split_dir=args.split_dir, max_spread=args.max_spread,
                          min_size=args.min_size, rej=rej)
        n_tokens += n
        log(f"  {slug}: {n} tokens in {time.time()-t0:.0f}s "
            f"(peak RSS {rd.peak_rss_gb():.1f} GB)")

    report(agg1, aggN, aggM, args.shares, len(slugs), n_tokens, rej,
           args.max_spread, args.min_size)

    if args.json:
        out = {"games": slugs, "shares": args.shares, "fee_rates": list(FEE_RATES),
               "thetas_c": list(THETAS_C), "max_spread": args.max_spread,
               "min_size": args.min_size,
               "rejected": {str(k): v for k, v in rej.items()}, "configs": {}}
        for name, agg in (("one_share", agg1), ("sized", aggN),
                          ("maker", aggM)):
            out["configs"][name] = {
                str(th): {"trades": a.n, "shares": a.shares, "gross": a.gross,
                          "fee_basis": a.basis,
                          "net": {str(r): a.net(r) for r in FEE_RATES},
                          "breakeven_rate": a.breakeven_rate(),
                          "avg_hold_s": (a.hold / a.n) if a.n else None,
                          "avg_move": (a.move / a.shares) if a.shares else None}
                for th, a in agg.items()}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
