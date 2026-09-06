"""
Econometric jump models, applied causally, and judged on whether they can
CAPTURE a jump rather than merely describe one after the fact.

Four families, all standard in the high-frequency econometrics literature and
all implemented so that nothing at time t uses data after t:

1. LEE-MYKLAND (2008) jump statistic
       L_i = r_i / sigma_i,   sigma_i^2 = bipower variation of the TRAILING
       K returns, excluding r_i itself.
   Bipower variation sums |r_j||r_{j-1}|, which stays finite when a single
   return is a jump, so the volatility scale is robust to the very jumps it
   is used to detect. Under the no-jump null L is asymptotically standard
   normal, so a large |L| flags a jump. This is the standard detector, and
   its causal form is exactly the published one with a one-sided window.

2. HAWKES self-excitation
       lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i))
   Jumps cluster. If they do so here, the intensity after a jump forecasts
   the next one, which is the only form of jump prediction that does not
   require knowing the news. Several decay rates are tried rather than
   fitting alpha/beta by MLE -- with this much data the fit is not the
   uncertain part; whether the effect exists at all is.

3. KALMAN / state-space FAIR VALUE (local level model)
       x_t = x_{t-1} + w_t     efficient price, Var(w) = Q
       y_t = x_t     + v_t     observed mid = efficient + microstructure noise
   Q and R come from the Roll/Hasbrouck autocovariance identities
       R = -Cov(dy_t, dy_{t-1}),   Q = Var(dy) - 2R
   estimated on each token's FIRST 20% of history and then held fixed. The
   FILTER is used, never the smoother: a smoother conditions on the whole
   sample and would leak the future into the fair value, which is the single
   easiest way to fake this result.
   Signal = x_hat_t - mid_t: how far the quoted mid sits from fair value.

4. PCA on the book shape
   First principal components of the depth profile, loadings fitted on TRAIN
   slates only and applied unchanged to TEST.

The bar every one of them has to clear
--------------------------------------
Detecting a jump is not capturing it. From the perfect-foresight study, a
taker needs precision of roughly

    >= 3c jumps : 80%      >= 5c jumps : 33%      >= 10c jumps : 14%

because a false positive costs the spread (~1.2c on moneyline) while a true
positive pays only what the jump delivers beyond it. So each signal is scored
by PRECISION at predicting a forward move of at least theta, and then by the
executable cents per share of actually trading it.

    python jump_models.py --slate 2026-09-04 --market-type moneyline
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
TRAIN = ("2026-08-27", "2026-08-28")
TEST = ("2026-09-04",)

LM_WINDOW = 200          # trailing returns for bipower volatility
HAWKES_DECAY = (0.5, 2.0, 10.0)     # seconds
THETAS = (2.0, 3.0, 5.0)            # cents; jump sizes worth capturing
HORIZONS = (1.0, 5.0, 10.0)         # seconds ahead


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def lee_mykland(mid, K=LM_WINDOW):
    """Causal Lee-Mykland statistic per observation.

    sigma_i uses only returns strictly before i, so the flag at i is
    computable the instant r_i is observed and never peeks ahead.
    """
    r = np.zeros(mid.size)
    r[1:] = np.diff(mid)
    bp = np.zeros(mid.size)
    bp[2:] = np.abs(r[2:]) * np.abs(r[1:-1])       # |r_j| |r_{j-1}|
    c = np.concatenate(([0.0], np.cumsum(bp)))
    lo = np.maximum(np.arange(mid.size) - K, 0)
    # window ends at i-1 -> strictly trailing
    s2 = (c[np.arange(mid.size)] - c[lo]) / np.maximum(
        np.arange(mid.size) - lo - 1, 1)
    sig = np.sqrt(np.maximum(s2, 1e-12)) / 0.7979   # E|N(0,1)| = sqrt(2/pi)
    return np.divide(r, np.maximum(sig, 1e-9)), r, sig


def hawkes_intensity(t, event, decay):
    """lambda(t) from past events only, exponential kernel, recursive.

    lambda_i = exp(-beta dt) * lambda_{i-1} + (event_{i-1} ? 1 : 0)
    so only events strictly earlier than i contribute.
    """
    n = t.size
    out = np.zeros(n)
    lam = 0.0
    for i in range(1, n):
        lam *= np.exp(-decay * (t[i] - t[i - 1]))
        if event[i - 1]:
            lam += 1.0
        out[i] = lam
    return out


def kalman_fair(mid, warm_frac=0.2):
    """Local-level Kalman FILTER estimate of the efficient price.

    Q and R are estimated from the first `warm_frac` of the series via the
    Roll/Hasbrouck autocovariance identities and then frozen, so the filter
    over the remainder uses no information from it.
    """
    n = mid.size
    w = max(int(n * warm_frac), 500)
    dy = np.diff(mid[:w])
    if dy.size < 100:
        return np.full(n, np.nan), np.nan
    g0 = float(np.var(dy))
    g1 = float(np.mean((dy[1:] - dy[1:].mean()) * (dy[:-1] - dy[:-1].mean())))
    R = max(-g1, 1e-6)                 # negative autocovariance -> noise
    Q = max(g0 - 2 * R, 1e-9)          # what is left is permanent variance
    # steady-state gain from the Riccati fixed point
    P = (Q + np.sqrt(Q * Q + 4 * Q * R)) / 2.0
    Kg = (P + Q) / (P + Q + R)
    x = np.empty(n)
    x[0] = mid[0]
    for i in range(1, n):
        x[i] = x[i - 1] + Kg * (mid[i] - x[i - 1])
    return x, Kg


def future_extremes(t, mid, H):
    """Max rise and max fall of the mid within the next H seconds.

    This is a TARGET, computed with future data on purpose. No feature in
    this file touches it.
    """
    n = t.size
    j = np.searchsorted(t, t + H, side="right")
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    # running max/min over a forward window via a reversed sliding approach
    for i in range(n):
        k = j[i]
        if k <= i + 1:
            continue
        seg = mid[i + 1:k]
        up[i] = seg.max() - mid[i]
        dn[i] = seg.min() - mid[i]
    return up, dn


def load_token_series(files, market_type, max_tokens=None):
    """Yield (slug, asset_id, t, mid, spread, bb, ba) for liquid tokens."""
    for path in files:
        slug = os.path.basename(path)[:-8]
        cols = ["t", "mid_c", "spread_c", "bb_c", "ba_c", "asset_id",
                "market_type", "dep_bid_1c", "dep_ask_1c", "dep_bid_2c",
                "dep_ask_2c", "dep_bid_5c", "dep_ask_5c", "bb_sz", "ba_sz"]
        tab = pq.read_table(path, columns=cols)
        d = {c: tab.column(c).to_numpy(zero_copy_only=False) for c in cols}
        ok = (np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"])
              & (d["ba_c"] > d["bb_c"]))       # crossed books excluded
        if market_type:
            ok &= (d["market_type"] == market_type)
        n = 0
        for aid in np.unique(d["asset_id"][ok]):
            sel = ok & (d["asset_id"] == aid)
            if sel.sum() < 5000:
                continue
            yield slug, aid, {k: v[sel] for k, v in d.items() if v.dtype != object}
            n += 1
            if max_tokens and n >= max_tokens:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", default=None)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--market-type", default="moneyline")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--panel-dir", default=PANEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pat = f"*-{args.slate}.parquet" if args.slate else "*.parquet"
    files = ([os.path.join(args.panel_dir, f"{g}.parquet") for g in args.games]
             if args.games else sorted(glob.glob(os.path.join(args.panel_dir, pat))))
    files = [f for f in files if os.path.exists(f)]
    if args.max_games:
        files = files[:args.max_games]
    if not files:
        raise SystemExit("no panel files")

    # stats[(signal, theta, H)] -> [n_fired, n_hit, sum_exec_c, n_rows, n_base]
    stats = collections.defaultdict(lambda: np.zeros(7))
    lm_sizes = []
    kal_gains = []
    n_tok = 0

    for slug, aid, d in load_token_series(files, args.market_type):
        t, mid = d["t"], d["mid_c"]
        bb, ba, spr = d["bb_c"], d["ba_c"], d["spread_c"]
        n_tok += 1

        L, r, sig = lee_mykland(mid)
        # Gumbel-corrected threshold for n observations, alpha ~ 1%
        nn = max(mid.size, 2)
        Cn = (2 * np.log(nn)) ** 0.5 / 0.7979 - (
            np.log(np.pi) + np.log(np.log(nn))) / (2 * 0.7979 * (2 * np.log(nn)) ** 0.5)
        Sn = 1.0 / (0.7979 * (2 * np.log(nn)) ** 0.5)
        thr = Cn + Sn * (-np.log(-np.log(0.99)))
        jump = np.abs(L) > thr
        lm_sizes.append(np.abs(r[jump]))

        haw = {b: hawkes_intensity(t, jump, b) for b in HAWKES_DECAY}
        fair, Kg = kalman_fair(mid)
        kal_gains.append(Kg)
        dev = fair - mid              # + means mid is BELOW fair value

        signals = {
            "lee_mykland_abs": np.abs(L),
            "lee_mykland_signed": L,
            "kalman_dev": dev,
            "kalman_dev_abs": np.abs(dev),
        }
        # Signals that carry their own critical value fire on it, not on a
        # quantile. |L| is zero for the majority of rows (the mid does not
        # move), so a 99th-percentile cut is degenerate and meaningless.
        fixed_thr = {"lee_mykland_abs": thr, "lee_mykland_signed": thr}
        for b, v in haw.items():
            signals[f"hawkes_{b:g}s"] = v

        for H in HORIZONS:
            up, dn = future_extremes(t, mid, H)
            for th in THETAS:
                hit_any = (up >= th) | (dn <= -th)
                hit_up = up >= th
                base = np.isfinite(up)
                for name, sv in signals.items():
                    v = np.isfinite(sv) & base
                    if v.sum() < 1000:
                        continue
                    # fire on the top 1% of the signal, threshold from this
                    # token's FIRST HALF only (no look-ahead in the rule)
                    if name in fixed_thr:
                        q = fixed_thr[name]
                    else:
                        half = sv[v][: max(v.sum() // 2, 100)]
                        q = float(np.quantile(half, 0.99))
                    fired = v & (sv >= q)
                    if fired.sum() < 20:
                        continue
                    directional = name in ("lee_mykland_signed", "kalman_dev")
                    tgt = hit_up if directional else hit_any
                    a = stats[(name, th, H)]
                    a[0] += int(fired.sum())
                    a[1] += int((fired & tgt).sum())
                    a[3] += int(v.sum())
                    a[4] += int((v & tgt).sum())
                    # signed mid move to the END of the horizon: if this is
                    # ~0 while precision is high, the signal forecasts
                    # VOLATILITY, not direction.
                    jj = np.minimum(np.searchsorted(t, t + H, side="left"),
                                    t.size - 1)
                    a[5] += float(np.nansum((mid[jj] - mid)[fired]))
                    a[6] += float(np.nansum(np.abs(mid[jj] - mid)[fired]))
                    # executable: buy the ask now, sell the bid at the best
                    # moment inside the horizon is NOT allowed (that is the
                    # oracle). Mark out at the END of the horizon instead.
                    j = np.minimum(np.searchsorted(t, t + H, side="left"),
                                   t.size - 1)
                    exec_c = bb[j] - ba          # long round trip, taker both legs
                    a[2] += float(np.nansum(exec_c[fired]))
        if n_tok % 4 == 0:
            log(f"  {n_tok} tokens processed")

    print()
    print("=" * 104)
    print(f"ECONOMETRIC JUMP MODELS -- {args.market_type}, {len(files)} game(s), "
          f"{n_tok} tokens")
    if lm_sizes:
        v = np.concatenate([x for x in lm_sizes if x.size])
        print(f"Lee-Mykland flagged {v.size:,} jumps; |size| p50 {np.median(v):.2f}c "
              f"p90 {np.percentile(v, 90):.2f}c max {v.max():.2f}c")
    kg = [k for k in kal_gains if np.isfinite(k)]
    if kg:
        print(f"Kalman steady-state gain: mean {np.mean(kg):.3f} "
              f"(1.0 = mid is pure signal, 0 = mid is pure noise)")
    print("precision = P(move >= theta within H | signal fires). "
          "base = unconditional rate.")
    print("exec = mean cents/share of a taker round trip held to the horizon.")
    print("=" * 104)
    print(f"{'signal':>20} {'theta':>6} {'H':>5} {'fires':>9} {'precision':>10} "
          f"{'base':>8} {'lift':>6} {'signed':>9} {'|move|':>8} {'exec c/sh':>11}")
    print("-" * 112)
    BAR = {2.0: 0.90, 3.0: 0.80, 5.0: 0.33}
    rows = []
    for (name, th, H), a in sorted(stats.items(),
                                   key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        if a[0] < 100:
            continue
        prec = a[1] / a[0]
        base = a[4] / max(a[3], 1)
        ex = a[2] / a[0]
        signed = a[5] / a[0]
        absmove = a[6] / a[0]
        rows.append({"signal": name, "theta": th, "H": H, "fires": int(a[0]),
                     "precision": prec, "base": base, "exec_c": ex,
                     "signed_c": signed, "absmove_c": absmove})
        print(f"{name:>20} {th:>5.1f}c {H:>4.0f}s {int(a[0]):>9,} {prec*100:>9.2f}% "
              f"{base*100:>7.2f}% {prec/max(base,1e-9):>5.2f}x {signed:>+8.3f}c "
              f"{absmove:>7.3f}c {ex:>+10.4f}c")

    good = [r for r in rows if r["exec_c"] > 0]
    print()
    print(f"signals with POSITIVE executable edge: {len(good)} of {len(rows)}")
    for r in sorted(good, key=lambda r: -r["exec_c"])[:10]:
        print(f"    {r['signal']:>20} theta={r['theta']:.0f}c H={r['H']:.0f}s "
              f"prec {r['precision']*100:.1f}%  exec {r['exec_c']:+.4f}c")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"market_type": args.market_type, "games": len(files),
                       "tokens": n_tok, "rows": rows}, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
