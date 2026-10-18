"""Steps 3-9: Lee-Mykland jump detection on 1-minute prediction-market mids.

The test implemented is Lee & Mykland (2008), "Jumps in Financial Markets: A
New Nonparametric Test and Jump Dynamics", Review of Financial Studies 21(6).
No z-score, rolling-sigma, percentage-return or Bollinger substitute is used.

Statistic
---------
Log returns on the 1-minute mid:

    r_i = log(m_i) - log(m_{i-1})

Local volatility from realised BIPOWER variation over the K-2 products
immediately preceding i (LM eq. 8), which is robust to jumps because it pairs
adjacent absolute returns:

    sigma_hat^2(i) = 1/(K-2) * sum_{j=i-K+2}^{i-1} |r_j| * |r_{j-1}|

Every term uses returns strictly BEFORE i, so the window is trailing only --
never centred. The statistic (LM eq. 7) is

    L(i) = r_i / sigma_hat(i)

Threshold
---------
Under the null of no jump, |L| behaves like |N(0,1)|/c with c = sqrt(2/pi),
because bipower variation estimates c^2 * sigma^2 rather than sigma^2. LM
Lemma 1 gives the extreme-value limit: with n tested observations,

    C_n = (2 log n)^(1/2)/c - (log(pi) + log(log n)) / (2c (2 log n)^(1/2))
    S_n = 1 / (c (2 log n)^(1/2))

and (max|L| - C_n)/S_n converges to a standard Gumbel. So observation i is a
jump when

    |L(i)| > C_n + S_n * beta_star,    beta_star = -log(-log(1 - alpha))

This is a family-wise threshold over the n observations actually tested, i.e.
it already accounts for multiple testing. Direction is the sign of r_i.

Handling of this dataset specifically
-------------------------------------
* series boundary   Each asset_id is an independent contract. Returns, bipower
                    windows and statistics never cross an asset boundary.
* gaps              A return is computed only between two ADJACENT minutes that
                    both carry a quote. Where build_minute_bars.py left a bar
                    missing (silence beyond the staleness cap) the return is
                    undefined and the window simply contains fewer products.
* warm-up           The first observations of each asset have no statistic and
                    are reported as warm-up, never as "no jump".
* partial windows   Real gaps mean a K-2 window is not always full. A statistic
                    is emitted only when at least `--min-frac-window` of the
                    products are present; this is documented rather than
                    silently imputed.
* sessions          Recording sessions are separate files and separate assets,
                    so session boundaries are already series boundaries.

    python detect_lee_mykland_jumps.py --K 600 --alpha 0.01
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache")
C_BP = np.sqrt(2.0 / np.pi)           # E|N(0,1)|
MIN_MS = 60_000
HKT_MS = 8 * 3600 * 1000


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def lm_threshold(n, alpha):
    """LM Lemma 1 extreme-value critical value for n tested observations."""
    if n < 3:
        return np.nan, np.nan, np.nan
    ln = np.log(n)
    root = np.sqrt(2.0 * ln)
    c_n = root / C_BP - (np.log(np.pi) + np.log(ln)) / (2.0 * C_BP * root)
    s_n = 1.0 / (C_BP * root)
    beta = -np.log(-np.log(1.0 - alpha))
    return c_n + s_n * beta, c_n, s_n


def returns_for(mid, scale):
    """Return definition. Lee-Mykland is derived for log-price, but these are
    BOUNDED probability contracts and log explodes at the boundary: at p=0.01 a
    single half-tick is a 0.41 log return, versus 0.0055 at p=0.90, so the test
    would spend its power on ticks in near-resolved contracts. Arithmetic
    differences are in the natural payoff unit (probability points) and are
    uniform across the price range, which is why they are the default here.
    LM's standardisation by locally-estimated bipower volatility is unchanged."""
    if scale == "log":
        return np.r_[np.nan, np.diff(np.log(mid))]
    if scale == "logit":
        z = np.log(mid / (1.0 - mid))
        return np.r_[np.nan, np.diff(z)]
    return np.r_[np.nan, np.diff(mid)]          # "diff": probability points


def lm_for_asset(g, K, min_frac, min_nz_pairs, scale="diff"):
    """Trailing-window Lee-Mykland statistic for one asset's minute series."""
    g = g.sort_values("minute")
    minute = g.minute.to_numpy(np.int64)
    mid = g.mid.to_numpy(np.float64)

    # return only between ADJACENT minutes that both carry a quote
    adjacent = np.empty(len(g), bool)
    adjacent[0] = False
    adjacent[1:] = (minute[1:] - minute[:-1]) == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = returns_for(mid, scale)
    lr[~adjacent] = np.nan
    lr[~np.isfinite(lr)] = np.nan

    r = pd.Series(lr)
    # p_j = |r_j| * |r_{j-1}|  (only where both returns exist)
    p = r.abs() * r.abs().shift(1)
    win = max(K - 2, 2)
    need = max(int(np.ceil(min_frac * win)), 2)
    # window ending at i-1 => shift(1) then roll. Strictly trailing.
    ps = p.shift(1)
    sigma2 = ps.rolling(win, min_periods=need).mean()
    # Guard against a structurally degenerate estimator. 72% of minute returns
    # are exactly zero (discrete tick prices, flat most minutes), and bipower
    # multiplies ADJACENT |r|, so a window can contain almost no non-zero
    # products. sigma_hat then collapses towards 0 and L = r/sigma_hat explodes
    # to ~1e8, which is an artefact of a vanishing denominator, not a jump.
    # Lee-Mykland's asymptotics assume sigma > 0, so where the window carries
    # too few non-zero products no statistic is emitted.
    n_nz = ps.gt(0).rolling(win, min_periods=1).sum()
    sigma = np.sqrt(sigma2)
    sigma[n_nz < min_nz_pairs] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        L = r / sigma

    out = pd.DataFrame({
        "minute": minute, "mid": mid, "log_return": r.to_numpy(),
        "local_volatility": sigma.to_numpy(),
        "LM_stat": L.to_numpy(),
        "n_window_pairs": ps.rolling(win, min_periods=1).count().to_numpy(),
        "n_window_nonzero_pairs": n_nz.to_numpy(),
    })
    for c in ("asset_id", "label", "event_slug", "market_type", "line",
              "outcome", "session", "n_events", "staleness_s", "carry_min"):
        if c in g.columns:
            out[c] = g[c].to_numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=30,
                    help="trailing bipower window in minutes. 30 was "
                         "chosen from K_sweep_in_game.csv, not from the "
                         "paper: it gives the tightest agreement between "
                         "sigma_hat and realised volatility (IQR 1.30-1.67 "
                         "vs 0.99-1.86 at K=120) while still testing 72%% "
                         "of in-game returns. The paper K=600 assumes "
                         "multi-day equity series; a baseball contract "
                         "lives a median of 84 in-game minutes.")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-frac-window", type=float, default=0.5)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--return-scale", default="diff",
                    choices=["diff", "log", "logit"],
                    help="'diff' = arithmetic change in probability points "
                         "(default, suits bounded contracts); 'log' reproduces "
                         "the textbook log-return version.")
    ap.add_argument("--in-game-only", action="store_true", default=True,
                    help="keep only minutes between first pitch and the final "
                         "out, from data/game_windows.json. Off-window minutes "
                         "are pre-game (barely moving, which poisons the "
                         "volatility warm-up) or post-resolution stale quotes.")
    ap.add_argument("--all-minutes", dest="in_game_only",
                    action="store_false",
                    help="undo --in-game-only and test pre-game and "
                         "post-resolution minutes too (reproduces the "
                         "first run; not recommended).")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--min-nonzero-pairs", type=int, default=7,
                    help="minimum NON-ZERO |r_j||r_{j-1}| products required in "
                         "the bipower window before a statistic is emitted. "
                         "Guards a vanishing denominator on a discrete, mostly "
                         "flat price; set 0 to disable and reproduce the "
                         "unguarded run.")
    ap.add_argument("--min-nonzero-frac", type=float, default=0.0,
                    help="keep only assets whose share of NON-ZERO minute "
                         "returns is at least this. Lee-Mykland assumes a "
                         "continuous diffusion; a series that is flat most "
                         "minutes violates that badly, because bipower pairs "
                         "adjacent |r| and is zero whenever either minute did "
                         "not move. 0 = keep everything (the default, so the "
                         "headline run is unfiltered).")
    a = ap.parse_args()

    RES = a.outdir or os.path.join(ROOT, "results", "lee_mykland")
    EX = os.path.join(RES, "examples")
    os.makedirs(EX, exist_ok=True)

    B = pd.read_parquet(os.path.join(CACHE, "minute_bars.parquet"))
    log("loaded {:,} minute bars, {:,} assets".format(len(B), B.asset_id.nunique()))

    if a.in_game_only:
        import json
        wf = os.path.join(ROOT, "data", "game_windows.json")
        W = json.load(open(wf, encoding="utf-8"))
        w = pd.DataFrame([{"event_slug": k,
                           "g_start": pd.Timestamp(v["start_utc"]).value // 10**6,
                           "g_end": pd.Timestamp(v["end_utc"]).value // 10**6}
                          for k, v in W.items()])
        n0 = len(B)
        B = B.merge(w, on="event_slug", how="inner")
        ms = B.minute * MIN_MS
        B = B[(ms >= B.g_start) & (ms <= B.g_end)].drop(columns=["g_start", "g_end"])
        log("in-game trim: {:,} -> {:,} bars ({:,} assets)"
            .format(n0, len(B), B.asset_id.nunique()))

    if a.min_nonzero_frac > 0:
        keep = []
        for aid, g in B.groupby("asset_id", sort=False, observed=True):
            g = g.sort_values("minute")
            m = g.mid.to_numpy(np.float64)
            mn = g.minute.to_numpy(np.int64)
            adj = np.r_[False, (mn[1:] - mn[:-1]) == 1]
            lr = np.r_[np.nan, np.diff(np.log(m))]
            lr[~adj] = np.nan
            v = lr[np.isfinite(lr)]
            if len(v) and float((v != 0).mean()) >= a.min_nonzero_frac:
                keep.append(aid)
        B = B[B.asset_id.isin(keep)]
        log("kept %d assets with >= %.0f%% non-zero minute returns"
            % (len(keep), 100 * a.min_nonzero_frac))

    rows = []
    for _, g in B.groupby("asset_id", sort=False, observed=True):
        rows.append(lm_for_asset(g, a.K, a.min_frac_window, a.min_nonzero_pairs,
                                 a.return_scale))
    D = pd.concat(rows, ignore_index=True)
    D["ts_hkt"] = pd.to_datetime(D.minute * MIN_MS + HKT_MS, unit="ms")

    tested = np.isfinite(D.LM_stat.to_numpy())
    n = int(tested.sum())
    thr, c_n, s_n = lm_threshold(n, a.alpha)
    log("tested observations n = {:,}  -> C_n {:.4f}  S_n {:.4f}  threshold {:.4f}"
        .format(n, c_n, s_n, thr))

    D["critical_threshold"] = thr
    absL = np.abs(D.LM_stat.to_numpy())
    D["jump"] = np.where(tested & (absL > thr), 1, 0).astype(np.int8)
    D["jump_direction"] = 0
    pos = (D.jump == 1) & (D.log_return > 0)
    neg = (D.jump == 1) & (D.log_return < 0)
    D.loc[pos, "jump_direction"] = 1
    D.loc[neg, "jump_direction"] = -1
    D["tested"] = tested

    cols = ["ts_hkt", "minute", "session", "asset_id", "label", "event_slug",
            "market_type", "line", "outcome", "mid", "log_return",
            "local_volatility", "LM_stat", "critical_threshold", "jump",
            "jump_direction", "tested", "n_events", "staleness_s", "carry_min",
            "n_window_pairs"]
    cols = [c for c in cols if c in D.columns]
    out_csv = os.path.join(RES, "lee_mykland_jumps.csv")
    D[cols].to_csv(out_csv, index=False)
    log("wrote %s ({:,} rows)".format(len(D)) % out_csv)

    J = D[D.jump == 1]
    stats = dict(
        K=a.K, alpha=a.alpha, return_scale=a.return_scale,
        in_game_only=bool(a.in_game_only),
        min_nonzero_pairs=a.min_nonzero_pairs,
        threshold=float(thr), C_n=float(c_n), S_n=float(s_n),
        total_minute_bars=int(len(D)),
        bars_with_price=int(np.isfinite(D.mid.to_numpy()).sum()),
        valid_returns=int(np.isfinite(D.log_return.to_numpy()).sum()),
        tested_observations=n,
        warmup_or_untestable=int(len(D) - n),
        assets_total=int(D.asset_id.nunique()),
        assets_tested=int(D.loc[tested, "asset_id"].nunique()),
        total_jumps=int(len(J)),
        positive_jumps=int((J.jump_direction == 1).sum()),
        negative_jumps=int((J.jump_direction == -1).sum()),
        pct_of_tested_minutes=float(100.0 * len(J) / max(n, 1)),
    )
    # --- diagnostics that decide whether these labels are trustworthy ---
    vr = D.log_return.to_numpy()
    vr = vr[np.isfinite(vr)]
    stats["frac_zero_minute_returns"] = float((vr == 0).mean())
    tr = D.loc[tested, "log_return"].to_numpy()
    stats["frac_zero_returns_among_tested"] = float((tr == 0).mean())
    # resolution proxy: how close to the END of the contract's life a jump sits
    life = D.groupby("asset_id").minute.transform("max") - D.minute
    D["minutes_to_series_end"] = life
    if len(J):
        lj = life[D.jump == 1]
        stats["frac_jumps_in_last_10min_of_series"] = float((lj <= 10).mean())
        stats["frac_jumps_at_mid_below_0.05"] = float((J.mid < 0.05).mean())
        stats["frac_jumps_at_mid_above_0.95"] = float((J.mid > 0.95).mean())
        stats["frac_jumps_on_carried_bar"] = float((J.n_events == 0).mean())
        stats["median_abs_return_all_tested"] = float(np.median(np.abs(tr)))
    # regime lag: forward realised vol vs the trailing bipower estimate. A ratio
    # far above 1 means sigma_hat is calibrated on a quieter period than the one
    # being tested, which inflates every statistic.
    Dv = D.sort_values(["asset_id", "minute"])
    fwd = (Dv.groupby("asset_id").log_return
           .transform(lambda s_: s_.shift(-1).rolling(30, min_periods=10).std()))
    m_ok = fwd.notna() & (Dv.local_volatility > 0)
    if int(m_ok.sum()) > 100:
        ratio = (fwd[m_ok] / Dv.local_volatility[m_ok])
        stats["sigma_regime_lag_median"] = float(ratio.median())
        stats["sigma_regime_lag_p25"] = float(ratio.quantile(.25))
        stats["sigma_regime_lag_p75"] = float(ratio.quantile(.75))
    if len(J):
        days = D.loc[tested, "ts_hkt"].dt.date.nunique()
        stats.update(
            distinct_days=int(days),
            avg_jumps_per_day=float(len(J) / max(days, 1)),
            median_abs_return=float(J.log_return.abs().median()),
            mean_abs_return=float(J.log_return.abs().mean()),
            max_positive_return=float(J.log_return.max()),
            max_negative_return=float(J.log_return.min()),
        )
    log(" | ".join("%s=%s" % (k, v) for k, v in stats.items()))

    # ---------------- Step 5: main price graph with jumps marked -------------
    primary = None
    if len(J) and not a.no_plots:
        cand = (D[tested].groupby("asset_id")
                .agg(n_tested=("LM_stat", "size"),
                     n_jumps=("jump", "sum")).reset_index())
        cand = cand[cand.n_jumps > 0].sort_values(["n_tested", "n_jumps"],
                                                  ascending=False)
        if len(cand):
            primary = cand.asset_id.iloc[0]

    def price_plot(sub, title, path, marker_size=70):
        sub = sub.sort_values("minute")
        has = np.isfinite(sub.mid.to_numpy())
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(sub.ts_hkt[has], sub.mid[has], lw=0.9, color="#1f77b4",
                label="Price (1-min mid)", zorder=1)
        up = sub[sub.jump_direction == 1]
        dn = sub[sub.jump_direction == -1]
        ax.scatter(up.ts_hkt, up.mid, marker="^", s=marker_size, color="#2ca02c",
                   edgecolor="black", linewidth=0.4, zorder=3,
                   label="Positive Lee-Mykland jump (n=%d)" % len(up))
        ax.scatter(dn.ts_hkt, dn.mid, marker="v", s=marker_size, color="#d62728",
                   edgecolor="black", linewidth=0.4, zorder=3,
                   label="Negative Lee-Mykland jump (n=%d)" % len(dn))
        ax.set_xlabel("time (HKT)")
        ax.set_ylabel("mid price (probability)")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    if primary is not None:
        sub = D[D.asset_id == primary]
        lbl = str(sub.label.iloc[0])
        npos = int((sub.jump_direction == 1).sum())
        nneg = int((sub.jump_direction == -1).sum())
        price_plot(sub,
                   "Lee-Mykland jumps  |  %s\nK=%d, alpha=%.3f, threshold=%.3f  "
                   "|  +%d / -%d / %d total"
                   % (lbl, a.K, a.alpha, thr, npos, nneg, npos + nneg),
                   os.path.join(RES, "lee_mykland_jumps_full.png"))
        log("main graph: asset %s (%s)" % (primary, lbl))

        # ---------------- Step 7: the statistic itself ----------------------
        s2 = sub[np.isfinite(sub.LM_stat.to_numpy())]
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(s2.ts_hkt, s2.LM_stat, lw=0.8, color="#444444",
                label="$L(i) = r_i/\\hat{\\sigma}(i)$")
        ax.axhline(thr, color="#d62728", ls="--", lw=1.2,
                   label="critical threshold $\\pm$%.3f" % thr)
        ax.axhline(-thr, color="#d62728", ls="--", lw=1.2)
        ax.scatter(s2[s2.jump == 1].ts_hkt, s2[s2.jump == 1].LM_stat, s=40,
                   color="#ff7f0e", edgecolor="black", linewidth=0.4, zorder=3,
                   label="detected jump")
        ax.set_xlabel("time (HKT)")
        ax.set_ylabel("Lee-Mykland statistic")
        ax.set_title("Lee-Mykland statistic  |  %s  |  K=%d, alpha=%.3f"
                     % (lbl, a.K, a.alpha), fontsize=11)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "lee_mykland_statistic.png"), dpi=130)
        plt.close(fig)

    # per-session overview so the full dataset is still visible
    for sess, gs in (D.groupby("session", sort=True, observed=True)
                     if not a.no_plots else []):
        gj = gs[gs.jump == 1]
        if not len(gj):
            continue
        top = (gs[gs.jump == 1].groupby("asset_id").size()
               .sort_values(ascending=False).head(1).index)
        if not len(top):
            continue
        sub = gs[gs.asset_id == top[0]]
        price_plot(sub, "Lee-Mykland jumps  |  session %s  |  %s"
                   % (sess, str(sub.label.iloc[0])),
                   os.path.join(RES, "session_%s.png" % sess))

    # ---------------- Step 6: zoomed examples ------------------------------
    if len(J) and not a.no_plots:
        pick = pd.concat([
            J.assign(k=J.LM_stat).nlargest(a.n_examples, "k"),
            J.assign(k=J.LM_stat).nsmallest(a.n_examples, "k"),
        ]).drop_duplicates(subset=["asset_id", "minute"])
        pick = pick.sort_values("LM_stat", key=np.abs, ascending=False)
        # spread across dates where possible
        pick["d"] = pick.ts_hkt.dt.date
        pick = (pick.groupby("d", group_keys=False).head(4)
                .head(a.n_examples).reset_index(drop=True))
        for i, ev in pick.iterrows():
            sub = D[(D.asset_id == ev.asset_id)
                    & (D.minute >= ev.minute - 30)
                    & (D.minute <= ev.minute + 30)].sort_values("minute")
            has = np.isfinite(sub.mid.to_numpy())
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.plot(sub.ts_hkt[has], sub.mid[has], lw=1.3, color="#1f77b4",
                    marker="o", ms=2.5, label="mid")
            mk = "^" if ev.jump_direction == 1 else "v"
            col = "#2ca02c" if ev.jump_direction == 1 else "#d62728"
            ax.scatter([ev.ts_hkt], [ev.mid], marker=mk, s=190, color=col,
                       edgecolor="black", zorder=4,
                       label="%s jump" % ("positive" if ev.jump_direction == 1
                                          else "negative"))
            ax.axvline(ev.ts_hkt, color=col, ls="--", lw=0.9, alpha=0.6)
            ax.set_title("%s\n%s  |  dir %+d  |  r=%+.4f  |  LM=%+.2f  "
                         "(threshold %.2f)"
                         % (str(ev.label)[:70],
                            ev.ts_hkt.strftime("%Y-%m-%d %H:%M HKT"),
                            int(ev.jump_direction), ev.log_return, ev.LM_stat,
                            thr), fontsize=9)
            ax.set_xlabel("time (HKT)")
            ax.set_ylabel("mid price")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(EX, "jump_%03d.png" % (i + 1)), dpi=130)
            plt.close(fig)
        log("wrote %d zoomed examples" % len(pick))

    # ---------------- Step 9: time of day ----------------------------------
    if len(J) and not a.no_plots:
        fig, ax = plt.subplots(figsize=(13, 4.6))
        mod_all = (D.loc[tested, "ts_hkt"].dt.hour * 60
                   + D.loc[tested, "ts_hkt"].dt.minute)
        mod_j = J.ts_hkt.dt.hour * 60 + J.ts_hkt.dt.minute
        bins = np.arange(0, 24 * 60 + 15, 15)
        tot, _ = np.histogram(mod_all, bins=bins)
        jm, _ = np.histogram(mod_j, bins=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(tot > 0, jm / tot, np.nan)
        centres = (bins[:-1] + 7.5) / 60.0
        ax.bar(centres, rate * 100, width=0.22, color="#1f77b4",
               label="jump rate (%% of tested minutes)")
        ax2 = ax.twinx()
        ax2.plot(centres, tot, color="#ff7f0e", lw=1.1, label="tested minutes")
        ax2.set_ylabel("tested minutes in bin", color="#ff7f0e")
        ax.set_xlabel("hour of day (HKT) -- these contracts trade continuously; "
                      "there is no exchange open/close")
        ax.set_ylabel("jump rate (%)")
        ax.set_title("Lee-Mykland jump rate by time of day (15-min bins, HKT)  "
                     "|  K=%d" % a.K)
        ax.set_xlim(0, 24)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "jumps_by_time_of_day.png"), dpi=130)
        plt.close(fig)

    # ---------------- Step 8: summary --------------------------------------
    top20 = pd.DataFrame()
    if len(J):
        top20 = (J.assign(absL=J.LM_stat.abs()).nlargest(20, "absL")
                 [["ts_hkt", "label", "mid", "log_return", "jump_direction",
                   "LM_stat", "local_volatility", "staleness_s", "n_events"]])
        top20.to_csv(os.path.join(RES, "top20_jumps.csv"), index=False)
    pd.Series(stats).to_csv(os.path.join(RES, "jump_statistics.csv"))
    with open(os.path.join(RES, "jump_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("# Lee-Mykland jump detection — summary\n\n")
        fh.write("Generated %s. K=%d, alpha=%.3f, threshold=%.4f "
                 "(C_n=%.4f, S_n=%.4f, c=sqrt(2/pi)=%.4f).\n\n"
                 % (time.strftime("%Y-%m-%d %H:%M"), a.K, a.alpha, thr, c_n,
                    s_n, C_BP))
        fh.write("| statistic | value |\n|---|---|\n")
        for k, v in stats.items():
            fh.write("| %s | %s |\n" % (k, ("%.6g" % v) if isinstance(v, float)
                                        else v))
        if len(top20):
            fh.write("\n## Top 20 jumps by |LM|\n\n")
            fh.write(top20.to_markdown(index=False))
            fh.write("\n")
    print()
    print(pd.Series(stats).to_string())
    if len(top20):
        print("\n=== top 20 by |LM| ===")
        print(top20.to_string(index=False))
    log("outputs in %s" % RES)


if __name__ == "__main__":
    sys.exit(main())
