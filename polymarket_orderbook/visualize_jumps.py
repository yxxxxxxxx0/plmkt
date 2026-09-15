"""Plot where the model expects a jump, against what the price actually did.

Three stacked panels per market, sharing a time axis:

  1  mid price, with the model's TRADE decisions marked (up / down) and the
     actual >=k-tick moves shaded, so agreement and disagreement are visible
     directly rather than through an aggregate statistic
  2  the model's class probabilities -- P(up), P(down), and the P(up)-P(down)
     edge it actually trades on, with the validation-chosen tau drawn in
  3  realised forward move in ticks, with the +/-k-tick label threshold and
     the round-trip cost drawn in

The third panel is the one that tends to be sobering: a prediction can be
directionally right and still sit inside the cost band, which is most of why
a high hit rate has not translated into money.

Rows the model is not allowed to trade (stale book, thin touch, wide spread,
no contiguous history) are drawn faded, so the plot shows the eligible subset
honestly rather than implying the model saw everything.

    python visualize_jumps.py --session 2026-09-10
    python visualize_jumps.py --match mlb-tex-sea-2026-09-10 --minutes 30
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from jump_model import frame_offsets
from jump_split import JD, TICK, load
from polymarket_fees import taker_fee_ticks
from taker_signal import batcher, label3, prepare, touch_backing

OUT = os.path.join(JD, "plots")


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="2026-09-10",
                    help="which trimmed session to load (keeps memory low)")
    ap.add_argument("--match", default=None,
                    help="slug to plot; default = the busiest in the session")
    ap.add_argument("--market-type", default="spread",
                    help="spread carries the signal; moneyline is at chance")
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--start-frac", type=float, default=0.45,
                    help="where in the match to start the window, 0-1")
    ap.add_argument("--model", default="cnn_direction")
    ap.add_argument("--label-ticks", type=int, default=4)
    ap.add_argument("--min-backing", type=float, default=200.0)
    ap.add_argument("--max-spread", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch
    ckpt = os.path.join(JD, f"{a.model}.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint at {ckpt}")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = blob.get("config", {})
    from taker_signal import build_dircnn
    OFF, STR, _, lookback = frame_offsets(cfg.get("fine", 24),
                                          cfg.get("coarse", 24),
                                          cfg.get("stride", 8))
    model = build_dircnn(len(OFF), d=cfg.get("d_model", 64))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    k = cfg.get("k", a.label_ticks)
    log(f"loaded {a.model} ({blob.get('nparam', 0):,} params, k={k})")

    tau = a.tau
    if tau is None:
        p = os.path.join(JD, "taker_comparison.csv")
        if os.path.exists(p):
            t = pd.read_csv(p)
            r = t[t.model == a.model]
            tau = float(r.iloc[0].tau) if len(r) else 0.9
        else:
            tau = 0.9

    F, T, tr, va, te = load(["moneyline", "total", "spread"],
                            lookback=lookback, log=log, max_book_age_s=5.0,
                            use_trimmed=True, sessions=a.session)
    F, h = prepare(F, 5.0, "sports", a.max_spread)
    elig = np.zeros(len(F), bool)
    elig[np.concatenate([tr, va, te])] = True
    bk = touch_backing(T)

    slug = F.series.astype(str).str.split("|").str[0].to_numpy()
    mt = F.mt.to_numpy().astype(str)
    if a.match is None:
        cand = pd.Series(slug[elig & (mt == a.market_type)]).value_counts()
        if cand.empty:
            raise SystemExit("no eligible rows for that market type")
        a.match = cand.index[0]
    log(f"plotting {a.match} ({a.market_type})")

    sel = (slug == a.match) & (mt == a.market_type)
    if not sel.any():
        raise SystemExit(f"no rows for {a.match} / {a.market_type}")
    # one series (one line) so the time axis is a single book
    sids = pd.Series(F.sid.to_numpy()[sel]).value_counts()
    sid = sids.index[0]
    sel &= F.sid.to_numpy() == sid

    idx_all = np.where(sel)[0]
    n_win = int(a.minutes * 60 * 1000 / 200)
    s0 = int(len(idx_all) * a.start_frac)
    idx = idx_all[s0:s0 + n_win]
    if len(idx) < 500:
        idx = idx_all[-min(n_win, len(idx_all)):]
    log(f"  {len(idx):,} grid rows ({len(idx)*0.2/60:.1f} min)")

    # score every row in the window that has a usable history window
    can = elig[idx] & (bk[idx] >= a.min_backing)
    y3 = label3(F.signed_ticks.to_numpy(np.float64), k)
    mb = batcher(T, F, OFF, STR, y3)
    scor = idx[can]
    P = np.full((len(idx), 3), np.nan)
    if len(scor):
        outp = []
        with torch.no_grad():
            for s in range(0, len(scor), 1024):
                img, sc, _ = mb(scor[s:s + 1024])
                outp.append(torch.softmax(model(img, sc), dim=-1).numpy())
        P[can] = np.concatenate(outp)
    log(f"  scored {can.sum():,} of {len(idx):,} rows "
        f"({can.mean():.0%} tradeable: fresh book, backed touch, tight spread)")

    ts = F.ts.to_numpy()[idx]
    tmin = (ts - ts[0]) / 60000.0
    mid = F.mid.to_numpy(np.float64)[idx]
    signed = F.signed_ticks.to_numpy(np.float64)[idx]
    cost = F.cost_taker.to_numpy(np.float64)[idx]
    edge = P[:, 2] - P[:, 0]
    side = np.where(edge > tau, 1, np.where(edge < -tau, -1, 0))

    fig, ax = plt.subplots(3, 1, figsize=(16, 11), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 2, 2]))

    # ---- panel 1: price + decisions ------------------------------------- #
    ax[0].plot(tmin, mid, lw=1.0, color="#222", zorder=3)
    big = np.abs(signed) >= k
    ax[0].fill_between(tmin, mid.min(), mid.max(), where=big, color="#ffd9d9",
                       step="mid", zorder=0,
                       label=f"actual move >= {k} ticks (next 5s)")
    up = side == 1
    dn = side == -1
    ax[0].scatter(tmin[up], mid[up], s=16, marker="^", color="#0a8f3c",
                  zorder=4, label=f"model: BUY (edge > {tau:.2f})")
    ax[0].scatter(tmin[dn], mid[dn], s=16, marker="v", color="#c1121f",
                  zorder=4, label=f"model: SELL (edge < -{tau:.2f})")
    ax[0].scatter(tmin[~can], mid[~can], s=4, color="#bbb", zorder=1,
                  label="not tradeable (stale/thin/wide)")
    ax[0].set_ylabel("mid price")
    ax[0].set_title(f"{a.match}  |  {a.market_type}  |  {a.model}   "
                    f"(tau={tau:.2f}, chosen on validation)")
    ax[0].legend(loc="upper left", fontsize=8, ncol=2)
    ax[0].grid(alpha=0.25)

    # ---- panel 2: probabilities ------------------------------------------ #
    ax[1].plot(tmin, P[:, 2], lw=0.9, color="#0a8f3c", label="P(up >= k)")
    ax[1].plot(tmin, P[:, 0], lw=0.9, color="#c1121f", label="P(down >= k)")
    ax[1].plot(tmin, edge, lw=1.3, color="#1d3557",
               label="edge = P(up) - P(down)")
    ax[1].axhline(tau, ls="--", lw=0.9, color="#0a8f3c")
    ax[1].axhline(-tau, ls="--", lw=0.9, color="#c1121f")
    ax[1].axhline(0, lw=0.6, color="#999")
    ax[1].set_ylabel("probability / edge")
    ax[1].set_ylim(-1.05, 1.05)
    ax[1].legend(loc="upper left", fontsize=8, ncol=3)
    ax[1].grid(alpha=0.25)

    # ---- panel 3: realised move vs what it must clear -------------------- #
    ax[2].plot(tmin, signed, lw=0.8, color="#444",
               label="realised move at +5s (ticks)")
    ax[2].axhline(k, ls=":", color="#0a8f3c", label=f"+/-{k} tick label")
    ax[2].axhline(-k, ls=":", color="#c1121f")
    ax[2].fill_between(tmin, -cost, cost, color="#ffe8b3", zorder=0,
                       label="round-trip cost band (spread + both fees)")
    ax[2].set_ylabel("ticks")
    ax[2].set_xlabel("minutes into the plotted window")
    ax[2].legend(loc="upper left", fontsize=8, ncol=3)
    ax[2].grid(alpha=0.25)

    # ---- what happened, in numbers --------------------------------------- #
    traded = side != 0
    mv = signed != 0
    hit = (((side > 0) == (signed > 0))[traded & mv].mean()
           if (traded & mv).any() else np.nan)
    pnl = np.where(traded, side * signed - cost, 0.0)
    txt = (f"rows {len(idx):,}   tradeable {can.sum():,}   "
           f"signals {traded.sum():,}   hit {hit:.0%}   "
           f"P&L {pnl.sum():+.0f} ticks   "
           f"mean cost {cost.mean():.2f} ticks   "
           f"fee {taker_fee_ticks(mid.mean(), TICK, 'sports'):.2f}/leg")
    fig.text(0.5, 0.005, txt, ha="center", fontsize=9, color="#333")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    fn = os.path.join(OUT, f"jumps_{a.match}_{a.market_type}.png")
    fig.savefig(fn, dpi=130)
    log(f"\nwrote {fn}")
    log(f"  {txt}")


if __name__ == "__main__":
    main()
