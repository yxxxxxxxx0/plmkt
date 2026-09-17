"""Sweep the HOLD horizon: where does the collapse signal actually pay?

Two clocks were being conflated in this project and they must be kept apart.

  lead time   how far BEFORE the price move the signal appears. Measured in
              pre_jump_seconds.py: about 6 s at 200 ms resolution.
  hold time   how long the position is open. That is what this script sweeps.

The temptation is to stretch the hold until the arithmetic works. At +30 s the
median move on a tight-spread collapse is ~1.0 tick against a ~1.5 tick round
trip, so it loses; at +300 s the move is ~4 ticks and it wins. But the jump
itself is over in about 10 s, so everything past +10 s is not the jump -- it is
subsequent game drift. The null test in this directory showed exactly that:
entering at a RANDOM tight moment captured 3.0 of those 4.0 ticks. Stretching
the hold does not trade the signal, it trades the game.

So this sweeps every horizon and reports, for each, the three numbers that
decide it:

  move/cost           how many round trips the move is worth
  break-even hit      the direction accuracy needed to break even
  measured EV         what a trained direction model actually earns, held out

and the same for the two nulls (random tight entry, coin-flip direction), so
the contribution of the signal is separated from the contribution of the clock.

Everything is causal: features end AT the entry instant, the collapse detector
is a trailing rule, and the split is chronological by whole session.

    python direction_hold.py
    python direction_hold.py --holds 10,30,60 --max-spread 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "jump_prediction"))
sys.path.insert(0, HERE)
import features as FE  # noqa: E402
from collapse_classify import TRACK, build_sequences  # noqa: E402

JD = os.path.join(ROOT, "data", "jump")
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(ROOT, "results", "makinen", "direction_hold")
GRID_MS = 200
TICK = 0.01


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def exit_at(E, hold_s):
    """Mid and spread at entry + hold_s. The exit must be a valid quote on the
    SAME contract, otherwise the event is dropped for this horizon."""
    step = int(round(hold_s * 1000 / GRID_MS))
    n_ev = len(E)
    ok = np.zeros(n_ev, bool)
    mid_out = np.full(n_ev, np.nan)
    spr_out = np.full(n_ev, np.nan)
    for sess, Es in E.groupby("session", sort=False):
        F = pq.read_table(os.path.join(JD, "feat_%s_trimmed.parquet" % sess),
                          columns=["sid", "mid", "spread_ticks", "valid"]).to_pandas()
        sid = F.sid.to_numpy()
        mid = F.mid.to_numpy(np.float64)
        spt = F.spread_ticks.to_numpy(np.float64)
        val = F.valid.to_numpy(bool)
        n = len(F)
        pos = Es.index.to_numpy()
        r0 = Es.row.to_numpy()
        r1 = r0 + step
        good = r1 < n
        r1c = np.clip(r1, 0, n - 1)
        good &= sid[r1c] == sid[r0]
        good &= val[r1c] & val[r0]
        ok[pos] = good
        mid_out[pos] = np.where(good, mid[r1c], np.nan)
        spr_out[pos] = np.where(good, spt[r1c], np.nan)
    return ok, mid_out, spr_out


def cluster_boot(pnl, games, n=1000, seed=0):
    """CI on mean P&L, resampling whole GAMES. Events inside one game share the
    same scoring shocks, so treating them as independent would overstate
    precision badly."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(games)
    idx = [np.where(games == g)[0] for g in uniq]
    draws = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(idx), len(idx))
        draws[b] = pnl[np.concatenate([idx[i] for i in pick])].mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holds", default="10,20,30,60,120,300,600")
    ap.add_argument("--max-spread", type=float, default=3.0)
    ap.add_argument("--pre-s", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    os.makedirs(RES, exist_ok=True)
    holds = [float(h) for h in a.holds.split(",")]

    E = pd.read_parquet(os.path.join(CACHE, "collapse_events.parquet")).reset_index(drop=True)
    log("collapse events {:,}".format(len(E)))
    E = E[E.spread_ticks <= a.max_spread].reset_index(drop=True)
    log("tight entry (spread <= {:g} ticks): {:,}".format(a.max_spread, len(E)))

    # sequences built ONCE; only the exit changes with the horizon
    X, keep = build_sequences(E, a.pre_s, a.stride)
    E = E.loc[keep].reset_index(drop=True)
    log("with a complete 20 s pre-window: {:,}".format(len(E)))

    inst = X[:, -1, :]
    traj = np.concatenate([inst,
                           X[:, -1, :] - X[:, -5, :],
                           X[:, -1, :] - X[:, -13, :],
                           X[:, -1, :] - X[:, 0, :],
                           X.min(axis=1), X.max(axis=1), X.std(axis=1)], axis=1)
    names = (["%s_now" % c for c in TRACK] + ["%s_d1s" % c for c in TRACK]
             + ["%s_d5s" % c for c in TRACK] + ["%s_d20s" % c for c in TRACK]
             + ["%s_min" % c for c in TRACK] + ["%s_max" % c for c in TRACK]
             + ["%s_std" % c for c in TRACK])
    del X

    sess = sorted(E.session.unique())
    split = np.where(E.session.isin(sess[:-2]), "train",
                     np.where(E.session.isin(sess[-2:-1]), "val", "test"))
    games_all = E.series.astype(str).str.split("|").str[0].to_numpy()
    spread_in = E.spread_ticks.to_numpy(np.float64)
    log("sessions %s   test = %s" % (sess, sess[-1]))

    rows = []
    for h in holds:
        ok, mid_out, spr_out = exit_at(E, h)
        if ok.sum() < 2000:
            log("+%gs: only %d events survive, skipping" % (h, ok.sum()))
            continue
        move = (mid_out - E.mid.to_numpy(np.float64)) / TICK
        cost = (spread_in + spr_out) / 2.0
        y = (move > 0).astype(int)

        tr = ok & (split == "train")
        va = ok & (split == "val")
        te = ok & (split == "test")
        if te.sum() < 500 or va.sum() < 500:
            log("+%gs: not enough held-out events, skipping" % h)
            continue

        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                               min_child_samples=100, subsample=.8, subsample_freq=1,
                               colsample_bytree=.8, random_state=0, n_jobs=8,
                               verbose=-1)
        m.fit(traj[tr], y[tr])
        pv = m.predict_proba(traj[va])[:, 1]
        pt = m.predict_proba(traj[te])[:, 1]
        auc = roc_auc_score(y[te], pt)

        # confidence gate chosen on VALIDATION only, then frozen
        gate, gate_ev = 0.0, -1e9
        for g in np.arange(0.0, 0.35, 0.01):
            s = np.abs(pv - 0.5) >= g
            if s.sum() < 300:
                break
            ev = (np.where(pv[s] > 0.5, 1.0, -1.0) * move[va][s] - cost[va][s]).mean()
            if ev > gate_ev:
                gate_ev, gate = ev, g

        mv, ct, gm = move[te], cost[te], games_all[te]
        amv, act = np.abs(mv).mean(), ct.mean()
        be = (1 + act / amv) / 2 if amv > 0 else np.nan

        def ev_of(sel, side):
            pnl = side * mv[sel] - ct[sel]
            lo, hi = cluster_boot(pnl, gm[sel], a.boot)
            return pnl.mean(), (side * mv[sel] > 0).mean(), lo, hi

        allx = np.ones(te.sum(), bool)
        e_all, h_all, lo_a, hi_a = ev_of(allx, np.where(pt > 0.5, 1.0, -1.0))
        g_sel = np.abs(pt - 0.5) >= gate
        if g_sel.sum() >= 100:
            e_g, h_g, lo_g, hi_g = ev_of(g_sel, np.where(pt[g_sel] > 0.5, 1.0, -1.0))
        else:
            e_g = h_g = lo_g = hi_g = np.nan
        rng = np.random.default_rng(0)
        e_c, h_c, lo_c, hi_c = ev_of(allx, rng.choice([-1.0, 1.0], te.sum()))
        e_b, h_b, _, _ = ev_of(allx, np.ones(te.sum()))

        rows.append(dict(
            hold_s=h, n_test=int(te.sum()),
            med_move=float(np.median(np.abs(mv))), mean_move=float(amv),
            med_cost=float(np.median(ct)), mean_cost=float(act),
            move_over_cost=float(amv / act), break_even_hit=float(be),
            dir_roc=float(auc), model_hit=float(h_all), model_ev=float(e_all),
            model_ci_lo=lo_a, model_ci_hi=hi_a,
            gate=float(gate), gated_n=int(g_sel.sum()), gated_hit=float(h_g),
            gated_ev=float(e_g), gated_ci_lo=lo_g, gated_ci_hi=hi_g,
            coinflip_ev=float(e_c), alwaysbuy_hit=float(h_b),
            alwaysbuy_ev=float(e_b)))
        log("+%4gs n=%6d  move %.2ft cost %.2ft  ratio %.2f  need %.1f%%  "
            "ROC %.3f  model hit %.3f  EV %+.3ft [%+.3f,%+.3f]"
            % (h, te.sum(), amv, act, amv / act, 100 * be, auc, h_all,
               e_all, lo_a, hi_a))

    R = pd.DataFrame(rows)
    tag = "s%g" % a.max_spread
    R.to_csv(os.path.join(RES, "hold_sweep_%s.csv" % tag), index=False)
    json.dump(dict(max_spread=a.max_spread, pre_s=a.pre_s, holds=holds,
                   n_events=int(len(E)), sessions=list(sess),
                   test_session=sess[-1]),
              open(os.path.join(RES, "config_%s.json" % tag), "w"), indent=2)

    print()
    show = ["hold_s", "n_test", "med_move", "med_cost", "move_over_cost",
            "break_even_hit", "dir_roc", "model_hit", "model_ev",
            "model_ci_lo", "model_ci_hi", "coinflip_ev"]
    print(R[show].to_string(index=False, float_format=lambda v: "%.3f" % v))
    print("\nwrote", os.path.join(RES, "hold_sweep_%s.csv" % tag))


if __name__ == "__main__":
    sys.exit(main())
