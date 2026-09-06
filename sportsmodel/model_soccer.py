"""
Dixon-Coles (1997) for soccer, walk-forward, scored against real baselines.

The model
---------
Goals are Poisson with team strengths and a home advantage:

    lambda = exp(atk_home - def_away + home_adv)      home goals
    mu     = exp(atk_away - def_home)                 away goals

Dixon-Coles adds two things to plain Poisson, and both matter:

  1. A dependence correction tau on the four low scores (0-0, 1-0, 0-1, 1-1),
     because independent Poisson under-predicts draws -- and the draw is a
     third of the question being asked here.
  2. Exponential time decay on the likelihood, so a match from three seasons
     ago counts less than one from last month. Team strength is not constant
     and a model that assumes it is will be confidently wrong after a
     transfer window.

Identifiability: attack parameters are constrained to sum to zero, otherwise
a constant can move between attack and defence without changing the fit.

Evaluation
----------
Strictly walk-forward within each league: fit on matches strictly BEFORE a
cutoff date, predict the next block, roll forward. Nothing is fitted on data
it later scores.

Scored three ways, because accuracy alone is a poor guide when one class
holds 44% of the mass:
    accuracy   share of H/D/A calls that were right
    log loss   proper score; punishes confident errors
    Brier      proper score; the multiclass version

against three baselines: always-home, the league's base rates, and plain
Poisson without the DC correction.

    python model_soccer.py
"""
import argparse
import collections
import csv
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "data", "soccer.csv")
MAXG = 10          # goals grid for the outcome probabilities


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction, vectorized over matches."""
    out = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return np.clip(out, 1e-10, None)


def fit(hidx, aidx, hg, ag, w, n_teams, use_dc=True, x0=None):
    """MLE for attack/defence/home-advantage (and rho if use_dc)."""
    def unpack(p):
        atk = np.empty(n_teams)
        atk[:-1] = p[:n_teams - 1]
        atk[-1] = -atk[:-1].sum()          # sum-to-zero constraint
        dfn = p[n_teams - 1:2 * n_teams - 1]
        return atk, dfn, p[2 * n_teams - 1], (p[2 * n_teams] if use_dc else 0.0)

    def nll(p):
        atk, dfn, ha, rho = unpack(p)
        lam = np.exp(atk[hidx] - dfn[aidx] + ha)
        mu = np.exp(atk[aidx] - dfn[hidx])
        lam = np.clip(lam, 1e-8, 20)
        mu = np.clip(mu, 1e-8, 20)
        ll = (-lam + hg * np.log(lam)) + (-mu + ag * np.log(mu))
        if use_dc:
            ll = ll + np.log(tau(hg, ag, lam, mu, rho))
        return -np.sum(w * ll)

    k = 2 * n_teams + (1 if use_dc else 0)
    if x0 is None or x0.size != k:
        x0 = np.zeros(k)
        x0[2 * n_teams - 1] = 0.25          # home advantage prior
    bnds = [(-3, 3)] * (n_teams - 1) + [(-3, 3)] * n_teams + [(-1, 1)]
    if use_dc:
        bnds += [(-0.2, 0.2)]
    r = minimize(nll, x0, method="L-BFGS-B", bounds=bnds,
                 options={"maxiter": 400})
    return r.x


def shrink(P, base, eps):
    """Blend model probabilities toward the base rate.

    A Poisson score model is overconfident in the tails and, on a truncated
    goal grid, can assign a numerically ZERO probability to an outcome that
    then happens -- observed here as H/D/A = 87.6/12.4/0.0 with Away winning.
    In soccer any side can win, so zero is never the right answer and a
    single such call dominates the log loss. Shrinking toward the base rate
    puts a floor under every outcome. `eps` is chosen on TRAINING data only.
    """
    return (1.0 - eps) * P + eps * base


def pick_eps(P_tr, y_tr, base):
    """Shrinkage that minimises log loss on the training block."""
    best, best_ll = 0.02, None
    for eps in (0.0, 0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35):
        Q = shrink(P_tr, base, eps)
        ll = -np.mean(np.log(np.clip(Q[np.arange(len(y_tr)), y_tr], 1e-15, 1)))
        if best_ll is None or ll < best_ll:
            best, best_ll = eps, ll
    return best


def outcome_probs(atk, dfn, ha, rho, h, a, use_dc):
    lam = float(np.clip(np.exp(atk[h] - dfn[a] + ha), 1e-8, 20))
    mu = float(np.clip(np.exp(atk[a] - dfn[h]), 1e-8, 20))
    px = poisson.pmf(np.arange(MAXG + 1), lam)
    py = poisson.pmf(np.arange(MAXG + 1), mu)
    M = np.outer(px, py)
    if use_dc:
        M[0, 0] *= 1.0 - lam * mu * rho
        M[0, 1] *= 1.0 + lam * rho
        M[1, 0] *= 1.0 + mu * rho
        M[1, 1] *= 1.0 - rho
        M = np.clip(M, 1e-12, None)
    M /= M.sum()
    ph = float(np.tril(M, -1).sum())        # home goals > away goals
    pd = float(np.trace(M))
    pa = float(np.triu(M, 1).sum())
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--half-life", type=float, default=180.0,
                    help="days over which a match's weight halves")
    ap.add_argument("--burn-in", type=int, default=400,
                    help="matches per league before predictions start")
    ap.add_argument("--refit-every", type=int, default=40)
    args = ap.parse_args()

    rows = []
    with open(args.src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    by_lg = collections.defaultdict(list)
    for r in rows:
        by_lg[r["league"]].append(r)

    preds = []          # (p_h, p_d, p_a, actual, league, model)
    xi = math.log(2) / args.half_life

    for lg, ms in sorted(by_lg.items()):
        ms.sort(key=lambda r: r["date"])
        teams = sorted({m["home"] for m in ms} | {m["away"] for m in ms})
        tidx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)
        dates = np.array([time.mktime(time.strptime(m["date"], "%Y-%m-%d"))
                          for m in ms]) / 86400.0
        hidx = np.array([tidx[m["home"]] for m in ms])
        aidx = np.array([tidx[m["away"]] for m in ms])
        hg = np.array([int(m["hg"]) for m in ms])
        ag = np.array([int(m["ag"]) for m in ms])

        warm = {True: None, False: None}
        i = args.burn_in
        n_fit = 0
        while i < len(ms):
            block = min(args.refit_every, len(ms) - i)
            t_cut = dates[i]
            w = np.exp(-xi * (t_cut - dates[:i]))
            for use_dc in (True, False):
                p = fit(hidx[:i], aidx[:i], hg[:i], ag[:i], w, n_teams,
                        use_dc=use_dc, x0=warm[use_dc])
                warm[use_dc] = p
                atk = np.empty(n_teams)
                atk[:-1] = p[:n_teams - 1]
                atk[-1] = -atk[:-1].sum()
                dfn = p[n_teams - 1:2 * n_teams - 1]
                ha = p[2 * n_teams - 1]
                rho = p[2 * n_teams] if use_dc else 0.0
                # base rate and shrinkage from the TRAINING window only
                res_tr = [ms[q]["result"] for q in range(i)]
                base = np.array([res_tr.count(c) / max(len(res_tr), 1)
                                 for c in ("H", "D", "A")])
                base = base / base.sum()
                tr_idx = list(range(max(0, i - 400), i))
                P_tr = np.array([outcome_probs(atk, dfn, ha, rho, hidx[q],
                                               aidx[q], use_dc) for q in tr_idx])
                y_tr = np.array([("H", "D", "A").index(ms[q]["result"])
                                 for q in tr_idx])
                eps = pick_eps(P_tr, y_tr, base)
                for j in range(i, i + block):
                    ph, pd, pa = outcome_probs(atk, dfn, ha, rho,
                                               hidx[j], aidx[j], use_dc)
                    q = shrink(np.array([[ph, pd, pa]]), base, eps)[0]
                    preds.append((q[0], q[1], q[2], ms[j]["result"], lg,
                                  "dixon_coles" if use_dc else "poisson"))
            n_fit += 1
            i += block
        log(f"  {lg}: {len(ms)} matches, {n_fit} refits")

    # ---- scoring ----------------------------------------------------------
    print()
    print("=" * 88)
    print("SOCCER: walk-forward, 3-way outcome (Home / Draw / Away)")
    print("=" * 88)
    labels = ["H", "D", "A"]
    base_counts = collections.Counter(p[3] for p in preds if p[5] == "dixon_coles")
    tot = sum(base_counts.values())
    base_rate = np.array([base_counts[l] / tot for l in labels])

    def score(rows_, name):
        P = np.array([[r[0], r[1], r[2]] for r in rows_])
        y = np.array([labels.index(r[3]) for r in rows_])
        pick = P.argmax(axis=1)
        acc = float((pick == y).mean())
        ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))))
        Y = np.zeros_like(P)
        Y[np.arange(len(y)), y] = 1
        brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
        return {"name": name, "n": len(y), "acc": acc, "ll": ll, "brier": brier}

    dc = [p for p in preds if p[5] == "dixon_coles"]
    po = [p for p in preds if p[5] == "poisson"]
    y_all = np.array([labels.index(r[3]) for r in dc])
    res = [score(dc, "Dixon-Coles"), score(po, "Poisson (no DC)")]
    # baselines
    always_h = np.tile(np.array([1.0, 0.0, 0.0]), (len(dc), 1))
    always_h = np.clip(always_h, 1e-6, 1)
    always_h /= always_h.sum(axis=1, keepdims=True)
    base_p = np.tile(base_rate, (len(dc), 1))
    for nm, P in (("always Home", always_h), ("league base rates", base_p)):
        pick = P.argmax(axis=1)
        acc = float((pick == y_all).mean())
        ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y_all)), y_all], 1e-15, 1))))
        Y = np.zeros_like(P)
        Y[np.arange(len(y_all)), y_all] = 1
        res.append({"name": nm, "n": len(y_all), "acc": acc, "ll": ll,
                    "brier": float(np.mean(np.sum((P - Y) ** 2, axis=1)))})

    print(f"  {'model':>20} {'n':>8} {'accuracy':>10} {'log loss':>10} {'Brier':>9}")
    print("  " + "-" * 62)
    for r in sorted(res, key=lambda r: r["ll"]):
        print(f"  {r['name']:>20} {r['n']:>8,} {r['acc']*100:>9.2f}% "
              f"{r['ll']:>10.4f} {r['brier']:>9.4f}")

    print()
    print("  Dixon-Coles by predicted class:")
    P = np.array([[r[0], r[1], r[2]] for r in dc])
    pick = P.argmax(axis=1)
    for k, lab in enumerate(labels):
        m = pick == k
        if m.sum():
            print(f"    predicts {lab}: {int(m.sum()):>6,} times, "
                  f"right {float((y_all[m] == k).mean())*100:>5.1f}%, "
                  f"mean confidence {P[m, k].mean()*100:>5.1f}%")
    print()
    print("  CALIBRATION -- mean predicted vs actual frequency (DC):")
    for k, lab in enumerate(labels):
        print(f"    {lab}: model says {P[:,k].mean()*100:>5.2f}% on average, "
              f"actual {float((y_all==k).mean())*100:>5.2f}%")
    print()
    print("  reliability by confidence decile (Home):")
    ph = P[:,0]; yh = (y_all==0).astype(float)
    order = np.argsort(ph)
    for d in range(10):
        lo, hi = d*len(order)//10, (d+1)*len(order)//10
        idx = order[lo:hi]
        print(f"    decile {d+1:>2}: predicted {ph[idx].mean()*100:>5.1f}%  "
              f"actual {yh[idx].mean()*100:>5.1f}%  n={len(idx):,}")
    worst = np.argsort(P[np.arange(len(y_all)), y_all])[:5]
    print()
    print("  most confidently wrong predictions (P assigned to the truth):")
    for w in worst:
        print(f"    p(truth)={P[w, y_all[w]]*100:.3f}%  probs H/D/A="
              f"{P[w,0]*100:.1f}/{P[w,1]*100:.1f}/{P[w,2]*100:.1f}  actual={labels[y_all[w]]}")
    print()
    print(f"  NOTE: the model predicted a DRAW {int((pick==1).sum()):,} times "
          f"out of {len(pick):,}.")
    print("  Draws are ~25% of outcomes but rarely the single most likely one,")
    print("  so an argmax classifier almost never calls them. That is correct")
    print("  behaviour for accuracy and useless for betting the draw.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
