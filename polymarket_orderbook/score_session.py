"""Score a bundled model on a session it never trained on.

This is the out-of-sample test the study was missing. Everything so far was
fit and evaluated inside one recording (2026-08-30), split only by time, with
79 of 85 series appearing on both sides -- and the diagnostics found that
93.5% of the taker P&L came from 10 of 49 series with only 15 profitable, plus
a +0.112 train-to-test AUC gap. A temporal split cannot detect either problem.

The 2026-09-09 recording is a genuine holdout: 15 different games, a different
date, zero series overlap with the training session.

Scoring loads only the target session, so the tensor stays memory-mapped
rather than concatenating both sessions into ~3GB of RAM.

Output is written in the same form taker_signal.py produces
(takeredge_/takeridx_), so simulate_taker.py can consume it unchanged.

    python score_session.py --session hkt0909 --model cnn_direction
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jump_model import frame_offsets
from jump_split import JD, TICK, load
from model_registry import load as load_bundle
from taker_signal import (batcher, label3, prepare, sweep, taker_pnl,
                          touch_backing)


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="hkt0909",
                    help="substring of the feat_*.parquet to score")
    ap.add_argument("--model", default="cnn_direction")
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--label-ticks", type=int, default=None,
                    help="default: the k the bundle was trained with")
    ap.add_argument("--min-backing", type=float, default=200.0)
    ap.add_argument("--max-spread", type=float, default=2.0)
    ap.add_argument("--max-book-age-s", type=float, default=5.0)
    ap.add_argument("--category", default="sports")
    ap.add_argument("--market-types", nargs="*",
                    default=["moneyline", "total", "spread"])
    ap.add_argument("--tag", default=None,
                    help="suffix for the saved edges; default _<session>")
    a = ap.parse_args()
    pd.set_option("display.width", 240)

    import torch
    model, man = load_bundle(a.model, a.version)
    cfg = man["train_config"]
    k = a.label_ticks if a.label_ticks is not None else cfg.get("k", 4)
    log(f"model {a.model}_v{man['version']}  ({man['nparam']:,} params, "
        f"trained with k={cfg.get('k')} ticks)")
    log(f"  trained on: {man['data_spec'].get('feat_files')}")

    OFF, STR, _, lookback = frame_offsets(cfg.get("fine", 24),
                                          cfg.get("coarse", 24),
                                          cfg.get("stride", 8))
    F, T, tr, va, te = load(a.market_types, lookback=lookback, log=log,
                            max_book_age_s=a.max_book_age_s,
                            sessions=a.session)
    F, h = prepare(F, 5.0, a.category, a.max_spread)

    # every eligible row in this session -- the temporal split is irrelevant
    # here because nothing is being fitted
    elig = np.sort(np.concatenate([tr, va, te]))
    elig = elig[F.tight.to_numpy()[elig]]
    log(f"eligible and tight: {len(elig):,}")
    bk = touch_backing(T)
    F["backing"] = bk
    elig = elig[bk[elig] >= a.min_backing]
    log(f"  backed >= ${a.min_backing:,.0f} both sides: {len(elig):,}")
    if len(elig) == 0:
        raise SystemExit("nothing to score after filters")

    signed = F.signed_ticks.to_numpy(np.float64)
    log(f"  P(|move| >= {k}) here: {(np.abs(signed[elig]) >= k).mean():.4f}   "
        f"mean |move| {np.abs(signed[elig]).mean():.2f} ticks   "
        f"round trip {F.cost_taker.to_numpy()[elig].mean():.2f} ticks")

    # Directional models emit 3 classes and are scored as P(up) - P(down);
    # magnitude models emit one logit and are scored as P(|move| >= 2 ticks).
    # The manifest's builder is the discriminator, so this works for any
    # bundle without a flag to get wrong.
    builder = man["architecture"].get("builder", "")
    is_direction = builder == "taker_signal.build_dircnn"
    log(f"  model kind: {'directional (3-class)' if is_direction else 'magnitude (binary)'}")

    if is_direction:
        mb = batcher(T, F, OFF, STR, label3(signed, k))
    else:
        from collapse_cnn import make_batcher
        mb0, _ = make_batcher(T, F, OFF, STR,
                              cfg.get("variant", "raw"),
                              np.zeros(len(F), np.float32))
        mb = lambda j: mb0(j)[:2]

    log(f"scoring {len(elig):,} rows...")
    t0 = time.time()
    edges = []
    with torch.no_grad():
        for s in range(0, len(elig), 1024):
            out = mb(elig[s:s + 1024])
            img, sc = out[0], out[1]
            if is_direction:
                q = torch.softmax(model(img, sc), dim=-1).numpy()
                edges.append(q[:, 2] - q[:, 0])
            else:
                edges.append(torch.sigmoid(model(img, sc)).numpy())
    e = np.concatenate(edges)
    log(f"  done in {time.time()-t0:.0f}s")

    if not is_direction:
        # magnitude: report the maker-gate economics, which is the use that
        # survived its controls, and stop -- the taker sweep below is
        # meaningless for a model that does not predict a side
        from jump_split import bootstrap_pnl_opp, econ_eval, save_row
        y = F.y.to_numpy()[elig]
        auc_m = (roc_auc_score(y, e) if len(np.unique(y)) > 1 else float("nan"))
        R, base = econ_eval(e, F, elig)
        best = R.loc[R.pnl_per_opportunity.idxmax()]
        lo, hi = bootstrap_pnl_opp(e, F, elig, float(best.threshold))
        log(f"\n{'='*100}\nOUT-OF-SAMPLE MAGNITUDE, session '{a.session}'"
            f"\n{'='*100}")
        log(f"  AUC {auc_m:.4f}   base rate {y.mean():.4f}   "
            f"n {len(elig):,}")
        log(f"  quote-blind {base:+.5f}/share   gated "
            f"{best.pnl_per_opportunity:+.5f} at threshold "
            f"{best.threshold:.2f}   CI [{lo:+.5f}, {hi:+.5f}]")
        mt_m = F.mt.to_numpy().astype(str)[elig]
        log(f"\n  by market type:")
        log(f"  {'market':>12} {'n':>9} {'base':>7} {'AUC':>8} {'blind':>9} "
            f"{'gated':>9}")
        for m in ["moneyline", "total", "spread"]:
            s = mt_m == m
            if s.sum() < 500 or len(np.unique(y[s])) < 2:
                continue
            Rm, bm = econ_eval(e[s], F, elig[s],
                               thresholds=np.array([float(best.threshold)]))
            log(f"  {m:>12} {s.sum():>9,} {y[s].mean():>7.3f} "
                f"{roc_auc_score(y[s], e[s]):>8.4f} {bm:>9.4f} "
                f"{Rm.iloc[0].pnl_per_opportunity:>9.4f}")
        tag_m = a.tag or f"_{a.session}"
        np.save(os.path.join(JD, f"preds_{a.model}{tag_m}.npy"),
                e.astype(np.float32))
        np.save(os.path.join(JD, f"testidx_{a.model}{tag_m}.npy"), elig)
        save_row(dict(model=f"{a.model}{tag_m}", auc=auc_m,
                      base_rate=float(y.mean()), pnl_always=base,
                      threshold=float(best.threshold),
                      pnl_per_opp=float(best.pnl_per_opportunity),
                      pnl_opp_lo=lo, pnl_opp_hi=hi, beats_zero=bool(lo > 0),
                      n_test=len(elig), session=a.session))
        log(f"\nsaved preds -> preds_{a.model}{tag_m}.npy")
        return 0

    tag = a.tag or f"_{a.session}"
    np.save(os.path.join(JD, f"takeredge_{a.model}{tag}.npy"),
            e.astype(np.float32))
    np.save(os.path.join(JD, f"takeridx_{a.model}{tag}.npy"), elig)

    # ---- how does it look out of sample? --------------------------------- #
    mv = signed[elig] != 0
    auc = (roc_auc_score((signed[elig][mv] > 0).astype(int), e[mv])
           if mv.sum() > 200 and len(np.unique(signed[elig][mv] > 0)) > 1
           else float("nan"))
    tcsv = os.path.join(JD, "taker_comparison.csv")
    tau = 0.90
    if os.path.exists(tcsv):
        t = pd.read_csv(tcsv)
        r = t[t.model == a.model]
        if len(r):
            tau = float(r.iloc[0].tau)
    log(f"\n{'='*100}\nOUT-OF-SAMPLE, session '{a.session}'\n{'='*100}")
    log(f"  sign AUC {auc:.4f}   (in-sample test block was 0.7896)")
    log(f"  moved rows {mv.sum():,} of {len(elig):,} ({mv.mean():.3%})")

    R = sweep(e, F, elig, np.round(np.arange(0.0, 0.96, 0.05), 3))
    log(f"\n  P&L per opportunity in ticks, by trade threshold "
        f"(tau chosen in training was {tau:.2f})")
    log(R.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    pnl, side, traded = taker_pnl(e, F, elig, tau)
    hit = (((side > 0) == (signed[elig] > 0))[traded & mv].mean()
           if (traded & mv).any() else float("nan"))
    log(f"\n  at the trained tau={tau:.2f}: traded {traded.mean():.3%}, "
        f"hit {hit:.4f}, pnl/opp {pnl.mean():+.4f} ticks")

    mt = F.mt.to_numpy().astype(str)[elig]
    log(f"\n  by market type:")
    log(f"  {'market':>12} {'n':>9} {'moved%':>8} {'signAUC':>9} "
        f"{'trade%':>8} {'pnl/opp':>9}")
    for m in ["moneyline", "total", "spread"]:
        s = mt == m
        if s.sum() < 500:
            continue
        m2 = mv & s
        au = (roc_auc_score((signed[elig][m2] > 0).astype(int), e[m2])
              if m2.sum() > 200 and len(np.unique(signed[elig][m2] > 0)) > 1
              else float("nan"))
        pn, sd, td = taker_pnl(e[s], F, elig[s], tau)
        log(f"  {m:>12} {s.sum():>9,} {mv[s].mean():>8.3%} {au:>9.4f} "
            f"{td.mean():>8.3%} {pn.mean():>9.4f}")

    with open(os.path.join(JD, f"oos_{a.model}{tag}.json"), "w") as fh:
        json.dump(dict(model=a.model, version=man["version"], session=a.session,
                       tau=tau, sign_auc=None if np.isnan(auc) else auc,
                       n_scored=len(elig), pnl_per_opp=float(pnl.mean()),
                       trade_frac=float(traded.mean()),
                       config=vars(a), ts=time.strftime("%Y-%m-%dT%H:%M:%S")),
                  fh, indent=2, default=str)
    log(f"\nsaved edges -> takeredge_{a.model}{tag}.npy  "
        f"(simulate_taker.py --model {a.model}{tag} will walk the clock)")


if __name__ == "__main__":
    main()
