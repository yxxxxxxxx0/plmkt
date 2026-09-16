"""Tick-by-tick replay: what would the detector actually have caught, live?

Everything measured so far was batch -- whole arrays, vectorised rolling
windows, train/test masks. That can hide two things a live system cannot avoid:

  * accidental lookahead in a rolling statistic,
  * the fact that a live system must decide AT the tick, with only a buffer of
    the past, and cannot revisit.

So this walks the held-out session forward one 200 ms slot at a time, per
series, maintaining all state incrementally in Python:

  * a 60-second deque of near-touch dollars for the collapse baseline
  * a 20-second ring buffer of book features for the model input
  * a refractory timer

Per-row feature decoding is done up front, which is legitimate: decoding a book
snapshot into spread/depth/imbalance uses only that row and has no time
dimension. Everything with a time dimension -- the baseline, the collapse rule,
the buffers, the alert -- is computed sequentially inside the loop.

Ground truth (did a durable move follow?) is computed afterwards and used ONLY
for scoring, never fed back.

    python live_replay.py --threshold-pct 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

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
RES = os.path.join(ROOT, "results", "makinen", "live_replay")
GRID_MS = 200


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def train_model(min_move, pre_s, stride):
    """Train on the earlier sessions exactly as the batch study did."""
    import lightgbm as lgb
    E = pd.read_parquet(os.path.join(CACHE, "collapse_events.parquet")).reset_index(drop=True)
    E["y"] = (E.durable_move >= min_move).astype(int)
    X, keep = build_sequences(E, pre_s, stride)
    E = E.loc[keep].reset_index(drop=True)
    y = E.y.to_numpy(int)
    sess = sorted(E.session.unique())
    tr = E.session.isin(sess[:-1]).to_numpy()      # everything before the replay session
    M = np.concatenate([X[:, -1, :], X[:, -1, :] - X[:, -5, :],
                        X[:, -1, :] - X[:, 0, :], X.min(1), X.max(1), X.std(1)], axis=1)
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                           min_child_samples=100, subsample=.8, subsample_freq=1,
                           colsample_bytree=.8, random_state=0, n_jobs=8,
                           verbose=-1, class_weight="balanced").fit(M[tr], y[tr])
    log("trained on %s sessions, %s events" % (len(sess) - 1, "{:,}".format(int(tr.sum()))))
    return m, sess[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-move", type=float, default=0.02)
    ap.add_argument("--pre-s", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--baseline-s", type=float, default=60.0)
    ap.add_argument("--drop-frac", type=float, default=0.25)
    ap.add_argument("--min-base", type=float, default=200.0)
    ap.add_argument("--refractory-s", type=float, default=30.0)
    ap.add_argument("--horizon-s", type=float, default=10.0)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--threshold-pct", type=float, default=2.0,
                    help="alert on the top X%% of scores, calibrated on the "
                         "TRAINING sessions only")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    model, replay_sess = train_model(a.min_move, a.pre_s, a.stride)
    log("replaying %s tick by tick" % replay_sess)

    # threshold frozen from training-session scores, never from the replay
    E = pd.read_parquet(os.path.join(CACHE, "collapse_events.parquet"))
    Etr = E[E.session != replay_sess]
    Xtr, keep = build_sequences(Etr.reset_index(drop=True), a.pre_s, a.stride)
    Mtr = np.concatenate([Xtr[:, -1, :], Xtr[:, -1, :] - Xtr[:, -5, :],
                          Xtr[:, -1, :] - Xtr[:, 0, :], Xtr.min(1), Xtr.max(1),
                          Xtr.std(1)], axis=1)
    thr = float(np.quantile(model.predict_proba(Mtr)[:, 1], 1 - a.threshold_pct / 100.0))
    log("alert threshold %.4f (top %.1f%% of TRAIN scores)" % (thr, a.threshold_pct))
    del Xtr, Mtr

    fp = os.path.join(JD, "feat_%s_trimmed.parquet" % replay_sess)
    tp = os.path.join(JD, "lob_%s_trimmed.npy" % replay_sess)
    F = pq.read_table(fp, columns=["ts", "sid", "series", "mid", "spread_ticks",
                                   "valid"]).to_pandas()
    F["row"] = np.arange(len(F), dtype=np.int64)
    F = F[F.valid.to_numpy(bool)].reset_index(drop=True)
    T = np.load(tp, mmap_mode="r")
    log("decoding %s book snapshots (per-row, no time dimension)"
        % "{:,}".format(len(F)))
    d = FE.decode(np.asarray(T[F.row.to_numpy()]), F.mid.to_numpy(np.float64))
    S = pd.DataFrame(FE.state_features(d, spread_ticks=F.spread_ticks.to_numpy()))
    FEAT = S[TRACK].to_numpy(np.float32)
    NEAR = (np.expm1(S.log_bid_usd_within_2t.to_numpy())
            + np.expm1(S.log_ask_usd_within_2t.to_numpy()))
    MID = F.mid.to_numpy(np.float64)
    TS = F.ts.to_numpy(np.int64)
    SID = F.sid.to_numpy()

    base_n = int(a.baseline_s * 1000 / GRID_MS)
    buf_n = int(a.pre_s * 1000 / GRID_MS)
    refr_n = int(a.refractory_s * 1000 / GRID_MS)
    step = a.stride

    alerts, events = [], []
    t0 = time.time()
    for sid in np.unique(SID):
        idx = np.where(SID == sid)[0]
        base = deque(maxlen=base_n)
        buf = deque(maxlen=buf_n)
        last_fire = -10 ** 9
        for pos, i in enumerate(idx):
            buf.append(FEAT[i])
            near = NEAR[i]
            # ---- decide using ONLY what is in the buffers ----
            fired = False
            if len(base) >= base_n // 2 and len(buf) == buf_n:
                b = float(np.median(base))
                if b >= a.min_base and near < a.drop_frac * b \
                        and (pos - last_fire) >= refr_n:
                    last_fire = pos
                    fired = True
            base.append(near)          # appended AFTER the test: baseline is trailing
            if not fired:
                continue
            W = np.asarray(buf, np.float32)[::step]
            if W.shape[0] < buf_n // step:
                continue
            v = np.concatenate([W[-1], W[-1] - W[-5], W[-1] - W[0],
                                W.min(0), W.max(0), W.std(0)])[None, :]
            score = float(model.predict_proba(v)[0, 1])
            events.append((sid, i, pos, score))
            if score >= thr:
                alerts.append((sid, i, pos, score))
    log("replay done in %.1fs -- %s collapse events, %s alerts"
        % (time.time() - t0, "{:,}".format(len(events)), "{:,}".format(len(alerts))))

    # ---- ground truth, computed afterwards, used only for scoring ----
    hor = int(a.horizon_s * 1000 / GRID_MS)
    hold = int(a.hold_s * 1000 / GRID_MS)
    def realised(i, sid):
        idx = np.where(SID == sid)[0]
        p = np.searchsorted(idx, i)
        seg = idx[p + 1:p + 1 + hor + hold + 1]
        if len(seg) < hor + hold:
            return np.nan
        m0 = MID[i]
        fwd = MID[seg[:hor]]
        k = int(np.argmax(np.abs(fwd - m0)))
        later = MID[seg[k:k + hold + 1]]
        return float(np.min(np.abs(later - m0))) if len(later) else 0.0

    ev = pd.DataFrame(events, columns=["sid", "i", "pos", "score"])
    ev["durable"] = [realised(int(r.i), int(r.sid)) for r in ev.itertuples()]
    ev = ev[np.isfinite(ev.durable)]
    ev["real"] = (ev.durable >= a.min_move).astype(int)
    ev["alert"] = (ev.score >= thr).astype(int)
    ev.to_csv(os.path.join(RES, "replay_events.csv"), index=False)

    hours = (TS.max() - TS.min()) / 3.6e6
    n_series = len(np.unique(SID))
    tp_ = int(((ev.alert == 1) & (ev.real == 1)).sum())
    fp_ = int(((ev.alert == 1) & (ev.real == 0)).sum())
    fn_ = int(((ev.alert == 0) & (ev.real == 1)).sum())
    print("\n" + "=" * 70)
    print("LIVE REPLAY -- %s, %d contracts, %.1f wall-clock hours"
          % (replay_sess, n_series, hours))
    print("=" * 70)
    print("  collapse events detected live : {:,}".format(len(ev)))
    print("  of which genuinely real       : {:,} ({:.1%})".format(int(ev.real.sum()),
                                                                   ev.real.mean()))
    print("  alerts raised                 : {:,}".format(int(ev.alert.sum())))
    print("\n  precision (alert was real)    : %.4f" % (tp_ / max(tp_ + fp_, 1)))
    print("  recall (real ones alerted)    : %.4f" % (tp_ / max(tp_ + fn_, 1)))
    print("  alerts per contract-hour      : %.2f" % (len(ev[ev.alert == 1]) /
                                                      max(hours * n_series, 1e-9)))
    print("  false alerts per contract-hour: %.2f" % (fp_ / max(hours * n_series, 1e-9)))
    print("  real jumps MISSED             : {:,}".format(fn_))
    from sklearn.metrics import roc_auc_score, average_precision_score
    if ev.real.nunique() > 1:
        print("\n  ROC-AUC (streamed scores)     : %.4f" % roc_auc_score(ev.real, ev.score))
        print("  PR-AUC                        : %.4f (base %.4f)"
              % (average_precision_score(ev.real, ev.score), ev.real.mean()))
    print("\nwrote", os.path.join(RES, "replay_events.csv"))


if __name__ == "__main__":
    sys.exit(main())
