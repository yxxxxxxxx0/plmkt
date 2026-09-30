"""Training and scoring loop shared by the multi-market CNN (m2) and the attention model (m3).

Train: every train-session game panel is held in memory; negatives are subsampled (NEG) and all
positives kept; 10% of train games are held out for early stopping on average precision.
Test: every sample (default every 200 ms) of every test game is scored, game by game -- needed so
the debounced trades are the ones a live system would have taken.
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from common import (list_panels, load_panel, samples, is_test, seq_batch, game_state, yes_bats, GAME_FEATS, NEG_SCALE,
                    GRID, report, save_report, PanelCache, K_MAX, HIST, SEQ_CH)

torch.set_num_threads(max(1, os.cpu_count() - 2))
NEG = 0.01 * NEG_SCALE


def game_vec(pan, ks, ts):
    G = game_state(pan['meta'][0]['series'].split('|')[0], (pan['s0'] + ts) * GRID)
    yb = np.array([yes_bats(pan['meta'][k], G[i:i + 1, GAME_FEATS.index('g_is_top')])[0] for i, k in enumerate(ks)])
    return np.nan_to_num(np.c_[G, yb, np.isnan(G).any(1)]).astype(np.float32)


def load_train(smoke, rng):
    games = []
    for sess, slug in list_panels(smoke):
        if is_test(sess): continue
        pan = load_panel(sess, slug)
        smp = samples(pan, neg_rate=NEG, rng=rng)
        if len(smp): games.append((pan, smp, game_vec(pan, smp[:, 0], smp[:, 1])))
    return games


def batches(games, idx, bs, rng=None):
    """idx: (n, 2) array of (game index, row) pairs."""
    if rng is not None: idx = idx[rng.permutation(len(idx))]
    for s in range(0, len(idx), bs):
        chunk = idx[s:s + bs]
        Xs, Rs, Gs, Ys = [], [], [], []
        for gi in np.unique(chunk[:, 0]):
            rows = chunk[chunk[:, 0] == gi, 1]; pan, smp, gv = games[gi]
            X, R = seq_batch(pan, smp[rows, 0], smp[rows, 1])
            Xs.append(X); Rs.append(R); Gs.append(gv[rows]); Ys.append(smp[rows, 2])
        yield (torch.from_numpy(np.concatenate(Xs)), torch.from_numpy(np.concatenate(Rs)),
               torch.from_numpy(np.concatenate(Gs)), torch.from_numpy(np.concatenate(Ys).astype(np.float32)))


def norm_stats(games, rng, n=20000):
    idx = np.array([(g, r) for g in range(len(games)) for r in range(len(games[g][1]))])
    idx = idx[rng.choice(len(idx), min(n, len(idx)), replace=False)]
    X = np.concatenate([b[0].numpy() for b in batches(games, idx, 2048)])
    mask = X[..., SEQ_CH - 2] > 0                                   # 'present' channel
    mu = np.array([X[..., c][mask].mean() for c in range(SEQ_CH)], np.float32)
    sd = np.array([X[..., c][mask].std() + 1e-3 for c in range(SEQ_CH)], np.float32)
    mu[SEQ_CH - 2], sd[SEQ_CH - 2] = 0, 1                           # keep the mask as 0/1
    return torch.from_numpy(mu), torch.from_numpy(sd)


def fit(model, games, smoke, rng, epochs=8, bs=256, lr=1e-3):
    idx = np.array([(g, r) for g in range(len(games)) for r in range(len(games[g][1]))])
    gid = np.array([games[g][0]['meta'][0]['series'].split('|')[0] for g in idx[:, 0]])
    ug = np.unique(gid); val_g = set(rng.choice(ug, max(1, len(ug) // 10), replace=False))
    va = np.isin(gid, list(val_g)); tr_idx, va_idx = idx[~va], idx[va]
    y = np.array([games[g][1][r, 2] for g, r in idx])
    pos = max(y[~va].mean(), 1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(50.0, float(np.sqrt((1 - pos) / pos)))))
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    best, state, bad = -1, None, 0
    for ep in range(1 if smoke else epochs):
        t0 = time.time(); model.train()
        for X, R, G, Y in batches(games, tr_idx, bs, rng):
            opt.zero_grad(); loss = lossf(model(X, R, G), Y); loss.backward(); opt.step()
        p, yv = predict_idx(model, games, va_idx)
        ap = average_precision_score(yv, p) if yv.sum() else 0.0
        print(f'  epoch {ep}: val AP {ap:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if ap > best: best, state, bad = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2: break
    model.load_state_dict(state)
    return model


def predict_idx(model, games, idx):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for X, R, G, Y in batches(games, idx, 2048):
            ps.append(torch.sigmoid(model(X, R, G)).numpy()); ys.append(Y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def score_test(model, smoke):
    rows = []
    for sess, slug in list_panels(smoke):
        if not is_test(sess): continue
        pan = load_panel(sess, slug); smp = samples(pan)
        if not len(smp): continue
        g = [(pan, smp, game_vec(pan, smp[:, 0], smp[:, 1]))]
        p, _ = predict_idx(model, g, np.c_[np.zeros(len(smp), int), np.arange(len(smp))])
        rows.append(pd.DataFrame(dict(sess=sess, slug=slug, k=smp[:, 0], t=smp[:, 1], p=p, y_large=smp[:, 2], y_mark=smp[:, 3])))
        print(' scored', sess, slug, len(smp), flush=True)
    return pd.concat(rows, ignore_index=True)


def run(name, make_model, smoke):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    games = load_train(smoke, rng)
    print(f'{name}: {len(games)} train games, {sum(len(g[1]) for g in games)} samples, '
          f'{sum(int(g[1][:, 2].sum()) for g in games)} large-positive', flush=True)
    mu, sd = norm_stats(games, rng)
    model = make_model(mu, sd, games[0][2].shape[1])
    model = fit(model, games, smoke, rng)
    del games
    R = score_test(model, smoke)
    print(f'\n{name}: test AUC {roc_auc_score(R.y_large, R.p):.3f}')
    t = report(name, R, panels=PanelCache()); save_report(name, t)
    return t
