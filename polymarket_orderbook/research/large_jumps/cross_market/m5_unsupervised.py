"""Method 5: what 'modes' does a game's whole book move in, and do any of them precede large jumps?

A game-state vector: for each of up to 10 markets (fixed order: moneyline, spreads, totals) its
spread vs normal, pulled flag, 4 s depth drain, |5 s move|, oriented 5 s move, log depth, and a
presence flag. On train games: PCA (10 components), k-means (8 clusters) and a small autoencoder.
Then, on test games:
  * per cluster (one vector per second): share of time and how often a large jump starts somewhere
    in the game within the label window;
  * the autoencoder's reconstruction error as an "unusual book" score;
  * a per-market tree with own + game-state features, with and without these unsupervised features,
    scored with the standard trade table -- the test of whether they add anything tradable.

    python research/large_jumps/cross_market/m5_unsupervised.py [--smoke]
"""
import os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import (list_panels, load_panel, samples, is_test, report, save_report, PanelCache, smoke_flag,
                    K_MAX, HIST, H_LO, H_HI, OUT, NEG_SCALE)
from features import features, market_arrays

smoke = smoke_flag(sys.argv)
rng = np.random.default_rng(0); torch.manual_seed(0)
PER, GSTEP = 7, 5                      # 7 values per market; game-level vectors once per second
NAMES = ['spr_rel', 'pulled', 'drain4', 'absmove5', 'dirmove5', 'logdep', 'present']


def game_vectors(pan, A, ts):
    meta = pan['meta']; K = len(meta)
    order = sorted(range(K), key=lambda j: (meta[j]['role'], meta[j]['line'] or 0, meta[j]['side']))[:K_MAX]
    V = np.zeros((len(ts), K_MAX * PER), np.float32)
    for slot, j in enumerate(order):
        mid = A['mid'][j]
        v = np.c_[np.clip(A['spr_rel'][j, ts], 0, 20), A['pulled'][j, ts],
                  np.clip(A['dep'][j, ts] / np.maximum(A['dep'][j, ts - 20], 1), 0, 3),
                  np.abs(mid[ts] - mid[ts - 25]) / .01, (mid[ts] - mid[ts - 25]) / .01 * A['ori'][j],
                  np.log1p(A['dep'][j, ts]), A['present'][j, ts]]
        V[:, slot * PER:(slot + 1) * PER] = np.nan_to_num(v)
    return V


def game_label(pan, ts):
    st = np.sort(np.concatenate(list(pan['starts_large']) + [np.array([], np.int64)]))
    if not len(st): return np.zeros(len(ts), bool)
    i = np.searchsorted(st, ts + H_LO + 1)
    return (i < len(st)) & (st[np.minimum(i, len(st) - 1)] <= ts + H_HI)


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 8))
        self.d = nn.Sequential(nn.ReLU(), nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, d))

    def forward(self, x): return self.d(self.e(x))


def unsup(V):
    """PCA components, cluster and autoencoder error for game vectors V."""
    Xs = sc.transform(V); Z = pca.transform(Xs)
    with torch.no_grad():
        x = torch.from_numpy(Xs.astype(np.float32)); err = ((ae(x) - x) ** 2).mean(1).numpy()
    D = {f'u_pc{i}': Z[:, i] for i in range(Z.shape[1])}
    D['u_cluster'] = km.predict(Z); D['u_ae_err'] = err
    return pd.DataFrame(D)


panels = list_panels(smoke)

# ---- pass 1: train games -> game vectors (unsupervised fit) and per-market training rows ----
trV, tr_rows = [], []
for sess, slug in panels:
    if is_test(sess): continue
    pan = load_panel(sess, slug); A = market_arrays(pan)
    S = pan['P'].shape[1]
    trV.append(game_vectors(pan, A, np.arange(HIST, S - H_HI - 1, GSTEP * 5)))
    smp = samples(pan, neg_rate=0.05 * NEG_SCALE, rng=rng)
    if len(smp):
        X = features(pan, smp, A, with_cross=False)
        X['_V'] = list(game_vectors(pan, A, smp[:, 1]))
        X['slug'], X['y_large'] = slug, smp[:, 2]
        tr_rows.append(X)
    print(sess, slug, flush=True)
trV = np.concatenate(trV)
sc = StandardScaler().fit(trV)
pca = PCA(10, random_state=0).fit(sc.transform(trV))
km = KMeans(8, n_init=5, random_state=0).fit(pca.transform(sc.transform(trV)))
print('PCA explained variance:', pca.explained_variance_ratio_.round(3).tolist())
Xtr = torch.from_numpy(sc.transform(trV).astype(np.float32))
ae = AE(Xtr.shape[1]); opt = torch.optim.Adam(ae.parameters(), 1e-3)
for ep in range(2 if smoke else 15):
    for i in torch.randperm(len(Xtr)).split(1024):
        opt.zero_grad(); loss = ((ae(Xtr[i]) - Xtr[i]) ** 2).mean(); loss.backward(); opt.step()

TR = pd.concat(tr_rows, ignore_index=True)
TR = pd.concat([TR.drop(columns='_V'), unsup(np.stack(TR._V.to_numpy()))], axis=1)
BASE = [c for c in TR.columns if c.startswith(('own_', 'g_'))]
UNS = [c for c in TR.columns if c.startswith('u_')]
SETS = {'m5_trees_own+game': BASE, 'm5_trees_own+game+unsupervised': BASE + UNS}
games = TR.slug.unique(); val = set(rng.choice(games, max(1, len(games) // 10), replace=False)); isv = TR.slug.isin(val)
models = {}
for name, cols in SETS.items():
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(TR.loc[~isv, cols], TR.y_large[~isv], eval_set=[(TR.loc[isv, cols], TR.y_large[isv])],
          eval_metric='average_precision', callbacks=[lgb.early_stopping(200, verbose=False)])
    models[name] = m
del TR

# ---- pass 2: test games -> cluster table (1 s) and per-market scores (every sample) ----
g_cl, g_y, g_err, g_pc1, cent = [], [], [], [], []
scores = {n: [] for n in SETS}
for sess, slug in panels:
    if not is_test(sess): continue
    pan = load_panel(sess, slug); A = market_arrays(pan)
    S = pan['P'].shape[1]; ts = np.arange(HIST, S - H_HI - 1, GSTEP)
    V = game_vectors(pan, A, ts); U = unsup(V)
    g_cl.append(U.u_cluster.to_numpy()); g_err.append(U.u_ae_err.to_numpy()); g_pc1.append(U.u_pc0.to_numpy())
    g_y.append(game_label(pan, ts)); cent.append(pd.DataFrame(V).assign(cluster=U.u_cluster.to_numpy()))
    smp = samples(pan)
    if len(smp):
        X = pd.concat([features(pan, smp, A, with_cross=False), unsup(game_vectors(pan, A, smp[:, 1]))], axis=1)
        ids = pd.DataFrame(dict(sess=sess, slug=slug, k=smp[:, 0].astype(np.int16), t=smp[:, 1].astype(np.int32),
                                y_large=smp[:, 2].astype(np.int8), y_mark=smp[:, 3].astype(np.int8)))
        for name, cols in SETS.items():
            scores[name].append(ids.assign(p=models[name].predict_proba(X[cols])[:, 1].astype(np.float32)))
    print(' scored', sess, slug, flush=True)

cl, y = np.concatenate(g_cl), np.concatenate(g_y)
base = y.mean()
print(f'\ngame-level: a large jump starts somewhere in the game within the label window on {base:.2%} of test seconds')
tab = pd.DataFrame(dict(cluster=cl, y=y)).groupby('cluster').y.agg(time_share='size', p_large='mean')
tab['time_share'] /= len(cl); tab['lift'] = tab.p_large / base
C = pd.concat(cent).groupby('cluster').mean()
for i, nm in enumerate(NAMES[:5]):
    tab[f'mean_{nm}'] = C[[s * PER + i for s in range(K_MAX)]].mean(1)
print(tab.round(3).to_string()); tab.to_csv(os.path.join(OUT, 'm5_clusters.csv'))
print(f'autoencoder error as a score: AUC {roc_auc_score(y, np.concatenate(g_err)):.3f}; '
      f'PCA component 1: AUC {roc_auc_score(y, np.concatenate(g_pc1)):.3f}')

cache = PanelCache()
for name in SETS:
    R = pd.concat(scores[name], ignore_index=True)
    print(f'\n{name}: test AUC {roc_auc_score(R.y_large, R.p):.3f}')
    save_report(name, report(name, R, panels=cache))
