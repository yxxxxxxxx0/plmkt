"""Method 5: what 'modes' does a game's whole book move in, and do any of them precede large jumps?

A game-state vector every second: for each of up to 10 markets (fixed order: moneyline, spreads,
totals) its spread vs normal, pulled flag, 4 s depth drain, |5 s move|, oriented 5 s move, log depth,
and a presence flag. On train games: PCA (10 components), k-means (8 clusters) and a small
autoencoder. Then, on test games:
  * per cluster: share of time and how often a large jump starts somewhere in the game 2-8 s later;
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
                    K_MAX, HIST, H_LO, H_HI, EVERY, OUT, logit)
from features import features, market_arrays

smoke = smoke_flag(sys.argv)
rng = np.random.default_rng(0); torch.manual_seed(0)
PER = 7


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
    st = np.sort(np.concatenate([s for s in pan['starts_large']] + [np.array([], np.int64)]))
    if not len(st): return np.zeros(len(ts), bool)
    i = np.searchsorted(st, ts + H_LO + 1)
    return (i < len(st)) & (st[np.minimum(i, len(st) - 1)] <= ts + H_HI)


# ---- game-level vectors ----
trV, teV, teY, teKey = [], [], [], []
per_market = {'tr': [], 'te': []}
for sess, slug in list_panels(smoke):
    pan = load_panel(sess, slug); A = market_arrays(pan)
    S = pan['P'].shape[1]; ts = np.arange(HIST, S - H_HI - 1, EVERY)
    V = game_vectors(pan, A, ts)
    test = is_test(sess)
    if test:
        teV.append(V); teY.append(game_label(pan, ts)); teKey.append(pd.DataFrame(dict(sess=sess, slug=slug, t=ts)))
    else:
        trV.append(V[::5])
    smp = samples(pan, neg_rate=1.0 if test else 0.05, rng=rng)
    if len(smp):
        X = features(pan, smp, A, with_cross=False)
        gi = np.searchsorted(ts, smp[:, 1])
        X = pd.concat([X, pd.DataFrame(V[np.clip(gi, 0, len(V) - 1)], columns=[f'v{i}' for i in range(V.shape[1])])], axis=1)
        X['sess'], X['slug'], X['k'], X['t'], X['y_large'], X['y_mark'] = sess, slug, smp[:, 0], smp[:, 1], smp[:, 2], smp[:, 3]
        per_market['te' if test else 'tr'].append(X)
    print(sess, slug, flush=True)
trV = np.concatenate(trV); teV = np.concatenate(teV); teY = np.concatenate(teY); teKey = pd.concat(teKey, ignore_index=True)

sc = StandardScaler().fit(trV)
pca = PCA(10, random_state=0).fit(sc.transform(trV))
km = KMeans(8, n_init=5, random_state=0).fit(pca.transform(sc.transform(trV)))
print('PCA explained variance:', pca.explained_variance_ratio_.round(3).tolist())


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 8))
        self.d = nn.Sequential(nn.ReLU(), nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, d))

    def forward(self, x): return self.d(self.e(x))


Xtr = torch.from_numpy(sc.transform(trV).astype(np.float32))
ae = AE(Xtr.shape[1]); opt = torch.optim.Adam(ae.parameters(), 1e-3)
for ep in range(2 if smoke else 15):
    for i in torch.randperm(len(Xtr)).split(1024):
        opt.zero_grad(); loss = ((ae(Xtr[i]) - Xtr[i]) ** 2).mean(); loss.backward(); opt.step()
with torch.no_grad():
    Xte = torch.from_numpy(sc.transform(teV).astype(np.float32)); err = ((ae(Xte) - Xte) ** 2).mean(1).numpy()

Z = pca.transform(sc.transform(teV)); cl = km.predict(Z)
base = teY.mean()
print(f'\ngame-level: large jump starts somewhere in the game 2-8 s later on {base:.2%} of test seconds')
tab = pd.DataFrame(dict(cluster=cl, y=teY)).groupby('cluster').y.agg(time_share='size', p_large='mean')
tab['time_share'] /= len(cl); tab['lift'] = tab.p_large / base
names = ['spr_rel', 'pulled', 'drain4', 'absmove5', 'dirmove5', 'logdep', 'present']
cent = pd.DataFrame(teV).groupby(cl).mean()
for i, nm in enumerate(names[:5]):
    tab[f'mean_{nm}'] = cent[[s * PER + i for s in range(K_MAX)]].mean(1)
print(tab.round(3).to_string())
tab.to_csv(os.path.join(OUT, 'm5_clusters.csv'))
print(f'autoencoder error as a score: AUC {roc_auc_score(teY, err):.3f}; PCA comp 1: AUC {roc_auc_score(teY, Z[:, 0]):.3f}')

# ---- does it add anything tradable? per-market trees with and without the unsupervised features ----
TR = pd.concat(per_market['tr'], ignore_index=True); TE = pd.concat(per_market['te'], ignore_index=True)
for D in (TR, TE):
    V = D[[c for c in D.columns if c.startswith('v') and c[1:].isdigit()]].to_numpy(np.float32)
    Zd = pca.transform(sc.transform(V))
    for i in range(Zd.shape[1]): D[f'u_pc{i}'] = Zd[:, i]
    D['u_cluster'] = km.predict(Zd)
    with torch.no_grad():
        xv = torch.from_numpy(sc.transform(V).astype(np.float32)); D['u_ae_err'] = ((ae(xv) - xv) ** 2).mean(1).numpy()
ID = ['sess', 'slug', 'k', 't', 'y_large', 'y_mark']
BASE = [c for c in TR.columns if c.startswith(('own_', 'g_'))]
UNS = [c for c in TR.columns if c.startswith('u_')]
games = TR.slug.unique(); val = set(rng.choice(games, max(1, len(games) // 10), replace=False)); isv = TR.slug.isin(val)
cache = PanelCache()
for name, cols in (('m5_trees_own+game', BASE), ('m5_trees_own+game+unsupervised', BASE + UNS)):
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(TR.loc[~isv, cols], TR.y_large[~isv], eval_set=[(TR.loc[isv, cols], TR.y_large[isv])],
          eval_metric='average_precision', callbacks=[lgb.early_stopping(200, verbose=False)])
    p = m.predict_proba(TE[cols])[:, 1]
    print(f'\n{name}: test AUC {roc_auc_score(TE.y_large, p):.3f}')
    t = report(name, TE[ID].assign(p=p), panels=cache); save_report(name, t)
