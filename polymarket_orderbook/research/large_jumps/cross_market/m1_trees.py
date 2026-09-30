"""Method 1: gradient-boosted trees on hand-built own-market, cross-market and game-state features.

Three nested feature sets, same samples, same split, so the value of the other markets is measured
directly: OWN, OWN+CROSS, OWN+CROSS+GAME. Trained on train sessions with negatives subsampled;
scored on EVERY second of the test sessions.

    python research/large_jumps/cross_market/m1_trees.py [--smoke]
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import list_panels, load_panel, samples, is_test, report, save_report, PanelCache, smoke_flag, OUT
from features import features, market_arrays

smoke = smoke_flag(sys.argv)
NEG = 0.05
rng = np.random.default_rng(0)
tr_parts, te_parts = [], []
for sess, slug in list_panels(smoke):
    pan = load_panel(sess, slug); A = market_arrays(pan)
    test = is_test(sess)
    smp = samples(pan, neg_rate=1.0 if test else NEG, rng=rng)
    if not len(smp): continue
    X = features(pan, smp, A)
    X['sess'], X['slug'], X['k'], X['t'] = sess, slug, smp[:, 0], smp[:, 1]
    X['y_large'], X['y_mark'] = smp[:, 2], smp[:, 3]
    (te_parts if test else tr_parts).append(X)
    print(sess, slug, len(X), flush=True)
TR, TE = pd.concat(tr_parts, ignore_index=True), pd.concat(te_parts, ignore_index=True)
ID = ['sess', 'slug', 'k', 't', 'y_large', 'y_mark']
OWN = [c for c in TR.columns if c.startswith('own_')]
CROSS = [c for c in TR.columns if c.startswith('x_')]
GAME = [c for c in TR.columns if c.startswith('g_')]
print(f'train {len(TR)} samples ({int(TR.y_large.sum())} large-positive) | test {len(TE)} ({int(TE.y_large.sum())})')

games = TR.slug.unique(); val = set(rng.choice(games, max(1, len(games) // 10), replace=False))
isv = TR.slug.isin(val)
cache = PanelCache()
tables = []
for name, cols in (('m1_trees_own', OWN), ('m1_trees_own+cross', OWN + CROSS), ('m1_trees_own+cross+game', OWN + CROSS + GAME)):
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(TR.loc[~isv, cols], TR.y_large[~isv], eval_set=[(TR.loc[isv, cols], TR.y_large[isv])],
          eval_metric='average_precision', callbacks=[lgb.early_stopping(200, verbose=False)])
    p = m.predict_proba(TE[cols])[:, 1]
    print(f'\n{name}: test AUC {roc_auc_score(TE.y_large, p):.3f}')
    t = report(name, TE[ID].assign(p=p), panels=cache); tables.append(t); save_report(name, t)
    imp = pd.Series(m.booster_.feature_importance('gain'), index=cols)
    (imp / imp.sum()).sort_values(ascending=False).to_csv(os.path.join(OUT, f'{name}_importance.csv'))
    print('top features:', (imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_dict())
