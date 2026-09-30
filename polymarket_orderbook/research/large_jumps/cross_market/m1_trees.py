"""Method 1: gradient-boosted trees on hand-built own-market, cross-market and game-state features.

Three nested feature sets, same samples, same split, so the value of the other markets is measured
directly: OWN, OWN+CROSS, OWN+CROSS+GAME. Trained on train sessions with negatives subsampled;
then every test game is scored sample by sample (kept in memory only as scores).

    python research/large_jumps/cross_market/m1_trees.py [--smoke]
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import list_panels, load_panel, samples, is_test, report, save_report, PanelCache, smoke_flag, OUT, NEG_SCALE
from features import features, market_arrays

smoke = smoke_flag(sys.argv)
NEG = 0.05 * NEG_SCALE
rng = np.random.default_rng(0)
panels = list_panels(smoke)

tr_parts = []
for sess, slug in panels:
    if is_test(sess): continue
    pan = load_panel(sess, slug)
    smp = samples(pan, neg_rate=NEG, rng=rng)
    if not len(smp): continue
    X = features(pan, smp, market_arrays(pan))
    X['slug'], X['y_large'] = slug, smp[:, 2]
    tr_parts.append(X)
    print(sess, slug, len(X), flush=True)
TR = pd.concat(tr_parts, ignore_index=True)
OWN = [c for c in TR.columns if c.startswith('own_')]
CROSS = [c for c in TR.columns if c.startswith('x_')]
GAME = [c for c in TR.columns if c.startswith('g_')]
SETS = {'m1_trees_own': OWN, 'm1_trees_own+cross': OWN + CROSS, 'm1_trees_own+cross+game': OWN + CROSS + GAME}
print(f'train {len(TR)} samples ({int(TR.y_large.sum())} large-positive)')

games = TR.slug.unique(); val = set(rng.choice(games, max(1, len(games) // 10), replace=False))
isv = TR.slug.isin(val)
models = {}
for name, cols in SETS.items():
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(TR.loc[~isv, cols], TR.y_large[~isv], eval_set=[(TR.loc[isv, cols], TR.y_large[isv])],
          eval_metric='average_precision', callbacks=[lgb.early_stopping(200, verbose=False)])
    models[name] = m
    imp = pd.Series(m.booster_.feature_importance('gain'), index=cols)
    (imp / imp.sum()).sort_values(ascending=False).to_csv(os.path.join(OUT, f'{name}_importance.csv'))
    print(name, 'top features:', (imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_dict())
del TR

scores = {n: [] for n in SETS}
for sess, slug in panels:
    if not is_test(sess): continue
    pan = load_panel(sess, slug); smp = samples(pan)
    if not len(smp): continue
    X = features(pan, smp, market_arrays(pan))
    ids = pd.DataFrame(dict(sess=sess, slug=slug, k=smp[:, 0].astype(np.int16), t=smp[:, 1].astype(np.int32),
                            y_large=smp[:, 2].astype(np.int8), y_mark=smp[:, 3].astype(np.int8)))
    for name, cols in SETS.items():
        scores[name].append(ids.assign(p=models[name].predict_proba(X[cols])[:, 1].astype(np.float32)))
    print(' scored', sess, slug, len(X), flush=True)

cache = PanelCache()
for name in SETS:
    R = pd.concat(scores[name], ignore_index=True)
    print(f'\n{name}: test AUC {roc_auc_score(R.y_large, R.p):.3f}')
    save_report(name, report(name, R, panels=cache))
