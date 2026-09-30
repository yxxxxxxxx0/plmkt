import os
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
A=pd.read_pickle(SP+r'\alerts_ds.pkl'); F=[c for c in A.columns if c.startswith('f_')]
print('alerts',len(A),'| followed by up-mark',A.y_mark.mean().round(4),'large',A.y_large.mean().round(4),'real either way',A.y_real.mean().round(4))
P=A[A.pnl.notna()]
print('\nP&L per alert, buy YES ask, sell bid 30s later (ticks):')
for nm,msk in [('followed by up-mark',P.y_mark),('  of which large',P.y_large),('real jump but no up-mark (mostly down)',P.y_real&~P.y_mark),('nothing followed',~P.y_real&~P.y_mark)]:
    print(f'  {nm:40s} n {msk.sum():7d} share {msk.mean():6.1%} mean {P[msk].pnl.mean():+6.2f}t  contributes {P[msk].pnl.sum()/len(P):+.2f}t per alert')
print(f'  all: mean {P.pnl.mean():+.2f}t')
# spread+fee cost of a no-move trade
print('  entry spread median ticks', (P.f_spr_rel*P.f_norm_spr).median())
print('\nsingle-feature AUC for y_mark (0.5 = useless), and medians:')
out=[]
for c in F:
    x=A[c].to_numpy(float); ok=np.isfinite(x)
    a=roc_auc_score(A.y_mark[ok],x[ok]); out.append((c,a,np.nanmedian(x[A.y_mark]),np.nanmedian(x[~A.y_mark]),roc_auc_score(A.y_real[ok],x[ok])))
o=pd.DataFrame(out,columns=['feature','auc_upmark','med_conv','med_not','auc_real']); o['sep']=(o.auc_upmark-.5).abs()
print(o.sort_values('sep',ascending=False).round(3).to_string(index=False))
tr=A.sess<'books_2026-09-20'; te=~tr
print('\ntrain sessions',A[tr].sess.nunique(),'alerts',tr.sum(),'| test sessions',A[te].sess.nunique(),'alerts',te.sum())
for y in ('y_mark','y_large','y_real'):
    mdl=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
    Xtr=A.loc[tr,F].assign(mtype=A.loc[tr,'mtype'].astype('category')); Xte=A.loc[te,F].assign(mtype=pd.Categorical(A.loc[te,'mtype'],categories=Xtr.mtype.cat.categories))
    mdl.fit(Xtr,A.loc[tr,y]); p=mdl.predict_proba(Xte)[:,1]; yt=A.loc[te,y].to_numpy(); pn=A.loc[te,'pnl'].to_numpy()
    print(f'\n{y}: test AUC {roc_auc_score(yt,p):.3f}, base rate {yt.mean():.2%}')
    for q in (50,10,5,1,0.2):
        k=p>=np.percentile(p,100-q)
        print(f'  top {q:>4}% of alerts: n {k.sum():6d} precision {yt[k].mean():6.2%} ({yt[k].mean()/yt.mean():.1f}x)  P&L buying YES {np.nanmean(pn[k]):+.2f}t')
    if y=='y_mark':
        imp=pd.Series(mdl.booster_.feature_importance('gain'),index=Xtr.columns).sort_values(ascending=False); print('  top features by gain:',(imp/imp.sum()).head(8).round(3).to_dict())
