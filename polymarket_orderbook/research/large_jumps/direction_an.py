import os, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
R=pd.read_pickle(os.path.join(S,'direction.pkl')); R=R[R.is_top.notna()]
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995)))
R['dlo']=(lg(R.p0+R.d30)-lg(R.p0)).abs(); R['large']=R.dlo>=0.46
def ci(k,n):
    p=k/n; se=np.sqrt(p*(1-p)/n); return f'{p:.1%} [{p-1.96*se:.1%}, {p+1.96*se:.1%}] n={n}'
T=R[R.mtype!='total'].copy()
T['yes_bats']=np.where(T.yes_side=='away',T.is_top==1,T.is_top==0)
print('MONEYLINE + SPREAD real jumps, rule "up if YES team is batting"')
for nm,x in [('all real',T),('large (log-odds>=0.46)',T[T.large]),('moneyline',T[T.mtype=='moneyline']),('moneyline large',T[(T.mtype=='moneyline')&T.large]),('spread large',T[(T.mtype=='spread')&T.large])]:
    hit=(x.yes_bats==x.up); print(f'  {nm:24s} accuracy {ci(hit.sum(),len(x))} | up base rate {x.up.mean():.1%}')
print('  by session, large:', T[T.large].groupby('sess').apply(lambda x: round((x.yes_bats==x.up).mean(),2)).to_dict())
O=R[R.mtype=='total']
print('\nTOTALS (YES = Over) share of real jumps that go UP:', ci(O.up.sum(),len(O)), '| large:', ci(O[O.large].up.sum(),O.large.sum()))
F=['bid_ratio','ask_ratio','bid_top_ratio','ask_top_ratio','bid_half_s','ask_half_s','imb_end','imb_start']
R['asym']=np.log(R.ask_ratio.clip(1e-3))-np.log(R.bid_ratio.clip(1e-3)); R['asym_top']=np.log(R.ask_top_ratio.clip(1e-3))-np.log(R.bid_top_ratio.clip(1e-3))
R['half_diff']=R.ask_half_s-R.bid_half_s; F+= ['asym','asym_top','half_diff']
print('\nDRAIN features before the jump starts: AUC for up vs down (0.5 = no clue)')
for c in F:
    print(f'  {c:14s} all {roc_auc_score(R.up,R[c]):.3f}  large {roc_auc_score(R[R.large].up,R[R.large][c]):.3f}')
print('  simple rule "up if ask side drained more than bid side": accuracy', ci(((R.asym<0)==R.up).sum(),len(R)), '| large', ci(((R[R.large].asym<0)==R[R.large].up).sum(),R.large.sum()))
tr=R.sess<'books_2026-09-18'; te=~tr
R['yes_bats']=np.where(R.mtype=='total',np.nan,np.where(R.yes_side=='away',R.is_top==1,R.is_top==0).astype(float))
for nm,cols in [('drain only',F),('batting only',['yes_bats']),('drain + batting',F+['yes_bats'])]:
    X=R[cols].assign(mtype=R.mtype.astype('category'))
    mdl=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=200,verbose=-1).fit(X[tr],R.up[tr])
    p=mdl.predict_proba(X[te])[:,1]; y=R.up[te].to_numpy(); L=R.large[te].to_numpy()
    print(f'  model {nm:16s} test AUC {roc_auc_score(y,p):.3f} acc {((p>.5)==y).mean():.1%} | large: AUC {roc_auc_score(y[L],p[L]):.3f} acc {((p[L]>.5)==y[L]).mean():.1%}')
