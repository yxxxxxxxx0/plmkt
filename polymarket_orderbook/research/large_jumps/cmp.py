import os
import numpy as np, pandas as pd
from scipy.stats import rankdata
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
F=pd.read_pickle(SP+r'\micro.pkl')
F['disturbed']=(F.jump_regime=='disturbed').astype(float)
feats=[c for c in F.columns if c.startswith(('pre_','dur_','post_')) and c not in ('pre_kind','post_kind')]+['jump_s','disturbed','entry_ask_usd','exit_bid_usd']
def auc(x,y):
    k=np.isfinite(x); x,y=x[k],y[k]; r=rankdata(x); n1=y.sum(); n0=len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
y=F.large.to_numpy(); gs=F.slug.to_numpy(); ug=np.unique(gs); rng=np.random.default_rng(0)
G={g:np.flatnonzero(gs==g) for g in ug}
boots=[np.concatenate([G[g] for g in rng.choice(ug,len(ug))]) for _ in range(200)]
out=[]
for c in feats:
    x=F[c].to_numpy(float); a=auc(x,y); b=[auc(x[i],y[i]) for i in boots]
    out.append(dict(feature=c,med_large=np.nanmedian(x[y]),med_rest=np.nanmedian(x[~y]),auc=a,lo=np.percentile(b,2.5),hi=np.percentile(b,97.5)))
o=pd.DataFrame(out); o['sep']=(o.auc-.5).abs(); print(o.sort_values('sep',ascending=False).round(3).to_string(index=False))
print(pd.crosstab(F.market_type,F.large,normalize='index').round(3))
print(pd.crosstab(F.jump_regime,F.large,normalize='columns').round(3))
print(pd.crosstab(F.pre_kind,F.large,normalize='columns').round(3))
