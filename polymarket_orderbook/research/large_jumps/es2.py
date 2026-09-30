import os
import numpy as np, pandas as pd
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
C=np.load(SP+r'\es_curves.npy'); M=pd.read_pickle(SP+r'\es_meta.pkl')
FEATS=['spr_rel','pulled','log_bid_top','log_ask_top','log_bid5','log_ask5','imb5','upd','absdmid']
t=np.arange(-120,60)
print(M.kind.value_counts().to_dict()); print('entry->start lead s', M.groupby('kind').lead_s.describe()[['50%','75%']])
rng=np.random.default_rng(1); ug=M.slug.unique(); G={g:np.flatnonzero(M.slug.to_numpy()==g) for g in ug}
kinds=M.kind.to_numpy()
def means(idx):
    return {k:np.nanmean(C[idx[kinds[idx]==k]],0) for k in ('large','rest','base')}
full=means(np.arange(len(M)))
B=[]
for _ in range(200):
    idx=np.concatenate([G[g] for g in rng.choice(ug,len(ug))]); mm=means(idx)
    B.append(np.stack([mm['large']-mm['base'],mm['rest']-mm['base'],mm['large']-mm['rest']]))
B=np.array(B); lo,hi=np.percentile(B,[2.5,97.5],0)
np.savez(SP+r'\es_summary.npz',t=t,large=full['large'],rest=full['rest'],base=full['base'],lo=lo,hi=hi)
def onset(c,f):   # earliest second before 0 from which the diff's CI excludes 0 continuously up to t=-1
    sig=(lo[c,:,f]>0)|(hi[c,:,f]<0); k=np.flatnonzero(t==-1)[0]
    if not sig[k]: return None
    j=k
    while j>0 and sig[j-1]: j-=1
    return int(t[j])
lags=[-120,-60,-30,-15,-10,-5,-3,-2,-1,0,2,5,10,30]
for f,nm in enumerate(FEATS):
    print(f'\n{nm}: onset vs base  large {onset(0,f)}  rest {onset(1,f)}  | large vs rest {onset(2,f)}')
    print(pd.DataFrame({k:full[k][[np.flatnonzero(t==l)[0] for l in lags],f] for k in ('large','rest','base')},index=lags).round(3).T.to_string())
