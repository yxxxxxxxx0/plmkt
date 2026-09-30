import os
import numpy as np, pandas as pd
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
d=pd.read_pickle(SP+r'\xmkt.pkl'); rng=np.random.default_rng(0)
def co(d,W,shift=False):
    out=np.zeros(len(d),int)
    for slug,g in d.groupby('slug'):
        T={s:np.sort(x.ts.to_numpy()) for s,x in g.groupby('series')}
        if shift: T={s:np.sort(t+rng.choice([-1,1])*rng.integers(180e3,600e3)) for s,t in T.items()}
        for s,x in g.groupby('series'):
            c=np.zeros(len(x),int)
            for s2,t2 in T.items():
                if s2==s: continue
                i=np.searchsorted(t2,x.ts.to_numpy()-W); c+=(i<len(t2))&(t2[np.minimum(i,len(t2)-1)]<=x.ts.to_numpy()+W)
            out[d.index.get_indexer(x.index)]=c
    return out
print('candidates',len(d),'games',d.slug.nunique(),'markets',d.series.nunique())
for W in (5000,10000):
    d['co']=co(d,W); d['co0']=co(d,W,True)
    for lab in ('real30','real60'):
        x=d[d[lab].notna()]; base=x[lab].mean()
        print(f'\n±{W/1e3:.0f}s, truth={lab}: labelled {len(x)} ({len(x)/len(d):.0%}), base real rate {base:.1%}')
        t=x.groupby(x.co.clip(upper=6))[lab].agg(['size','mean']); t['null_share']=x.co0.clip(upper=6).value_counts(normalize=True).sort_index()
        t['share']=t['size']/len(x); print(t.round(3).to_string())
        c=x.co>=2
        print(f'  co>=2: P(real)={x[c][lab].mean():.1%} vs co<2: {x[~c][lab].mean():.1%} | recall of real {c[x[lab]==1].mean():.0%} | fakes passing {c[x[lab]==0].mean():.0%} | share co>=2 by chance {(x.co0>=2).mean():.0%}')
        pl=x.pulled
        print(f'  among pull-spike candidates: co>=2 P(real) {x[c&pl][lab].mean():.1%} vs {x[~c&pl][lab].mean():.1%} ; among quoted moves {x[c&~pl][lab].mean():.1%} vs {x[~c&~pl][lab].mean():.1%}')
d.to_pickle(SP+r'\xmkt_co.pkl')
