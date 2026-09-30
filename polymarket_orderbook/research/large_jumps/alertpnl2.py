import os
# Trade every depth-drain alert: buy at the ask, sell at the bid once the book is quoted again H s later.
import numpy as np, pandas as pd, pyarrow.parquet as pq
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
LAG=20; STILL=25; DEB=50; THR=[0.5,0.25,0.10]; HS=[50,150]; FEE=0.05; STAKE=10
fee=lambda p: FEE*p*(1-p)/0.01
m=pd.read_csv('results/makinen/single_jumps/marks.csv'); m['mkt']=m.series.str.rsplit('|',n=1).str[0]
rows=[]
for sess,g in m.groupby('session'):
    f=pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet',columns=['mid','spread_ticks','ts','series']).to_pandas()
    f['mkt']=f.series.str.rsplit('|',n=1).str[0]; f=f[f.series.isin(set(g.series))]
    lob=np.load(f'data/jump/lob_{sess}_trimmed.npy',mmap_mode='r')
    for mk,s in f.groupby('series',sort=False):
        s=s.sort_values('ts'); ts=s.ts.to_numpy(); n=len(ts)
        L=np.asarray(lob[s.index.to_numpy()],np.float64)
        spr=s.spread_ticks.to_numpy(); mid=s.mid.to_numpy(); norm=np.median(spr); cut=norm+max(2,norm)
        bid=np.round(mid-spr*.005,4); ask=np.round(mid+spr*.005,4)
        bu=np.expm1(L[:,1,0]); au=np.expm1(L[:,3,0])
        dep=(np.expm1(L[:,1])*(np.abs(L[:,0])<=5)).sum(1)+(np.expm1(L[:,3])*(np.abs(L[:,2])<=5)).sum(1)
        contig=pd.Series(np.r_[1,np.diff(ts)==200]).rolling(STILL+LAG).min().to_numpy()==1
        rb=pd.Series(bid).rolling(STILL); ra=pd.Series(ask).rolling(STILL)
        still=(((rb.max()-rb.min())<=.011)&((ra.max()-ra.min())<=.011)&(pd.Series(spr).rolling(STILL).max()<=cut)).to_numpy()&contig
        r=np.full(n,np.nan); r[LAG:]=dep[LAG:]/np.maximum(dep[:-LAG],1)
        quoted=np.flatnonzero(spr<=cut)
        for t in THR:
            last=-10**9
            for i in np.flatnonzero(still&(r<t)):
                if i-last<DEB: continue
                last=i
                for H in HS:
                    j=quoted[np.searchsorted(quoted,i+H)] if np.searchsorted(quoted,i+H)<len(quoted) else -1
                    if j<0 or j-i>H+300 or ts[j]-ts[i]!=(j-i)*200: continue
                    # token A: buy ask_i, sell bid_j.  token B (mirror): buy 1-bid_i, sell 1-ask_j
                    pa=(bid[j]-ask[i])/0.01-fee(ask[i])-fee(bid[j])
                    pb=(ask[i]-ask[j]-(bid[i]-bid[j])*0+ (bid[i]-ask[j]) - (ask[i]-ask[j]))/0.01  # = (bid_i-ask_j)/tick
                    pb=(bid[i]-ask[j])/0.01-fee(1-bid[i])-fee(1-ask[j])
                    okA=au[i]>=STAKE; okB=bu[i]/max(bid[i],1e-3)*(1-bid[i])>=STAKE
                    rows.append(dict(sess=sess,mkt=mk,thr=t,H=H*0.2,pa=pa if okA else np.nan,pb=pb if okB else np.nan,
                                     dA=ask[i],sprd=spr[i]))
    print(sess,len(rows),flush=True)
pd.DataFrame(rows).to_pickle(SP+r'\alertpnl2.pkl')
