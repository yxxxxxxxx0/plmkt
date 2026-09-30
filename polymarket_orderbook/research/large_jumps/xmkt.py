import os
# Hypothesis: a jump confirmed by a similar jump in >=2 OTHER markets of the same game is real.
# Candidates are raw mid moves (fakes included); "real" = still displaced once the book is quoted 30s/60s later.
import numpy as np, pandas as pd, pyarrow.parquet as pq, glob
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
STILL=25; LOOK=50; J=0.03
rows=[]
for p in sorted(glob.glob('data/jump/feat_books_*_trimmed.parquet')):
    sess=p.split('feat_')[1].replace('_trimmed.parquet','')
    f=pq.read_table(p,columns=['mid','spread_ticks','ts','series']).to_pandas()
    for sr,s in f.groupby('series',sort=False):
        s=s.sort_values('ts'); ts=s.ts.to_numpy(); n=len(ts)
        if n<1000: continue
        spr=s.spread_ticks.to_numpy(); mid=s.mid.to_numpy(); norm=np.median(spr); cut=norm+max(2,norm)
        bid=mid-spr*.005; ask=mid+spr*.005
        contig=np.r_[True,np.diff(ts)==200]
        rb=pd.Series(bid).rolling(STILL); ra=pd.Series(ask).rolling(STILL)
        still=(((rb.max()-rb.min())<=.011)&((ra.max()-ra.min())<=.011)&(pd.Series(spr).rolling(STILL).max()<=cut)
               &(pd.Series(contig).rolling(STILL).min()==1)).to_numpy()
        quoted=np.flatnonzero(spr<=cut)
        ends=np.flatnonzero(still[:-1]&~still[1:])          # last slot of a still level
        last=-10**9
        for i in ends:
            if i<=last or i+LOOK>=n or not contig[i+1:i+LOOK].all(): continue
            dv=mid[i+1:i+LOOK+1]-mid[i]; k=np.flatnonzero(np.abs(dv)>=J-1e-9)
            if not len(k): continue
            k=k[0]; sg=np.sign(dv[k]); last=i+k
            lab={}
            for H in (150,300):
                q=np.searchsorted(quoted,i+H)
                if q>=len(quoted) or quoted[q]-(i+H)>300 or ts[quoted[q]]-ts[i]!=(quoted[q]-i)*200: lab[H]=np.nan; continue
                d=(mid[quoted[q]]-mid[i])*sg
                lab[H]=1.0 if d>=0.02-1e-9 else (0.0 if d<=0.01+1e-9 else np.nan)
            rows.append(dict(sess=sess,slug=sr.split('|')[0],series=sr,ts=int(ts[i]),size=abs(dv[k]),
                             pulled=spr[i+1+k]>cut,real30=lab[150],real60=lab[300]))
    print(sess,len(rows),flush=True)
pd.DataFrame(rows).to_pickle(SP+r'\xmkt.pkl')
