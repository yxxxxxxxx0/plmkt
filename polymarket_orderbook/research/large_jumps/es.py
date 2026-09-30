import os
# Event study: book features per second around the jump START (first slot after entry
# where the book leaves the entry level), large vs rest vs quiet-plateau baseline.
import sys; sys.path.insert(0,'research/makinen'); sys.path.insert(0,'.')
import numpy as np, pandas as pd, pyarrow.parquet as pq
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
PRE,POST=600,300            # slots of 200ms: -120s .. +60s
m=pd.read_csv('results/makinen/single_jumps/marks.csv')
lg=lambda p: np.log(p/(1-p)); m['large']=(lg(m.exit_bid.clip(.005,.995))-lg(m.entry_ask.clip(.005,.995)))>=0.46
rng=np.random.default_rng(0)
FEATS=['spr_rel','pulled','log_bid_top','log_ask_top','log_bid5','log_ask5','imb5','upd','absdmid']
recs=[]; curves=[]
for sess,g in m.groupby('session'):
    f=pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet',columns=['mid','spread_ticks','imb5','ts','series']).to_pandas()
    f=f[f.series.isin(set(g.series))]
    lob=np.load(f'data/jump/lob_{sess}_trimmed.npy',mmap_mode='r')
    for sr,s in f.groupby('series',sort=False):
        s=s.sort_values('ts'); ts=s.ts.to_numpy(); L=np.asarray(lob[s.index.to_numpy()],np.float64)
        spr=s.spread_ticks.to_numpy(); mid=s.mid.to_numpy(); norm=np.median(spr); cut=norm+max(2,norm)
        bid=np.round(mid-spr*.005,4); ask=np.round(mid+spr*.005,4)
        near=lambda side: np.log1p((np.expm1(L[:,side*2+1])*(np.abs(L[:,side*2])<=5)).sum(1))
        top=L[:,:,0]
        upd=np.r_[0,(np.abs(np.diff(top,axis=0))>1e-6).any(1)].astype(float)
        X=np.c_[spr/norm,(spr>cut),L[:,1,0],L[:,3,0],near(0),near(1),s.imb5.to_numpy(),upd,np.r_[0,np.abs(np.diff(mid))]/0.01]
        n=len(ts); ok=np.r_[True,np.diff(ts)==200]
        def win(i):
            if i-PRE<0 or i+POST>n or not ok[i-PRE+1:i+POST].all(): return None
            return X[i-PRE:i+POST].reshape(-1,5,X.shape[1]).mean(1)   # 1s bins
        gs=g[g.series==sr]; starts=[]
        for r in gs.itertuples():
            e=np.searchsorted(ts,r.ts_entry); b0,a0=bid[e],ask[e]; k=e
            while k<n-1 and abs(bid[k]-b0)<=.011 and abs(ask[k]-a0)<=.011 and spr[k]<=cut: k+=1
            starts.append(k); w=win(k)
            if w is not None: curves.append(w); recs.append(dict(kind='large' if r.large else 'rest',slug=r.slug,lead_s=(k-e)*.2))
        # baseline: quiet 5s plateaus (same rule as an entry) >=3 min from any mark
        rb=pd.Series(bid).rolling(25); ra=pd.Series(ask).rolling(25)
        still=((rb.max()-rb.min())<=.011)&((ra.max()-ra.min())<=.011)&(pd.Series(spr).rolling(25).max()<=cut)
        far=np.ones(n,bool)
        for k in starts: far[max(0,k-900):k+900]=False
        cand=np.flatnonzero(still.to_numpy()&far)
        for i in rng.choice(cand,min(len(cand),len(gs)),replace=False) if len(cand) else []:
            w=win(i)
            if w is not None: curves.append(w); recs.append(dict(kind='base',slug=s.series.iat[0].split('|')[0],lead_s=np.nan))
    print(sess,len(recs),flush=True)
np.save(SP+r'\es_curves.npy',np.array(curves)); pd.DataFrame(recs).to_pickle(SP+r'\es_meta.pkl')
