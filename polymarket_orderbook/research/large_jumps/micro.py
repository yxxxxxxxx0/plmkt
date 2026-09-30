import os
import sys, glob; sys.path.insert(0,'research/makinen'); sys.path.insert(0,'.')
import numpy as np, pandas as pd, pyarrow.parquet as pq
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
m=pd.read_csv('results/makinen/single_jumps/marks.csv')
lg=lambda p: np.log(p/(1-p)); m['d']=lg(m.exit_bid.clip(.005,.995))-lg(m.entry_ask.clip(.005,.995))
m['large']=m.d>=0.46
C=['spread_ticks','imb1','imb5','microprice_dev','log_bid_usd','log_ask_usd','rv_25','rv_150','ofi_25','dmid_25','book_age_ms','ts','series']
rows=[]
for sess,g in m.groupby('session'):
    f=pq.read_table(f'data/jump/feat_{sess}_trimmed.parquet',columns=C).to_pandas()
    f=f[f.series.isin(set(g.series))]
    lob=np.load(f'data/jump/lob_{sess}_trimmed.npy',mmap_mode='r')
    for sr,s in f.groupby('series',sort=False):
        s=s.sort_values('ts'); ts=s.ts.to_numpy(); ix=s.index.to_numpy()
        spr=s.spread_ticks.to_numpy(); norm=np.median(spr); cut=norm+max(2,norm)
        chg=np.r_[0,(np.diff(spr)!=0)|(np.diff(s.imb1.to_numpy())!=0)]
        for r in g[g.series==sr].itertuples():
            e=np.searchsorted(ts,r.ts_entry); j=np.searchsorted(ts,r.ts_jump_end); x=np.searchsorted(ts,r.ts_exit)
            p0=np.searchsorted(ts,r.ts_entry-30000); p5=np.searchsorted(ts,r.ts_entry-5000)
            a=np.searchsorted(ts,r.ts_jump_end+30000)
            L=np.asarray(lob[ix[e]],np.float64)
            near=lambda side: np.expm1(L[side*2+1][np.abs(L[side*2])<=5]).sum()
            rows.append(dict(idx=r.Index,
              pre_spread=spr[e], pre_spread_rel=spr[e]/norm, pre_pulled_30s=(spr[p0:e]>cut).mean() if e>p0 else np.nan,
              pre_spread_max_5s=spr[p5:e+1].max(), pre_updates_30s=chg[p0:e].sum(),
              pre_imb1=s.imb1.iat[e], pre_imb5=s.imb5.iat[e], pre_microdev=s.microprice_dev.iat[e],
              pre_depth5_bid=near(0), pre_depth5_ask=near(1), pre_logdepth_bid=s.log_bid_usd.iat[e], pre_logdepth_ask=s.log_ask_usd.iat[e],
              pre_rv25=s.rv_25.iat[e], pre_rv150=s.rv_150.iat[e], pre_ofi25=s.ofi_25.iat[e], pre_dmid25=s.dmid_25.iat[e],
              pre_book_age=s.book_age_ms.iat[e],
              dur_spread_max=spr[e:j+1].max(), dur_pulled=(spr[e:j+1]>cut).mean(),
              post_pulled_30s=(spr[j:a]>cut).mean(), post_spread_rel=spr[x]/norm))
    print(sess,flush=True)
F=m.join(pd.DataFrame(rows).set_index('idx'))
F['pre_price_dist']=(F.entry_ask-.5).abs(); F['normal_spread_']=F.normal_spread
F.to_pickle(SP+r'\micro.pkl')
