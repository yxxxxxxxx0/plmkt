"""Executable bid/ask event study for match jumps versus related team futures."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT=Path('kaggle_cross_market_download/linked_cross_market_sports')
REPORT=ROOT/'reports'; REPORT.mkdir(exist_ok=True)
DELAY=3.0; WINDOW=3.0; QUOTE_WAIT=10.0; COOLDOWN=60.0
THRESHOLDS=(.03,.05,.10); HORIZONS=(10,30,60,300)

def scale(ts):
    m=float(np.nanmedian(ts))
    return 1e9 if m>1e17 else 1e6 if m>1e14 else 1e3 if m>1e11 else 1.
def book(rows,cid,yes_token,want_yes):
    x=rows[rows.market_id.eq(cid)].copy()
    if x.empty:return x
    y=x.token_id.astype(str).eq(str(yes_token)).to_numpy()
    mid=np.where(y,x.mid_price,1-x.mid_price)
    bid=np.where(y,x.best_bid,1-x.best_ask); ask=np.where(y,x.best_ask,1-x.best_bid)
    if not want_yes: mid,bid,ask=1-mid,1-ask,1-bid
    x=x.assign(p=mid,bid=bid,ask=ask)[['timestamp_received','p','bid','ask']].dropna()
    return x.sort_values('timestamp_received',kind='stable').drop_duplicates('timestamp_received',keep='last')
def signal_indices(x,thr):
    t=x.t.to_numpy(); p=x.p.to_numpy(); old=np.searchsorted(t,t-WINDOW,side='right')-1
    out=[]; last=-np.inf
    for i,j in enumerate(old):
        if j>=0 and p[i]-p[j]>=thr and t[i]-last>=COOLDOWN:
            out.append((i,float(p[i]-p[j]))); last=t[i]
    return out
def relation(row):
    mq=row.match_question.lower(); fq=row.future_question.lower()
    if row.category=='CS2' and 'esl pro league' in mq and 'esl pro league' in fq:return 'direct_same_competition'
    if row.category=='football':return 'possibly_direct_competition'
    if row.category=='NBA':return 'same_season_team_signal'
    if row.category=='MLB':return 'preseason/team_strength_signal'
    return 'cross_tournament_strength_signal'

def main():
    pairs=pd.read_csv('cross_market_universe/verified_pairs.csv')
    meta=pq.read_table(ROOT/'metadata/candidate_metadata.parquet').to_pandas().set_index('condition_id')
    pairs['day']=pd.to_datetime(pairs.match_end_date,utc=True).dt.strftime('%Y-%m-%d')
    trades=[]; audits=[]
    for day,g in pairs.groupby('day'):
        path=ROOT/'orderbook'/f'orderbook_{day}.parquet'
        if not path.exists():continue
        ids=pd.unique(g[['match_condition_id','future_condition_id']].values.ravel()).tolist()
        tab=ds.dataset(path,format='parquet').to_table(columns=['timestamp_received','market_id','token_id','best_bid','best_ask','mid_price'],filter=ds.field('market_id').isin(ids))
        rows=tab.to_pandas(); del tab
        cache={}
        for cid in ids:
            if cid not in meta.index:continue
            cache[(cid,True)]=book(rows,cid,meta.loc[cid].clob_token_id_yes,True)
            cache[(cid,False)]=book(rows,cid,meta.loc[cid].clob_token_id_yes,False)
        for r in g.itertuples(index=False):
            m=cache.get((r.match_condition_id,bool(r.team_is_yes)),pd.DataFrame())
            f=cache.get((r.future_condition_id,True),pd.DataFrame())
            base={'category':r.category,'team':r.team,'match_question':r.match_question,'future_question':r.future_question,'relationship':relation(r)}
            if m.empty or f.empty:
                for th in THRESHOLDS: audits.append({**base,'threshold':th,'match_rows':len(m),'future_rows':len(f),'signals':0,'entries':0,'status':'missing_overlap'})
                continue
            sc=scale(m.timestamp_received.to_numpy(dtype=float)); m=m.assign(t=m.timestamp_received/sc); f=f.assign(t=f.timestamp_received/sc)
            event=pd.Timestamp(r.match_end_date).timestamp(); m=m[(m.t>=event-6*3600)&(m.t<=event+6*3600)]
            ft=f.t.to_numpy(); entries={}
            for th in THRESHOLDS:
                sigs=signal_indices(m,th) if len(m) else []; entered=0
                for si,jump in sigs:
                    st=float(m.t.iloc[si]); ei=int(np.searchsorted(ft,st+DELAY))
                    if ei>=len(f) or ft[ei]-(st+DELAY)>QUOTE_WAIT:continue
                    ask=float(f.ask.iloc[ei])
                    if not 0<ask<1:continue
                    entered+=1
                    for h in HORIZONS:
                        xi=int(np.searchsorted(ft,ft[ei]+h))
                        if xi>=len(f) or ft[xi]-(ft[ei]+h)>QUOTE_WAIT:continue
                        bid=float(f.bid.iloc[xi])
                        if not 0<bid<1:continue
                        trades.append({**base,'threshold':th,'signal_time_utc':pd.to_datetime(st,unit='s',utc=True),'match_jump':jump,'entry_ask':ask,'horizon_s':h,'exit_bid':bid,'pnl_per_share':bid-ask,'pnl_5_shares':5*(bid-ask),'return_on_cost':(bid-ask)/ask})
                audits.append({**base,'threshold':th,'match_rows':len(m),'future_rows':len(f),'signals':len(sigs),'entries':entered,'status':'tested'})
        print(day,'rows',len(rows),'pairs',len(g),flush=True)
    td=pd.DataFrame(trades); ad=pd.DataFrame(audits)
    td.to_csv(REPORT/'cross_category_trades.csv',index=False); ad.to_csv(REPORT/'cross_category_signal_audit.csv',index=False)
    if len(td):
        s=td.groupby(['category','relationship','threshold','horizon_s']).agg(trades=('pnl_per_share','size'),wins=('pnl_per_share',lambda x:int((x>0).sum())),win_rate=('pnl_per_share',lambda x:(x>0).mean()),mean_pnl_per_share=('pnl_per_share','mean'),median_pnl_per_share=('pnl_per_share','median'),total_pnl_5_shares=('pnl_5_shares','sum'),mean_return=('return_on_cost','mean')).reset_index()
    else:s=pd.DataFrame()
    s.to_csv(REPORT/'cross_category_summary.csv',index=False)
    cov=ad.groupby(['category','status']).agg(pair_threshold_tests=('threshold','size'),signals=('signals','sum'),entries=('entries','sum')).reset_index(); cov.to_csv(REPORT/'cross_category_coverage.csv',index=False)
    print('\nCOVERAGE\n',cov.to_string(index=False)); print('\nSUMMARY\n',s.to_string(index=False))
if __name__=='__main__':main()
