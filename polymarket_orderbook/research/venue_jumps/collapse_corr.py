"""Trade at the joint book collapse, BEFORE any price condition.
Alert T: both venues' total top-5 depth drops below RATIO x its own median over the previous 60s->5s,
onsets within 2 s; T = later onset. Filter: corr of the two venues' log depth ratio over [T-20, T].
Direction (no future info): outcome/side whose book emptied most, both venues summed:
asks gone -> buy, bids gone -> sell. Oracle = best outcome+direction in hindsight (upper bound)."""
import json, gzip, os, numpy as np, pandas as pd
VIEW='data/view'; idx=json.load(open(f'{VIEW}/index.json')); ev=json.load(open(f'{VIEW}/jump_events.json',encoding='utf-8'))
FEE={'pm':0.05,'ks':0.07}; fee=lambda v,p: FEE[v]*p*(1-p); RATIO=0.5
def ratio(D):
    out=np.full(len(D),np.nan)
    for i in range(60,len(D)):
        w=D[i-60:i-4]; w=w[~np.isnan(w)]
        if len(w)>=28 and np.median(w)>0 and not np.isnan(D[i]): out[i]=D[i]/np.median(w)
    return out
def onsets(r):
    on=np.nan_to_num(r,nan=9)<RATIO; o=[]; last=-999
    for i in np.flatnonzero(on[1:]&~on[:-1])+1:
        if i-last>=30: o.append(i); last=i
    return np.array(o,int)
rows=[]
for tag,sess in idx.items():
  for gi in sess['games']:
    f=f"{VIEW}/{tag}/aligned/{gi['slug']}.csv.gz"
    if not os.path.exists(f): print('missing',f); continue
    d=pd.read_csv(f); t=d['t_ms'].to_numpy(); n=len(t)
    keys=sorted({c.split('_')[0] for c in d.columns if c.endswith('_pm_mid')})
    col=lambda c: d[c].to_numpy(float) if c in d else np.full(n,np.nan)
    side=lambda k,v,s: np.nansum([np.nan_to_num(col(f'{k}_{v}_{s}{l}_sz')) for l in range(1,6)],axis=0)
    B={(k,v,s):side(k,v,s) for k in keys for v in ('pm','ks') for s in ('bid','ask')}
    P={(k,v,x):pd.Series(col(f'{k}_{v}_{x}1' if x!='mid' else f'{k}_{v}_mid')).ffill().to_numpy() for k in keys for v in ('pm','ks') for x in ('bid','ask','mid')}
    tot={v:sum(B[(k,v,s)] for k in keys for s in ('bid','ask')).astype(float) for v in ('pm','ks')}
    for v in tot: tot[v][tot[v]<=0]=np.nan
    r={v:ratio(tot[v]) for v in tot}; op,ok=onsets(r['pm']),onsets(r['ks'])
    good=[e['t'] for e in ev if e['game']==gi['slug'] and e['best']['30']>=0]
    big=[e['t'] for e in ev if e['game']==gi['slug'] and e['cat'] in ('goal','event')]
    single=len(op)+len(ok)
    for i in op:
      if not len(ok): break
      j=ok[np.argmin(np.abs(ok-i))]
      if abs(j-i)>2: continue
      T=max(i,j)
      if T<80 or T+40>=n: continue
      a,b=np.log(r['pm'][T-20:T+1]),np.log(r['ks'][T-20:T+1]); m=~np.isnan(a)&~np.isnan(b)
      c=float(np.corrcoef(a[m],b[m])[0,1]) if m.sum()>=10 and a[m].std()>0 and b[m].std()>0 else 0.0
      # direction from which side emptied: drop vs own median 60->5 s ago, summed over venues
      score={}
      for k in keys:
        drop={s:sum(1-B[(k,v,s)][T]/max(1,np.median(B[(k,v,s)][T-60:T-4])) for v in ('pm','ks')) for s in ('bid','ask')}
        score[k]=drop['ask']-drop['bid']          # >0: asks vanished -> price up
      k_dir=max(score,key=lambda k:abs(score[k])); sg=np.sign(score[k_dir]) or 1
      pre=max(abs(P[(k,'pm','mid')][T]-P[(k,'pm','mid')][T-5]) for k in keys)
      def pnl(k,s,v,delay,H):
          e=T+delay; x=e+H
          pin,pout=(P[(k,v,'ask')][e],P[(k,v,'bid')][x]) if s>0 else (1-P[(k,v,'bid')][e],1-P[(k,v,'ask')][x])
          return pout-pin-fee(v,pin)-fee(v,pout)
      R=dict(game=gi['slug'],cc=c,pre=pre,good=any(-10_000<=x-t[T]<=60_000 for x in good),big=any(-10_000<=x-t[T]<=60_000 for x in big))
      for v in ('pm','ks'):
        for dl in (0,1):
          for H in (10,30):
            R[f'mod_{v}{dl}_{H}']=pnl(k_dir,sg,v,dl,H)
            R[f'orc_{v}{dl}_{H}']=max(pnl(k,s,v,dl,H) for k in keys for s in (1,-1))
      # was the book-side direction right? (sign of the actual mid move T -> T+30 on that outcome)
      mv=P[(k_dir,'pm','mid')][T+30]-P[(k_dir,'pm','mid')][T]
      R['dir_ok']=np.nan if mv==0 or np.isnan(mv) else float(np.sign(mv)==sg)
      rows.append(R)
    rows.append(dict(game=gi['slug'],single=single))
D=pd.DataFrame(rows); S=D['single'].sum(); D=D[D['single'].isna()]
print(f"{int(S)} single-venue collapses; {len(D)} joint (both venues within 2 s), 12 games")
print(f"price already moved >=2c on PM in the 5 s before the alert: {np.mean(D.pre>=0.02):.0%} of joint alerts\n")
def summ(sub,lab):
    print(f"{lab:16s} n={len(sub):3d}  ->good jump {int(sub.good.sum()):3d} ({sub.good.mean():4.0%})  goal/event {int(sub.big.sum()):2d}  "
          f"dir right {np.nanmean(sub.dir_ok):4.0%} | PM 1s delay hold 30s: model {100*np.nanmean(sub.mod_pm1_30):+5.1f}c  oracle {100*np.nanmean(sub.orc_pm1_30):+5.1f}c"
          f" | KS 0s: model {100*np.nanmean(sub.mod_ks0_30):+5.1f}c  oracle {100*np.nanmean(sub.orc_ks0_30):+5.1f}c")
summ(D,'all joint')
for lo,hi in ((-1,0),(0,0.3),(0.3,0.6),(0.6,0.8),(0.8,1.01)):
    summ(D[(D.cc>=lo)&(D.cc<hi)],f'corr {lo:+.1f}..{hi:.1f}')
print('\nfilter corr>=0.6 and price NOT yet moved (<2c):')
summ(D[(D.cc>=0.6)&(D.pre<0.02)],'corr>=.6, early')
summ(D[(D.cc<0.6)&(D.pre<0.02)],'corr<.6, early')
g=D[(D.cc>=0.6)]
print(f"\ncorr>=0.6, model PM 1s: hold 10s {100*np.nanmean(g.mod_pm1_10):+.1f}c, 30s {100*np.nanmean(g.mod_pm1_30):+.1f}c; winners {int((g.mod_pm1_30>0).sum())}/{len(g)}")
