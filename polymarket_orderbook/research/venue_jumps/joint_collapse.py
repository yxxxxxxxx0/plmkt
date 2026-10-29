"""Joint-collapse taker strategy. Alert at T = second venue's collapse onset (depth < 0.5x its own
median over the previous 60s->5s; both onsets within 2 s), direction = outcome whose mid moved the
same way on BOTH venues over [T-5, T]; require min(|move_pm|, |move_ks|) >= AGREE.
Trade on one venue: buy at ask at T+delay, sell at bid at T+delay+H, taker fee both legs."""
import json, numpy as np
VIEW='data/view'; idx=json.load(open(f'{VIEW}/index.json')); ev=json.load(open(f'{VIEW}/jump_events.json',encoding='utf-8'))
FEE={'pm':0.05,'ks':0.07}; fee=lambda v,p: FEE[v]*p*(1-p)
arr=lambda v: np.array([np.nan if x is None else x for x in v],float)
def ffill(a):
    i=np.where(~np.isnan(a),np.arange(len(a)),0); np.maximum.accumulate(i,out=i); return a[i]
def ratio(D):
    D=arr(D); out=np.full(len(D),np.nan)
    for i in range(60,len(D)):
        w=D[i-60:i-4]; w=w[~np.isnan(w)]
        if len(w)>=28 and np.median(w)>0 and not np.isnan(D[i]): out[i]=D[i]/np.median(w)
    return out
def onsets(r,thr=0.5):
    on=np.nan_to_num(r,nan=9)<thr; o=[]; last=-999
    for i in np.flatnonzero(on[1:]&~on[:-1])+1:
        if i-last>=30: o.append(i); last=i
    return np.array(o)
alerts=[]
for sess in idx.values():
  for gi in sess['games']:
    g=json.load(open(f"{VIEW}/{gi['file']}",encoding='utf-8')); t=np.array(g['t']); n=len(t)
    good=[e['t'] for e in ev if e['game']==gi['slug'] and e['best']['30']>=0]
    goal=[e['t'] for e in ev if e['game']==gi['slug'] and e['cat'] in ('goal','event')]
    op,ok=onsets(ratio(g['depth']['pm'])),onsets(ratio(g['depth']['ks']))
    S={k:{x:ffill(arr(s[x])) for x in s} for k,s in g['series'].items()}
    for i in op:
      if not len(ok): break
      j=ok[np.argmin(np.abs(ok-i))]
      if abs(j-i)>2: continue
      T=max(i,j)
      if T<10 or T+40>=n: continue
      best=None
      for k,s in S.items():
        a=s['pm_mid'][T]-s['pm_mid'][T-5]; b=s['ks_mid'][T]-s['ks_mid'][T-5]
        if np.isnan(a) or np.isnan(b) or np.sign(a)!=np.sign(b) or a==0: continue
        m=min(abs(a),abs(b))
        if best is None or m>best[0]: best=(m,k,np.sign(a))
      if best is None: best=(0.0,None,0)
      alerts.append(dict(game=gi['slug'],mlb=gi['slug'].startswith('mlb'),T=T,t=t[T],agree=best[0],k=best[1],sg=best[2],S=S,
                         good=any(abs(t[T]-x)<=60_000 for x in good),goal=any(abs(t[T]-x)<=60_000 for x in goal)))
def pnl(a,v,delay,H):
    if a['k'] is None: return np.nan
    s=a['S'][a['k']]; e=a['T']+delay; x=e+H
    if a['sg']>0: pin,pout=s[f'{v}_ask'][e],s[f'{v}_bid'][x]
    else: pin,pout=1-s[f'{v}_bid'][e],1-s[f'{v}_ask'][x]
    return pout-pin-fee(v,pin)-fee(v,pout)
print(f'{len(alerts)} joint collapses (both venues within 2 s) across 12 games\n')
print('agree = both venues moved >= X the same way in the 5 s up to the alert')
print('  X    alerts  ->good jump  goal/event | P&L per trade, cents, by venue / delay / hold')
for X in (0.0,0.01,0.02,0.03,0.05):
  A=[a for a in alerts if a['k'] is not None and a['agree']>=X]
  if not A: continue
  line=f"{100*X:3.0f}c  {len(A):6d}  {sum(a['good'] for a in A):4d} ({np.mean([a['good'] for a in A]):3.0%})  {sum(a['goal'] for a in A):4d}     |"
  for v,dl in (('pm',0),('pm',1),('ks',0),('ks',1)):
    for H in (10,30):
      p=[pnl(a,v,dl,H) for a in A]; p=[x for x in p if not np.isnan(x)]
      line+=f" {v}{dl}s/{H}s {100*np.mean(p):+5.1f}"
  print(line)
X=0.02; A=[a for a in alerts if a['k'] is not None and a['agree']>=X]
print(f'\nAt X={100*X:.0f}c, PM with 1 s delay, hold 10 s: win rate and break-even')
for H in (10,30):
  p=np.array([pnl(a,'pm',1,H) for a in A]); p=p[~np.isnan(p)]
  print(f'  hold {H}s: trades {len(p)}, winners {int((p>0).sum())}, mean {100*p.mean():+.2f}c, mean of winners {100*p[p>0].mean():+.1f}c, losers {100*p[p<=0].mean():+.1f}c')
print('\ndelay cost on PM (same alerts, hold 10 s): ' + '  '.join(f"{d}s {100*np.nanmean([pnl(a,'pm',d,10) for a in A]):+.2f}c" for d in (0,1,2,3)))
print('football only, PM 1 s delay hold 10 s: %+.2fc   MLB, PM 0 s: %+.2fc' % (
  100*np.nanmean([pnl(a,'pm',1,10) for a in A if not a['mlb']]), 100*np.nanmean([pnl(a,'pm',0,10) for a in A if a['mlb']] or [np.nan])))

print('\nWhere the loss comes from (X=2c, hold 30 s, entry delay 1 s):')
for v in ('pm','ks'):
  for lab,sub in (('became good jump',[a for a in A if a['good']]),('did not',[a for a in A if not a['good']])):
    mv=[];sp_in=[];sp_out=[];pl=[]
    for a in sub:
      s=a['S'][a['k']]; e=a['T']+1; x=e+30
      mv.append(a['sg']*(s[f'{v}_mid'][x]-s[f'{v}_mid'][e])); sp_in.append(s[f'{v}_ask'][e]-s[f'{v}_bid'][e]); sp_out.append(s[f'{v}_ask'][x]-s[f'{v}_bid'][x]); pl.append(pnl(a,v,1,30))
    print(f"  {v} {lab:17s} n={len(sub):3d}  mid move {100*np.nanmean(mv):+5.1f}c  spread at entry mean {100*np.nanmean(sp_in):4.1f}c (p90 {100*np.nanpercentile(sp_in,90):4.1f})  at exit {100*np.nanmean(sp_out):4.1f}c  P&L {100*np.nanmean(pl):+5.1f}c")
# move already done before the alert vs. still to come
pre=[a['agree'] for a in A]
post=[a['sg']*(a['S'][a['k']]['pm_mid'][a['T']+31]-a['S'][a['k']]['pm_mid'][a['T']+1]) for a in A]
print(f"\n  move already made by the alert (min of venues, 5 s): {100*np.mean(pre):.1f}c   PM mid move after entry: {100*np.nanmean(post):+.1f}c")
