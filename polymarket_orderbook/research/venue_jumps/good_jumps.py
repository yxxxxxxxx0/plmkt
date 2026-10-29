"""Good jump = trading it breaks even: buy at the ask when the 30 s move starts, sell at the bid
H s after it ends, taker fee both legs (PM sports 0.05*p(1-p), Kalshi 0.07*p(1-p)). Oracle direction."""
import json, numpy as np, datetime as dt
VIEW='data/view'; marks=json.load(open(f'{VIEW}/score_marks.json',encoding='utf-8')); idx=json.load(open(f'{VIEW}/index.json'))
FEE={'pm':0.05,'ks':0.07}; HS=(10,30,60)
arr=lambda v: np.array([np.nan if x is None else x for x in v],float)
def ffill(a):
    i=np.where(~np.isnan(a),np.arange(len(a)),0); np.maximum.accumulate(i,out=i); return a[i]
fee=lambda v,p: FEE[v]*p*(1-p)
ev=[]
for sess in idx.values():
  for gi in sess['games']:
    g=json.load(open(f"{VIEW}/{gi['file']}",encoding='utf-8')); t=np.array(g['t']); n=len(t); M=marks[gi['slug']]
    goals=[s['t'] for s in M['scores']]; js=[]
    for k,s in g['series'].items():
      for v in ('pm','ks'):
        bid,ask,mid=(ffill(arr(s[f'{v}_{x}'])) for x in ('bid','ask','mid'))
        d=np.full(n,np.nan); d[30:]=mid[30:]-mid[:-30]; last=-999
        for i in np.flatnonzero(np.abs(np.nan_to_num(d))>=0.06):
          if i-last<120: continue
          last=i
          if i+max(HS)>=n: continue
          e=i-30; up=d[i]>0; pnl={}
          for H in HS:
            x=i+H
            if up: pin,pout=ask[e],bid[x]
            else: pin,pout=1-bid[e],1-ask[x]   # selling = buying the other side
            pnl[H]=pout-pin-fee(v,pin)-fee(v,pout)
          js.append(dict(t=t[i],k=k,v=v,move=float(d[i]),pnl=pnl,spread_in=float(ask[e]-bid[e])))
    # one event per cluster of jumps within 60 s (all outcomes, both venues)
    js.sort(key=lambda j:j['t']); cl=[]
    for j in js:
      if cl and j['t']-cl[-1][-1]['t']<=60_000: cl[-1].append(j)
      else: cl.append([j])
    for c in cl:
      t0=c[0]['t']; goal=next((s['label'] for s in M['scores'] if s['matched'] and -60_000<=t0-s['t']<=180_000),None)
      if goal is None: goal=next((s['label'] for s in M['scores'] if -60_000<=t0-s['t']<=180_000),None)
      m=(t0-M['kickoff_fit_ms'])/60000; clock=m if m<=48 else m-M['clock_param_min']
      ev.append(dict(game=gi['slug'],t=t0,clock=clock,goal=goal,best={H:max(j['pnl'][H] for j in c) for H in HS},
                     move=max(abs(j['move']) for j in c),venues=''.join(sorted({j['v'] for j in c}))))
print(f'{len(ev)} jump events ({sum(e["goal"] is not None for e in ev)} near a goal/run)\n')
print('exit  | goal events: good / bad | non-goal: good / bad | mean P&L best trade, cents (goal / non-goal)')
for H in HS:
  g=[e for e in ev if e['goal']]; o=[e for e in ev if not e['goal']]
  gg=sum(e['best'][H]>=0 for e in g); og=sum(e['best'][H]>=0 for e in o)
  print(f'+{H:2d}s  | {gg:3d} / {len(g)-gg:3d}               | {og:3d} / {len(o)-og:3d}            | {100*np.mean([e["best"][H] for e in g]):+.1f} / {100*np.mean([e["best"][H] for e in o]):+.1f}')
H=30; print(f'\nGood non-goal events at +{H}s exit (clock = approx match minute):')
for e in sorted([e for e in ev if not e['goal'] and e['best'][H]>=0],key=lambda e:(e['game'],e['t'])):
  print(f"  {e['game']:24s} {dt.datetime.fromtimestamp(e['t']/1000,dt.UTC):%H:%M:%S} ~{e['clock']:5.1f}'  move {100*e['move']:4.1f}c  P&L {100*e['best'][H]:+5.1f}c  venues {e['venues']}")
json.dump(ev,open(f'{VIEW}/jump_events.json','w',encoding='utf-8'),ensure_ascii=False,default=float)

# ---- categorise every event: goal / known match event / late-game (85'+) / unexplained
for e in ev:
  M=marks[e['game']]; oth=next((o['label'] for o in M['other_events'] if abs(e['t']-o['t'])<=180_000),None)
  e['cat']='goal' if e['goal'] else 'event' if oth else 'late 85+' if (e['clock']>=85 and not e['game'].startswith('mlb')) else 'mlb in-play' if e['game'].startswith('mlb') else 'unexplained'
  e['why']=e['goal'] or oth or ''
H=30; print(f'\nBy cause, exit +{H}s   good / bad   (mean P&L, cents)')
for c in ('goal','event','late 85+','mlb in-play','unexplained'):
  x=[e for e in ev if e['cat']==c]; g=sum(e['best'][H]>=0 for e in x)
  print(f'  {c:12s} {g:3d} / {len(x)-g:3d}   ({100*np.mean([e["best"][H] for e in x]):+.1f})')
print('\nKnown-event jumps:')
for e in ev:
  if e['cat']=='event': print(f"  {e['game']:24s} ~{e['clock']:5.1f}'  {e['why']:45s} move {100*e['move']:4.1f}c  {'GOOD' if e['best'][H]>=0 else 'bad '} {100*e['best'][H]:+.1f}c")
json.dump(ev,open(f'{VIEW}/jump_events.json','w',encoding='utf-8'),ensure_ascii=False,default=float)
