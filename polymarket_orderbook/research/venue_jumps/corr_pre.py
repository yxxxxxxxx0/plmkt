import json, numpy as np
VIEW='data/view'; marks=json.load(open(f'{VIEW}/score_marks.json',encoding='utf-8')); idx=json.load(open(f'{VIEW}/index.json'))
arr=lambda v: np.array([np.nan if x is None else x for x in v],float)
def ffill(a):
    i=np.where(~np.isnan(a),np.arange(len(a)),0); np.maximum.accumulate(i,out=i); return a[i]
def corr(a,b):
    ok=~np.isnan(a)&~np.isnan(b); a,b=a[ok],b[ok]
    if len(a)<10 or a.std()==0 or b.std()==0: return 0.0
    return float(np.corrcoef(a,b)[0,1])
rows=[]; rng=np.random.default_rng(0)
for sess in idx.values():
  for gi in sess['games']:
    g=json.load(open(f"{VIEW}/{gi['file']}",encoding='utf-8')); t=np.array(g['t']); n=len(t)
    goals=[s['t'] for s in marks[gi['slug']]['scores']]
    for k,s in g['series'].items():
      pm,ks=ffill(arr(s['pm_mid'])),ffill(arr(s['ks_mid']))
      for me,other in ((pm,ks),(ks,pm)):
        d=np.full(n,np.nan); d[30:]=me[30:]-me[:-30]
        # 5 s price changes on both venues, correlated over the 2 min around the jump
        r5m=np.full(n,np.nan); r5m[5:]=me[5:]-me[:-5]; r5o=np.full(n,np.nan); r5o[5:]=other[5:]-other[:-5]
        last=-999; J=[]
        for i in np.flatnonzero(np.abs(np.nan_to_num(d))>=0.06):
          if i-last>=120 and 90<i<n-310: J.append(i)
          if i-last>=120: last=i
        for i in J:
          c=corr(r5m[i-90:i-30:5], r5o[i-90:i-30:5])  # 60 s ending where the 30 s jump starts
          hold=abs(me[i+300]-me[i-30])
          kind='score' if any(-60_000<=t[i]-gt<=180_000 for gt in goals) else 'other'
          rows.append((kind,c,abs(d[i]),hold))
        live=np.flatnonzero(~np.isnan(r5m)&~np.isnan(r5o)); live=live[(live>90)&(live<n-90)]
        for i in rng.choice(live,size=min(len(live),3*len(J)),replace=False) if len(live) else []:
          rows.append(('random',corr(r5m[i-90:i-30:5],r5o[i-90:i-30:5]),0,0))
K=np.array([r[0] for r in rows]); C=np.array([r[1] for r in rows]); H=np.array([r[3] for r in rows])
for k in ('score','other','random'):
  c=C[K==k]; print(f'{k:6s} n={len(c):4d} median corr {np.median(c):.2f}  share >=0.5: {np.mean(c>=0.5):.0%}  >=0.8: {np.mean(c>=0.8):.0%}')
print('\nfilter: keep jump if corr >= thr')
S,O=K=='score',K=='other'; big=H>=0.10
for thr in (0.3,0.5,0.7,0.8,0.9):
  keep=C>=thr
  print(f' thr {thr}: goals kept {(keep&S).sum()}/{S.sum()}  non-goal kept {(keep&O).sum()}/{O.sum()}  share of kept that are goals {(keep&S).sum()/max(1,(keep&(S|O)).sum()):.0%}'
        f'  | jumps still >=10c after 5min kept {(keep&big&(S|O)).sum()}/{(big&(S|O)).sum()}, small-and-gone kept {(keep&~big&(S|O)).sum()}/{(~big&(S|O)).sum()}')
