import json, numpy as np
VIEW='data/view'; marks=json.load(open(f'{VIEW}/score_marks.json',encoding='utf-8')); idx=json.load(open(f'{VIEW}/index.json'))
arr=lambda v: np.array([np.nan if x is None else x for x in v],float)
def ffill(a):
    i=np.where(~np.isnan(a),np.arange(len(a)),0); np.maximum.accumulate(i,out=i); return a[i]
rng=np.random.default_rng(0); out={W:{'score':[], 'noise':[], 'random':[]} for W in (5,15,30)}
for sess in idx.values():
  for gi in sess['games']:
    g=json.load(open(f"{VIEW}/{gi['file']}",encoding='utf-8')); t=np.array(g['t']); n=len(t)
    goals=[s['t'] for s in marks[gi['slug']]['scores']]
    for k,s in g['series'].items():
      pm,ks=ffill(arr(s['pm_mid'])),ffill(arr(s['ks_mid']))
      for me,other in ((pm,ks),(ks,pm)):
        d=np.full(n,np.nan); d[30:]=me[30:]-me[:-30]
        jumps=[]; last=-999
        for i in np.flatnonzero(np.abs(np.nan_to_num(d))>=0.06):
          if i-last>=120 and 60<i<n-60: jumps.append(i)
          if i-last>=120: last=i
        live=np.flatnonzero(~np.isnan(d)&~np.isnan(other)); live=live[(live>60)&(live<n-60)]
        rand=rng.choice(live, size=min(len(live),len(jumps)*3), replace=False) if len(live) else []
        for W in (5,15,30):
          od=lambda i: max((abs(other[j]-other[j-30]) if np.sign(other[j]-other[j-30])==np.sign(d[i]) or d[i]==0 else 0) for j in range(i-W,i+W+1))
          for i in jumps:
            real=any(-60_000<=t[i]-gt<=180_000 for gt in goals); out[W]['score' if real else 'noise'].append(od(i))
          for i in rand: out[W]['random'].append(max(abs(other[j]-other[j-30]) for j in range(i-W,i+W+1)))
for W,v in out.items():
  print(f'±{W}s', '  '.join(f"{k}: n={len(x)} other≥3c {np.mean(np.array(x)>=0.03):.0%} ≥6c {np.mean(np.array(x)>=0.06):.0%}" for k,x in v.items()))
