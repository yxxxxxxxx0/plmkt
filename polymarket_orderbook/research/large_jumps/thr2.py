import os
import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings('ignore')
m=pd.read_csv('results/makinen/single_jumps/marks.csv')
lg=lambda p: np.log(p/(1-p))
m['d']=lg(m.exit_bid.clip(.005,.995))-lg(m.entry_ask.clip(.005,.995))
m=m.sort_values(['slug','ts_entry']).reset_index(drop=True); m['n']=m.groupby('slug').cumcount()+1
A=pd.concat([pd.read_csv(f) for f in glob.glob('results/makinen/mlb_plays/plays_*.csv')])
print('unmatched games:',sorted(set(m.slug)-set(A.slug)))
m=m[m.slug.isin(set(A.slug))].reset_index(drop=True)
P=A[A.is_scoring]; pad=30e3
def lab(shift=0):
    y=np.zeros(len(m),bool)
    for slug,g in m.groupby('slug'):
        e=np.sort(P[P.slug==slug].end_ms.to_numpy())+shift
        a=g.ts_entry.to_numpy()-pad; b=g.ts_jump_end.to_numpy()+pad
        i=np.minimum(np.searchsorted(e,a),max(len(e)-1,0))
        y[g.index]=(len(e)>0)&(e[i]>=a)&(e[i]<=b)
    return y
m['y']=lab(); m['y0']=lab(20*60e3)
print('marks',len(m),'games',m.slug.nunique(),'sessions',m.session.nunique(),'scoring plays',len(P),'marks on a run',m.y.sum())
bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,1.0,1.2,1.5,9]
t=m.groupby(pd.cut(m.d,bins)).agg(n=('y','size'),p_run=('y','mean'),p_null=('y0','mean'))
t['pct_above']=[(m.d>b).mean()*100 for b in bins[:-1]]
print(t.round(3).to_string())
def stats(g):
    o=np.argsort(-g.d.to_numpy()); y=g.y.to_numpy()[o]; d=g.d.to_numpy()[o]
    tpr=np.cumsum(y)/y.sum(); fpr=np.cumsum(~y)/(~y).sum(); k=np.argmax(tpr-fpr)
    # logistic fit on d: 50% point
    X=np.c_[np.ones(len(g)),g.d]; w=np.zeros(2); Y=g.y.to_numpy()
    for _ in range(30):
        p=1/(1+np.exp(-X@w)); H=X.T@(X*(p*(1-p))[:,None]); w+=np.linalg.solve(H,X.T@(Y-p))
    return np.trapezoid(tpr,fpr), d[k], tpr[k], fpr[k], -w[0]/w[1], (np.log(4)-w[0])/w[1]
auc,yj,tp,fp,c50,c80=stats(m)
print(f'AUC {auc:.3f} | Youden cut {yj:.3f} (catches {tp:.0%} of run jumps, {fp:.0%} of others) | 50% cut {c50:.3f} | 80% cut {c80:.3f}')
rng=np.random.default_rng(0); gs=m.slug.unique(); B=[]
for _ in range(300):
    s=rng.choice(gs,len(gs)); B.append(stats(pd.concat([m[m.slug==x] for x in s],ignore_index=True)))
B=np.array(B)
for j,nm in [(0,'AUC'),(1,'Youden'),(4,'50%'),(5,'80%')]: print(nm,'95% CI',np.percentile(B[:,j],[2.5,97.5]).round(3))
for c in (yj,c50,c80): print(f'cut {c:.3f}: keeps {(m.d>=c).mean():.1%} of marks ({(m.d>=c).sum()}), of which runs {m[m.d>=c].y.mean():.0%}')
print(m.groupby('session').apply(lambda g: pd.Series(stats(g)[:1]+stats(g)[4:5],index=['auc','cut50'])).round(2).T.to_string())
want={'mlb-pit-stl-2026-08-29':[51,53,55],'mlb-sea-col-2026-09-18':[57],'mlb-sea-tor-2026-08-29':[13,29]}
ex=pd.concat([m[(m.slug==s)&m.n.isin(ns)] for s,ns in want.items()])
print(ex[['slug','n','d','y']].round(2).to_string(index=False))
