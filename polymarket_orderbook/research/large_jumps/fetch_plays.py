import os
import sys, time; sys.path.insert(0,'research/makinen'); sys.path.insert(0,'.')
import pandas as pd
from pathlib import Path
from align_mlb_plays import slug_to_gamepk, fetch_plays
out=Path('results/makinen/mlb_plays')
m=pd.read_csv('results/makinen/single_jumps/marks.csv',usecols=['session','slug'])
for sess,g in m.groupby('session'):
    f=out/f'plays_{sess}.csv'; old=pd.read_csv(f)
    miss=sorted(set(g.slug)-set(old.slug))
    if not miss: continue
    pks=slug_to_gamepk(miss, sess.replace('books_',''))
    fr=[old]
    for s,pk in pks.items():
        fr.append(fetch_plays(s,pk)); time.sleep(0.4)
    pd.concat(fr,ignore_index=True).to_csv(f,index=False)
    print(sess,'added',sorted(pks),'still missing',sorted(set(miss)-set(pks)),flush=True)
