import json, os, sys, numpy as np
VIEW = 'data/view'
marks = json.load(open(f'{VIEW}/score_marks.json', encoding='utf-8'))
idx = json.load(open(f'{VIEW}/index.json'))
arr = lambda v: np.array([np.nan if x is None else x for x in v], float)
def ffill(a):
    i = np.where(~np.isnan(a), np.arange(len(a)), 0); np.maximum.accumulate(i, out=i); return a[i]
rows = []
for sess in idx.values():
  for gi in sess['games']:
    g = json.load(open(f"{VIEW}/{gi['file']}", encoding='utf-8')); t = np.array(g['t']); n = len(t)
    m = marks[gi['slug']]; goals = [s['t'] for s in m['scores']]
    for k, s in g['series'].items():
      pm, ks = ffill(arr(s['pm_mid'])), ffill(arr(s['ks_mid']))
      for me, other, name in ((pm, ks, 'PM'), (ks, pm, 'KS')):
        d = np.full(n, np.nan); d[30:] = me[30:] - me[:-30]
        last = -999
        for i in np.flatnonzero(np.abs(np.nan_to_num(d)) >= 0.06):
          if i - last < 120 or i + 300 >= n: last = i if i - last >= 120 else last; continue
          last = i; sg = np.sign(d[i])
          # other venue: best same-direction move over 30 s windows ending within +-30 s of this jump
          o = max((sg * (other[j] - other[j - 30]) for j in range(i - 30, i + 31) if not np.isnan(other[j]) and not np.isnan(other[j - 30])), default=np.nan)
          hold = sg * (me[i + 300] - me[i - 30])  # still there 5 min later?
          real = any(-60_000 <= t[i] - gt <= 180_000 for gt in goals)
          rows.append((name, real, o, hold, d[i]))
R = np.array([(r[0] == 'PM', r[1], r[2], r[3]) for r in rows], float)
def rep(sel, lab):
  r = R[sel]; print(f'{lab:28s} n={len(r):4d}  scores={int(r[:,1].sum()):3d}')
print(f'{len(R)} single-venue jumps (6c+/30s), {int(R[:,1].sum())} within a score window')
for thr in (0.02, 0.03, 0.05):
  agree = np.nan_to_num(R[:,2], nan=-1) >= thr
  tp = (agree & (R[:,1]==1)).sum(); fp = (agree & (R[:,1]==0)).sum(); fn = (~agree & (R[:,1]==1)).sum()
  print(f'other venue moves >={thr*100:.0f}c same way: flagged {agree.sum()}, real {tp} (precision {tp/max(1,agree.sum()):.0%}), scores missed {fn}/{tp+fn}')
for v, lab in ((1,'PM'),(0,'KS')):
  r = R[R[:,0]==v]; print(lab, 'jumps', len(r), 'real', int(r[:,1].sum()))
for real in (1,0):
  r = R[R[:,1]==real]; a = np.nan_to_num(r[:,2], nan=-1) >= 0.03
  print('score' if real else 'noise', ' agree&held>=3c:', int((a & (r[:,3]>=0.03)).sum()), ' agree only:', int(a.sum()), ' held only:', int((r[:,3]>=0.03).sum()), ' of', len(r))
