"""Build one panel per game: every recorded YES market on the shared 200 ms clock.

Reads data/jump/feat_*_trimmed.parquet and lob_*_trimmed.npy (read-only) and the marks; writes
research/large_jumps/cache/xm/panel_<session>_<slug>.npz.

    python research/large_jumps/cross_market/build_panel.py [--smoke]
"""
import json, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
from common import (ROOT, IN, MARKS, GRID, ROLES, LARGE_LOGODDS, panel_path, sessions, is_test, smoke_flag, logit)

smoke = smoke_flag(sys.argv)
meta_yes = pd.read_csv(os.path.join(IN, 'yes_meta.csv'), dtype={'yes': str}).drop_duplicates('yes').set_index('yes')
marks = pd.read_csv(MARKS)
marks['large'] = (logit(marks.exit_bid) - logit(marks.entry_ask)) >= LARGE_LOGODDS

todo = sessions()
if smoke:
    todo = [s for s in todo if not is_test(s)][-1:] + [s for s in todo if is_test(s)][:1]
for sess in todo:
    f = pq.read_table(os.path.join(ROOT, 'data', 'jump', f'feat_{sess}_trimmed.parquet'),
                      columns=['mid', 'spread_ticks', 'ts', 'series']).to_pandas()
    lob = np.load(os.path.join(ROOT, 'data', 'jump', f'lob_{sess}_trimmed.npy'), mmap_mode='r')
    f['slug'] = f.series.str.split('|').str[0]; f['role'] = f.series.str.split('|').str[1]
    f = f[f.role.isin(ROLES)]
    ms = marks[marks.session == sess]
    games = sorted(f.slug.unique())
    if smoke: games = games[:6]
    for slug in games:
        g = f[f.slug == slug]
        series = sorted(g.series.unique())
        s0 = int(g.ts.min() // GRID); S = int(g.ts.max() // GRID) - s0 + 1
        K = len(series)
        P = np.zeros((K, S, 7), np.float16); MID = np.full((K, S), np.nan, np.float32)
        meta, starts, larges = [], [], []
        for k, sr in enumerate(series):
            s = g[g.series == sr].sort_values('ts')
            idx = (s.ts.to_numpy() // GRID - s0).astype(np.int64)
            L = np.asarray(lob[s.index.to_numpy()], np.float32)
            spr = s.spread_ticks.to_numpy(np.float64); mid = s.mid.to_numpy(np.float64)
            norm = float(np.median(spr)); cut = norm + max(2, norm)
            near = lambda side: np.log1p((np.expm1(L[:, side * 2 + 1]) * (np.abs(L[:, side * 2]) <= 5)).sum(1))
            P[k, idx, 0] = np.clip(spr / norm, 0, 60)
            P[k, idx, 1] = spr > cut
            P[k, idx, 2] = near(0); P[k, idx, 3] = near(1)
            P[k, idx, 4] = L[:, 1, 0]; P[k, idx, 5] = L[:, 3, 0]
            P[k, idx, 6] = 1
            MID[k, idx] = mid
            aid = sr.split('|')[-1]; side = meta_yes.yes_side.get(aid, 'unknown')
            ori = {'away': 1.0, 'Over': 1.0, 'home': -1.0, 'Under': -1.0}.get(side, 0.0)
            line = sr.split('|')[2]
            meta.append(dict(series=sr, role=ROLES[sr.split('|')[1]], line=None if line == 'None' else float(line),
                             side=side, ori=ori, norm=norm))
            # where each marked jump starts: the first slot after entry that leaves the entry level
            bid = mid - spr * .005; ask = mid + spr * .005; tsr = s.ts.to_numpy()
            st, lg = [], []
            for x in ms[ms.series == sr].itertuples():
                e = min(np.searchsorted(tsr, x.ts_entry), len(tsr) - 1); j = e
                while j < len(tsr) - 1 and abs(bid[j] - bid[e]) <= .011 and abs(ask[j] - ask[e]) <= .011 and spr[j] <= cut:
                    j += 1
                st.append(int(tsr[j] // GRID - s0)); lg.append(bool(x.large))
            st = np.array(st, np.int64); lg = np.array(lg, bool); o = np.argsort(st)
            starts.append(st[o]); larges.append(st[o][lg[o]])
        np.savez_compressed(panel_path(sess, slug), P=P, mid=MID, s0=np.int64(s0), meta=json.dumps(meta),
                            **{f'start_{k}': starts[k] for k in range(K)}, **{f'large_{k}': larges[k] for k in range(K)})
    print(sess, len(games), 'games', flush=True)
