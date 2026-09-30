"""YES/NO token pairs and what YES means (team side, Over/Under) for every recorded game.

Writes results/large_jumps/inputs/token_pairs.csv (slug, yes, no) and yes_meta.csv
(yes token -> outcome name, and 'away' / 'home' / 'Over' / 'Under'). Gamma's outcomes[0] is the
YES token, which is the one the recorder keeps; team sides come from statsapi's schedule.

    python research/large_jumps/fetch_tokens.py
"""
import json, os, sys, time
import pandas as pd
sys.path.insert(0, '.')
import plan_slate as ps

IN = 'results/large_jumps/inputs'
slugs = sorted(pd.read_csv('results/makinen/single_jumps/marks.csv', usecols=['slug']).slug.unique())
rows, teams = [], []
for slug in slugs:
    ev = ps.gamma_event(slug); time.sleep(0.15)
    for m in (ev or {}).get('markets', []):
        ids = json.loads(m.get('clobTokenIds') or '[]'); oc = json.loads(m.get('outcomes') or '[]')
        if len(ids) == 2 and len(oc) == 2:
            rows.append(dict(slug=slug, yes=ids[0], no=ids[1], yes_name=oc[0], no_name=oc[1], question=m.get('question')))
for d in sorted({s[-10:] for s in slugs}):
    for g in ps.schedule(d):
        for a in ps.candidates(g['away_abbr']):
            for h in ps.candidates(g['home_abbr']):
                teams.append(dict(slug=f'mlb-{a}-{h}-{d}', away_name=g['away_name'], home_name=g['home_name']))
M = pd.DataFrame(rows).merge(pd.DataFrame(teams).drop_duplicates('slug'), on='slug', how='left')
M['yes_side'] = M.apply(lambda r: 'away' if r.yes_name == r.away_name else ('home' if r.yes_name == r.home_name else r.yes_name), axis=1)
M[['slug', 'yes', 'no']].to_csv(os.path.join(IN, 'token_pairs.csv'), index=False)
M.drop(columns='no').to_csv(os.path.join(IN, 'yes_meta.csv'), index=False)
print(len(M), 'markets;', M.yes_side.value_counts().to_dict())
