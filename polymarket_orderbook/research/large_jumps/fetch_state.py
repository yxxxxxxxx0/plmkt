import os
# Game-state timeline per game from statsapi: one row each time the state changes, stamped with
# statsapi's own endTime for that event. Looking up "latest row at or before t" is causal: the feed
# lags the market, so an in-progress pitch's result cannot leak in.
import sys, glob, time, datetime as dt
import pandas as pd, requests
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
ms = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000 if s else None

games = pd.concat([pd.read_csv(f, usecols=['slug', 'gamePk']) for f in glob.glob('results/makinen/mlb_plays/plays_*.csv')]).drop_duplicates()
rows = []
for slug, pk in games.itertuples(index=False):
    try:
        P = requests.get(FEED.format(pk=pk), timeout=90).json()['liveData']['plays']['allPlays']
    except Exception as e:
        print('warn', slug, e); continue
    away = home = 0; bases = (0, 0, 0); outs = 0; half = None
    for p in P:
        ab = p['about']
        if (ab['inning'], ab['isTopInning']) != half:              # new half-inning
            half = (ab['inning'], ab['isTopInning']); bases = (0, 0, 0); outs = 0
        state = dict(slug=slug, inning=ab['inning'], is_top=ab['isTopInning'], away=away, home=home,
                     on1=bases[0], on2=bases[1], on3=bases[2])
        n_pitch = 0
        for e in p['playEvents']:
            c = e.get('count', {}); n_pitch += bool(e.get('isPitch'))
            t = ms(e.get('endTime'))
            if t is None: continue
            rows.append(dict(state, t=t, kind='pitch' if e.get('isPitch') else 'event', balls=c.get('balls', 0),
                             strikes=c.get('strikes', 0), outs=c.get('outs', outs), pitches_pa=n_pitch))
        # end of plate appearance: new count, bases, score
        mu = p['matchup']; r = p['result']
        away, home = r.get('awayScore', away), r.get('homeScore', home)
        outs = p['count']['outs']
        bases = (int('postOnFirst' in mu), int('postOnSecond' in mu), int('postOnThird' in mu))
        if outs >= 3: bases = (0, 0, 0)
        t = ms(ab.get('endTime'))
        if t is not None:
            rows.append(dict(slug=slug, inning=ab['inning'], is_top=ab['isTopInning'], away=away, home=home,
                             on1=bases[0], on2=bases[1], on3=bases[2], t=t, kind='pa_end',
                             balls=0, strikes=0, outs=min(outs, 3), pitches_pa=0))
    time.sleep(0.3)
print(len(rows), 'state rows over', games.slug.nunique(), 'games')
pd.DataFrame(rows).sort_values(['slug', 't']).to_csv(os.path.join(IN, 'game_state.csv'), index=False)
