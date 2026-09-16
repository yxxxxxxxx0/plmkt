from pathlib import Path
import gc,re,shutil
import pandas as pd
import polars as pl

I=Path('/kaggle/input'); O=Path('/kaggle/working/linked_cross_market_sports')
NBA="Hawks Celtics Nets Hornets Bulls Cavaliers Mavericks Nuggets Pistons Warriors Rockets Pacers Clippers Lakers Grizzlies Heat Bucks Timberwolves Pelicans Knicks Thunder Magic 76ers Suns|Trail Blazers|Kings Spurs Raptors Jazz Wizards".replace('|',' ').split()
NBA.remove('Trail'); NBA.remove('Blazers'); NBA += ['Trail Blazers']
MLB="Diamondbacks Athletics Braves Orioles|Red Sox|Cubs|White Sox|Reds Guardians Rockies Tigers Astros Royals Angels Dodgers Marlins Brewers Twins Mets Yankees Phillies Pirates Padres Giants Mariners Cardinals Rays Rangers|Blue Jays|Nationals".replace('|',' ').split()
for x in ['Red','Sox','White','Blue','Jays']: 
    while x in MLB: MLB.remove(x)
MLB += ['Red Sox','White Sox','Blue Jays']

def norm(x):
    x=(x or '').lower().replace('�','-'); x=re.sub(r'\b(the|fc|cf|afc|cfc)\b',' ',x)
    return re.sub(r'[^a-z0-9]+',' ',x).strip()
def fteam(q):
    m=re.match(r'(?i)^Will (?:the )?(.+?) (?:win|make|finish|reach|qualify|be relegated)\b',q or '')
    return m.group(1).strip() if m else None
def cat(q):
    if re.search(r'(?i)NBA|Eastern Conference Finals|Western Conference Finals',q): return 'NBA'
    if re.search(r'(?i)World Series|American League Championship|National League Championship|\b(?:AL|NL) (?:East|West|Central) title',q): return 'MLB'
    if re.search(r'(?i)Premier League|Champions League|La Liga|Serie A|Bundesliga|Ligue 1|MLS|FA Cup|Carabao Cup',q): return 'football'
    if re.search(r'(?i)ESL Pro League|Counter-Strike',q): return 'CS2'
    if re.search(r'(?i)LCK|LPL|League of Legends|\bLoL\b',q): return 'LoL'
def aliases(team,c):
    a={norm(team)}
    if c=='NBA': a|={norm(x) for x in NBA if norm(team).endswith(norm(x))}
    if c=='MLB': a|={norm(x) for x in MLB if norm(team).endswith(norm(x))}
    return a
def root():
    x=[p.parent for p in I.rglob('labels') if (p/'market_targets.parquet').exists() and (p.parent/'orderbook').is_dir()]
    if len(x)!=1: raise RuntimeError(x)
    return x[0]
def main():
    r=root(); d=pl.read_parquet(r/'labels/market_targets.parquet').to_pandas(); d['end_dt']=pd.to_datetime(d.end_date,errors='coerce',utc=True)
    fs=[]
    for z in d.itertuples(index=False):
        t=fteam(z.question); c=cat(z.question or '')
        if t and c: fs.append((c,t,z.condition_id,z.question))
    cs=[]; lo=pd.Timestamp('2026-03-06',tz='UTC'); hi=pd.Timestamp('2026-03-27',tz='UTC')
    for z in d.itertuples(index=False):
        if pd.isna(z.end_dt) or not lo<=z.end_dt<hi: continue
        q=z.question or ''; m=re.match(r'^([^:]+?) vs\. ([^:]+?)$',q)
        if m:
            cs += [('NBA',z,m.group(1),m.group(2),None),('MLB',z,m.group(1),m.group(2),None)]
        m=re.match(r'(?i)^Will (.+?) win on 2026-\d\d-\d\d\??$',q)
        if m: cs.append(('football',z,None,None,norm(m.group(1))))
        for c,prefix in [('CS2','Counter-Strike'),('LoL','LoL')]:
            m=re.match(rf'(?i)^{prefix}: (.+?) vs (.+?) \(BO\d\) -',q)
            if m and not re.search(r'(?i)Map \d|Handicap',q): cs.append((c,z,m.group(1),m.group(2),None))
    pairs=[]
    for c,t,fc,fq in fs:
        aa=aliases(t,c)
        for cc,z,a,b,dated in cs:
            if cc!=c: continue
            yes=None
            if dated in aa: yes=True
            if a is not None:
                if norm(a) in aa: yes=True
                elif norm(b) in aa: yes=False
            if yes is not None: pairs.append({'category':c,'team':t,'match_condition_id':z.condition_id,'match_question':z.question,'match_end_date':z.end_date,'team_is_yes':yes,'future_condition_id':fc,'future_question':fq})
    p=pd.DataFrame(pairs).drop_duplicates(); ids=pl.Series(pd.unique(p[['match_condition_id','future_condition_id']].values.ravel()).tolist())
    print(p.groupby('category').agg(pairs=('team','size'),teams=('team','nunique'),matches=('match_condition_id','nunique'),futures=('future_condition_id','nunique')))
    if O.exists(): shutil.rmtree(O)
    (O/'metadata').mkdir(parents=True); (O/'orderbook').mkdir()
    p.to_csv(O/'metadata/verified_pairs.csv',index=False)
    pl.from_pandas(d[d.condition_id.isin(ids.to_list())].drop(columns=['end_dt'])).write_parquet(O/'metadata/candidate_metadata.parquet')
    report=[]; files=sorted((r/'orderbook').glob('orderbook_*.parquet'))
    for n,s in enumerate(files,1):
        t=O/'orderbook'/s.name; print(f'[{n}/{len(files)}] {s.name}',flush=True)
        q=pl.scan_parquet(s).filter(pl.col('market_id').is_in(ids)); q.sink_parquet(t,compression='zstd',maintain_order=True)
        rows=pl.scan_parquet(t).select(pl.len()).collect().item()
        if not rows: t.unlink()
        report.append({'file':s.name,'rows':rows,'size_mb':t.stat().st_size/2**20 if t.exists() else 0}); del q; gc.collect()
    pl.DataFrame(report).write_csv(O/'extraction_report.csv')
    a=shutil.make_archive('/kaggle/working/linked_cross_market_sports','zip',O.parent,O.name)
    print(a,Path(a).stat().st_size/2**30,'GiB')
if __name__=='__main__': main()
