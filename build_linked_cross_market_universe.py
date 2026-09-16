from pathlib import Path
import re
import pandas as pd
import pyarrow.parquet as pq

META = Path("kaggle_metadata_full/market_targets.parquet")
OUT = Path("cross_market_universe")

NBA = ["Hawks","Celtics","Nets","Hornets","Bulls","Cavaliers","Mavericks","Nuggets","Pistons","Warriors","Rockets","Pacers","Clippers","Lakers","Grizzlies","Heat","Bucks","Timberwolves","Pelicans","Knicks","Thunder","Magic","76ers","Suns","Trail Blazers","Kings","Spurs","Raptors","Jazz","Wizards"]
MLB = ["Diamondbacks","Athletics","Braves","Orioles","Red Sox","Cubs","White Sox","Reds","Guardians","Rockies","Tigers","Astros","Royals","Angels","Dodgers","Marlins","Brewers","Twins","Mets","Yankees","Phillies","Pirates","Padres","Giants","Mariners","Cardinals","Rays","Rangers","Blue Jays","Nationals"]

def norm(x):
    x=x.lower().replace("�","-")
    x=re.sub(r"\b(the|fc|cf|afc|cfc)\b"," ",x)
    return re.sub(r"[^a-z0-9]+"," ",x).strip()

def future_team(q):
    m=re.match(r"(?i)^Will (?:the )?(.+?) (?:win|make|finish|reach|qualify|be relegated)\b",q)
    return m.group(1).strip() if m else None

def category(q):
    if re.search(r"(?i)NBA|Eastern Conference Finals|Western Conference Finals",q): return "NBA"
    if re.search(r"(?i)World Series|American League Championship|National League Championship|\b(?:AL|NL) (?:East|West|Central) title",q): return "MLB"
    if re.search(r"(?i)Premier League|Champions League|La Liga|Serie A|Bundesliga|Ligue 1|MLS|FA Cup|Carabao Cup",q): return "football"
    if re.search(r"(?i)ESL Pro League|Counter-Strike",q): return "CS2"
    if re.search(r"(?i)LCK|LPL|League of Legends|\bLoL\b",q): return "LoL"
    return None

def aliases(team, cat):
    out={norm(team)}
    if cat=="NBA": out |= {norm(x) for x in NBA if norm(team).endswith(norm(x))}
    return {x for x in out if x}

def main():
    d=pq.read_table(META).to_pandas()
    d["end_dt"]=pd.to_datetime(d.end_date,errors="coerce",utc=True)
    futures=[]
    for r in d.itertuples(index=False):
        team=future_team(r.question or ""); cat=category(r.question or "")
        if team and cat: futures.append((cat,team,r.condition_id,r.question,r.clob_token_id_yes,r.clob_token_id_no))
    candidates=[]
    window_start=pd.Timestamp("2026-03-06",tz="UTC"); window_end=pd.Timestamp("2026-03-27",tz="UTC")
    for r in d.itertuples(index=False):
        if pd.isna(r.end_dt) or not (window_start <= r.end_dt < window_end): continue
        q=r.question or ""
        m=re.match(r"^([^:]+?) vs\. ([^:]+?)$",q)
        if m:
            for cat in ("NBA","MLB"):
                candidates.append((cat,r,m.group(1),m.group(2),None))
        m=re.match(r"(?i)^Will (.+?) win on 2026-\d\d-\d\d\??$",q)
        if m: candidates.append(("football",r,None,None,norm(m.group(1))))
        for cat,prefix in (("CS2","Counter-Strike"),("LoL","LoL")):
            m=re.match(rf"(?i)^{prefix}: (.+?) vs (.+?) \(BO\d\) -",q)
            if m and not re.search(r"(?i)Map \d|Handicap",q): candidates.append((cat,r,m.group(1),m.group(2),None))
    full_teams={c:{norm(t) for cc,t,*_ in futures if cc==c} for c in ("NBA","MLB")}
    nba_nicknames={norm(x) for x in NBA}
    pairs=[]
    for cat,team,fcid,fq,fy,fn in futures:
        aa=aliases(team,cat)
        for candidate_cat,r,side1,side2,dated_team in candidates:
            if candidate_cat != cat: continue
            q=r.question or ""; team_is_yes=None
            if dated_team is not None and dated_team in aa: team_is_yes=True
            if side1 is not None:
                ns=[norm(side1),norm(side2)]
                if cat=="MLB" and not all(x in full_teams["MLB"] for x in ns): continue
                if cat=="NBA" and not all(x in nba_nicknames for x in ns): continue
                if ns[0] in aa: team_is_yes=True
                elif ns[1] in aa: team_is_yes=False
            if team_is_yes is None: continue
            pairs.append({"category":cat,"team":team,"match_condition_id":r.condition_id,"match_question":q,"match_end_date":r.end_date,"team_is_yes":team_is_yes,"future_condition_id":fcid,"future_question":fq})
    p=pd.DataFrame(pairs).drop_duplicates()
    OUT.mkdir(exist_ok=True)
    p.to_csv(OUT/"verified_pairs.csv",index=False)
    print(p.groupby("category").agg(pairs=("team","size"),teams=("team","nunique"),matches=("match_condition_id","nunique"),futures=("future_condition_id","nunique")))
    print("unique IDs",pd.unique(p[["match_condition_id","future_condition_id"]].values.ravel()).size)
    print(p.groupby("category").head(5)[["category","team","match_question","future_question"]].to_string(index=False))

if __name__=="__main__": main()
