from pathlib import Path
import re

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path("polymarket_sports")
meta = pq.read_table(ROOT / "metadata" / "sports_markets.parquet").to_pandas()
q = meta.question.fillna("")

groups = {
    "NBA": r"NBA|basketball",
    "MLB": r"MLB|baseball",
    "football": r"NFL|Premier League|Champions League|La Liga|Serie A|Bundesliga|Ligue 1|MLS|soccer|football",
    "CS2": r"Counter-Strike|CS2|ESL Pro League",
}
future_re = re.compile(r"^Will (.+?) (?:win|become|finish|make|reach|qualify)", re.I)
match_re = re.compile(r"(?:^|: )(.+?) vs (.+?)(?: \(| - |$)", re.I)

for name, pattern in groups.items():
    subset = meta[q.str.contains(pattern, case=False, regex=True)].copy()
    futures = []
    matches = []
    for row in subset.itertuples(index=False):
        text = row.question
        fm = future_re.search(text)
        mm = match_re.search(text)
        if fm and re.search(r"win|champion|playoff|final|division|conference|league|cup|season|tournament", text, re.I):
            futures.append((fm.group(1).strip(), row.condition_id, text))
        if mm and not re.search(r"Map \d|Game \d|Handicap|Total|Spread|Both teams|Goals|Points|Kills", text, re.I):
            matches.append((mm.group(1).strip(), mm.group(2).strip(), row.condition_id, text))
    print(f"\n=== {name}: {len(subset)} metadata rows, {len(futures)} future candidates, {len(matches)} main-match candidates ===")
    for x in futures[:30]: print("FUTURE", x)
    print("MATCH SAMPLES")
    for x in matches[:20]: print(x)
