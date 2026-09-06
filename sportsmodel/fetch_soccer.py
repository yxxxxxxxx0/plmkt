"""
Fetch historical soccer results (openfootball, free, no key).

football-data.co.uk -- the usual source, and the one that carries bookmaker
odds -- returns 503 from this machine, so results come from openfootball
instead. That costs the ability to compare directly against closing odds, so
the soccer test here measures MODEL ACCURACY, not whether the model beats a
market. Those are different claims and are kept apart deliberately.

    python fetch_soccer.py
Output: data/soccer.csv  (date, league, season, home, away, hg, ag, result)
"""
import csv
import io
import json
import os
import sys
import time

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "soccer.csv")
ROOT = "https://raw.githubusercontent.com/openfootball/football.json/master"
LEAGUES = ["en.1", "es.1", "de.1", "it.1", "fr.1"]
SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
           "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (research)"
    rows = []
    for season in SEASONS:
        for lg in LEAGUES:
            url = f"{ROOT}/{season}/{lg}.json"
            try:
                r = s.get(url, timeout=30)
            except requests.RequestException:
                continue
            if r.status_code != 200:
                continue
            try:
                data = r.json()
            except json.JSONDecodeError:
                continue
            n = 0
            for m in data.get("matches", []):
                sc = (m.get("score") or {}).get("ft")
                if not sc or len(sc) != 2:
                    continue
                hg, ag = int(sc[0]), int(sc[1])
                rows.append({
                    "date": m.get("date"), "league": lg, "season": season,
                    "home": m.get("team1"), "away": m.get("team2"),
                    "hg": hg, "ag": ag,
                    "result": "H" if hg > ag else ("A" if ag > hg else "D"),
                })
                n += 1
            print(f"  {season} {lg}: {n} matches", flush=True)
            time.sleep(0.1)

    rows.sort(key=lambda r: (r["date"] or "", r["league"]))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {}
    for r in rows:
        res[r["result"]] = res.get(r["result"], 0) + 1
    tot = len(rows)
    print(f"\n{tot:,} matches -> {OUT}")
    print("  base rates: " + "  ".join(
        f"{k}={v/tot*100:.1f}%" for k, v in sorted(res.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
