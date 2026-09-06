"""
Fetch historical MLB results from the MLB Stats API (free, no key).

Regular-season games only, and only those that finished with a decision.
MLB has no draws in the modern regular season, so the outcome is binary --
which is why the soccer and baseball tests below are scored differently.

    python fetch_mlb.py --seasons 2018 2019 2021 2022 2023 2024 2025 2026
Output: data/mlb.csv  (date, season, home, away, hr, ar, result)
"""
import argparse
import csv
import os
import sys
import time

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "mlb.csv")
API = "https://statsapi.mlb.com/api/v1/schedule"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int,
                    default=[2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    s = requests.Session()
    rows = []
    for yr in args.seasons:
        r = s.get(API, params={"sportId": 1, "startDate": f"{yr}-03-01",
                               "endDate": f"{yr}-11-15", "gameType": "R"},
                  timeout=60)
        if r.status_code != 200:
            print(f"  {yr}: HTTP {r.status_code}", flush=True)
            continue
        n = 0
        for day in r.json().get("dates", []):
            for g in day.get("games", []):
                if (g.get("status") or {}).get("detailedState") != "Final":
                    continue
                t = g.get("teams") or {}
                h, a = t.get("home") or {}, t.get("away") or {}
                hs, as_ = h.get("score"), a.get("score")
                if hs is None or as_ is None or hs == as_:
                    continue
                rows.append({
                    "date": day.get("date"), "season": yr,
                    "home": (h.get("team") or {}).get("name"),
                    "away": (a.get("team") or {}).get("name"),
                    "hr": hs, "ar": as_,
                    "result": "H" if hs > as_ else "A",
                })
                n += 1
        print(f"  {yr}: {n} games", flush=True)
        time.sleep(0.2)

    rows.sort(key=lambda r: (r["date"] or "", r["home"] or ""))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    hw = sum(1 for r in rows if r["result"] == "H")
    print(f"\n{len(rows):,} games -> {args.out}")
    print(f"  home win rate {hw/len(rows)*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
