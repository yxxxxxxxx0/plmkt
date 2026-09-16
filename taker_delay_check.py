"""Read-only audit of Polymarket's live ``seconds_delay`` market field.

No orders are created. Gamma discovers current markets and the CLOB market
endpoint reports the matching delay. Dataset-category mode checks one current
representative market for every sports group in the downloaded dataset.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
import re
from pathlib import Path

import requests

GAMMA_SPORTS = "https://gamma-api.polymarket.com/sports"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_MARKET = "https://clob.polymarket.com/markets/{}"

CATEGORY_SPORT_CODE = {
    "Counter-Strike / CS2": "cs2", "NHL": "nhl", "NBA": "nba",
    "Formula 1": "f1", "Dota 2": "dota2", "Valorant": "val",
    "Esports (generic)": "cs2", "League of Legends": "lol",
    "Cricket": "cricipl", "Premier League": "epl", "MLS": "mls",
    "UFC": "ufc", "NFL": "nfl", "NCAA": "ncaab", "Ligue 1": "fl1",
    "Champions League": "ucl", "La Liga": "lal", "Serie A": "sea",
    "Bundesliga": "bun", "EPL": "epl", "Baseball": "mlb",
    "Football": "nfl", "Basketball": "nba", "Rugby": "ruprem",
    "WNBA": "wnba", "MLB": "mlb",
}


def get_json(session: requests.Session, url: str, params=None, retries: int = 3):
    error = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(error)


def accepting_market(events: list[dict]) -> tuple[dict, dict, str] | None:
    candidates = []
    for event in events:
        for market in event.get("markets", []):
            if (market.get("conditionId") and not market.get("closed", False)
                    and market.get("active", True) and market.get("acceptingOrders", True)):
                text = f"{event.get('title', '')} | {market.get('question', '')}"
                matchup = bool(re.search(r"\bvs\.?\b|\bversus\b", text, re.I))
                game_like = matchup or bool(re.search(
                    r"grand prix|\bgame\s*\d*\b|\bmatch\b|\bmap\s*\d+\b|\bfight\b",
                    text, re.I,
                ))
                # Actual scheduled contests outrank long-dated futures. Preserve
                # source order within a class because Gamma generally ranks relevance.
                candidates.append((2 if matchup else 1 if game_like else 0, event, market,
                                   "matchup" if matchup else "game-like" if game_like else "future/prop"))
    if not candidates:
        return None
    _, event, market, kind = max(candidates, key=lambda row: row[0])
    return event, market, kind


def audit_dataset_categories(dataset_root: Path, output: Path) -> list[dict]:
    coverage = dataset_root / "reports" / "sports_coverage_by_classifier.csv"
    categories = [r["sports_group"] for r in csv.DictReader(coverage.open(encoding="utf-8"))]
    session = requests.Session()
    sports = get_json(session, GAMMA_SPORTS)
    by_code = {row["sport"]: row for row in sports}
    event_cache: dict[int, list[dict]] = {}
    clob_cache: dict[str, dict] = {}
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    results: list[dict] = []

    for category in categories:
        code = CATEGORY_SPORT_CODE.get(category)
        base = {
            "checked_utc": now, "dataset_category": category, "sport_code": code or "",
            "official_sport_name": "", "primary_tag_id": "", "event_slug": "",
            "question": "", "market_type": "", "condition_id": "", "seconds_delay": "", "status": "",
        }
        sport = by_code.get(code) if code else None
        if not sport:
            base["status"] = "no official sport-code mapping"
            results.append(base)
            continue
        tag = int(sport["primaryTagId"])
        base.update(official_sport_name=sport["name"], primary_tag_id=tag)
        try:
            if tag not in event_cache:
                event_cache[tag] = get_json(
                    session, GAMMA_EVENTS,
                    {"tag_id": tag, "active": "true", "closed": "false", "limit": 100},
                )
            chosen = accepting_market(event_cache[tag])
            if chosen is None:
                base["status"] = "no currently accepting market"
            else:
                event, market, market_type = chosen
                cid = market["conditionId"]
                if cid not in clob_cache:
                    clob_cache[cid] = get_json(session, CLOB_MARKET.format(cid))
                base.update(
                    event_slug=event.get("slug", ""), question=market.get("question", ""),
                    market_type=market_type,
                    condition_id=cid, seconds_delay=clob_cache[cid].get("seconds_delay", ""),
                    status="checked",
                )
        except Exception as exc:
            base["status"] = f"error: {exc}"
        results.append(base)
        print(f"{category:<24} {str(base['seconds_delay']):>4}s  {base['status']:<30} {base['question'][:70]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-categories", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path("polymarket_sports"))
    parser.add_argument("--output", type=Path,
                        default=Path("polymarket_sports/reports/current_taker_delay_audit.csv"))
    args = parser.parse_args()
    if not args.dataset_categories:
        parser.error("use --dataset-categories")
    rows = audit_dataset_categories(args.dataset_root, args.output)
    checked = [r for r in rows if r["status"] == "checked"]
    delays: dict[object, int] = {}
    for row in checked:
        delay = row["seconds_delay"]
        delays[delay] = delays.get(delay, 0) + 1
    print(f"\nChecked {len(checked)}/{len(rows)} categories; report: {args.output}")
    print("Delay tally:", delays)


if __name__ == "__main__":
    main()
