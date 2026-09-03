"""
Discovers and classifies Polymarket markets for a list of MLB game event
slugs, using the Gamma API only (no hard-coded condition IDs or token IDs —
everything is read live).

For each event slug, this fetches the event and every market belonging to
it, then classifies each market using Polymarket's own `sportsMarketType`
field. The mapping was read off live events, not guessed:

    moneyline                        -> "moneyline"
    spreads                          -> "spread"
    totals                           -> "total"
    nrfi                             -> "first_inning_run"
    baseball_team_first_five_spread   -> "first_five_spread"
    baseball_team_first_five_total    -> "first_five_total"
    baseball_game_extra_innings       -> "extra_innings"

The first four are the "core" set. The last three were previously dropped on
the floor even though they carry live order books -- a 2026-08-27 event has
17 order-book markets, of which the core set is only 9. `core_only=True`
restores the old, narrower behaviour.

Anything with no recognised `sportsMarketType`, or with no order book, is
still skipped, and `discover_target_markets` reports what it skipped rather
than staying silent about it.
"""

import json
import re

import requests

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

CORE_SPORTS_MARKET_TYPES = {
    "moneyline": "moneyline",
    "spreads": "spread",
    "totals": "total",
    "nrfi": "first_inning_run",
}

EXTRA_SPORTS_MARKET_TYPES = {
    "baseball_team_first_five_spread": "first_five_spread",
    "baseball_team_first_five_total": "first_five_total",
    "baseball_game_extra_innings": "extra_innings",
}

ALL_SPORTS_MARKET_TYPES = {**CORE_SPORTS_MARKET_TYPES, **EXTRA_SPORTS_MARKET_TYPES}

# market types that carry a numeric line worth parsing
LINED_TYPES = {"spread", "total", "first_five_spread", "first_five_total"}

# kept for backwards compatibility with anything importing the old name
TARGET_SPORTS_MARKET_TYPES = CORE_SPORTS_MARKET_TYPES

# Pulls the trailing numeric line out of e.g. "Spread -1.5", "O/U 9.5",
# "Spread: New York Yankees (-1.5)".
_LINE_RE = re.compile(r"\(?(-?\d+(?:\.\d+)?)\)?\s*$")


def _parse_line(market):
    for text in (market.get("groupItemTitle"), market.get("question")):
        if not text:
            continue
        m = _LINE_RE.search(text.strip())
        if m:
            return float(m.group(1))
    return None


def fetch_event(slug):
    resp = requests.get(GAMMA_EVENTS_URL, params={"slug": slug}, timeout=15)
    resp.raise_for_status()
    rows = resp.json()
    return rows[0] if rows else None


def classify_market(market, core_only=False):
    """Returns (market_type, line) for a market we want, else None."""
    table = CORE_SPORTS_MARKET_TYPES if core_only else ALL_SPORTS_MARKET_TYPES
    market_type = table.get(market.get("sportsMarketType"))
    if market_type is None:
        return None
    line = _parse_line(market) if market_type in LINED_TYPES else None
    return market_type, line


def build_target_market(match, market, market_type, line):
    outcomes = json.loads(market.get("outcomes") or "[]")
    token_ids = json.loads(market.get("clobTokenIds") or "[]")

    errors = []
    if len(outcomes) != 2:
        errors.append(f"expected 2 outcomes, got {len(outcomes)}")
    if len(token_ids) != 2:
        errors.append(f"expected 2 token ids, got {len(token_ids)} (only one side visible)")

    return {
        "event_slug": match["slug"],
        "match": match["match"],
        "scheduled_time_hkt": match["time_hkt"],
        "sport": "MLB",
        "market_type": market_type,
        "market_question": market.get("question"),
        "market_slug": market.get("slug"),
        "line": line,
        "condition_id": market.get("conditionId"),
        "closed": bool(market.get("closed")),
        "accepting_orders": market.get("acceptingOrders", True),
        "outcomes": outcomes,
        "token_ids": token_ids,
        "yes_asset_id": token_ids[0] if len(token_ids) > 0 else None,
        "no_asset_id": token_ids[1] if len(token_ids) > 1 else None,
        "errors": errors,
    }


def discover_target_markets(matches, core_only=False):
    """Queries the Gamma API for every match's event.

    Returns (target_markets, events_found, discovery_errors). Markets that
    were skipped are reported too, so silently missing a whole category of
    order book (which is what happened to the first-five-innings and
    extra-innings markets) is visible instead of invisible.
    """
    target_markets = []
    events_found = 0
    discovery_errors = []
    skipped = {}

    for match in matches:
        try:
            event = fetch_event(match["slug"])
        except Exception as e:
            discovery_errors.append(f"{match['match']} ({match['slug']}): fetch failed: {e}")
            continue
        if event is None:
            discovery_errors.append(f"{match['match']} ({match['slug']}): event not found")
            continue
        events_found += 1

        for market in event.get("markets", []):
            classified = classify_market(market, core_only=core_only)
            if classified is None:
                smt = market.get("sportsMarketType") or "(none)"
                if market.get("enableOrderBook"):
                    skipped[smt] = skipped.get(smt, 0) + 1
                continue
            if not market.get("enableOrderBook"):
                discovery_errors.append(
                    f"{match['match']} / {market.get('question')!r}: no order book, skipped")
                continue
            market_type, line = classified
            record = build_target_market(match, market, market_type, line)
            if record["errors"]:
                discovery_errors.append(
                    f"{match['match']} / {market_type} {record['market_question']!r}: "
                    + "; ".join(record["errors"])
                )
            target_markets.append(record)

    if skipped:
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(skipped.items()))
        discovery_errors.append(
            f"[note] skipped {sum(skipped.values())} order-book markets by type: {detail}")

    return target_markets, events_found, discovery_errors
