"""
Builds a single combined multi-match index from every rebuilt game in
data/uniform/*.json (see rebuild_multi.py), for the multi-match viewer.

For each game this adds the human-readable metadata the raw recorder never
stored (real match/league/date info, and -- critically -- what a token's
recorded "YES"/"NO" outcome actually corresponds to, e.g. a team name or
"Over"/"Under") by fetching the event once from the Gamma API and matching
each asset's condition_id to its market's outcomes list (outcomes[0] is
always the YES token, outcomes[1] the NO token -- the same assumption
discovery.py / multi_recorder.py make when subscribing).

Gamma responses are cached to data/discovery_cache.json so re-running this
after rebuilding more games doesn't re-fetch events it already has.

Ticks here are exact: each match's tick_times is carried straight through from
the rebuild, with no resampling of its own.

Both the uniform files and the payloads are handled as streams, never loaded
whole -- see the streaming-reader section below for why.

WARNING -- payload size. With the live_recorder jsonl source, one game's
payload is ~2.5 GB even at the default --depth 30, because 4.2M runs is simply
a lot of runs. No browser will fetch that. Depth is not the lever it once was:
books average ~23 levels/side, so --depth 30 trims almost nothing, and even a
brutal --depth 5 only reaches ~590 MB. Cutting this to a fetchable size means
reducing the number of RUNS (e.g. sampling the 100ms grid rather than every
microsecond-level change), which is a viewer behaviour change and is
deliberately not done here.

Usage:
    python build_match_index.py
"""

import argparse
import glob
import json
import os
import re

import ijson
import orjson
import requests

import discovery

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UNIFORM_DIR = os.path.join(DATA_DIR, "uniform")
DISCOVERY_CACHE_PATH = os.path.join(DATA_DIR, "discovery_cache.json")

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

_SLUG_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})$")
_TEAMS_SPLIT_RE = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)


def parse_line(raw):
    """The depth CSV round-trips `line` as text, so it arrives as a string
    ('-1.5', '8.5') or '' for markets that have no line. Coerce to a number
    so markets sort numerically: sorting the raw strings puts O/U 10.5 ahead
    of O/U 8.5, and -1.5 ahead of -2.5. It also removes a latent TypeError,
    since the old sort key compared int 0 against those strings whenever a
    market type mixed lined and unlined markets."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def fetch_event_cached(slug, cache):
    if slug in cache:
        return cache[slug]
    resp = requests.get(GAMMA_EVENTS_URL, params={"slug": slug}, timeout=15)
    resp.raise_for_status()
    rows = resp.json()
    event = rows[0] if rows else None
    cache[slug] = event
    return event


def build_condition_outcomes(event):
    """condition_id -> (yes_label, no_label, market_type, market_question, line, group_item_title)"""
    out = {}
    for m in event.get("markets", []) if event else []:
        smt = m.get("sportsMarketType")
        market_type = discovery.ALL_SPORTS_MARKET_TYPES.get(smt)
        if market_type is None:
            continue
        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
        except json.JSONDecodeError:
            outcomes = []
        if len(outcomes) != 2:
            continue
        out[m["conditionId"]] = {
            "yes_label": outcomes[0],
            "no_label": outcomes[1],
            "market_type": market_type,
            "market_question": m.get("question"),
            "group_item_title": m.get("groupItemTitle"),
        }
    return out


def market_label_for(market_type, market_question, group_item_title):
    if market_type == "moneyline":
        return "Moneyline"
    if market_type == "first_inning_run":
        return "First Inning Run"
    if market_type == "extra_innings":
        return "Extra Innings"
    if market_type in ("first_five_spread", "first_five_total"):
        # the question names both the team and the line; groupItemTitle alone
        # is ambiguous the same way plain "Spread -1.5" is
        return market_question or group_item_title
    if market_type == "total":
        # groupItemTitle ("O/U 9.5") is already unique per line for a match.
        return group_item_title or market_question
    if market_type == "spread":
        # groupItemTitle alone ("Spread -1.5") is NOT unique -- both teams
        # can have a market at the same magnitude (Guardians -1.5 vs Angels
        # -1.5 are different conditions). The full question always names
        # the team, so use it instead.
        return market_question or group_item_title
    return market_question or group_item_title


def safe_label(label, market_type, condition_id):
    """A market must always be selectable. Non-MLB events (and any market
    whose Gamma metadata we could not resolve) have no question text to fall
    back on, which produced a literal `null` in the Market dropdown."""
    if label:
        return label
    if market_type and market_type != "other":
        return market_type.replace("_", " ").title()
    return f"Market {(condition_id or '?')[:10]}"


# ---------------------------------------------------------------------------
# Streaming reader for data/uniform/*.json
#
# These files are no longer small. With the live_recorder jsonl source, one
# game is ~2.7 GB of JSON (4.2M runs x ~46 price levels each). json.load() on
# that would need roughly 10-20x the file size in Python objects -- tens of GB
# -- so the whole file is never materialised. Instead:
#
#   load_uniform_header()  reads everything EXCEPT `runs` (scalars, per-asset
#                          metadata, tick_times) -- tens of MB, not GB.
#   iter_uniform_runs()    yields (asset_id, runs) one asset at a time, so only
#                          one asset's books are live at once.
#
# Both rely on `runs` being the last key in the object, which is how both
# rebuild_duckdb.write_game_json and rebuild_multi's json.dump emit it.
# use_float=True matters: ijson defaults to Decimal, which is far slower and
# much larger, and would change the numbers written to the payload.
# ---------------------------------------------------------------------------

def load_uniform_header(path):
    """Everything in a uniform file except `runs`."""
    header = {"tick_times": [], "assets": {}}
    with open(path, "rb") as f:
        for prefix, event, value in ijson.parse(f, use_float=True):
            if prefix == "runs":
                break
            if event not in ("string", "number", "boolean", "null"):
                continue
            if "." not in prefix:
                header[prefix] = value
            elif prefix == "tick_times.item":
                header["tick_times"].append(value)
            elif prefix.startswith("assets."):
                _, asset_id, field = prefix.split(".", 2)
                header["assets"].setdefault(asset_id, {})[field] = value
    missing = {"event_slug", "start_ts", "end_ts"} - set(header)
    if missing:
        raise SystemExit(f"{path}: malformed uniform file, missing {sorted(missing)}")
    return header


def iter_uniform_runs(path):
    """Yields (asset_id, runs) for each asset, one asset at a time."""
    with open(path, "rb") as f:
        yield from ijson.kvitems(f, "runs", use_float=True)


def grid_times(start_ts, end_ts, grid):
    """The resampled tick axis: uniform `grid`-second steps over the window."""
    out = []
    n = int((end_ts - start_ts) / grid) + 1
    for k in range(n + 1):
        t = start_ts + k * grid
        if t > end_ts:
            break
        out.append(round(t, 6))
    if not out or out[-1] < round(end_ts, 6):
        out.append(round(end_ts, 6))
    return out


def resample_runs(runs, times):
    """Reduces one asset's runs to at most one state per entry in `times`.

    The archive keeps every real change -- 4.2M runs for one game, which is
    both an unfetchable download and far too many for the viewer to draw. This
    walks the (contiguous, sorted) runs and the (sorted) grid together in one
    pass, keeping the book that was in force at each grid instant and merging
    consecutive grid steps whose book did not change. That merge is why the
    result is usually far smaller than len(times): during quiet stretches
    nothing is emitted at all.

    Lossy in time by construction -- changes between two grid instants are not
    represented -- and lossless in depth. data/uniform/*.json keeps everything.
    """
    if not runs:
        return []
    out = []
    i = 0
    for t in times:
        while i + 1 < len(runs) and runs[i]["t_end"] <= t:
            i += 1
        r = runs[i]
        if out and out[-1]["bids"] == r["bids"] and out[-1]["asks"] == r["asks"]:
            out[-1]["t_end"] = t
            continue
        out.append({
            "t_start": t, "t_end": t,
            "bids": r["bids"], "asks": r["asks"],
            "best_bid": r["best_bid"], "best_ask": r["best_ask"],
        })
    # Each emitted run must span up to the next one, and the last to the end.
    for k in range(len(out) - 1):
        out[k]["t_end"] = out[k + 1]["t_start"]
    out[-1]["t_end"] = max(out[-1]["t_end"], times[-1])
    return out


def cap_runs(runs, depth_cap):
    """Trims each run's book to the top `depth_cap` levels per side for the
    embedded viewer copy. The authoritative full-depth book stays in
    data/uniform/*.json -- this only bounds what gets inlined into the HTML,
    since full depth across every match is far too large to embed."""
    if not depth_cap:
        return runs
    out = []
    for r in runs:
        out.append({**r, "bids": r["bids"][:depth_cap], "asks": r["asks"][:depth_cap]})
    return out


def build_match(slug, uniform_data, event, grid=0):
    condition_outcomes = build_condition_outcomes(event) if event else {}

    date = None
    m = _SLUG_DATE_RE.search(slug)
    if m:
        date = m.group(1)

    match_name = (event or {}).get("title") or uniform_data.get("match") or slug
    sport = None
    if event and event.get("series"):
        sport = event["series"][0].get("title")
    sport = sport or "MLB"

    teams = _TEAMS_SPLIT_RE.split(match_name)
    teams = [t.strip() for t in teams if t.strip()]

    # market_type -> list of markets; each market -> list of tokens
    market_types = {}
    tokens_flat = {}

    assets = uniform_data.get("assets", {})

    # group assets by (market_type, condition_id) -> that's one specific market
    markets_by_key = {}
    for asset_id, meta in assets.items():
        condition_id = meta.get("condition_id")
        co = condition_outcomes.get(condition_id, {})
        market_type = co.get("market_type") or meta.get("market_type")
        market_question = co.get("market_question") or meta.get("market_question")
        group_item_title = co.get("group_item_title")

        outcome = meta.get("outcome")  # "YES" / "NO" as recorded
        if outcome == "YES":
            label = co.get("yes_label", "Yes")
        elif outcome == "NO":
            label = co.get("no_label", "No")
        else:
            label = outcome or asset_id[:8]

        market_key = condition_id or f"{market_type}:{market_question}:{meta.get('line')}"
        market_label = safe_label(
            market_label_for(market_type, market_question, group_item_title),
            market_type, condition_id)

        markets_by_key.setdefault((market_type, market_key), {
            "market_type": market_type,
            "market_key": market_key,
            "market_label": market_label,
            "market_question": market_question,
            "line": parse_line(meta.get("line")),
            "condition_id": condition_id,
            "tokens": [],
        })["tokens"].append({
            "asset_id": asset_id,
            "label": label,
            "outcome": outcome,
        })

        # No "runs" here: the per-tick bulk never enters the match structure.
        # write_payload() streams it straight from the uniform file to disk.
        tokens_flat[asset_id] = {
            "label": label,
            "outcome": outcome,
            "market_type": market_type,
            "market_label": market_label,
        }

    for (market_type, _key), market in markets_by_key.items():
        market["tokens"].sort(key=lambda t: t["outcome"] != "YES")  # YES first
        market_types.setdefault(market_type, []).append(market)

    for mt, mlist in market_types.items():
        # numeric line first (unlined markets last), then label for stability
        mlist.sort(key=lambda m: (m["line"] is None, m["line"] if m["line"] is not None else 0.0,
                                   m["market_label"] or ""))

    return {
        "event_slug": slug,
        "event_id": (event or {}).get("id"),
        "sport": sport,
        "date": date,
        "match_name": match_name,
        "teams": teams,
        "start_ts": uniform_data["start_ts"],
        "end_ts": uniform_data["end_ts"],
        "kickoff_ts": uniform_data.get("kickoff_ts"),
        "game_end_ts": uniform_data.get("game_end_ts"),
        "game_status": uniform_data.get("game_status"),
        # The viewer's tick axis. When resampling, the slider must step over
        # the same grid the runs were reduced onto -- keeping the archive's
        # 1.8M-entry axis would give a slider whose steps mostly show an
        # identical book, and 30 MB of tick_times to download for nothing.
        "tick_times": (grid_times(uniform_data["start_ts"], uniform_data["end_ts"], grid)
                       if grid else uniform_data["tick_times"]),
        "n_ticks": len(grid_times(uniform_data["start_ts"], uniform_data["end_ts"], grid)
                       if grid else uniform_data["tick_times"]),
        "tick_grid_seconds": grid or None,
        "n_ticks_archive": len(uniform_data["tick_times"]),
        "n_real_ticks_in_window": uniform_data.get("n_real_ticks_in_window"),
        "n_real_ticks_before_window": uniform_data.get("n_real_ticks_before_window"),
        "n_real_ticks_after_window": uniform_data.get("n_real_ticks_after_window"),
        "market_types": market_types,
        "tokens": tokens_flat,
    }


def write_payload(uniform_path, payload_path, match, depth_cap=0, grid=0):
    """Writes one match's payload, streaming the runs asset-by-asset.

    The manifest holds everything needed to populate the selectors and the
    overview table -- match identity, market/token structure, tick counts --
    but none of the bulk. The payload holds the per-tick data (tick_times plus
    every token's run-length-encoded book) and is fetched only when that match
    is actually opened.

    Written incrementally for the same reason the uniform file is read
    incrementally: at this data volume one match's runs are gigabytes, so
    neither the assembled dict nor json.dumps' output string can be held whole.

    Written with orjson rather than json.dump: json.dump defaults to
    separators=(', ', ': '), and at ~4M runs of ~50 numbers each those two
    padding bytes per element added ~280 MB of pure whitespace -- enough to
    make the depth-capped payload LARGER than the full-depth file it came from.
    """
    wanted = set(match["tokens"])
    times = match["tick_times"]
    n_runs = 0
    with open(payload_path, "wb") as f:
        f.write(b'{"event_slug":')
        f.write(orjson.dumps(match["event_slug"]))
        f.write(b',"tick_times":')
        f.write(orjson.dumps(times))
        f.write(b',"runs":{')
        first = True
        for asset_id, runs in iter_uniform_runs(uniform_path):
            if asset_id not in wanted:
                continue
            if not first:
                f.write(b",")
            first = False
            if grid:
                runs = resample_runs(runs, times)
            runs = cap_runs(runs, depth_cap)
            f.write(orjson.dumps(asset_id))
            f.write(b":")
            f.write(orjson.dumps(runs))
            n_runs += len(runs)
            del runs
        f.write(b"}}")
    return n_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform-dir", default=UNIFORM_DIR)
    ap.add_argument("--viewer-dir", default=os.path.join(DATA_DIR, "viewer"),
                     help="where the manifest + per-match payloads are written")
    ap.add_argument("--no-fetch", action="store_true",
                     help="don't hit the Gamma API; use only cached discovery + raw asset meta")
    ap.add_argument("--grid", type=float, default=1.0,
                     help="viewer tick resolution in seconds (0 = every real change). "
                          "The archive keeps every microsecond-level change -- 4.2M runs "
                          "for one game, ~2.5 GB, which no browser can fetch or draw. "
                          "Resampling onto a uniform grid is the only lever that actually "
                          "shrinks that; --depth barely moves it (books average ~23 "
                          "levels/side). data/uniform/*.json is unaffected.")
    ap.add_argument("--depth", type=int, default=30,
                     help="levels per side to inline into the viewer index (0 = full depth). "
                          "Full depth for every match is far too large to embed in one HTML; "
                          "data/uniform/*.json always keeps the complete book.")
    args = ap.parse_args()

    cache = {}
    if os.path.exists(DISCOVERY_CACHE_PATH):
        with open(DISCOVERY_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

    files = sorted(glob.glob(os.path.join(args.uniform_dir, "*.json")))
    print(f"Found {len(files)} rebuilt game(s) in {args.uniform_dir}")

    # Each match's payload is written as soon as it is built, rather than
    # collecting every match first: holding several games' runs at once is what
    # the streaming reader exists to avoid.
    os.makedirs(args.viewer_dir, exist_ok=True)
    manifest_matches = []
    total_payload = 0
    for path in files:
        slug = os.path.splitext(os.path.basename(path))[0]
        header = load_uniform_header(path)

        event = None
        if not args.no_fetch:
            try:
                event = fetch_event_cached(slug, cache)
            except Exception as e:
                print(f"  [warn] {slug}: Gamma fetch failed ({e}); falling back to raw YES/NO labels")
        else:
            event = cache.get(slug)

        match = build_match(slug, header, event, grid=args.grid)
        n_markets = sum(len(v) for v in match["market_types"].values())
        res = f"{args.grid}s grid" if args.grid else "every real change"
        print(f"  {slug}: {match['match_name']} -- {n_markets} markets, "
              f"{len(match['tokens'])} tokens, {match['n_ticks']:,} ticks "
              f"({res}, archive has {match['n_ticks_archive']:,})")

        payload_path = os.path.join(args.viewer_dir, f"{slug}.json")
        n_runs = write_payload(path, payload_path, match,
                               depth_cap=args.depth, grid=args.grid)
        size = os.path.getsize(payload_path)
        total_payload += size
        print(f"    payload: {n_runs:,} runs -> {size/1e6:.1f} MB")

        entry = dict(match)
        entry.pop("tick_times", None)
        manifest_matches.append(entry)

    with open(DISCOVERY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    manifest_path = os.path.join(args.viewer_dir, "index.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"matches": manifest_matches}, f)

    print(f"\nWrote {manifest_path} "
          f"({os.path.getsize(manifest_path)/1e6:.2f} MB manifest, {len(manifest_matches)} matches)")
    print(f"Wrote {len(manifest_matches)} per-match payloads to {args.viewer_dir} "
          f"({total_payload/1e6:.1f} MB total, loaded on demand)")


if __name__ == "__main__":
    main()
