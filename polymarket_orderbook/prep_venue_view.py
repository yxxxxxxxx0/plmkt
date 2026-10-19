"""
Builds the data behind venue_viewer.html: per game, both venues' top of book
on one 1-second timeline plus every trade on either venue, in YES terms.

For each finished session (data/sessions/<tag>.json whose recordings are
decompressed in data/live/):
  1. data/view/<tag>/ gets session.json and hard links to the raw files
     (same volume, so no extra disk), which is the folder layout
     align_session.py expects.
  2. align_session.py writes aligned/<pm_slug>.csv.gz on a --step-ms grid.
  3. That grid plus both trade tapes become data/view/<tag>/<pm_slug>.view.json.
data/view/index.json lists every session and game for the viewer.

Sessions still recording are skipped: their files are incomplete.

Usage:
    python prep_venue_view.py                 # every finished session not yet built
    python prep_venue_view.py --force         # rebuild everything
    python prep_venue_view.py 2026-10-19_0508_mlb
"""
import argparse
import csv
import gzip
import json
import os

from align_session import align_session

BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(BASE, "data", "live")
SESS = os.path.join(BASE, "data", "sessions")
VIEW = os.path.join(BASE, "data", "view")


def raw_files(tag):
    names = [os.path.join(LIVE, f"books_{tag}.jsonl")]
    names += [os.path.join(LIVE, "kalshi", f"{k}_{tag}.jsonl") for k in ("kbooks", "ktrades", "kpolls")]
    return names


def finished(tag):
    # compress_raw.py only runs after both recorders exit, so an .xz means done
    return os.path.exists(os.path.join(LIVE, f"books_{tag}.jsonl.xz"))


def link_session(tag, spec_path):
    folder = os.path.join(VIEW, tag)
    os.makedirs(os.path.join(folder, "kalshi"), exist_ok=True)
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    with open(os.path.join(folder, "session.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    for src in raw_files(tag):
        sub = "kalshi" if os.path.basename(src).startswith("k") else ""
        dst = os.path.join(folder, sub, os.path.basename(src))
        if os.path.exists(src) and not os.path.exists(dst):
            os.link(src, dst)
    return folder, spec


def pm_trades(path, pairs):
    """Polymarket prints for the paired tokens, in YES terms of the paired
    token: a print on the complementary token of the same condition is 1-p."""
    key_of = {p["pm_asset"]: p["kalshi"].rsplit("-", 1)[-1] for p in pairs if p["pm_asset"]}
    cond_of, rows, pending = {}, [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"et":"trade"' not in line:
                continue
            r = json.loads(line)
            aid, cond = r["asset_id"], r.get("condition_id")
            if aid in key_of:
                cond_of[cond] = aid
            pending.append(r)
    for r in pending:
        aid = r["asset_id"]
        if aid in key_of:
            key, price, side = key_of[aid], r["price"], r.get("aggressor")
        elif r.get("condition_id") in cond_of:
            key, price = key_of[cond_of[r["condition_id"]]], 1 - r["price"]
            side = {"buy": "sell", "sell": "buy"}.get(r.get("aggressor"))
        else:
            continue
        rows.append([r["recv"], key, round(price, 4), round(r["size"], 2), side])
    rows.sort(key=lambda x: x[0])
    return rows


def ks_trades(path, tickers):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["ticker"] not in tickers:
                continue
            side = {"yes": "buy", "no": "sell"}.get(r.get("taker_side"))
            rows.append([r["ts_ms"], r["ticker"].rsplit("-", 1)[-1], round(r["yes_price"], 4),
                         round(r["size"], 2), side])
    rows.sort(key=lambda x: x[0])
    return rows


def num(v):
    return round(float(v), 4) if v not in ("", None) else None


def build_game(folder, tag, game):
    slug = game["pm_slug"]
    aligned = os.path.join(folder, "aligned", f"{slug}.csv.gz")
    mpath = os.path.join(folder, "aligned", f"{slug}.mapping.json")
    if not os.path.exists(aligned):
        return None
    with open(mpath, encoding="utf-8") as f:
        mapping = json.load(f)
    pairs = [p for p in mapping["pairs"] if p["pm_asset"]]
    keys = [p["kalshi"].rsplit("-", 1)[-1] for p in pairs]

    levels = mapping["levels"]
    t, depth = [], {"pm": [], "ks": []}
    series = {k: {c: [] for c in ("pm_bid", "pm_ask", "pm_mid", "ks_bid", "ks_ask", "ks_mid")} for k in keys}
    with gzip.open(aligned, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t.append(int(row["t_ms"]))
            for k in keys:
                for v in ("pm", "ks"):
                    ok = row.get(f"{k}_{v}_ok") == "1"
                    s = series[k]
                    s[f"{v}_bid"].append(num(row[f"{k}_{v}_bid1"]) if ok else None)
                    s[f"{v}_ask"].append(num(row[f"{k}_{v}_ask1"]) if ok else None)
                    s[f"{v}_mid"].append(num(row[f"{k}_{v}_mid"]) if ok else None)
            # analyze_collapse.py's depth: top-N bid + ask size, summed over
            # every outcome; unknown if any outcome's state is unknown
            for v in ("pm", "ks"):
                if all(row.get(f"{k}_{v}_ok") == "1" for k in keys):
                    depth[v].append(round(sum(float(row[f"{k}_{v}_{sd}{i}_sz"] or 0) for k in keys
                                              for sd in ("bid", "ask") for i in range(1, levels + 1))))
                else:
                    depth[v].append(None)

    view = {
        "session": tag, "slug": slug, "title": game["title"], "league": game.get("league"),
        "kickoff_ms": int(__import__("datetime").datetime.fromisoformat(game["kickoff"]).timestamp() * 1000),
        "step_ms": mapping["step_ms"],
        "outcomes": [{"key": k, "kalshi": p["kalshi"], "name": p["kalshi_outcome"],
                      "pm_question": p["pm_question"], "pm_outcome": p["pm_outcome_name"]}
                     for k, p in zip(keys, pairs)],
        "t": t, "series": series, "depth": depth, "depth_levels": levels,
        "pm_trades": pm_trades(os.path.join(folder, f"books_{tag}.jsonl"), pairs),
        "ks_trades": ks_trades(os.path.join(folder, "kalshi", f"ktrades_{tag}.jsonl"),
                               {p["kalshi"] for p in pairs}),
        "pm_gaps": mapping["pm_reconnect_gaps"], "ks_gaps": mapping["kalshi_gaps"],
    }
    out = os.path.join(folder, f"{slug}.view.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(view, f, separators=(",", ":"))
    return {"slug": slug, "title": game["title"], "league": game.get("league"),
            "kickoff_ms": view["kickoff_ms"], "outcomes": len(keys), "rows": len(t),
            "pm_trades": len(view["pm_trades"]), "ks_trades": len(view["ks_trades"]),
            "file": f"{tag}/{slug}.view.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--step-ms", type=int, default=1000)
    ap.add_argument("--levels", type=int, default=5, help="book levels summed into depth")
    args = ap.parse_args()

    os.makedirs(VIEW, exist_ok=True)
    idx_path = os.path.join(VIEW, "index.json")
    try:
        with open(idx_path, encoding="utf-8") as f:
            index = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        index = {}

    tags = args.tags or sorted(f[:-5] for f in os.listdir(SESS)
                               if f.endswith(".json") and f != "state.json")
    for tag in tags:
        if not finished(tag):
            print(f"{tag}: still recording (or never compressed); skipped")
            continue
        if not all(os.path.exists(p) for p in raw_files(tag)[:1] + raw_files(tag)[3:]):
            print(f"{tag}: not decompressed; run decompress_raw.py first")
            continue
        if tag in index and not args.force:
            print(f"{tag}: already built")
            continue
        folder, spec = link_session(tag, os.path.join(SESS, f"{tag}.json"))
        align_session(folder, levels=args.levels, step_ms=args.step_ms, pm_time="recv",
                      kalshi_time="recv", max_gap_ms=2000)
        games = [g for g in (build_game(folder, tag, g) for g in spec["games"]) if g]
        index[tag] = {"sport": spec["sport"], "games": games}
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=1)
        print(f"{tag}: {len(games)} game(s) ready")


if __name__ == "__main__":
    main()
