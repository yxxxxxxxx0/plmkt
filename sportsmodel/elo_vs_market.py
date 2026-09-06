"""
The only test that matters: does Elo beat the Polymarket PRICE?

Beating "always pick home" is table stakes. A model is only worth money if it
prices a game better than the market does, because the market is what you
trade against. This scores both on the same games.

Elo is built strictly from games BEFORE each match, so its pre-game number is
causal. The market number is the Polymarket moneyline mid at first pitch,
taken from the recorded books.

Team-name matching is the fiddly part: MLB Stats API gives full club names
("Toronto Blue Jays") while the Polymarket slug carries abbreviations
("mlb-tor-kc-..."). The map is built from the abbreviations the schedule
endpoint itself supplies, so nothing is hand-guessed.

Sample size is the limit and is stated plainly: only the games with recorded
order books qualify, which is a few dozen. That is enough to see a large
effect and nowhere near enough to see a small one.

    python elo_vs_market.py
"""
import collections
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "..", "polymarket_orderbook", "data", "panel")
TOKMAP = os.path.join(BASE, "..", "polymarket_orderbook", "data", "token_map.json")
SRC = os.path.join(BASE, "data", "mlb.csv")
K, HFA, REGRESS = 4.0, 24.0, 0.30


def team_abbr_map():
    """abbreviation -> full club name, straight from the MLB API."""
    r = requests.get("https://statsapi.mlb.com/api/v1/teams",
                     params={"sportId": 1}, timeout=40)
    out = {}
    for t in r.json().get("teams", []):
        ab = (t.get("abbreviation") or "").lower()
        fc = (t.get("fileCode") or "").lower()
        nm = t.get("name")
        if nm:
            if ab:
                out[ab] = nm
            if fc:
                out.setdefault(fc, nm)
    return out


def main():
    rows = []
    with open(SRC, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: (r["date"], r["home"]))

    abbr = team_abbr_map()
    tokens = {}
    if os.path.exists(TOKMAP):
        with open(TOKMAP, "r", encoding="utf-8") as f:
            tokens = json.load(f)["tokens"]

    # market price at first pitch, per (date, home, away)
    market = {}
    for path in sorted(glob.glob(os.path.join(PANEL, "*.parquet"))):
        slug = os.path.basename(path)[:-8]
        parts = slug.split("-")
        if len(parts) < 5 or parts[0] != "mlb":
            continue
        away_ab, home_ab = parts[1], parts[2]
        date = "-".join(parts[3:6])
        home = abbr.get(home_ab)
        away = abbr.get(away_ab)
        if not home or not away:
            continue
        tab = pq.read_table(path, columns=["t", "mid_c", "bb_c", "ba_c",
                                           "asset_id", "market_type"])
        d = {c: tab.column(c).to_numpy(zero_copy_only=False)
             for c in tab.column_names}
        ok = (np.isfinite(d["bb_c"]) & np.isfinite(d["ba_c"])
              & (d["ba_c"] > d["bb_c"]) & (d["market_type"] == "moneyline"))
        if ok.sum() < 2000:
            continue
        chosen = None
        for aid in np.unique(d["asset_id"][ok]):
            info = tokens.get(aid) or {}
            nm = (info.get("outcome_name") or "")
            if nm and home.split()[-1].lower() in nm.lower():
                chosen = aid
                break
        if chosen is None:
            continue
        sel = ok & (d["asset_id"] == chosen)
        if sel.sum() < 500:
            continue
        # first 2% of the in-game window ~ around first pitch
        mid = d["mid_c"][sel]
        market[(date, home, away)] = float(np.median(mid[:max(len(mid)//50, 20)])) / 100.0

    print(f"market prices recovered for {len(market)} games")

    R = collections.defaultdict(lambda: 1500.0)
    season = None
    pairs = []
    for g in rows:
        if g["season"] != season:
            season = g["season"]
            for t in list(R):
                R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - REGRESS)
        h, a = g["home"], g["away"]
        e = 1.0 / (1.0 + 10 ** (-((R[h] + HFA - R[a]) / 400.0)))
        y = 1 if g["result"] == "H" else 0
        key = (g["date"], h, a)
        if key in market:
            pairs.append((e, market[key], y, h, a, g["date"]))
        upd = K * (y - e)
        R[h] += upd
        R[a] -= upd

    if not pairs:
        raise SystemExit("no overlap between Elo history and recorded markets")

    E = np.clip(np.array([p[0] for p in pairs]), 1e-6, 1 - 1e-6)
    M = np.clip(np.array([p[1] for p in pairs]), 1e-6, 1 - 1e-6)
    Y = np.array([p[2] for p in pairs])

    def sc(p, name):
        acc = float(((p > 0.5).astype(int) == Y).mean())
        ll = float(-np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))
        br = float(np.mean((p - Y) ** 2))
        return name, acc, ll, br

    print()
    print("=" * 78)
    print(f"ELO vs THE MARKET, head to head on {len(Y)} games")
    print(f"home wins in this sample: {Y.mean()*100:.1f}%")
    print("=" * 78)
    print(f"  {'':>22} {'accuracy':>10} {'log loss':>10} {'Brier':>9}")
    print("  " + "-" * 54)
    for nm, acc, ll, br in (sc(E, "Elo model"), sc(M, "Polymarket price"),
                            sc(np.full_like(E, Y.mean()), "sample base rate")):
        print(f"  {nm:>22} {acc*100:>9.1f}% {ll:>10.4f} {br:>9.4f}")

    agree = ((E > 0.5) == (M > 0.5))
    print()
    print(f"  Elo and the market pick the same side in {agree.mean()*100:.0f}% "
          f"of games ({int(agree.sum())}/{len(Y)})")
    dis = ~agree
    if dis.sum():
        print(f"  when they DISAGREE ({int(dis.sum())} games): "
              f"Elo right {float(((E[dis]>0.5).astype(int)==Y[dis]).mean())*100:.0f}%, "
              f"market right {float(((M[dis]>0.5).astype(int)==Y[dis]).mean())*100:.0f}%")
    gap = E - M
    print(f"  mean |Elo - market| = {np.abs(gap).mean()*100:.1f} pp   "
          f"max {np.abs(gap).max()*100:.1f} pp")
    print()
    print("  A model only earns money where it DISAGREES with the price, and only")
    print("  if it is right more often there. With this sample size that column")
    print("  is indicative, not conclusive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
