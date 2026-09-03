"""
Load one match's full-depth order book history for simulation / backtesting.

Which source to use
-------------------
data/uniform/<slug>.json is the self-contained per-match archive: every real
recorded tick, full depth, all tokens, one file. It is the right thing to
point analysis at -- but it is 1.4-2.7 GB of JSON, so json.load() on it needs
tens of GB of RAM and will die. Always stream it (load_match below does).

If you have the raw data/live/by_game/<slug>.jsonl instead, load_match_fast()
rebuilds the identical runs in ~1/5 the time (parity-enforced by
test_rebuild_parity.py). Same numbers, different road.

Both give you, per asset: a sorted array of run start times plus the runs, so
"what was the book at time T" is a binary search, not a replay.

Usage:
    python sim_load.py mlb-kc-tor-2026-08-27
"""
import bisect
import sys

import build_match_index as bmi
import rebuild_duckdb as rd
import rebuild_multi as rm

UNIFORM = "data/uniform/{}.json"


class Book:
    """One token's run-length-encoded book history."""
    def __init__(self, asset_id, meta, runs):
        self.asset_id = asset_id
        self.meta = meta
        self.runs = runs
        self.starts = [r["t_start"] for r in runs]

    def at(self, t):
        """Full book in force at unix time t (None before the first run)."""
        i = bisect.bisect_right(self.starts, t) - 1
        return self.runs[i] if i >= 0 else None

    def mid(self, t):
        r = self.at(t)
        if not r:
            return None
        bb, ba = r["best_bid"], r["best_ask"]
        if bb is not None and ba is not None:
            return (bb + ba) / 2
        return bb if bb is not None else ba

    def __len__(self):
        return len(self.runs)


def load_match(slug, assets=None):
    """Stream data/uniform/<slug>.json. `assets` = set of asset_ids to keep
    (None = all). Keeping only what you need is what makes this fit in RAM."""
    path = UNIFORM.format(slug)
    header = bmi.load_uniform_header(path)
    books = {}
    for aid, runs in bmi.iter_uniform_runs(path):
        if assets is not None and aid not in assets:
            del runs
            continue
        books[aid] = Book(aid, header["assets"].get(aid, {}), runs)
    return header, books


def load_match_fast(slug, window_mode="game", assets=None):
    """Same result, rebuilt from data/live/by_game/<slug>.jsonl via DuckDB."""
    windows = rm.load_game_windows()
    header, boundaries, ts_adj, cols, start_ts, end_ts = rd.prepare_game(
        slug, windows.get(slug, {}), window_mode=window_mode)
    _, _, bpx, bsz, boff, apx, asz, aoff, _ = cols
    books = {}
    for aid, lo, hi in boundaries:
        if assets is not None and aid not in assets:
            continue
        runs = rd.build_asset_runs(lo, ts_adj[aid], boff, bpx, bsz,
                                   aoff, apx, asz, start_ts, end_ts, depth=0)
        books[aid] = Book(aid, header["assets"].get(aid, {}), runs)
    return header, books


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "mlb-kc-tor-2026-08-27"
    header, books = load_match_fast(slug)

    print(f"{slug}")
    print(f"  window   {header['start_ts']:.3f} -> {header['end_ts']:.3f} "
          f"({(header['end_ts']-header['start_ts'])/60:.1f} min)")
    print(f"  ticks    {len(header['tick_times']):,}")
    print(f"  tokens   {len(books)}")
    print(f"  runs     {sum(len(b) for b in books.values()):,}")

    ml = [b for b in books.values() if b.meta.get("market_type") == "moneyline"]
    print(f"\n  moneyline tokens: {len(ml)}")
    t = header["start_ts"] + 3600     # one hour into the game
    for b in ml:
        r = b.at(t)
        print(f"    {b.meta.get('outcome'):<4} runs={len(b):>9,}  "
              f"mid={b.mid(t)}  bid={r['best_bid']} ask={r['best_ask']}  "
              f"levels={len(r['bids'])}/{len(r['asks'])}")
    if len(ml) == 2:
        a, c = ml[0].at(t), ml[1].at(t)
        print(f"    complement check: {a['best_bid']} + {c['best_ask']} = "
              f"{a['best_bid'] + c['best_ask']:.3f}  (expect 1.000)")
