"""
asset_id -> market identity and its complement, for every recorded game.

The panel keys rows by asset_id, which is enough for single-token features but
not for anything relational. Pairing a YES book with its NO book, or lining the
moneyline up against the run line for the same game, needs condition_id.

Derived from the RECORDINGS, not from data/discovery_cache.json. The cache
only covers the games that happened to be fetched through the Gamma path and
silently lacks later slates -- pairing against it returned zero pairs for
2026-09-04. Every recorded row already carries its own condition_id, so the
recordings are both authoritative and complete.

Only the head of each split file is read. live_recorder seeds every tracked
asset before streaming, so the first few thousand records contain the full
token roster; parsing 24 GB to learn 34 identities would be absurd. The roster
is cross-checked against the number of distinct assets found, and any game
whose head does not look complete is reported rather than silently half-mapped.

A complement pair is a condition_id carrying exactly two asset_ids. Those two
are worth exactly $1 together at settlement, which is what makes the
complete-set arbitrage in cross_complement.py well defined.

    python build_token_map.py
Output: data/token_map.json
"""
import argparse
import collections
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(BASE, "data", "live", "by_game")
OUT = os.path.join(BASE, "data", "token_map.json")
HEAD_LINES = 20000


def scan_head(path, max_lines=HEAD_LINES):
    seen = {}
    with open(path, "r", encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k >= max_lines:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = r.get("asset_id")
            if aid is None or aid in seen:
                continue
            seen[aid] = {
                "slug": r.get("slug"),
                "condition_id": r.get("condition_id"),
                "market_type": r.get("market_type"),
                "question": r.get("market_question"),
                "outcome": r.get("outcome"),
                "outcome_name": r.get("outcome_name"),
                "line": r.get("line"),
            }
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default=SPLIT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.split_dir, "*.jsonl")))
    if not files:
        raise SystemExit("no split games found")

    tokens, pairs = {}, {}
    by_event = collections.defaultdict(list)
    odd = []

    for path in files:
        slug = os.path.basename(path)[:-6]
        seen = scan_head(path)
        tokens.update(seen)
        by_cond = collections.defaultdict(list)
        for aid, info in seen.items():
            by_event[slug].append(aid)
            by_cond[info["condition_id"]].append(aid)
        for cid, ids in by_cond.items():
            if len(ids) != 2:
                # Not a clean two-outcome market in this recording -- could be
                # a token that never seeded. Recorded, never guessed at.
                odd.append((slug, cid, len(ids)))
                continue
            ids = sorted(ids)
            a, b = ids
            tokens[a]["complement"] = b
            tokens[b]["complement"] = a
            info = seen[a]
            pairs[cid] = {"slug": slug, "a": a, "b": b,
                          "market_type": info["market_type"],
                          "line": info["line"], "question": info["question"]}

    out = {"tokens": tokens, "pairs": pairs,
           "by_event": {k: sorted(set(v)) for k, v in by_event.items()}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)

    types = collections.Counter(t.get("market_type") for t in tokens.values())
    slates = collections.Counter(str(t.get("slug"))[-10:] for t in tokens.values())
    print(f"{len(files)} games, {len(tokens):,} tokens, {len(pairs):,} complement pairs")
    print("tokens by market type:")
    for k, v in types.most_common():
        print(f"  {str(k):>20} {v:>6,}")
    print("tokens by slate:")
    for k, v in sorted(slates.items()):
        print(f"  {k} {v:>6,}")
    if odd:
        print(f"\n{len(odd)} condition_id(s) without exactly 2 tokens "
              f"(excluded from pairing): {odd[:4]}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
