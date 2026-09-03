"""
Reports each market's `seconds_delay` -- the matching delay Polymarket applies
to taker orders -- for one or more event slugs.

Why it matters here: a non-zero seconds_delay means a taker order does not
execute against the book you can see at that instant, it executes against the
book `seconds_delay` later. Any backtest that assumes immediate fills against
the recorded book will be optimistic by exactly that much, so the value has to
be known per market before the recordings are used for simulation.

The value comes from the CLOB API (one request per condition_id); Gamma only
supplies the market list.

Usage:
    python taker_delay_check.py --slate 2026-09-02        # tonight's HKT slate
    python taker_delay_check.py mlb-sd-cin-2026-09-01
"""
import argparse
import collections
import sys
import time

import requests

GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com/markets/{}"


def event_markets(slug, session):
    r = session.get(GAMMA, params={"slug": slug}, timeout=25)
    r.raise_for_status()
    rows = r.json()
    return rows[0]["markets"] if rows else []


def seconds_delay(cid, session, retries=3):
    for i in range(retries):
        try:
            r = session.get(CLOB.format(cid), timeout=25)
            if r.status_code == 200:
                return r.json().get("seconds_delay"), None
            err = f"HTTP {r.status_code}"
        except Exception as e:
            err = f"{type(e).__name__}"
        time.sleep(0.5 * (i + 1))
    return None, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--slate", help="HKT date; resolve slugs via plan_slate")
    ap.add_argument("--quiet", action="store_true", help="only show non-zero delays")
    args = ap.parse_args()

    slugs = list(args.slugs)
    if args.slate:
        import plan_slate as ps
        seen = {}
        for us in sorted({args.slate, str(__import__("datetime").date.fromisoformat(args.slate)
                                          - __import__("datetime").timedelta(days=1))}):
            for g in ps.schedule(us):
                for a in ps.candidates(g["away_abbr"]):
                    for h in ps.candidates(g["home_abbr"]):
                        s = f"mlb-{a}-{h}-{g['us_date']}"
                        if s in seen:
                            continue
                        if ps.gamma_event(s):
                            seen[s] = True
                            break
                    else:
                        continue
                    break
        slugs = sorted(seen)
    if not slugs:
        raise SystemExit("no slugs; pass them or use --slate")

    session = requests.Session()
    tally = collections.Counter()
    per_game = {}
    errors = []

    for slug in slugs:
        try:
            markets = event_markets(slug, session)
        except Exception as e:
            errors.append(f"{slug}: {e}")
            continue
        rows = []
        for m in markets:
            cid = m.get("conditionId")
            if not cid:
                continue
            d, err = seconds_delay(cid, session)
            if err:
                errors.append(f"{slug} {cid[:10]}: {err}")
            rows.append((m.get("question", "?"), cid, d))
            tally[d] += 1
        per_game[slug] = rows

        shown = [r for r in rows if not args.quiet or r[2]]
        print(f"\n=== {slug}  ({len(rows)} markets) ===")
        for q, cid, d in shown:
            flag = "" if d in (0, None) else "   <-- DELAYED"
            print(f"  {str(d):>6}s  {q[:66]:<66}{flag}")
        if args.quiet and not shown:
            print("  (all zero)")

    print("\n" + "=" * 72)
    print("seconds_delay across all markets checked:")
    for d, n in sorted(tally.items(), key=lambda kv: (kv[0] is None, kv[0])):
        print(f"  {str(d):>6} : {n} market(s)")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors[:10]:
            print("  " + e)


if __name__ == "__main__":
    sys.exit(main())
