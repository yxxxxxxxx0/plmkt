"""
Health and consistency report for one recorded session (both venues).

Reads the session from data/outbox/<tag>/ (compressed) or, while it is still
recording, from data/live/ (raw). Reports:

  1. Polymarket feed health per match: snapshots, longest silence.
  2. Kalshi polling health: rate, round trip, longest gap between answers,
     429s/errors, and how often a reply arrived after a later-sent one.
  3. Kalshi ordering: one-poll A->B->A flickers in the recorded (arrival
     order) stream, and how many involve an "overtook" reply.
  4. Cross-venue agreement: per outcome, Kalshi vs Polymarket midpoint
     sampled once a second (mean |diff|, share within 2c).
  5. Big moves (goals, red cards...): every >=10c midpoint move within 60 s
     on Polymarket, when Kalshi made the same move, and which venue led.

Usage:
    python check_session.py test4h
"""
import argparse
import bisect
import collections
import json
import lzma
import os
import statistics

from pairing import kalshi_codes, name_sim, pm_codes

BASE = os.path.dirname(os.path.abspath(__file__))


def open_any(*cands):
    for p in cands:
        if os.path.exists(p):
            return lzma.open(p, "rt", encoding="utf-8") if p.endswith(".xz") else open(p, encoding="utf-8")
    return None


def files(tag):
    ob = os.path.join(BASE, "data", "outbox", tag)
    live = os.path.join(BASE, "data", "live")
    def pick(sub, name):
        return [os.path.join(ob, sub, name + ".xz"), os.path.join(live, name),
                os.path.join(live, "kalshi", name)]
    return {
        "pm": pick("polymarket", f"books_{tag}.jsonl"),
        "kb": pick("kalshi", f"kbooks_{tag}.jsonl"),
        "kp": pick("kalshi", f"kpolls_{tag}.jsonl"),
        "kt": pick("kalshi", f"ktrades_{tag}.jsonl"),
    }


def mid(bids, asks):
    if not bids or not asks:
        return None
    return (bids[0][0] + asks[0][0]) / 2


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--move", type=float, default=0.10, help="big-move threshold (price units)")
    a = ap.parse_args()
    F = files(a.tag)

    # ---- Polymarket ---------------------------------------------------------
    pm = collections.defaultdict(list)       # (slug, question) -> [(t_ms, mid)]  YES token only
    pm_ts = collections.defaultdict(list)    # slug -> recv times
    with open_any(*F["pm"]) as f:
        for line in f:
            r = json.loads(line)
            pm_ts[r["slug"]].append(r["recv"])
            if r["outcome"] != "YES":
                continue
            m = mid(r["bids"], r["asks"])
            if m is not None:
                pm[(r["slug"], r.get("market_question") or r.get("outcome_name"))].append((r["recv"], m))
    print("1) POLYMARKET FEED")
    for slug, ts in sorted(pm_ts.items()):
        ts.sort()
        gap = max((b - a_) for a_, b in zip(ts, ts[1:])) / 1000 if len(ts) > 1 else float("nan")
        print(f"   {slug:<30} {len(ts):>8,} snapshots over {(ts[-1]-ts[0])/3.6e6:4.1f} h, "
              f"longest silence {gap:6.1f} s")

    # ---- Kalshi polls -------------------------------------------------------
    polls, status = [], []
    with open_any(*F["kp"]) as f:
        for line in f:
            r = json.loads(line)
            (status if r.get("type") == "status" else polls).append(r)
    ok = [p for p in polls if p.get("status") == 200]
    rtt = [p["t_recv"] - p["t_send"] for p in ok]
    recv = sorted(p["t_recv"] for p in ok)
    gaps = [b - a_ for a_, b in zip(recv, recv[1:])]
    span = (recv[-1] - recv[0]) / 1000 if recv else 0
    print("\n2) KALSHI POLLING")
    print(f"   {len(polls):,} requests over {span/3600:.1f} h ({len(polls)/max(span,1):.1f}/s); "
          f"{len(ok):,} ok, {sum(p.get('status') == 429 for p in polls)} rate-limited, "
          f"{sum(p.get('status') not in (200, 429) for p in polls)} errors")
    print(f"   round trip p50 {pct(rtt,.5):.0f} ms, p95 {pct(rtt,.95):.0f}, p99 {pct(rtt,.99):.0f}, max {max(rtt):.0f}")
    print(f"   gap between answers p50 {pct(gaps,.5):.0f} ms, p99 {pct(gaps,.99):.0f}, "
          f"longest {max(gaps)/1000:.1f} s")
    ov = sum(1 for p in ok if p.get("overtook"))
    print(f"   arrived after a later-sent poll: {ov:,} ({ov/max(len(ok),1):.0%})")
    for s in status:
        if s.get("status") != "active":
            print(f"   status: {s['ticker']} -> {s['status']} at {s['t']}")

    # ---- Kalshi books -------------------------------------------------------
    kb = collections.defaultdict(list)       # ticker -> [(t_recv, t_send, mid, overtook, state)]
    kname = {}
    with open_any(*F["kb"]) as f:
        for line in f:
            r = json.loads(line)
            kname[r["ticker"]] = (r["event"], r["outcome"])
            st = json.dumps([r["bids"], r["asks"]])
            kb[r["ticker"]].append((r["t_recv"], r["t_send"], mid(r["bids"], r["asks"]), r.get("overtook"), st,
                                   r["poll_id"]))
    print("\n3) KALSHI ORDERING (recorded stream, arrival order)")
    tot = fl = flo = 0
    for tk, seq in kb.items():
        s = [x[4] for x in seq]
        tot += len(s) - 1
        for i in range(1, len(s) - 1):
            if s[i - 1] == s[i + 1] != s[i]:
                fl += 1
                flo += bool(seq[i][3] or seq[i + 1][3])
    print(f"   {tot:,} book changes; {fl:,} one-poll A->B->A flickers ({fl/max(tot,1):.1%}), "
          f"{flo:,} of them involve an overtaking reply")
    # Rebuild every reply's book (unchanged replies aren't written, so a ticker's
    # state at each poll is its last written change) and keep a state only once
    # two consecutive replies agree -- the rule the analysis should use.
    changes = collections.defaultdict(dict)        # poll_id -> ticker -> state
    for tk, sq in kb.items():
        for x in sq:
            changes[x[5]][tk] = x[4]
    cur, runs = {}, collections.defaultdict(list)
    for p in ok:                                   # kpolls is in arrival order
        for tk, st in changes.get(p["poll_id"], {}).items():
            cur[tk] = st
        for tk, st in cur.items():
            runs[tk].append(st)
    ctot = cfl = 0
    for tk, st in runs.items():
        out, run = [], 0
        for i, x in enumerate(st):
            run = run + 1 if i and x == st[i - 1] else 1
            if run >= 2 and (not out or out[-1] != x):
                out.append(x)
        ctot += len(out) - 1
        cfl += sum(1 for a_, b_, c_ in zip(out, out[1:], out[2:]) if a_ == c_)
    print(f"   confirmed by 2 consecutive replies: {ctot:,} changes, {cfl:,} A->B->A "
          f"({cfl/max(ctot,1):.1%})")

    # ---- cross-venue ----------------------------------------------------------
    print("\n4) KALSHI vs POLYMARKET MIDPOINT (sampled every 1 s)")
    pairs = []
    for tk, (ev, outcome) in sorted(kname.items()):
        best, bs = None, 0
        for k in pm:
            if kalshi_codes(ev) != pm_codes(k[0]):          # same game only
                continue
            q = (k[1] or "").lower()
            is_draw = "draw" in q
            if outcome.lower() in ("tie", "draw"):
                sc = 2.0 if is_draw else 0
            elif is_draw:
                continue                                       # draw question names both teams
            else:
                sc = name_sim(outcome, (k[1] or "").replace("Will ", "").split(" win")[0])
            if sc > bs:
                best, bs = k, sc
        if best and bs >= 0.8:
            pairs.append((tk, best))
    rows = []
    for tk, k in pairs:
        ks = sorted((x[0], x[2]) for x in kb[tk] if x[2] is not None)
        ps = sorted(pm[k])
        if not ks or not ps:
            continue
        kt = [x[0] for x in ks]
        pt = [x[0] for x in ps]
        t0, t1 = max(kt[0], pt[0]), min(kt[-1], pt[-1])
        diffs = []
        for t in range(int(t0), int(t1), 1000):
            i = bisect.bisect_right(kt, t) - 1
            j = bisect.bisect_right(pt, t) - 1
            if i >= 0 and j >= 0:
                diffs.append(abs(ks[i][1] - ps[j][1]))
        if diffs:
            rows.append((tk, k[0], statistics.mean(diffs), sum(d <= 0.02 for d in diffs) / len(diffs), len(diffs)))
            print(f"   {tk:<38} vs {k[0]:<26} mean|diff| {statistics.mean(diffs)*100:4.1f}c, "
                  f"within 2c {sum(d <= 0.02 for d in diffs)/len(diffs):4.0%}  ({len(diffs):,} s)")

    # ---- big moves --------------------------------------------------------------
    print(f"\n5) BIG MOVES (>= {a.move*100:.0f}c within 60 s on Polymarket) and Kalshi's timing")
    leads = []
    for tk, k in pairs:
        ps = sorted(pm[k])
        ks = sorted((x[0], x[2]) for x in kb[tk] if x[2] is not None)
        if not ps or not ks:
            continue
        pt = [x[0] for x in ps]
        last_evt = -1e18
        j0 = 0
        for i, (t, m) in enumerate(ps):
            while ps[j0][0] < t - 60000:
                j0 += 1
            base = ps[j0][1]
            if abs(m - base) >= a.move and t - last_evt > 300000:
                last_evt = t
                target = (base + m) / 2          # half-way crossing
                up = m > base
                # Polymarket half-way time
                tp = next(tt for tt, mm in ps[j0:i + 1] if (mm >= target if up else mm <= target))
                # Kalshi half-way time, searching +-120 s
                tk_ = next((tt for tt, mm in ks if tt >= tp - 120000 and tt <= tp + 120000
                            and (mm >= target if up else mm <= target)), None)
                lag = (tk_ - tp) / 1000 if tk_ else None
                if lag is not None:
                    leads.append(lag)
                print(f"   {k[0]:<26} {tk.split('-')[-1]:<4} {base:.2f}->{m:.2f}  "
                      + (f"Kalshi {lag:+6.1f} s vs Polymarket" if lag is not None else "Kalshi: no matching move"))
    if leads:
        print(f"   Kalshi minus Polymarket at half-way: median {statistics.median(leads):+.1f} s "
              f"over {len(leads)} move(s) (positive = Kalshi later)")


if __name__ == "__main__":
    main()
