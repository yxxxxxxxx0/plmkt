"""
Aligns one downloaded session's Polymarket and Kalshi order books onto a
single timeline, per game, so both venues can be compared at the same instant.

Timestamps: both recorders run on the same machine, so their receive times
(`recv` on Polymarket, `t_*_recv` on Kalshi) are on the same clock. That
clock is the common timeline. Exchange timestamps differ between venues
(Kalshi's book endpoint has none), so they are not used for alignment.

    Polymarket  t = recv                     (--pm-time ts: exchange time)
    Kalshi      t = t_first_recv of the      (--kalshi-time send|mid|recv:
                cleaned state                 the state existed somewhere in
                                              [t_first_send, t_first_recv])

Kalshi input is the cleaned stream (kbooks_clean_*, from clean_kalshi.py),
so one-poll flickers from out-of-order replies are already gone; it is
created here if missing.

Continuity: a book holds its last state until the next update (forward
fill), but only while the feed is known to be alive. Each venue/outcome
gets an `_ok` flag that is 0 when the state at that instant is unknown:
    - before the stream's first update
    - Polymarket: while its websocket was reconnecting (a re-seed of an
      already-seeded asset marks the silence before it as unknown)
    - Kalshi: when no successful reply arrived for more than --max-gap-ms
      (taken from kpolls), and once the market leaves "active"
`_age_ms` is how old the shown state is.

Output, per game, in <session>/aligned/:
    <pm_slug>.csv.gz   one row every --step-ms (default 100 ms), or one row
                       per update on either venue with --step-ms 0
    columns: t_ms, t_utc, then for every outcome K (Kalshi ticker suffix,
    e.g. BOU / SUN / TIE) and venue V in (pm, ks):
        K_V_bid1, K_V_bid1_sz, ... K_V_bidN, K_V_bidN_sz,
        K_V_ask1, K_V_ask1_sz, ... , K_V_mid, K_V_age_ms, K_V_ok
    prices are in YES terms (probability of K), sizes in contracts/shares
    <pm_slug>.mapping.json   which Polymarket token was paired with which
                             Kalshi ticker, gaps found, row count

Usage:
    python align_session.py ~/plmkt-recordings/2026-10-19_0000_football
    python align_session.py data/outbox/test4h --step-ms 1000 --levels 3
    python align_session.py <session> --step-ms 0          # event-driven
"""
import argparse
import bisect
import csv
import datetime as dt
import glob
import gzip
import json
import lzma
import os
import tempfile

from pairing import name_sim


def _open(path):
    return lzma.open(path, "rt", encoding="utf-8") if path.endswith(".xz") else open(path, encoding="utf-8")


def _one(folder, pattern):
    hits = glob.glob(os.path.join(folder, "**", pattern), recursive=True)
    return hits[0] if hits else None


def top(levels, n):
    return [[p, s] for p, s in levels[:n]]


# ---- gaps -------------------------------------------------------------------

def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class Gaps:
    """Sorted, merged (start, end) intervals; `inside(t)` for any t."""
    def __init__(self, intervals):
        self.iv = merge(intervals)
        self.starts = [a for a, _ in self.iv]

    def inside(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        return i >= 0 and t < self.iv[i][1]


# ---- Polymarket: one pass, split per game into small temp files --------------

def split_polymarket(path, slugs, levels, tmp, pm_time):
    """Streams the (large) Polymarket file once. Keeps only the top `levels`
    of each book, drops updates where those levels did not change, and
    writes per-game temp files. Returns (assets, gaps):
        assets  asset_id -> {slug, outcome, outcome_name, question}
        gaps    slug -> [(start, end)] reconnect silences"""
    outs = {s: open(os.path.join(tmp, f"pm_{s}.jsonl"), "w", encoding="utf-8") for s in slugs}
    assets, last_top, seeded = {}, {}, set()
    last_recv, gaps = {}, {s: [] for s in slugs}
    with _open(path) as f:
        for line in f:
            r = json.loads(line)
            slug = r["slug"]
            if slug not in outs:
                continue
            # main's live_recorder.py also writes conn / trade / tick_size_change
            # rows into this stream; only book states belong on the timeline,
            # but a reported outage is a gap just like a re-seed.
            if r.get("et") == "conn" and r.get("event") == "outage_end":
                gaps[slug].append((r["recv"] - int(r.get("outage_s", 0) * 1000), r["recv"]))
            if "bids" not in r or r.get("et") == "trade":
                continue
            aid = r["asset_id"]
            t = r["ts"] if pm_time == "ts" else r["recv"]
            if aid not in assets:
                assets[aid] = {"slug": slug, "outcome": r["outcome"],
                               "outcome_name": r.get("outcome_name"),
                               "question": r.get("market_question")}
            if r.get("et") == "seed":
                if aid in seeded and slug in last_recv and r["recv"] > last_recv[slug]:
                    gaps[slug].append((last_recv[slug], r["recv"]))
                seeded.add(aid)
            last_recv[slug] = r["recv"]
            book = (top(r["bids"], levels), top(r["asks"], levels))
            if last_top.get(aid) == book:
                continue
            last_top[aid] = book
            outs[slug].write(json.dumps([t, aid, book[0], book[1]], separators=(",", ":")) + "\n")
    for f in outs.values():
        f.close()
    return assets, gaps


# ---- Kalshi -----------------------------------------------------------------

def load_kalshi(clean_path, kpolls_path, levels, kalshi_time, max_gap_ms):
    """Returns (updates, meta, closed, gaps):
        updates  ticker -> [(t, bids, asks)] sorted by t
        meta     ticker -> {event, outcome}
        closed   ticker -> first time its status left "active"
        gaps     [(start, end)] stretches with no successful reply"""
    updates, meta = {}, {}
    with _open(clean_path) as f:
        for line in f:
            r = json.loads(line)
            tk = r["ticker"]
            if kalshi_time == "send":
                t = r["t_first_send"]
            elif kalshi_time == "mid":
                t = (r["t_first_send"] + r["t_first_recv"]) / 2
            else:
                t = r["t_first_recv"]
            meta[tk] = {"event": r["event"], "outcome": r["outcome"]}
            updates.setdefault(tk, []).append((t, top(r["bids"], levels), top(r["asks"], levels)))
    for u in updates.values():
        u.sort(key=lambda x: x[0])
    closed, recv = {}, []
    with _open(kpolls_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "status":
                if r.get("status") != "active" and r["ticker"] not in closed:
                    closed[r["ticker"]] = r["t"]
            elif r.get("status") == 200:
                recv.append(r["t_recv"])
    recv.sort()
    gaps = [(a, b) for a, b in zip(recv, recv[1:]) if b - a > max_gap_ms]
    return updates, meta, closed, gaps


# ---- pairing a Kalshi ticker with a Polymarket token -------------------------

def is_tie(name):
    return name.strip().lower() in ("tie", "draw")


def pm_asset_for(kalshi_outcome, slug_assets):
    """Football: three Yes/No markets (home win, away win, draw) -> the YES
    token of the matching market. MLB: one market whose two tokens are named
    after the teams -> the token named like the Kalshi outcome."""
    best, best_sc = None, 0.0
    for aid, a in slug_assets.items():
        name = (a["outcome_name"] or "").strip()
        if name.lower() in ("yes", "no"):
            if a["outcome"] != "YES":
                continue
            q = (a["question"] or "").lower()
            if is_tie(kalshi_outcome):
                sc = 2.0 if "draw" in q else 0.0
            elif "draw" in q:
                continue                      # the draw question names both teams
            else:
                sc = name_sim(kalshi_outcome, q.replace("will ", "").split(" win")[0])
        else:
            sc = name_sim(kalshi_outcome, name)
        if sc > best_sc:
            best, best_sc = aid, sc
    return (best, best_sc) if best_sc >= 0.6 else (None, best_sc)


# ---- one game ---------------------------------------------------------------

class Stream:
    """One venue's book for one outcome, stepped forward in time."""
    def __init__(self, key, venue, updates, gaps, closed_at=None):
        self.key, self.venue = key, venue
        self.u = updates                      # [(t, bids, asks)] sorted
        self.gaps = gaps
        self.closed_at = closed_at
        self.i = -1                           # index of the current state

    def advance(self, t):
        while self.i + 1 < len(self.u) and self.u[self.i + 1][0] <= t:
            self.i += 1

    def row(self, t, levels):
        if self.i < 0:
            return [""] * (4 * levels + 2) + [0]
        ts, bids, asks = self.u[self.i]
        cells = []
        for side in (bids, asks):
            for k in range(levels):
                cells += side[k] if k < len(side) else ["", ""]
        mid = round((bids[0][0] + asks[0][0]) / 2, 4) if bids and asks else ""
        ok = not self.gaps.inside(t) and (self.closed_at is None or t < self.closed_at)
        return cells + [mid, int(t - ts), int(ok)]

    def header(self, levels):
        p = f"{self.key}_{self.venue}"
        cols = []
        for side in ("bid", "ask"):
            for k in range(1, levels + 1):
                cols += [f"{p}_{side}{k}", f"{p}_{side}{k}_sz"]
        return cols + [f"{p}_mid", f"{p}_age_ms", f"{p}_ok"]


def align_game(game, out_dir, tmp, assets, pm_gaps, ks, levels, step_ms):
    slug = game["pm_slug"]
    updates, meta, closed, ks_gaps = ks
    events = set(game["kalshi_events"])
    tickers = sorted(tk for tk, m in meta.items() if m["event"] in events)
    slug_assets = {aid: a for aid, a in assets.items() if a["slug"] == slug}

    pm_updates = {}
    with open(os.path.join(tmp, f"pm_{slug}.jsonl"), encoding="utf-8") as f:
        for line in f:
            t, aid, bids, asks = json.loads(line)
            pm_updates.setdefault(aid, []).append((t, bids, asks))
    for u in pm_updates.values():
        u.sort(key=lambda x: x[0])

    streams, mapping = [], []
    pm_gap = Gaps(pm_gaps.get(slug, []))
    ks_gap = Gaps(ks_gaps)
    picks = {tk: pm_asset_for(meta[tk]["outcome"], slug_assets) for tk in tickers}
    # Names that differ completely ("Santander" vs "Real Racing Club", "Alkmaar"
    # vs "AZ") are paired by elimination: one unmatched team, one unused team
    # token (football: a non-draw YES token; MLB: a team-named token).
    used = {aid for aid, _ in picks.values() if aid}
    left = [tk for tk, (aid, _) in picks.items() if aid is None and not is_tie(meta[tk]["outcome"])]
    spare = [aid for aid, a in slug_assets.items() if aid not in used and (
        (a["outcome"] == "YES" and "draw" not in (a["question"] or "").lower())
        if (a["outcome_name"] or "").strip().lower() in ("yes", "no") else True)]
    if len(left) == 1 and len(spare) == 1:
        picks[left[0]] = (spare[0], 0.0)
        print(f"   {left[0]}: paired by elimination with {slug_assets[spare[0]]['question']!r}")
    for tk in tickers:
        key = tk.rsplit("-", 1)[-1]
        aid, sc = picks[tk]
        mapping.append({"kalshi": tk, "kalshi_outcome": meta[tk]["outcome"], "pm_asset": aid,
                        "pm_question": slug_assets.get(aid, {}).get("question"),
                        "pm_outcome_name": slug_assets.get(aid, {}).get("outcome_name"),
                        "score": round(sc, 2)})
        if aid is None or aid not in pm_updates:
            print(f"   ! {tk}: no Polymarket token matched (best score {sc:.2f}); skipped")
            continue
        streams.append(Stream(key, "pm", pm_updates[aid], pm_gap))
        streams.append(Stream(key, "ks", updates[tk], ks_gap, closed.get(tk)))
    if not streams:
        print(f"   ! {slug}: nothing to align")
        return

    # rows run from the moment every stream has a state to the last update
    t0 = max(s.u[0][0] for s in streams)
    t1 = max(s.u[-1][0] for s in streams)
    if step_ms > 0:
        times = range(int(t0) - int(t0) % step_ms + step_ms, int(t1) + 1, step_ms)
    else:
        times = sorted({u[0] for s in streams for u in s.u if t0 <= u[0] <= t1})

    out = os.path.join(out_dir, f"{slug}.csv.gz")
    n = 0
    with gzip.open(out, "wt", encoding="utf-8", newline="", compresslevel=6) as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "t_utc"] + [c for s in streams for c in s.header(levels)])
        for t in times:
            row = [int(t), dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).isoformat(timespec="milliseconds")]
            for s in streams:
                s.advance(t)
                row += s.row(t, levels)
            w.writerow(row)
            n += 1
    with open(os.path.join(out_dir, f"{slug}.mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"pm_slug": slug, "kalshi_events": sorted(events), "pairs": mapping,
                   "rows": n, "step_ms": step_ms, "levels": levels,
                   "t_start": int(t0), "t_end": int(t1),
                   "pm_reconnect_gaps": pm_gap.iv, "kalshi_gaps": ks_gap.iv,
                   "kalshi_closed": {tk: closed[tk] for tk in tickers if tk in closed}}, f, indent=1)
    print(f"   {slug:<30} {len(streams)//2} outcome(s), {n:,} rows, "
          f"{(t1-t0)/3.6e6:.1f} h, {len(pm_gap.iv)} pm gap(s), {len(ks_gap.iv)} kalshi gap(s) -> {out}")


# ---- session ----------------------------------------------------------------

def align_session(folder, levels, step_ms, pm_time, kalshi_time, max_gap_ms):
    with open(os.path.join(folder, "session.json"), encoding="utf-8") as f:
        spec = json.load(f)
    tag = spec["session"]
    pm_path = _one(folder, f"books_{tag}.jsonl*")
    kp_path = _one(folder, f"kpolls_{tag}.jsonl*")
    clean = _one(folder, f"kbooks_clean_{tag}.jsonl*")
    if not pm_path or not kp_path:
        raise SystemExit(f"{folder}: missing books_{tag} or kpolls_{tag}")
    if not clean:
        from clean_kalshi import clean_session
        clean = clean_session(folder)

    out_dir = os.path.join(folder, "aligned")
    os.makedirs(out_dir, exist_ok=True)
    games = [g for g in spec["games"] if g.get("pm_slug") and g.get("kalshi_events")]
    print(f"{tag}: {len(games)} game(s); reading Kalshi ...")
    ks = load_kalshi(clean, kp_path, levels, kalshi_time, max_gap_ms)
    print("   reading Polymarket (one pass) ...")
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        assets, pm_gaps = split_polymarket(pm_path, {g["pm_slug"] for g in games}, levels, tmp, pm_time)
        for g in games:
            align_game(g, out_dir, tmp, assets, pm_gaps, ks, levels, step_ms)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sessions", nargs="+", help="session folder(s), e.g. ~/plmkt-recordings/<tag>")
    ap.add_argument("--step-ms", type=int, default=100, help="grid step; 0 = one row per update")
    ap.add_argument("--levels", type=int, default=5, help="book levels kept per side")
    ap.add_argument("--pm-time", choices=("recv", "ts"), default="recv")
    ap.add_argument("--kalshi-time", choices=("recv", "mid", "send"), default="recv")
    ap.add_argument("--max-gap-ms", type=int, default=2000,
                    help="Kalshi: no reply for longer than this = state unknown")
    a = ap.parse_args()
    for s in a.sessions:
        align_session(os.path.expanduser(s), a.levels, a.step_ms, a.pm_time, a.kalshi_time, a.max_gap_ms)


if __name__ == "__main__":
    main()
