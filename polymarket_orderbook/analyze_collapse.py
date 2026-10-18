"""
Tests the two-venue collapse hypothesis on aligned sessions (align_session.py):

    An order-book collapse on Polymarket AND Kalshi at the same moment is
    more likely to precede a real event (goal, penalty, red card) than a
    collapse on one venue alone.

Definitions (per game, on the 100 ms grid, in-play window only):

  depth D(t)   total size of the top 5 bid + ask levels, summed over every
               outcome of the game, per venue
  collapse     D falls below RATIO x its median over [t-60 s, t-5 s];
               onsets at least 30 s apart
  "clean"      the price has not moved yet at the collapse (every outcome
               within 2c of its [t-30 s, t-5 s] median) -- otherwise the
               collapse is a consequence of a move, not a warning
  event        a price move that sticks: some outcome moves >= 7c from its
               [t-30 s, t-5 s] median within 15 s of the onset, is still >= 5c
               away after 60 s, on BOTH venues. Onset = first 3c deviation.
               There is no second-accurate goal feed, so this stands in for
               goals / penalties / red cards.
  signals      PM     clean Polymarket collapse
               KS     clean Kalshi collapse
               BOTH   PM and KS collapse onsets within --sync seconds;
                      fires at the later one (when both are known)
               PM_ONLY / KS_ONLY: a venue's collapse with no partner

  hit          an event onset within (0, --horizon] s after the signal
  precision    hits / signals;  base = share of random in-play moments
               followed by an event within the horizon;  lift = prec / base
  recall       share of events with a signal in the --horizon before onset

Usage:
    python analyze_collapse.py data/outbox/test4h ~/plmkt-recordings/<tag> ...
    python analyze_collapse.py ... --ratio 0.4 --horizon 10 --sync 2 --events-out ev.csv
"""
import argparse
import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

STEP = 0.1                    # s per row (align_session default grid)
INPLAY = {"football": 115 * 60, "mlb": 4 * 3600}


def rows(sec):
    return int(round(sec / STEP))


def load_game(path, kickoff, sport):
    d = pd.read_csv(path)
    keys = sorted({c.split("_")[0] for c in d.columns if c.endswith("_pm_mid")})
    g = {"t": d["t_ms"].to_numpy() / 1000, "keys": keys}
    for v in ("pm", "ks"):
        g[f"D_{v}"] = sum(d[f"{k}_{v}_{s}{i}_sz"].fillna(0).to_numpy()
                          for k in keys for s in ("bid", "ask") for i in range(1, 6)
                          if f"{k}_{v}_{s}{i}_sz" in d)
        g[f"ok_{v}"] = np.all([d[f"{k}_{v}_ok"].to_numpy() == 1 for k in keys], axis=0)
        g[f"mid_{v}"] = np.stack([d[f"{k}_{v}_mid"].to_numpy(dtype=float) for k in keys])
    k0 = kickoff.timestamp()
    g["inplay"] = (g["t"] >= k0) & (g["t"] <= k0 + INPLAY.get(sport, INPLAY["football"]))
    g["valid"] = g["inplay"] & g["ok_pm"] & g["ok_ks"]
    return g


def roll_median(x, win, lag):
    """median of x over [t-win-lag, t-lag) rows, for every t"""
    return pd.Series(x).rolling(win, min_periods=win // 2).median().shift(lag).to_numpy()


def onsets(flag, gap):
    idx, last = [], -10 ** 9
    for i in np.flatnonzero(flag & ~np.r_[False, flag[:-1]]):
        if i - last >= gap:
            idx.append(i)
            last = i
    return np.array(idx, dtype=int)


def price_dev(mid):
    """per row: max over outcomes of |mid - its [t-30 s, t-5 s] median|, and the base"""
    base = np.stack([roll_median(m, rows(25), rows(5)) for m in mid])
    return np.nanmax(np.abs(mid - base), axis=0), base


def find_events(g):
    dev, base = price_dev(g["mid_pm"])
    flag = (dev >= 0.03) & g["valid"]
    ev = []
    for i in onsets(flag, rows(120)):
        j15, j60 = min(i + rows(15), len(dev) - 1), min(i + rows(60), len(dev) - 1)
        b = base[:, i]
        ok = False
        for v in ("pm", "ks"):
            m15 = np.nanmax(np.abs(g[f"mid_{v}"][:, i:j15 + 1] - b[:, None]), axis=1)
            m60 = np.abs(g[f"mid_{v}"][:, j60] - b)
            ok_v = np.any((m15 >= 0.07) & (m60 >= 0.05))
            ok = ok_v if v == "pm" else (ok and ok_v)
        if ok:
            ev.append(i)
    return np.array(ev, dtype=int), dev


def collapses(g, v, ratio, dev):
    D = g[f"D_{v}"].astype(float)
    base = roll_median(D, rows(55), rows(5))
    flag = (D < ratio * base) & g["valid"] & (base > 0)
    on = onsets(flag, rows(30))
    return on[dev[on] < 0.02] if len(on) else on      # clean only


def pair(a, b, sync):
    """BOTH signals (fire at the later onset) and the unpaired rest of each."""
    both, ua, ub = [], set(a.tolist()), set(b.tolist())
    for i in a:
        near = b[np.abs(b - i) <= rows(sync)]
        if len(near):
            j = near[np.argmin(np.abs(near - i))]
            both.append(max(i, j))
            ua.discard(i)
            ub.discard(j)
    return np.array(sorted(set(both)), dtype=int), np.array(sorted(ua), dtype=int), np.array(sorted(ub), dtype=int)


def analyze(games, ratio, horizon, sync):
    H = rows(horizon)
    tot = {k: [0, 0] for k in ("PM", "KS", "BOTH", "PM_ONLY", "KS_ONLY")}
    base_hits = base_n = 0
    n_ev = 0
    rec = {k: 0 for k in tot}
    leads = {k: [] for k in tot}
    per_event = []
    for name, g in games:
        ev, dev = find_events(g)
        n_ev += len(ev)
        pm = collapses(g, "pm", ratio, dev)
        ks = collapses(g, "ks", ratio, dev)
        both, pm_only, ks_only = pair(pm, ks, sync)
        sig = {"PM": pm, "KS": ks, "BOTH": both, "PM_ONLY": pm_only, "KS_ONLY": ks_only}
        for k, s in sig.items():
            for i in s:
                hit = np.any((ev > i) & (ev <= i + H))
                tot[k][0] += 1
                tot[k][1] += hit
            for e in ev:
                pre = s[(s < e) & (s >= e - H)]
                if len(pre):
                    rec[k] += 1
                    leads[k].append((e - pre.max()) * STEP)
        # base rate: clean, valid moments followed by an event within H
        cand = np.flatnonzero(g["valid"] & (dev < 0.02))[::rows(1)]
        if len(cand) and not len(ev):
            base_n += len(cand)
        elif len(cand):
            nxt = np.searchsorted(ev, cand, side="right")
            nxt_t = np.where(nxt < len(ev), ev[np.minimum(nxt, len(ev) - 1)], 10 ** 12)
            base_hits += int(np.sum(nxt_t - cand <= H))
            base_n += len(cand)
        for e in ev:
            pre = lambda s: (e - s[(s < e) & (s >= e - H)].max()) * STEP if np.any((s < e) & (s >= e - H)) else None
            per_event.append({"game": name, "t_utc": dt.datetime.fromtimestamp(g["t"][e], dt.timezone.utc)
                              .strftime("%H:%M:%S.%f")[:-5], "pm_lead_s": pre(pm), "ks_lead_s": pre(ks),
                              "both_lead_s": pre(both)})
    base = base_hits / max(base_n, 1)
    return tot, rec, leads, n_ev, base, per_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="session folders containing aligned/")
    ap.add_argument("--ratio", type=float, default=0.4, help="collapse: depth < RATIO x recent median")
    ap.add_argument("--horizon", type=float, default=10, help="s after a signal an event must start")
    ap.add_argument("--sync", type=float, default=2, help="s between PM and KS onsets for BOTH")
    ap.add_argument("--min-ks-rate", type=float, default=5,
                    help="skip games whose Kalshi updates came slower than this many polls/s")
    ap.add_argument("--sweep", action="store_true", help="also print a small parameter sweep")
    ap.add_argument("--events-out", default=None, help="write the per-event table here (csv)")
    a = ap.parse_args()

    games, seen = [], set()
    for s in a.sessions:
        s = os.path.expanduser(s)
        spec = json.load(open(os.path.join(s, "session.json")))
        kick = {g["pm_slug"]: (dt.datetime.fromisoformat(g["kickoff"]), g["sport"]) for g in spec["games"]}
        kp = glob.glob(os.path.join(s, "kalshi", "kpolls_*.jsonl.xz"))
        rate = None
        if kp:
            import lzma
            ts = [json.loads(l)["t_recv"] for l in lzma.open(kp[0], "rt")
                  if '"type":"poll"' in l and '"status":200' in l]
            rate = len(ts) / max((max(ts) - min(ts)) / 1000, 1) if ts else 0
        for f in sorted(glob.glob(os.path.join(s, "aligned", "*.csv.gz"))):
            slug = os.path.basename(f)[:-7]
            if slug in seen:
                print(f"skip {slug} in {os.path.basename(s)} (already loaded from an earlier session)")
                continue
            if rate is not None and rate < a.min_ks_rate:
                print(f"skip {slug}: Kalshi only {rate:.1f} polls/s in {os.path.basename(s)}")
                continue
            k, sport = kick[slug]
            g = load_game(f, k, sport)
            if g["valid"].sum() < rows(600):
                print(f"skip {slug}: under 10 min of in-play data on both venues")
                continue
            seen.add(slug)
            games.append((slug, g))
            print(f"loaded {slug}: {g['valid'].sum() * STEP / 60:.0f} min in-play on both venues")
    if not games:
        raise SystemExit("no usable games")

    tot, rec, leads, n_ev, base, per_event = analyze(games, a.ratio, a.horizon, a.sync)
    print(f"\n{len(games)} games, {n_ev} events (sticky >=7c moves on both venues)")
    print(f"collapse < {a.ratio:.0%} of recent depth, clean (price not yet moved), "
          f"hit = event within {a.horizon:g} s, BOTH = onsets within {a.sync:g} s")
    print(f"base rate (random clean moment -> event within {a.horizon:g} s): {base:.2%}\n")
    print(f"{'signal':<9}{'signals':>9}{'hits':>6}{'precision':>11}{'lift':>8}{'recall':>9}{'median lead':>13}")
    for k in ("PM", "KS", "BOTH", "PM_ONLY", "KS_ONLY"):
        n, h = tot[k]
        p = h / n if n else float("nan")
        lead = f"{np.median(leads[k]):.1f} s" if leads[k] else "-"
        print(f"{k:<9}{n:>9}{h:>6}{p:>11.1%}{p / base if base else float('nan'):>8.1f}"
              f"{rec[k] / max(n_ev, 1):>9.0%}{lead:>13}")
    if a.events_out:
        pd.DataFrame(per_event).to_csv(a.events_out, index=False)
        print(f"\nper-event table -> {a.events_out}")
    if a.sweep:
        print("\nsweep: precision (signals)  for PM / KS / BOTH")
        for r in (0.3, 0.4, 0.5, 0.6):
            for hz in (5, 10, 20):
                t, _, _, _, b, _ = analyze(games, r, hz, a.sync)
                cell = "  ".join(f"{k} {t[k][1] / max(t[k][0], 1):5.1%} ({t[k][0]:>3})" for k in ("PM", "KS", "BOTH"))
                print(f"  ratio {r:.1f} horizon {hz:>2}s  base {b:5.2%}   {cell}")


if __name__ == "__main__":
    main()
