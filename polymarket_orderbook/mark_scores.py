"""Mark which price jumps in the venue viewer were caused by scoring.

Goal minutes come from match reports (sources in SCORES below). Match-clock
minutes are not wall-clock times, so for each game we fit the real kickoff
(and halftime length) that best lines the goals up with moves in the scoring
side's price, then snap each goal to the sharpest move near its estimate.
Every other jump >= JUMP is labelled noise (or a known non-goal event).

    python mark_scores.py      # writes data/view/score_marks.json
"""
import json
import os

import numpy as np

VIEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "view")
JUMP = 0.06   # a "jump" = mid change of 6c+ over 30 s, any outcome, either venue
LAG = 30      # seconds

# slug -> (sport, [(minute, scoring outcome key, label)], [(minute, label)] other events)
# Football: minute is the match clock; 46+ = second half. MLB: minute = inning
# (+.5 for the bottom half); Tampa Bay is the away side.
SCORES = {
    "lal-rea-sev-2026-10-18": ("football", [(30, "RMA", "Güler 1-0"), (37, "RMA", "Mbappé 2-0"),
        (63, "RMA", "Mbappé pen 3-0"), (80, "SEV", "Bueno 3-1"), (82, "RMA", "Diomande 4-1")], []),
    "fl1-lyo-ogc-2026-10-18": ("football", [(18, "NIC", "Wahi 0-1"), (33, "OL", "Šulc 1-1"),
        (36, "OL", "Tessmann 2-1")], [(70, "Nakamura red card (Lyon)"), (53, "Witsel hits bar (Nice)"),
        (74, "Wahi hits post (Nice)"), (95, "Wahi stoppage-time goal ruled out by VAR (offside)")]),
    "sea-juv-laz-2026-10-18": ("football", [(37, "LAZ", "Provstgaard 0-1"), (66, "LAZ", "Isaksen 0-2"),
        (89, "LAZ", "Dele-Bashiru 0-3")], []),
    "ere-nec-sit-2026-10-18": ("football", [(18, "FOR", "Descotte 0-1"), (28, "NIJ", "Kers 1-1"),
        (48, "FOR", "Descotte 1-2"), (93, "NIJ", "Willumsson 2-2")],
        [(88, "Fortuna penalty given (VAR)"), (90, "NEC penalty missed")]),
    "ere-twe-utr-2026-10-18": ("football", [(28, "TWE", "Weidmann 1-0"), (47, "FCU", "Alarcón 1-1"),
        (63, "FCU", "Alarcón 1-2")], []),
    "bra-gre-cru-2026-10-18": ("football", [(17, "GPA", "C. Vinícius pen 1-0"), (22, "CRU", "Kaio Jorge 1-1"),
        (41, "GPA", "Pavón 2-1"), (48, "CRU", "Kaio Jorge 2-2"), (61, "CRU", "Kaio Jorge 2-3"),
        (80, "CRU", "João Marcelo 2-4")], [(74, "Kannemann second yellow (Grêmio)")]),
    "bra-flu-san-2026-10-18": ("football", [(4, "FLU", "Castillo 1-0"), (49, "SAN", "Gabigol 1-1"),
        (54, "SAN", "Coutinho 1-2")], []),
    "mls-fcc-clb-2026-10-18": ("football", [(22, "CLB", "Tapia 0-1"), (54, "CLB", "Habroune 0-2")], []),
    "bra-bah-fla-2026-10-18": ("football", [(64, "FLA", "Jorginho pen 0-1"), (73, "FLA", "Plata 0-2"),
        (78, "FLA", "Paquetá 0-3")], []),
    "bra-cre-bra-2026-10-18": ("football", [(9, "RBB", "L. Barbosa 0-1"), (54, "CR", "Jajá 1-1")],
        [(10, "long VAR check on Barbosa goal"), (36, "Fernando goal ruled out by VAR (handball)")]),
    "mex-mon1-pac-2026-10-18": ("football", [(13, "MON", "Rossi 1-0"), (16, "MON", "Corona 2-0"),
        (75, "PAC", "Rondón 2-1")], [(86, "Pachuca goal disallowed (offside)"), (12, "Ocampos penalty saved")]),
    "mlb-tb-cle-2026-10-17": ("mlb", [(1, "TB", "DeLuca bases-loaded walk 1-0"), (2, "TB", "Aranda RBI single 2-0"),
        (3, "TB", "Hicks RBI single 3-0"), (6.5, "CLE", "DeLauter RBI groundout 3-1")], []),
}


def mid(series):
    """Average of the two venues' mids, forward-filled; NaN before any quote."""
    a = np.array([np.nan if v is None else v for v in series["pm_mid"]], float)
    b = np.array([np.nan if v is None else v for v in series["ks_mid"]], float)
    m = np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2))
    return m


def moves(m):
    """Price change over the trailing LAG seconds (NaN where either end is missing)."""
    d = np.full_like(m, np.nan)
    d[LAG:] = m[LAG:] - m[:-LAG]
    return d


def clock_s(sport, minute, p):
    """Seconds after fitted kickoff for a match-clock minute, under params p."""
    if sport == "mlb":   # p = inning length in minutes; score lands mid half-inning
        return ((minute - 1) * p + p / 4) * 60
    return (minute - 0.5) * 60 + (p * 60 if minute > 45 else 0)   # p = halftime gap incl. 1st-half stoppage


def win(a, lo, hi):
    """a[lo:hi] clamped to the array (a plain slice wraps on negative bounds)."""
    lo, hi = max(0, lo), min(len(a), hi)
    return a[lo:hi] if hi > lo else a[:0]


def fit(sport, t0, mv, goals, n):
    """Grid-search kickoff offset and clock param maximising the scoring sides' summed moves."""
    params = range(15, 25) if sport == "mlb" else range(15, 31)
    best = (-1e9, 0, 0)
    for p in params:
        for ko in range(0, 75 * 60, 15):   # real kickoff is at or after the recorded one
            s = 0.0
            for minute, key, _ in goals:
                i = t0 + ko + int(clock_s(sport, minute, p))
                w = win(mv[key], i - 60, i + 150)
                s += np.nanmax(w) if w.size and not np.all(np.isnan(w)) else 0
            if s > best[0]:
                best = (s, ko, p)
    return best[1], best[2]


def mark(g, sport, goals, others):
    n = len(g["t"])
    t0 = round((g["kickoff_ms"] - g["t"][0]) / 1000)   # index of recorded kickoff
    mids = {k: mid(s) for k, s in g["series"].items()}
    mv = {k: moves(m) for k, m in mids.items()}
    ko, p = fit(sport, t0, mv, goals, n)

    scores, floor = [], 0
    for minute, key, label in goals:   # in match order; each snaps at least 60 s after the previous
        est = t0 + ko + int(clock_s(sport, minute, p))
        lo, hi = max(LAG, est - 60, floor), min(n, est + 150)
        w = win(mv[key], lo, hi)
        if not w.size or np.all(np.isnan(w)) or np.nanmax(w) < 0.03:
            scores.append({"label": label, "team": key, "minute": minute, "t": g["t"][min(max(est, 0), n - 1)],
                           "matched": False, "move": None})
            continue
        j = lo + int(np.nanargmax(w))
        floor = j + 60
        scores.append({"label": label, "team": key, "minute": minute, "t": g["t"][j],
                       "matched": True, "move": round(float(w[j - lo]), 3)})
    other = [{"label": lab, "t": g["t"][min(max(t0 + ko + int(clock_s(sport, mi, p)), 0), n - 1)]}
             for mi, lab in others]

    # every jump: biggest |30 s move| over outcomes (venue-averaged mid), clustered 120 s apart
    stack = np.vstack([np.abs(np.nan_to_num(v)) for v in mv.values()])
    big = stack.max(0)
    jumps, last = [], -10**9
    for i in np.flatnonzero(big >= JUMP):
        if i - last < 120:
            continue
        last = i
        k = list(mv)[int(stack[:, i].argmax())]
        t = g["t"][i]
        near = [s for s in scores if s["matched"] and abs(s["t"] - t) <= 120_000]
        ev = [o for o in other if abs(o["t"] - t) <= 180_000]
        kind = "score" if near else "event" if ev else "noise"
        jumps.append({"t": t, "outcome": k, "move": round(float(mv[k][i]), 3), "kind": kind,
                      "why": near[0]["label"] if near else ev[0]["label"] if ev else ""})
    return {"kickoff_fit_ms": g["t"][0] + (t0 + ko) * 1000, "clock_param_min": p,
            "scores": scores, "other_events": other, "jumps": jumps}


def main():
    idx = json.load(open(os.path.join(VIEW, "index.json")))
    out = {}
    for sess in idx.values():
        for gi in sess["games"]:
            if gi["slug"] not in SCORES:
                continue
            g = json.load(open(os.path.join(VIEW, gi["file"]), encoding="utf-8"))
            sport, goals, others = SCORES[gi["slug"]]
            out[gi["slug"]] = r = mark(g, sport, goals, others)
            kinds = [j["kind"] for j in r["jumps"]]
            print(f"{gi['slug']:26s} goals matched {sum(s['matched'] for s in r['scores'])}/{len(r['scores'])}  "
                  f"jumps {len(kinds)}: {kinds.count('score')} score, {kinds.count('event')} event, {kinds.count('noise')} noise")
    json.dump(out, open(os.path.join(VIEW, "score_marks.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
