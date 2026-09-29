"""Mark every SINGLE upward jump that breaks even, read off the price graph.

Justin asked for the upward jumps that are large enough to break even marked
by hand-checkable rules on the price graph, with one condition: a mark must be
ONE jump, never two jumps glued together to reach break-even. The one
exception he allowed is two jumps between which the market microstructure did
not change; those are marked too, but drawn differently.

WHAT A JUMP IS HERE
-------------------
Everything runs on the executable prices, bid and ask, never the mid. On these
books the mid is dominated by quote pulls: on almost every pitch the makers
withdraw, the spread blows out to 40-90 ticks for a few seconds and the book
comes back where it was. That spikes the mid and moves nothing you could trade.

  level     where the market SAT. Two kinds:
              plateau  >= PLATEAU_S with the bid within one cent and the ask
                       within one cent, AND a quoted book (spread not blown
                       out, see regimes below). When the makers pull, the
                       emptied book often stands motionless for seconds with
                       the bid far below and the ask far above; that is the
                       middle of a jump, not a price anyone traded at.
              rest     inside a transition, >= PLATEAU_S with the BID within
                       one cent at a price >= MARGIN away from where the bid
                       stood on both surrounding plateaus. On spread markets
                       the ask flickers on every pitch, so the bid can sit
                       part-way up for many seconds without the whole book
                       ever being still. A long sells into the bid, so a bid
                       that sat part-way is a first move that could have been
                       banked: it ends a jump. An ask resting part-way while
                       bids are still absent does not.

  jump      the transition between two consecutive levels A -> B. Whatever
            happens in between -- a burst of steps, a blown-out book, pauses
            shorter than PLATEAU_S -- is one jump, because the market never
            sat anywhere. A move A -> B -> C is two jumps and is only ever
            priced as A -> B and B -> C separately.

  entry     buy at the ask standing in the last grid slot of A, i.e. the
            price on the book immediately before the jump began. (On a rest,
            whose ask may be flickering: the lowest ask that stood STAND_S
            within A's last PLATEAU_S, never before the previous mark's exit.)

  exit      sell at the highest bid on B that STOOD for at least STAND_S
            seconds, searched over B's first EXIT_S seconds. A bid that
            flickers for one 200ms slot cannot be sold into and is ignored.

  breaks    (exit bid - entry ask) in ticks minus the sports taker fee on both
  even      legs is > 0, AND the touch held at least $STAKE at both ends, so a
            $STAKE trade fills at exactly those two prices.

THE MICROSTRUCTURE EXCEPTION
----------------------------
The book is in one of two regimes. CALM: its spread is near that contract's
normal (in-game median) spread. DISTURBED: the spread is blown out -- median
spread above normal + max(2 ticks, normal) -- which is what makers pulling
their quotes looks like. The regime of a jump is the regime of the book during
its transition (an instant step is calm); the regime of a plateau is the
regime of the book while it sits there.

Two consecutive upward jumps A -> B -> C are marked as a COMBINED jump when
  * the book is in ONE regime throughout: during the first jump, on the pause
    B, and during the second jump -- the microstructure did not change over
    the two jumps (typically: the makers pulled out for the first step and had
    still not come back when the second one arrived);
  * A -> C breaks even by the same rule as a single jump;
  * neither A -> B nor B -> C breaks even on its own (otherwise the single
    mark already covers it);
  * the pause B lasts at most PAUSE_MAX_S. A calm book that sits for minutes
    and then ticks up again is drift, not one move that hesitated.
A pause where the book went back to normal is a genuine end of the first jump,
and a pair straddling it is not marked at all.

WHAT IS EXCLUDED, AND WHY
-------------------------
  recording silences   any mark whose window [entry - GAP_PAD_S, exit]
                       overlaps a >5s silence of the whole match stream from
                       data/continuity/. Across a silence the grid carries the
                       last book forward, so a flat plateau and then an instant
                       step is exactly what a dropped connection looks like.
  slow drifts          transitions longer than MAX_JUMP_S. A two-minute grind
                       with no pause is a drift, not a jump.
  thin touch           fewer than $STAKE at the entry ask or at the exit bid.

Outputs, under results/makinen/single_jumps/
  marks.csv            every marked jump (single and combined), one row each
  excluded.csv         break-even candidates dropped by the rules above
  candidates.parquet   both of the above together
  games/<slug>.png     one picture per game: every mark as its own zoomed
                       panel, in time order

No two marks on one contract share any stretch of price: check_no_overlap
asserts it on the output.

This is an ORACLE. It knows when each jump starts and ends; the marks are an
upper bound on what trading single jumps could earn, not a strategy.

    python research/makinen/mark_single_jumps.py
    python research/makinen/mark_single_jumps.py --sessions books_2026-09-18
    python research/makinen/mark_single_jumps.py --games mlb-atl-hou-2026-09-18
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from match_filter import load_windows  # noqa: E402
from mark_breakeven_exec import fee_ticks, SESSIONS  # noqa: E402

TICK = 0.01
GRID_MS = 200
TOL = 0.01 + 1e-6      # "within one cent", with float slack
PLATEAU_S = 5.0        # a level the market sat at for at least this long
STAND_S = 1.0          # an exit bid must have stood this long
EXIT_S = 30.0          # exit searched over this much of the post-jump plateau
MAX_JUMP_S = 60.0      # longer transitions are drifts, not jumps
GAP_PAD_S = 5.0        # a silence this close before entry also disqualifies
MARGIN = 0.015         # a one-sided rest must sit >= 2 cents from both ends
PAUSE_MAX_S = 30.0     # a combined mark's pause may last at most this long
STAKE = 10.0

OUT = ROOT / "results" / "makinen" / "single_jumps"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_session(session: str) -> pd.DataFrame:
    p = ROOT / "data" / "jump" / f"feat_{session}_trimmed.parquet"
    df = pq.read_table(p, columns=["mid", "spread_ticks", "ts", "series", "sid"]).to_pandas()
    lob = np.load(ROOT / "data" / "jump" / f"lob_{session}_trimmed.npy", mmap_mode="r")
    assert lob.shape[0] == len(df), "LOB tensor and feature grid are not row-aligned"
    # tensor channels [bid_px, bid_sz, ask_px, ask_sz]; sizes are log1p(dollars)
    df["bid_usd"] = np.expm1(np.asarray(lob[:, 1, 0], dtype=np.float64))
    df["ask_usd"] = np.expm1(np.asarray(lob[:, 3, 0], dtype=np.float64))
    # Touch prices. Exact: mid = (bid+ask)/2 and spread = ask-bid. Rounded to
    # 1e-4 to remove float32 noise; the finest tick is 0.001.
    half = df.spread_ticks.to_numpy(np.float64) * TICK / 2.0
    mid = df.mid.to_numpy(np.float64)
    df["bid"] = np.round(mid - half, 4)
    df["ask"] = np.round(mid + half, 4)
    df = df.sort_values(["sid", "ts"], kind="stable").reset_index(drop=True)
    return df


def load_gaps(session: str) -> dict:
    """slug -> array of (start_ms, end_ms) silences on the match stream."""
    p = ROOT / "data" / "continuity" / (session.replace("books_", "") + ".json")
    if not p.exists():
        raise SystemExit(f"no continuity scan for {session}: run continuity_scan.py first")
    out = {}
    for m in json.load(open(p, encoding="utf-8")):
        g = [(x["start"], x["end"]) for x in m.get("gaps", [])]
        out[m["slug"]] = np.array(g, dtype=np.int64).reshape(-1, 2)
    return out


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------

def plateaus(ts, bid, ask, min_s=PLATEAU_S):
    """Greedy left-to-right over runs of constant (bid, ask).

    A segment grows while its bid range and ask range both stay within one
    cent and the grid stays contiguous; it is a plateau if it lasts min_s.
    Returns inclusive grid index pairs.
    """
    n = len(ts)
    brk = np.empty(n, bool)
    brk[0] = True
    brk[1:] = (bid[1:] != bid[:-1]) | (ask[1:] != ask[:-1]) | (np.diff(ts) != GRID_MS)
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:] - 1, n - 1)
    rb, ra = bid[starts], ask[starts]
    contig = np.ones(len(starts), bool)
    contig[1:] = ts[starts[1:]] - ts[ends[:-1]] == GRID_MS
    need = int(round(min_s * 1000 / GRID_MS))
    out = []
    s0 = 0
    bmin = bmax = rb[0]
    amin = amax = ra[0]
    for r in range(1, len(starts)):
        b, a = rb[r], ra[r]
        nb0, nb1 = min(bmin, b), max(bmax, b)
        na0, na1 = min(amin, a), max(amax, a)
        if contig[r] and nb1 - nb0 <= TOL and na1 - na0 <= TOL:
            bmin, bmax, amin, amax = nb0, nb1, na0, na1
            continue
        if ends[r - 1] - starts[s0] + 1 >= need:
            out.append((starts[s0], ends[r - 1]))
        s0 = r
        bmin = bmax = b
        amin = amax = a
    if ends[-1] - starts[s0] + 1 >= need:
        out.append((starts[s0], ends[-1]))
    return out


def intermediate_rests(ts, bid, ask, a1, b0, min_s=PLATEAU_S):
    """Bid rests strictly inside the transition (a1, b0).

    A rest is >= min_s with the bid within one cent, at a level at least
    MARGIN away from where the bid stood on both surrounding plateaus. Only
    the bid can end an upward jump: it is the side a long sells into, so a bid
    that sat at an intermediate price is a first move you could have banked.
    An ask that sits part-way up while the bid is still absent is the book
    re-quoting from the top, not a pause in the move.
    """
    i0, i1 = a1 + 1, b0 - 1
    need = int(round(min_s * 1000 / GRID_MS))
    if i1 - i0 + 1 < need:
        return []
    found = []
    for x in (bid,):
        lo = min(x[a1], x[b0]) + MARGIN
        hi = max(x[a1], x[b0]) - MARGIN
        if lo > hi:
            continue
        seg = x[i0:i1 + 1]
        for j0, j1 in plateaus(ts[i0:i1 + 1], seg, seg, min_s):
            if lo <= np.median(seg[j0:j1 + 1]) <= hi:
                found.append([i0 + j0, i0 + j1])
    found.sort()
    merged = []
    for r in found:
        if merged and r[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return [(r0, r1, "rest") for r0, r1 in merged]


def best_standing_ask(ask, i0, i1, stand):
    """Lowest ask held for `stand` consecutive slots within [i0, i1]."""
    p, i = best_standing_bid(-ask, i0, i1, stand)
    return -p, i


def best_standing_bid(bid, i0, i1, stand):
    """Highest bid held for `stand` consecutive slots within [i0, i1].

    Returns (price, index of the first slot of that stand) or (nan, -1).
    """
    seg = bid[i0:i1 + 1]
    if len(seg) < stand:
        return np.nan, -1
    w = np.lib.stride_tricks.sliding_window_view(seg, stand).min(axis=1)
    k = int(np.argmax(w))
    return float(w[k]), i0 + k


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------

def net_ticks(entry_ask, exit_bid):
    return (exit_bid - entry_ask) / TICK - fee_ticks(entry_ask) - fee_ticks(exit_bid)


def overlaps(gaps, a, b):
    if len(gaps) == 0:
        return False
    return bool(np.any((gaps[:, 0] < b) & (gaps[:, 1] > a)))


def scan_series(s: pd.DataFrame, gaps) -> list[dict]:
    ts = s.ts.to_numpy(np.int64)
    bid = s.bid.to_numpy(np.float64)
    ask = s.ask.to_numpy(np.float64)
    spr = s.spread_ticks.to_numpy(np.float64)
    bu = s.bid_usd.to_numpy(np.float64)
    au = s.ask_usd.to_numpy(np.float64)
    if len(ts) < 100:
        return []
    normal = float(np.median(spr))
    cut = normal + max(2.0, normal)          # above this the book is disturbed

    # A still book is only a LEVEL if it is also a quoted one. When the makers
    # pull out, the emptied book often stands motionless for several seconds
    # with the bid far below and the ask far above; that is the middle of a
    # jump, not a price the market sat at. Such stretches are dropped here and
    # left to intermediate_rests, which keeps them only if one side genuinely
    # rests at a price between the surrounding levels.
    P = [(i0, i1) for i0, i1 in plateaus(ts, bid, ask)
         if np.median(spr[i0:i1 + 1]) <= cut]
    if len(P) < 2:
        return []
    stand = int(round(STAND_S * 1000 / GRID_MS))
    exit_n = int(round(EXIT_S * 1000 / GRID_MS))
    mid = (bid + ask) / 2.0

    def regime_of(i0, i1):
        if i1 < i0:                          # an instant step has no transition
            return "calm"
        return "disturbed" if np.median(spr[i0:i1 + 1]) > cut else "calm"

    # The levels the market sat at: every two-sided plateau, plus any
    # ONE-sided rest inside a transition at a level strictly between the two
    # plateaus. On spread markets the ask flickers on every pitch, so the bid
    # can sit at an intermediate price for many seconds without the whole book
    # ever being still; a seller could have got out there, so it ends a jump.
    levels = []
    for (a0, a1), (b0, b1) in zip(P[:-1], P[1:]):
        levels.append((a0, a1, "plateau"))
        levels += intermediate_rests(ts, bid, ask, a1, b0)
    levels.append((P[-1][0], P[-1][1], "plateau"))

    # one record per consecutive pair of levels
    steps = []
    prev_exit_end = -1                       # last slot the previous exit's bid stood
    for (a0, a1, ka), (b0, b1, kb) in zip(levels[:-1], levels[1:]):
        contiguous = ts[b0] - ts[a1] == (b0 - a1) * GRID_MS
        # Entry never precedes the previous step's exit on the same level, so
        # no stretch of price is ever inside two marks.
        lo = max(a0, a1 - int(PLATEAU_S * 1000 / GRID_MS) + 1, prev_exit_end + 1)
        if ka == "plateau" or lo > a1 - stand + 1:
            ea, ei = ask[a1], a1                 # the ask standing as the jump began
        else:                                    # a rest's ask may be flickering
            ea, ei = best_standing_ask(ask, lo, a1, stand)
        xb, xi = best_standing_bid(bid, b0, min(b1, b0 + exit_n - 1), stand)
        prev_exit_end = xi + stand - 1 if xi >= 0 else b0
        steps.append(dict(
            a0=a0, a1=a1, b0=b0, b1=b1, contiguous=contiguous,
            pre_kind=ka, post_kind=kb,
            entry_ask=ea, entry_i=ei, exit_bid=xb, exit_i=xi,
            jump_s=(ts[b0] - ts[a1]) / 1000.0,
            jump_regime=regime_of(a1 + 1, b0 - 1),
            pause_regime=regime_of(b0, b1),
            # on the bid: a rest's mid is inflated by the empty ask above it
            up=np.median(bid[b0:b1 + 1]) > np.median(bid[a0:a1 + 1]),
        ))

    def record(kind, first, last, mid_step=None):
        e_i, x_i = first["entry_i"], last["exit_i"]
        e_px, x_px = first["entry_ask"], last["exit_bid"]
        net = net_ticks(e_px, x_px)
        shares = STAKE / e_px
        reasons = []
        if not (first["contiguous"] and last["contiguous"]):
            reasons.append("grid_gap")
        if max(first["jump_s"], last["jump_s"]) > MAX_JUMP_S:
            reasons.append("drift")
        if au[e_i] < STAKE:
            reasons.append("thin_entry")
        if bu[x_i] / x_px < shares:
            reasons.append("thin_exit")
        if overlaps(gaps, ts[e_i] - GAP_PAD_S * 1000, ts[x_i]):
            reasons.append("recording_gap")
        r = dict(
            kind=kind, series=s.series.iat[0],
            ts_entry=int(ts[e_i]), ts_jump_end=int(ts[last["b0"]]), ts_exit=int(ts[x_i]),
            ts_pre_start=int(ts[first["a0"]]), ts_post_end=int(ts[last["b1"]]),
            entry_ask=e_px, entry_bid=bid[e_i], entry_spread=spr[e_i],
            exit_bid=x_px, exit_ask=ask[x_i], exit_spread=spr[x_i],
            gross_ticks=(x_px - e_px) / TICK,
            fee_ticks=float(fee_ticks(e_px) + fee_ticks(x_px)),
            net_ticks=net, pnl_usd=net * TICK * shares,
            jump_s=(ts[last["b0"]] - ts[e_i]) / 1000.0,
            jump_regime=first["jump_regime"], normal_spread=normal,
            pre_kind=first["pre_kind"], post_kind=last["post_kind"],
            entry_ask_usd=au[e_i], exit_bid_usd=bu[x_i],
            excluded=";".join(reasons),
        )
        if mid_step is not None:
            r.update(ts_pause_start=int(ts[mid_step["b0"]]), ts_pause_end=int(ts[mid_step["b1"]]),
                     pause_regime=mid_step["pause_regime"],
                     pause_s=(ts[mid_step["b1"]] - ts[mid_step["b0"]]) / 1000.0 + GRID_MS / 1000.0,
                     step1_net=net_ticks(first["entry_ask"], first["exit_bid"]),
                     step2_net=net_ticks(last["entry_ask"], last["exit_bid"]))
        return r

    out = []
    single_be = []
    for st in steps:
        ok = np.isfinite(st["exit_bid"]) and st["up"]
        be = ok and net_ticks(st["entry_ask"], st["exit_bid"]) > 0
        single_be.append(be)
        if be:
            out.append(record("single", st, st))
    used = -1                                # last step consumed by a kept combined
    for k in range(len(steps) - 1):
        s1, s2 = steps[k], steps[k + 1]
        if single_be[k] or single_be[k + 1] or k <= used:
            continue
        if not (s1["up"] and s2["up"] and np.isfinite(s2["exit_bid"])):
            continue
        # the pause is s1's post-level == s2's pre-level; the book must be
        # in one regime through jump 1, the pause and jump 2
        if not (s1["jump_regime"] == s1["pause_regime"] == s2["jump_regime"]):
            continue
        if (ts[s1["b1"]] - ts[s1["b0"]]) / 1000.0 + GRID_MS / 1000.0 > PAUSE_MAX_S:
            continue
        if net_ticks(s1["entry_ask"], s2["exit_bid"]) <= 0:
            continue
        r = record("combined", s1, s2, mid_step=s1)
        out.append(r)
        if r["excluded"] == "":
            used = k + 1        # A->B->C and B->C->D would both contain B->C
    return out


def check_no_overlap(c: pd.DataFrame) -> None:
    """No two kept marks on one contract may share any stretch of price.

    Checked on the output, not assumed from the construction: sort each
    contract's marks by entry and require every entry to come at or after the
    previous exit.
    """
    k = c[c.excluded == ""].sort_values(["series", "ts_entry"])
    same = k.series.to_numpy()[1:] == k.series.to_numpy()[:-1]
    bad = same & (k.ts_entry.to_numpy()[1:] < k.ts_exit.to_numpy()[:-1])
    assert not bad.any(), f"{int(bad.sum())} overlapping marks"
    assert (k.exit_bid > k.entry_ask).all(), "a mark that is not upward"
    assert (k.ts_exit > k.ts_entry).all(), "an exit that does not follow its entry"


def scan_session(session: str) -> pd.DataFrame:
    df = load_session(session)
    gaps = load_gaps(session)
    rows = []
    for _, s in df.groupby("sid", sort=False):
        slug = s.series.iat[0].split("|")[0]
        rows += scan_series(s, gaps.get(slug, np.empty((0, 2), np.int64)))
    out = pd.DataFrame(rows)
    if len(out):
        check_no_overlap(out)
        parts = out.series.str.split("|", expand=True)
        out.insert(0, "session", session)
        out.insert(1, "slug", parts[0])
        out.insert(2, "market_type", parts[1])
        out.insert(3, "line", parts[2])
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

# The repo's chart tokens (see mark_upward_breakeven.py for the checks). Kind
# is also carried by marker shape and by the panel label, never colour alone.
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
LINE = "#8d9299"
BAND = "#dcdedf"
GRID_C = "#e9e8e4"
KIND_C = {"single": "#2a78d6", "combined": "#eb6834"}
KIND_M = {"single": "o", "combined": "D"}
CTX_S = 40.0          # seconds of price shown either side of the mark
PANELS = (6, 4)       # rows x cols per sheet


def draw_panel(ax, m, s, t_game0):
    lo = m.ts_entry - CTX_S * 1000
    hi = m.ts_exit + CTX_S * 1000
    w = s[(s.ts >= lo) & (s.ts <= hi)]
    x = (w.ts.to_numpy() - m.ts_entry) / 1000.0
    c = KIND_C[m.kind]
    ax.fill_between(x, w.bid, w.ask, step="post", color=BAND, lw=0, zorder=1)
    ax.step(x, w.ask, where="post", color=LINE, lw=0.8, zorder=2)
    ax.step(x, w.bid, where="post", color=INK2, lw=0.9, zorder=2)
    xj = (m.ts_jump_end - m.ts_entry) / 1000.0
    ax.axvspan(0, xj, color=c, alpha=0.10, lw=0, zorder=0)
    if m.kind == "combined":
        p0 = (m.ts_pause_start - m.ts_entry) / 1000.0
        p1 = (m.ts_pause_end - m.ts_entry) / 1000.0
        # drawn above the book band so the pause stays visible
        ax.axvspan(p0, p1, facecolor="none", edgecolor=c, hatch="////",
                   lw=0, alpha=0.45, zorder=2.5)
    xe = (m.ts_exit - m.ts_entry) / 1000.0
    ax.plot([0, xe], [m.entry_ask, m.exit_bid], color=c, lw=2, zorder=3)
    ax.plot([0, xe], [m.entry_ask, m.exit_bid], ls="none", marker=KIND_M[m.kind],
            ms=6, mfc=c, mec=SURFACE, mew=1.5, zorder=4)
    ys = np.concatenate([w.bid.to_numpy(), w.ask.to_numpy()])
    ys = ys[np.isfinite(ys)]
    # frame on the traded levels; a blown-out book is allowed to run off-panel
    y0 = min(m.entry_bid, m.entry_ask, m.exit_bid) - 0.02
    y1 = max(m.exit_ask, m.exit_bid, m.entry_ask) + 0.02
    if len(ys):
        y0 = max(y0 - 0.03, min(y0, np.percentile(ys, 2)))
        y1 = min(y1 + 0.03, max(y1, np.percentile(ys, 98)))
    ax.set_ylim(y0, y1)
    ax.set_xlim(-CTX_S, xe + CTX_S)
    minute = (m.ts_entry - t_game0) / 60000.0 if t_game0 else float("nan")
    line = "" if m.line in (None, "None") else f" {m.line}"
    ax.set_title(f"{m.slug.replace('mlb-', '')}  {m.market_type}{line} ..{m.series[-4:]}  "
                 f"min {minute:.1f}", fontsize=7.5, color=INK, loc="left", pad=3)
    tag = ("SINGLE" if m.kind == "single" else f"COMBINED, pause {m.pause_s:.0f}s {m.pause_regime}")
    ax.text(0.02, 0.97, f"{tag}\nbuy {m.entry_ask:.3f} -> sell {m.exit_bid:.3f}\n"
            f"net {m.net_ticks:+.1f}t  \\${m.pnl_usd:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=c, lw=1))
    ax.tick_params(labelsize=6, colors=INK2, length=2)
    ax.grid(color=GRID_C, lw=0.5)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.set_facecolor(SURFACE)


def draw_sheets(marks: pd.DataFrame, outdir: Path, prefix="sheet") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    windows = load_windows()
    marks = marks.sort_values(["session", "slug", "ts_entry"]).reset_index(drop=True)
    per = PANELS[0] * PANELS[1]
    paths = []
    cache = {}
    for sheet, lo in enumerate(range(0, len(marks), per), start=1):
        chunk = marks.iloc[lo:lo + per]
        fig, axes = plt.subplots(*PANELS, figsize=(16, 19), facecolor=SURFACE)
        for ax in axes.flat[len(chunk):]:
            ax.set_visible(False)
        for ax, m in zip(axes.flat, chunk.itertuples()):
            if m.session not in cache:
                cache.clear()
                cache[m.session] = load_session(m.session)
            df = cache[m.session]
            s = df[df.series == m.series]
            draw_panel(ax, m, s, windows.get(m.slug, (None,))[0])
        first, last = chunk.iloc[0], chunk.iloc[-1]
        fig.suptitle(f"Upward jumps that break even at the touch, ${STAKE:.0f} a trade   "
                     f"sheet {sheet}   marks {lo + 1}-{lo + len(chunk)} of {len(marks)}   "
                     f"({first.slug} .. {last.slug})",
                     fontsize=11, color=INK, x=0.01, ha="left", y=0.997)
        legend = [
            Line2D([], [], color=INK2, lw=1, label="bid"),
            Line2D([], [], color=LINE, lw=1, label="ask"),
            Line2D([], [], color=KIND_C["single"], lw=2, marker="o", mec=SURFACE,
                   label="single jump: buy ask before, sell bid after"),
            Line2D([], [], color=KIND_C["combined"], lw=2, marker="D", mec=SURFACE,
                   label="combined: two jumps, book regime unchanged across the pause (hatched)"),
        ]
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.01, 0.988), ncol=4,
                   frameon=False, fontsize=8, labelcolor=INK)
        fig.supxlabel("seconds from entry (shaded: the jump itself)", fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.01, 1, 0.975))
        p = outdir / f"{prefix}_{sheet:02d}.png"
        fig.savefig(p, dpi=110, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
    return paths


def contract_label(m):
    line = "" if m.line in (None, "None") or pd.isna(m.line) else f" {m.line}"
    return f"{m.market_type}{line} ..{m.series[-4:]}"


def draw_overview(ax, s, ms, t_game0, show_x, xlim):
    """The whole match for one contract, with its marks where they happened.

    Only the QUOTED book is drawn (spread at or below the disturbed cut):
    during a quote pull the touch runs to 0.01 / 0.99 and would bury the
    price path under vertical spikes. Gaps in the line are those pulls.
    """
    spr = s.spread_ticks.to_numpy()
    normal = float(np.median(spr))
    cut = normal + max(2.0, normal)
    x = (s.ts.to_numpy() - t_game0) / 60000.0
    q = spr <= cut
    bid = np.where(q, s.bid.to_numpy(), np.nan)
    ask = np.where(q, s.ask.to_numpy(), np.nan)
    ax.fill_between(x, bid, ask, color=BAND, lw=0, zorder=1)
    ax.plot(x, ask, color=LINE, lw=0.6, zorder=2)
    ax.plot(x, bid, color=INK2, lw=0.7, zorder=2)
    for m in ms.itertuples():
        c = KIND_C[m.kind]
        xm = (m.ts_entry - t_game0) / 60000.0
        ax.plot([xm, xm], [m.entry_ask, m.exit_bid], color=c, lw=2.2, zorder=4,
                solid_capstyle="round")
        ax.plot([xm], [m.exit_bid], ls="none", marker=KIND_M[m.kind], ms=4.5, mfc=c,
                mec=SURFACE, mew=1, zorder=5)
        ax.annotate(str(m.n), (xm, m.exit_bid), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.5, color=INK2, zorder=6)
    lo = np.nanmin([np.nanpercentile(bid, 1), ms.entry_ask.min()])
    hi = np.nanmax([np.nanpercentile(ask, 99), ms.exit_bid.max()])
    pad = max(0.02, 0.08 * (hi - lo))
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax.set_xlim(*xlim)                     # one time axis for every contract
    ax.text(0.003, 0.95, f"{contract_label(ms.iloc[0])}   {len(ms)} mark(s), "
            f"net {ms.net_ticks.sum():.1f}t", transform=ax.transAxes, ha="left",
            va="top", fontsize=8, color=INK,
            bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    ax.tick_params(labelsize=6.5, colors=INK2, length=2, labelbottom=show_x)
    ax.grid(color=GRID_C, lw=0.5)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.set_facecolor(SURFACE)
    if show_x:
        ax.set_xlabel("minutes from first pitch", fontsize=8, color=INK2)


def draw_game(slug, gm, by_series, t_game0, path, cols=6):
    """One PNG per match.

    Top: every contract that has a mark, over the whole match, with each mark
    drawn where it happened (a vertical bar from the ask paid to the bid sold)
    and numbered. Below: every mark as its own zoomed panel, same numbers, in
    time order, so any bar on the match plot can be looked up and checked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    gm = gm.sort_values("ts_entry").reset_index(drop=True)
    gm["n"] = np.arange(1, len(gm) + 1)
    if t_game0 is None:
        t_game0 = int(gm.ts_entry.min())
    order = (gm.groupby("series").ts_entry.min().sort_values().index.tolist())
    order.sort(key=lambda sr: {"moneyline": 0, "spread": 1, "total": 2}.get(sr.split("|")[1], 3))
    n_ov = len(order)
    rows = int(np.ceil(len(gm) / cols))

    head, ov_h, gap, z_h = 0.95, 1.15, 0.75, 2.35   # inches
    H = head + n_ov * ov_h + 0.35 + gap + rows * z_h
    W = cols * 2.95
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)

    ov_top = 1 - head / H
    ov_bot = ov_top - (n_ov * ov_h + 0.35) / H
    g1 = fig.add_gridspec(n_ov, 1, left=0.035, right=0.995, top=ov_top, bottom=ov_bot + 0.35 / H,
                          hspace=0.12)
    t_lo = min(int(by_series[sr].ts.min()) for sr in order)
    t_hi = max(int(by_series[sr].ts.max()) for sr in order)
    xlim = ((t_lo - t_game0) / 60000.0 - 1, (t_hi - t_game0) / 60000.0 + 1)
    for r, sr in enumerate(order):
        ax = fig.add_subplot(g1[r, 0])
        draw_overview(ax, by_series[sr], gm[gm.series == sr], t_game0,
                      show_x=(r == n_ov - 1), xlim=xlim)

    z_top = ov_bot - gap / H
    g2 = fig.add_gridspec(rows, cols, left=0.035, right=0.995, top=z_top, bottom=0.25 / H,
                          hspace=0.42, wspace=0.28)
    for k, m in enumerate(gm.itertuples()):
        ax = fig.add_subplot(g2[k // cols, k % cols])
        draw_panel(ax, m, by_series[m.series], t_game0)
        ax.set_title(f"#{m.n}  " + ax.get_title(loc="left").split("  ", 1)[1],
                     fontsize=7.5, color=INK, loc="left", pad=3)
    fig.text(0.035, z_top + 0.30 / H, "Every mark, zoomed (x = seconds from entry; shaded = the jump; "
             "numbers match the bars above)", fontsize=10, color=INK, ha="left", va="bottom")

    n_s = int((gm.kind == "single").sum())
    n_c = int((gm.kind == "combined").sum())
    fig.text(0.005, 1 - 0.18 / H,
             f"{slug}   {n_s} single + {n_c} combined upward jump(s) that break even at the touch"
             f"   net {gm.net_ticks.sum():.1f} ticks, \\${gm.pnl_usd.sum():.2f} at "
             f"\\${STAKE:.0f} a trade (perfect-foresight marks, not a strategy)",
             fontsize=12, color=INK, ha="left", va="top")
    handles = [
        Line2D([], [], color=INK2, lw=1, label="bid"),
        Line2D([], [], color=LINE, lw=1, label="ask"),
        Line2D([], [], color=KIND_C["single"], lw=2, marker="o", mec=SURFACE,
               label="single jump: buy the ask before it, sell a bid that stood after it"),
        Line2D([], [], color=KIND_C["combined"], lw=2, marker="D", mec=SURFACE,
               label="combined: two jumps, book regime unchanged across the pause (hatched)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.42 / H),
               ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.995, 1 - 0.50 / H, "match plots show the quoted book only; gaps are quote pulls",
             fontsize=8, color=INK2, ha="right", va="top")
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def draw_all(marks: pd.DataFrame, figdir: Path, games=None) -> int:
    windows = load_windows()
    made = 0
    for sess, sm in marks.groupby("session"):
        if games:
            sm = sm[sm.slug.isin(games)]
        if not len(sm):
            continue
        df = load_session(sess)
        df = df[df.series.isin(set(sm.series))]
        by_series = {k: v for k, v in df.groupby("series", sort=False)}
        for slug, gm in sm.groupby("slug"):
            draw_game(slug, gm, by_series, windows.get(slug, (None,))[0],
                      figdir / f"{slug}.png")
            made += 1
        print(f"  drew {sess}: {sm.slug.nunique()} game(s)", flush=True)
        del df, by_series
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=SESSIONS)
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--games", nargs="*", default=None, help="only draw these slugs")
    ap.add_argument("--from-marks", action="store_true",
                    help="skip the scan and redraw from the saved candidates.parquet")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.from_marks:
        allc = pd.read_parquet(outdir / "candidates.parquet")
    else:
        frames = []
        for sess in args.sessions:
            f = scan_session(sess)
            n_ok = int((f.excluded == "").sum()) if len(f) else 0
            print(f"{sess}: {len(f)} break-even candidates, {n_ok} kept", flush=True)
            frames.append(f)
        allc = pd.concat(frames, ignore_index=True)
        allc.to_parquet(outdir / "candidates.parquet", index=False)
        allc[allc.excluded == ""].to_csv(outdir / "marks.csv", index=False)
        allc[allc.excluded != ""].to_csv(outdir / "excluded.csv", index=False)
    marks = allc[allc.excluded == ""].reset_index(drop=True)
    print(f"\nmarked: {len(marks)}  ({marks.kind.value_counts().to_dict()})")
    if not args.no_figures:
        figdir = outdir / "games"
        figdir.mkdir(exist_ok=True)
        print(f"wrote {draw_all(marks, figdir, args.games)} game figure(s) to {figdir}")


if __name__ == "__main__":
    main()
