"""
Event study: does ask-side liquidity collapse BEFORE upward price jumps?

Reads the tick-level L2 books in data/viewer_full (run-length-encoded full-depth
snapshots), finds upward mid-price jumps and comparable no-jump controls, and
measures how depth near the touch behaves around t=0.

Definitions, per the spec
------------------------
Reference prices are frozen at WINDOW START (t = -PRE seconds) and never move:
    a_ref = best ask at window start
    b_ref = best bid at window start
Freezing at window start rather than at t=0 is deliberate: a reference taken at
t=0 would be set by information from the end of the pre-period, which muddies
"what did the book look like before the jump".

    V_A(t,k)   ask volume resting at a_ref + k cents,  k = 0..10
    V_B(t,k)   bid volume resting at b_ref - k cents
    D_A(t,d)   sum of V_A over k*0.01 <= d      (d = 1,3,5,10 cents)
    D_B(t,d)   sum of V_B over k*0.01 <= d
    C_A(t,d)   -ln( (D_A(t,d)+eps) / (D_A(t-1s,d)+eps) )   positive = collapse
    I(t,d)     (D_B - D_A) / (D_B + D_A + eps)

Prices off the k=0..10 grid contribute nothing; missing levels are zero volume.

No look-ahead
-------------
Every signal at time t is computed from the book in force at or before t. The
D(t-1s) term uses the latest run at or before t-1s, as specified. Future prices
are used ONLY to label a window jump vs no-jump, never inside a signal.

Event definition
----------------
mid is sampled on a uniform grid. A jump starts at the first grid instant t0 of
a contiguous stretch where mid(t0 + HORIZON) - mid(t0) >= JUMP. Events are
separated by at least SEP seconds so windows never overlap.

Controls are drawn from instants where |mid(t+HORIZON) - mid(t)| <= FLAT and
which are at least SEP from any jump, sampled to match the jump count per token.

Usage:
    python liquidity_collapse.py --all
    python liquidity_collapse.py --match mlb-ari-sf-2026-08-27 --examples 3
"""
import argparse
import bisect
import csv
import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orjson

BASE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(BASE, "data", "viewer_full")
OUT = os.path.join(BASE, "data", "collapse")

TICK = 0.01
KMAX = 10                       # k = 0..10 cents
KS = np.arange(KMAX + 1)
DVALS = [0.01, 0.03, 0.05, 0.10]
EPS = 1.0                       # shares; matters only when a side empties


# ---------------------------------------------------------------- data access

def load_runs(slug, asset_id):
    """Runs for one token, deduped (chunks repeat boundary-straddling runs)."""
    adir = os.path.join(VIEWER, slug, asset_id)
    seen, out = set(), []
    for path in sorted(glob.glob(os.path.join(adir, "c*.json"))):
        with open(path, "rb") as f:
            for r in orjson.loads(f.read()):
                t = r["t_start"]
                if t in seen:
                    continue
                seen.add(t)
                out.append(r)
    out.sort(key=lambda r: r["t_start"])
    return out, [r["t_start"] for r in out]


def run_at(runs, ts, t):
    """Latest run at or before t. None if t precedes the first run."""
    j = bisect.bisect_right(ts, t) - 1
    return runs[j] if j >= 0 else None


def mid_of(r):
    if r is None:
        return np.nan
    bb, ba = r["best_bid"], r["best_ask"]
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    return bb if bb is not None else (ba if ba is not None else np.nan)


# ------------------------------------------------------------------- measures

def ladder(levels, ref, is_ask):
    """Volume at each k = 0..10 cents from ref. Off-grid prices ignored."""
    v = np.zeros(KMAX + 1)
    if not levels:
        return v
    for p, q in levels:
        d = (p - ref) if is_ask else (ref - p)
        k = int(round(d / TICK))
        if 0 <= k <= KMAX and abs(d - k * TICK) < TICK / 2:
            v[k] += q
    return v


def cum(v, d):
    return v[: int(round(d / TICK)) + 1].sum()


# --------------------------------------------------------------- event finding

def find_events(runs, ts, t0, t1, dt, jump, horizon, flat, sep, rng, max_ctrl):
    """Returns (jump_starts, control_starts). Future mid is used ONLY here."""
    grid = np.arange(t0, t1, dt)
    mids = np.array([mid_of(run_at(runs, ts, t)) for t in grid])
    step = int(round(horizon / dt))
    fwd = np.full(len(grid), np.nan)
    fwd[: len(grid) - step] = mids[step:] - mids[: len(grid) - step]

    # A candidate flags an interval in which the mid rises by `jump`, but the
    # move itself can land anywhere inside that horizon. Aligning t=0 on the
    # candidate would put the jump up to `horizon` seconds AFTER zero and leave
    # the actual pre-jump book outside the window -- which would silently
    # invalidate the whole before/after comparison. So refine each candidate to
    # the ONSET: the first instant the mid actually starts rising.
    is_jump = np.nan_to_num(fwd, nan=0.0) >= jump
    starts = []
    last = -1e18
    for i, flagged in enumerate(is_jump):
        if not flagged or (i > 0 and is_jump[i - 1]):
            continue
        base = mids[i]
        j = i
        hit = None
        while j < len(grid) and grid[j] <= grid[i] + horizon:
            if not np.isnan(mids[j]) and mids[j] - base >= jump:
                hit = j
                break
            j += 1
        if hit is None:
            continue
        onset = hit
        while onset > i and not np.isnan(mids[onset - 1]) and mids[onset - 1] > base + 1e-12:
            onset -= 1
        t_on = grid[onset]
        if t_on - last >= sep:
            starts.append(t_on)
            last = t_on

    flat_ok = np.abs(np.nan_to_num(fwd, nan=1e9)) <= flat
    cand = []
    for i, ok in enumerate(flat_ok):
        t = grid[i]
        if not ok or np.isnan(mids[i]):
            continue
        if any(abs(t - s) < sep for s in starts):
            continue
        cand.append(t)
    ctrl = []
    for t in cand:
        if not ctrl or t - ctrl[-1] >= sep:
            ctrl.append(t)
    if len(ctrl) > max_ctrl:
        ctrl = sorted(rng.sample(ctrl, max_ctrl))
    return starts, ctrl


# --------------------------------------------------------------- window metrics

def measure(runs, ts, t_event, pre, post, dt):
    """All per-window series. References frozen at window start."""
    t_start = t_event - pre
    r0 = run_at(runs, ts, t_start)
    if r0 is None or r0["best_ask"] is None or r0["best_bid"] is None:
        return None
    a_ref, b_ref = r0["best_ask"], r0["best_bid"]

    rel = np.arange(-pre, post + 1e-9, dt)
    n = len(rel)
    VA = np.zeros((n, KMAX + 1))
    VB = np.zeros((n, KMAX + 1))
    DA = {d: np.zeros(n) for d in DVALS}
    DB = {d: np.zeros(n) for d in DVALS}
    CA = {d: np.full(n, np.nan) for d in DVALS}
    CB = {d: np.full(n, np.nan) for d in DVALS}
    IM = {d: np.full(n, np.nan) for d in DVALS}
    bb = np.full(n, np.nan)
    ba = np.full(n, np.nan)
    mid = np.full(n, np.nan)

    for i, s in enumerate(rel):
        t = t_event + s
        r = run_at(runs, ts, t)
        if r is None:
            continue
        VA[i] = ladder(r["asks"], a_ref, True)
        VB[i] = ladder(r["bids"], b_ref, False)
        bb[i] = r["best_bid"] if r["best_bid"] is not None else np.nan
        ba[i] = r["best_ask"] if r["best_ask"] is not None else np.nan
        mid[i] = mid_of(r)

        rp = run_at(runs, ts, t - 1.0)          # latest obs at or before t-1s
        VAp = ladder(rp["asks"], a_ref, True) if rp else np.zeros(KMAX + 1)
        VBp = ladder(rp["bids"], b_ref, False) if rp else np.zeros(KMAX + 1)
        for d in DVALS:
            da, db = cum(VA[i], d), cum(VB[i], d)
            DA[d][i], DB[d][i] = da, db
            dap, dbp = cum(VAp, d), cum(VBp, d)
            CA[d][i] = -np.log((da + EPS) / (dap + EPS))
            CB[d][i] = -np.log((db + EPS) / (dbp + EPS))
            IM[d][i] = (db - da) / (db + da + EPS)

    return dict(rel=rel, a_ref=a_ref, b_ref=b_ref, VA=VA, VB=VB,
                DA=DA, DB=DB, CA=CA, CB=CB, IM=IM, bb=bb, ba=ba, mid=mid)


# ------------------------------------------------------------------- plotting

def plot_window(m, title, path):
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1.3, 1.3], hspace=0.42, wspace=0.22)
    rel = m["rel"]

    ax = fig.add_subplot(gs[0, :])
    ax.step(rel, m["ba"], where="post", color="#cf222e", lw=1.2, label="best ask")
    ax.step(rel, m["bb"], where="post", color="#1a7f37", lw=1.2, label="best bid")
    ax.step(rel, m["mid"], where="post", color="#1f4e9c", lw=1.0, ls="--", label="mid")
    ax.axhline(m["a_ref"], color="#cf222e", lw=0.7, ls=":", alpha=0.7)
    ax.axhline(m["b_ref"], color="#1a7f37", lw=0.7, ls=":", alpha=0.7)
    ax.set_ylabel("price"); ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_title("best bid / best ask / mid   (dotted = frozen refs)", fontsize=9, loc="left")

    ax = fig.add_subplot(gs[1, :])
    ax.step(rel, m["DA"][0.05], where="post", color="#cf222e", lw=1.3, label="$D_A(5c)$")
    ax.step(rel, m["DB"][0.05], where="post", color="#1a7f37", lw=1.3, label="$D_B(5c)$")
    ax.set_ylabel("cumulative depth"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("depth within 5c of the frozen refs", fontsize=9, loc="left")

    ax = fig.add_subplot(gs[2, :])
    for d, c in ((0.01, "#f0a202"), (0.03, "#e2571e"), (0.05, "#cf222e")):
        ax.step(rel, m["CA"][d], where="post", lw=1.1, color=c, label=f"$C_A({int(d*100)}c)$")
    ax.step(rel, m["CB"][0.05], where="post", lw=1.1, color="#1a7f37", label="$C_B(5c)$")
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_ylabel("collapse score"); ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.set_title("collapse  $-\\ln(D(t)/D(t-1s))$   positive = depth removed", fontsize=9, loc="left")

    ax = fig.add_subplot(gs[3, :])
    for d, c in ((0.01, "#f0a202"), (0.05, "#1f4e9c"), (0.10, "#6f42c1")):
        ax.step(rel, m["IM"][d], where="post", lw=1.1, color=c, label=f"$I({int(d*100)}c)$")
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_ylabel("imbalance"); ax.set_ylim(-1, 1); ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_title("depth imbalance  $(D_B-D_A)/(D_B+D_A)$", fontsize=9, loc="left")

    # heatmaps share one colour scale -- per-timestamp normalisation would hide
    # exactly the effect being tested (depth vanishing over time)
    vmax = max(np.percentile(m["VA"], 99) if m["VA"].any() else 1,
               np.percentile(m["VB"], 99) if m["VB"].any() else 1, 1)
    ext = [rel[0], rel[-1], KMAX + 0.5, -0.5]
    for col, (M, name) in enumerate(((m["VA"], "ask"), (m["VB"], "bid"))):
        ax = fig.add_subplot(gs[4, col])
        im = ax.imshow(M.T, aspect="auto", origin="upper", extent=ext,
                       cmap="magma", vmin=0, vmax=vmax)
        ax.axvline(0, color="w", lw=0.8, ls="--")
        ax.set_xlabel("t (s)"); ax.set_ylabel("k (cents from ref)")
        ax.set_title(f"{name} liquidity heatmap (shared scale)", fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.035)

    ax = fig.add_subplot(gs[5, :])
    ds = np.arange(0, KMAX + 1) * TICK
    for s, c in ((-2.0, "#2b2b2b"), (-1.5, "#6f42c1"), (-1.0, "#1f4e9c"),
                 (-0.5, "#e2571e"), (0.0, "#cf222e")):
        i = int(np.argmin(np.abs(rel - s)))
        if abs(rel[i] - s) > 1e-6:
            continue
        ax.plot(ds, np.cumsum(m["VA"][i]), marker="o", ms=3, lw=1.3, color=c,
                label=f"t = {s:+.1f}s")
    ax.set_xlabel("d (distance from $a_{ref}$)"); ax.set_ylabel("$D_A(d)$")
    ax.set_title("cumulative ask depth at times before t=0", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=5)

    for a in fig.axes[:4]:
        a.axvline(0, color="#444", lw=0.9, ls="--")
        a.grid(alpha=0.2, lw=0.4)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.905)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(J, N, rel, path, eps=EPS):
    def band(ax, stack, colour, label):
        if not len(stack):
            return
        A = np.vstack(stack)
        med = np.nanmedian(A, 0)
        lo = np.nanpercentile(A, 25, 0)
        hi = np.nanpercentile(A, 75, 0)
        ax.plot(rel, med, color=colour, lw=1.8, label=f"{label} (n={A.shape[0]})")
        ax.fill_between(rel, lo, hi, color=colour, alpha=0.15, lw=0)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    specs = [("normDA", "normalised $D_A(5c)$   (relative to window start)"),
             ("CA", "$C_A(5c)$   collapse score"),
             ("IM", "imbalance $I(5c)$"),
             ("dmid", "mid change since t=0")]
    for ax, (key, title) in zip(axes, specs):
        band(ax, [e[key] for e in J], "#cf222e", "jump")
        band(ax, [e[key] for e in N], "#1f4e9c", "no-jump")
        ax.axvline(0, color="#444", lw=0.9, ls="--")
        ax.axhline(0 if key != "normDA" else 1, color="#888", lw=0.7)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.2, lw=0.4)
    axes[-1].set_xlabel("time relative to t=0 (s)")
    fig.suptitle("Event study: jump vs no-jump   (median, shaded = IQR)",
                 fontsize=13, fontweight="bold", y=0.905)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--market", default="moneyline")
    ap.add_argument("--outcome", default="YES")
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--post", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--jump", type=float, default=0.02)
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--flat", type=float, default=0.005)
    ap.add_argument("--sep", type=float, default=60.0)
    ap.add_argument("--max-ctrl", type=int, default=40)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    with open(os.path.join(VIEWER, "index.json"), "rb") as f:
        matches = orjson.loads(f.read())["matches"]
    if args.match:
        matches = [m for m in matches if m["event_slug"] == args.match]
    elif not args.all:
        raise SystemExit("pass --match <slug> or --all")

    rows = []
    Jagg, Nagg = [], []
    rel_ref = None
    n_ex = {"jump": 0, "nojump": 0}

    for mi, mt in enumerate(matches, 1):
        slug = mt["event_slug"]
        mk = mt["market_types"].get(args.market)
        if not mk:
            continue
        toks = mk[0]["tokens"]
        tok = next((t for t in toks if t["outcome"] == args.outcome), toks[0])
        runs, ts = load_runs(slug, tok["asset_id"])
        if not runs:
            continue
        lo = mt["start_ts"] + args.pre + 1.5
        hi = mt["end_ts"] - args.post - args.horizon
        jumps, ctrls = find_events(runs, ts, lo, hi, args.dt, args.jump,
                                   args.horizon, args.flat, args.sep, rng, args.max_ctrl)
        print(f"[{mi}/{len(matches)}] {slug:<26} jumps={len(jumps):<4} controls={len(ctrls)}",
              flush=True)

        for kind, times in (("jump", jumps), ("nojump", ctrls)):
            for t0 in times:
                m = measure(runs, ts, t0, args.pre, args.post, args.dt)
                if m is None:
                    continue
                if rel_ref is None:
                    rel_ref = m["rel"]
                if len(m["rel"]) != len(rel_ref):
                    continue
                base = m["DA"][0.05][0]
                baseb = m["DB"][0.05][0]
                agg = dict(normDA=m["DA"][0.05] / (base + EPS),
                           normDB=m["DB"][0.05] / (baseb + EPS),
                           CA=m["CA"][0.05],
                           CB=m["CB"][0.05],
                           IM=m["IM"][0.05],
                           dmid=m["mid"] - m["mid"][int(args.pre / args.dt)])
                (Jagg if kind == "jump" else Nagg).append(agg)

                i0 = int(args.pre / args.dt)
                pre_mask = m["rel"] < 0
                rows.append(dict(
                    slug=slug, market=args.market, outcome=tok["outcome"],
                    kind=kind, t0=f"{t0:.6f}", a_ref=m["a_ref"], b_ref=m["b_ref"],
                    mid_at_0=round(float(m["mid"][i0]), 6),
                    mid_change=round(float(m["mid"][-1] - m["mid"][i0]), 6),
                    DA5_start=round(float(m["DA"][0.05][0]), 2),
                    DA5_at_0=round(float(m["DA"][0.05][i0]), 2),
                    DA5_ratio=round(float((m["DA"][0.05][i0] + EPS) / (m["DA"][0.05][0] + EPS)), 4),
                    DB5_at_0=round(float(m["DB"][0.05][i0]), 2),
                    CA5_max_pre=round(float(np.nanmax(m["CA"][0.05][pre_mask])), 4),
                    CA1_max_pre=round(float(np.nanmax(m["CA"][0.01][pre_mask])), 4),
                    CB5_max_pre=round(float(np.nanmax(m["CB"][0.05][pre_mask])), 4),
                    I5_at_0=round(float(m["IM"][0.05][i0]), 4),
                ))

                if n_ex[kind] < args.examples:
                    n_ex[kind] += 1
                    p = os.path.join(args.out,
                                     f"window_{kind}_{n_ex[kind]:02d}_{slug}.png")
                    plot_window(m, f"{kind.upper()}  ·  {mt['match_name']}  ·  "
                                   f"{tok['label']}  ·  t0={t0:.3f}", p)

    csv_path = os.path.join(args.out, "events.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if rel_ref is not None:
        plot_aggregate(Jagg, Nagg, rel_ref, os.path.join(args.out, "aggregate_event_study.png"))
        # Dump the stacked series so figures can be redrawn without repeating
        # the ~25 minute read over every match.
        np.savez_compressed(
            os.path.join(args.out, "aggregate.npz"), rel=rel_ref,
            **{f"J_{k}": np.vstack([e[k] for e in Jagg]) for k in Jagg[0]},
            **{f"N_{k}": np.vstack([e[k] for e in Nagg]) for k in Nagg[0]})

    nj = sum(1 for r in rows if r["kind"] == "jump")
    print(f"\nevents: {nj} jump, {len(rows)-nj} no-jump  ->  {csv_path}")
    print(f"plots : {args.out}")


if __name__ == "__main__":
    main()
