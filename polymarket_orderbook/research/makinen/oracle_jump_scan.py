"""
Model-free: find every durable jump in the price graph and price it honestly.

Justin's question: forget the classifier -- just look at the price series, find
all the jumps, and tell me which ones are big enough to break even. If that set
is empty, no classifier can help, because these are oracle bounds: a trader with
perfect foresight about when the jump comes, which way it goes, and when to get
out.

Differences from direction_hold.py, which swept fixed holds from collapse
entries:

  * the candidate set is NOT the collapse detector's -- every durable move in
    the price graph is enumerated, so the detector's recall cannot be the
    binding constraint;
  * the exit is the actual peak of the move (and, separately, the single best
    instant within 300s), not a fixed clock;
  * both ends of the round trip are charged at the spread actually standing at
    that instant, and the Polymarket sports taker fee is charged on both legs.

A warning about the omniscient exit. Picking the best of ~1500 future instants
is positive on a pure random walk, so that number is meaningless on its own. It
is reported only against matched random entries scored the same way: the
question is never "is the oracle positive" but "does entering at a jump beat
entering at an arbitrary moment".

Definitions follow the rest of the project: tick = 0.01, a durable move is
>= 2 ticks within 10s that is still >= 2 ticks away 3s after the peak, and one
jump is debounced over 30s so a single event is not counted many times.

Outputs, under results/makinen/oracle_jumps/
  events.parquet       one row per jump and per matched control
  summary.csv          the headline table
  jump_vs_random.csv   the decisive comparison
  by_entry_spread.csv  the ex-ante rule test
  who_pays.csv         what the jumps that do pay look like
  oracle_jumps.png     the picture
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

TICK = 0.01           # Polymarket price tick, in dollars
GRID_MS = 200         # the in-game grid the features were built on
J_TICKS = 2.0         # durable-move threshold, matches the rest of the project
WIN_S = 10.0          # search this far forward for the peak
HOLD_S = 3.0          # the displacement must survive this long past the peak
DEBOUNCE_S = 30.0     # one jump per this window, per series
EXIT_MAX_S = 300.0    # how far ahead the omniscient exit is allowed to look
SPORTS_FEE_RATE = 0.05

BUCKET_EDGES = [-0.01, 1.01, 2.01, 3.01, 5.01, 10.01, 20.01, 1e9]
BUCKET_NAMES = ["<=1", "2", "3", "4-5", "6-10", "11-20", ">20"]

def _sessions():
    """Every session that has a trimmed cache, discovered rather than listed.

    This was a hardcoded list of six, and it went stale the moment three more
    sessions were built: the scan kept describing a population that no longer
    existed, and the derived summaries next to it described a different one
    again. build_panel.py has always globbed; this now does too.
    """
    suffix = "_trimmed.parquet"
    d = ROOT / "data" / "jump"
    return sorted(p.name[len("feat_"):-len(suffix)]
                  for p in d.glob("feat_books_*" + suffix))


SESSIONS = _sessions()


def fee_ticks(price):
    """Polymarket sports taker fee at `price`, expressed in ticks.

    fee_dollars_per_share = rate * p * (1 - p); divide by TICK for ticks.
    Charged on every taker leg, so a round trip pays it twice.
    """
    p = np.clip(np.asarray(price, dtype=np.float64), 0.0, 1.0)
    return SPORTS_FEE_RATE * p * (1.0 - p) / TICK


def run_ids(sid: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Label contiguous 200ms stretches within one series.

    A window may only be used if its start and end share a run id: that rules
    out windows that silently bridge a recording gap or a series boundary.
    """
    brk = np.empty(len(ts), dtype=bool)
    brk[0] = True
    brk[1:] = (sid[1:] != sid[:-1]) | (np.diff(ts) != GRID_MS)
    return np.cumsum(brk)


def scan_session(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["mid", "spread_ticks", "ts", "series", "sid", "valid"])
    df = df.sort_values(["sid", "ts"], kind="stable").reset_index(drop=True)

    mid = df["mid"].to_numpy(np.float64)
    spr = df["spread_ticks"].to_numpy(np.float64)
    ts = df["ts"].to_numpy(np.int64)
    sid = df["sid"].to_numpy(np.int64)
    ok = df["valid"].to_numpy(bool) & np.isfinite(mid) & np.isfinite(spr)

    n = len(mid)
    rid = run_ids(sid, ts)
    W = int(WIN_S * 1000 / GRID_MS)      # 50 steps
    H = int(HOLD_S * 1000 / GRID_MS)     # 15 steps

    # Running max / min of the mid over the next W steps, with the index where
    # each extreme occurred, restricted to windows that stay inside one run.
    fmax = np.full(n, -np.inf)
    fmin = np.full(n, np.inf)
    amax = np.full(n, -1, dtype=np.int64)
    amin = np.full(n, -1, dtype=np.int64)

    idx = np.arange(n)
    for k in range(1, W + 1):
        nxt = idx + k
        good = np.zeros(n, dtype=bool)
        good[: n - k] = (rid[k:] == rid[: n - k]) & ok[k:]
        cand = np.where(good, np.concatenate([mid[k:], np.full(k, np.nan)]), np.nan)

        hit_hi = good & (cand > fmax)
        fmax = np.where(hit_hi, cand, fmax)
        amax = np.where(hit_hi, nxt, amax)

        hit_lo = good & (cand < fmin)
        fmin = np.where(hit_lo, cand, fmin)
        amin = np.where(hit_lo, nxt, amin)

    up = fmax - mid
    dn = mid - fmin
    take_up = up >= dn
    peak_move = np.where(take_up, up, dn)
    peak_idx = np.where(take_up, amax, amin)

    have = ok & (peak_idx >= 0) & np.isfinite(peak_move)
    # The persistence check sits HOLD_S past the peak and must stay in the run.
    hold_idx = np.where(have, np.minimum(peak_idx + H, n - 1), 0)
    have &= rid[hold_idx] == rid
    have &= ok[hold_idx]

    peak_ticks = peak_move / TICK
    held_disp = np.abs(mid[hold_idx] - mid) / TICK
    is_jump = have & (peak_ticks >= J_TICKS) & (held_disp >= J_TICKS)

    # Debounce: walk the candidates in time order, take one, skip DEBOUNCE_S.
    D = int(DEBOUNCE_S * 1000 / GRID_MS)
    keep = []
    last_sid, last_pos = -1, -(10**9)
    for t in np.flatnonzero(is_jump):
        if sid[t] != last_sid or t - last_pos >= D:
            keep.append(t)
            last_sid, last_pos = sid[t], t
    keep = np.array(keep, dtype=np.int64)
    if len(keep) == 0:
        return pd.DataFrame()

    # --- matched control -----------------------------------------------------
    # An omniscient exit is a maximum over ~1500 future instants, and on a pure
    # random walk that maximum is large and positive by construction. So jump
    # entries are only meaningful against random entries scored the SAME way.
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(ok & have)
    ctrl = np.sort(rng.choice(pool, size=min(len(keep), len(pool)), replace=False))

    both = np.concatenate([keep, ctrl])
    is_jump_flag = np.concatenate([np.ones(len(keep), bool), np.zeros(len(ctrl), bool)])

    p_idx = peak_idx[both]
    h_idx = hold_idx[both]

    entry_mid = mid[both]
    peak_mid = mid[p_idx]
    hold_mid = mid[h_idx]
    entry_spr = spr[both]
    peak_spr = spr[p_idx]
    hold_spr = spr[h_idx]

    # Gross capture, in ticks. Exiting at the peak is the best exit a system
    # that only knows about the jump could aim at; 3s later is what a system
    # reacting to the move would more realistically get.
    gross_peak = np.abs(peak_mid - entry_mid) / TICK
    gross_hold = np.abs(hold_mid - entry_mid) / TICK

    # A round trip crosses half the spread at each end.
    cost_peak = (entry_spr + peak_spr) / 2.0
    cost_hold = (entry_spr + hold_spr) / 2.0

    fee_peak = fee_ticks(entry_mid) + fee_ticks(peak_mid)
    fee_hold = fee_ticks(entry_mid) + fee_ticks(hold_mid)

    # --- the omniscient exit -------------------------------------------------
    # Perfect foresight over the exit too: within EXIT_MAX_S, pick the instant
    # maximising (displacement in your favour) - (half the spread you cross
    # there). The peak is NOT optimal, because the peak is when the book is
    # thinnest; this lets the oracle wait for liquidity to come back.
    E = int(EXIT_MAX_S * 1000 / GRID_MS)
    best_net = np.full(len(both), -np.inf)
    best_idx = np.full(len(both), -1, dtype=np.int64)
    best_dir = np.zeros(len(both), dtype=np.int64)
    offs = np.arange(1, E + 1)
    for lo in range(0, len(both), 4000):
        blk = both[lo : lo + 4000]
        ix = np.clip(blk[:, None] + offs[None, :], 0, n - 1)
        same = (rid[ix] == rid[blk][:, None]) & ok[ix] & (blk[:, None] + offs <= n - 1)
        disp = (mid[ix] - mid[blk][:, None]) / TICK
        half = spr[ix] / 2.0
        net_up = np.where(same, disp - half, -np.inf)
        net_dn = np.where(same, -disp - half, -np.inf)
        net = np.maximum(net_up, net_dn)
        j = np.argmax(net, axis=1)
        r = np.arange(len(blk))
        best_net[lo : lo + len(blk)] = net[r, j]
        best_idx[lo : lo + len(blk)] = ix[r, j]
        best_dir[lo : lo + len(blk)] = np.where(net_up[r, j] >= net_dn[r, j], 1, -1)
        del ix, same, disp, half, net_up, net_dn, net

    ser = df["series"].to_numpy()[both]
    parts = pd.Series(ser).str.split("|", expand=True)

    out = pd.DataFrame(
        {
            "session": path.stem.replace("feat_", "").replace("_trimmed", ""),
            "is_jump": is_jump_flag,
            "sid": sid[both],
            "series": ser,
            "slug": parts[0].to_numpy(),
            "market_type": parts[1].to_numpy(),
            "ts": ts[both],
            "ts_peak": ts[p_idx],
            "lead_to_peak_s": (ts[p_idx] - ts[both]) / 1000.0,
            "entry_mid": entry_mid,
            "peak_mid": peak_mid,
            "entry_spread": entry_spr,
            "peak_spread": peak_spr,
            "hold_spread": hold_spr,
            "gross_peak": gross_peak,
            "gross_hold": gross_hold,
            "cost_peak": cost_peak,
            "cost_hold": cost_hold,
            "fee_peak": fee_peak,
            "fee_hold": fee_hold,
            "signed": np.where(take_up[both], 1, -1),
            "best_exit_s": (ts[best_idx] - ts[both]) / 1000.0,
            "best_exit_spread": spr[best_idx],
            "best_exit_mid": mid[best_idx],
            "best_dir": best_dir,
        }
    )
    out["pnl_peak_nofee"] = out.gross_peak - out.cost_peak
    out["pnl_peak"] = out.pnl_peak_nofee - out.fee_peak
    out["pnl_hold_nofee"] = out.gross_hold - out.cost_hold
    out["pnl_hold"] = out.pnl_hold_nofee - out.fee_hold
    # best_net already carries the exit half-spread; entry pays its own half.
    out["pnl_best_nofee"] = best_net - entry_spr / 2.0
    out["fee_best"] = fee_ticks(entry_mid) + fee_ticks(out.best_exit_mid.to_numpy())
    out["pnl_best"] = out.pnl_best_nofee - out.fee_best
    return out


def clustered_ci(frame: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 0):
    """Bootstrap the mean of `col`, resampling whole games."""
    rng = np.random.default_rng(seed)
    games = frame.slug.unique()
    vals = frame[col].to_numpy()
    slugs = frame.slug.to_numpy()
    by = {g: vals[slugs == g] for g in games}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(games, size=len(games), replace=True)
        draws[b] = np.concatenate([by[g] for g in pick]).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=SESSIONS)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    ap.add_argument("--from-cache", action="store_true",
                    help="re-report from events.parquet without rescanning")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        frames = [pd.read_parquet(outdir / "events.parquet")]
        args.sessions = []
    else:
        frames = []
    for s in args.sessions:
        path = ROOT / "data" / "jump" / f"feat_{s}_trimmed.parquet"
        if not path.exists():
            print(f"  skip {s}: {path.name} missing")
            continue
        got = scan_session(path)
        print(f"  {s}: {int(got.is_jump.sum()):,} durable jumps "
              f"(+{int((~got.is_jump).sum()):,} matched controls)")
        frames.append(got)

    allrows = pd.concat(frames, ignore_index=True)
    if not args.from_cache:
        allrows.to_parquet(outdir / "events.parquet", index=False)

    # An entry too close to the end of a contiguous run has no exit inside the
    # window at all. Those rows carry -inf and would poison a mean.
    finite = np.isfinite(allrows.pnl_best) & np.isfinite(allrows.pnl_peak)
    if (~finite).any():
        print(f"  dropped {int((~finite).sum()):,} entries with no valid exit inside "
              f"{EXIT_MAX_S:.0f}s")
    allrows = allrows[finite].reset_index(drop=True)
    allrows["entry_bucket"] = pd.cut(allrows.entry_spread, BUCKET_EDGES, labels=BUCKET_NAMES)

    ev = allrows[allrows.is_jump].copy()
    ct = allrows[~allrows.is_jump].copy()

    rows = []
    for name, gross, cost, fee, pnl in [
        ("exit at the peak, no fee", "gross_peak", "cost_peak", None, "pnl_peak_nofee"),
        ("exit at the peak, with fee", "gross_peak", "cost_peak", "fee_peak", "pnl_peak"),
        ("exit 3s after the peak, no fee", "gross_hold", "cost_hold", None, "pnl_hold_nofee"),
        ("exit 3s after the peak, with fee", "gross_hold", "cost_hold", "fee_hold", "pnl_hold"),
        ("omniscient exit <=300s, no fee", "gross_peak", "cost_peak", None, "pnl_best_nofee"),
        ("omniscient exit <=300s, with fee", "gross_peak", "cost_peak", "fee_best", "pnl_best"),
    ]:
        p = ev[pnl]
        rows.append({
            "case": name,
            "n_jumps": len(ev),
            "mean_gross": ev[gross].mean(),
            "mean_cost": ev[cost].mean(),
            "mean_fee": ev[fee].mean() if fee else 0.0,
            "mean_pnl": p.mean(),
            "median_pnl": p.median(),
            "n_profitable": int((p > 0).sum()),
            "share_profitable": float((p > 0).mean()),
            "ticks_if_only_winners": float(p[p > 0].sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "summary.csv", index=False)
    print()
    print("JUMPS ONLY")
    print(summary.to_string(index=False))

    print()
    print("JUMP vs RANDOM ENTRY, scored identically")
    cmp_rows = []
    for label, frame in [("jump entry", ev), ("random entry", ct)]:
        cmp_rows.append({
            "entries": label,
            "n": len(frame),
            "ev_peak_exit": frame.pnl_peak.mean(),
            "ev_best_exit_nofee": frame.pnl_best_nofee.mean(),
            "ev_best_exit": frame.pnl_best.mean(),
            "median_best_exit": frame.pnl_best.median(),
            "share_profitable_best": float((frame.pnl_best > 0).mean()),
            "mean_best_exit_wait_s": frame.best_exit_s.mean(),
        })
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(outdir / "jump_vs_random.csv", index=False)
    print(cmp.to_string(index=False))
    d = cmp.ev_best_exit.iloc[0] - cmp.ev_best_exit.iloc[1]
    print(f"  jump entry advantage over a random entry: {d:+.3f} ticks")

    # The ex-ante question: entry spread is the one thing you can see BEFORE the
    # move. If some bucket of it has positive oracle EV, a rule exists.
    by = allrows.groupby(["entry_bucket", "is_jump"], observed=True).agg(
        n=("pnl_peak", "size"),
        mean_gross_peak=("gross_peak", "mean"),
        mean_cost_peak=("cost_peak", "mean"),
        mean_fee=("fee_peak", "mean"),
        peak_ev=("pnl_peak", "mean"),
        best_exit_ev=("pnl_best", "mean"),
        share_profitable_peak=("pnl_peak", lambda s: float((s > 0).mean())),
    )
    by.to_csv(outdir / "by_entry_spread.csv")
    print()
    print(by.to_string())

    # Who actually pays? Characterise the jumps that clear break-even on the
    # only exit a real system could aim at -- the peak of the move itself.
    w = ev[ev.pnl_peak > 0]
    prof = pd.DataFrame({
        "n_jumps": [len(ev)],
        "n_paying": [len(w)],
        "share_paying": [len(w) / len(ev)],
        "distinct_series_paying": [w.series.nunique()],
        "distinct_games_paying": [w.slug.nunique()],
        "median_move_paying": [w.gross_peak.median()],
        "median_move_all": [ev.gross_peak.median()],
        "median_entry_spread_paying": [w.entry_spread.median()],
        "median_entry_spread_all": [ev.entry_spread.median()],
        "median_peak_spread_paying": [w.peak_spread.median()],
        "median_peak_spread_all": [ev.peak_spread.median()],
        "total_ticks_if_you_took_only_these": [w.pnl_peak.sum()],
    })
    prof.to_csv(outdir / "who_pays.csv", index=False)
    print()
    print("THE JUMPS THAT PAY (exit at the peak, fee charged)")
    print(prof.T.to_string(header=False))

    # The moneyline slice pays far more often than the others, so it gets its
    # own look rather than a shrug -- matched on a 1-tick entry so the
    # comparison is not just "moneyline is tighter".
    tight = ev[ev.entry_spread <= 1.01]
    mt = tight.groupby("market_type").agg(
        n=("pnl_peak", "size"),
        peak_spread=("peak_spread", "mean"),
        move=("gross_peak", "mean"),
        cost=("cost_peak", "mean"),
        fee=("fee_peak", "mean"),
        ev_before_fee=("pnl_peak_nofee", "mean"),
        ev_after_fee=("pnl_peak", "mean"),
        pays_before_fee=("pnl_peak_nofee", lambda s: float((s > 0).mean())),
        pays_after_fee=("pnl_peak", lambda s: float((s > 0).mean())),
    )
    mt.to_csv(outdir / "tight_entry_by_market_type.csv")
    print()
    print("TIGHT ENTRY (spread <= 1 tick), BY MARKET TYPE")
    print(mt.round(3).to_string())

    ml = tight[tight.market_type == "moneyline"]
    lo, hi = clustered_ci(ml, "pnl_peak_nofee")
    lo2, hi2 = clustered_ci(ml, "pnl_peak")
    print()
    print(f"  moneyline, 1-tick entry, n={len(ml):,} over {ml.slug.nunique()} games")
    print(f"    before fee: {ml.pnl_peak_nofee.mean():+.3f} ticks  "
          f"game-clustered 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"    after fee : {ml.pnl_peak.mean():+.3f} ticks  "
          f"game-clustered 95% CI [{lo2:+.3f}, {hi2:+.3f}]")

    make_plot(allrows, outdir / "oracle_jumps.png")
    print(f"wrote {outdir}")


def make_plot(allrows: pd.DataFrame, path: Path) -> None:
    """Three panels: what a jump costs, whether it beats a coin toss of a
    moment, and the one slice where the binding constraint is the fee."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE = "#fcfcfb"
    INK, INK2 = "#0b0b0b", "#52514e"
    S1, S2 = "#2a78d6", "#eb6834"      # categorical slots 1, 2
    GOOD, BAD = "#0ca30c", "#d03b3b"   # status good / critical

    jumps = allrows[allrows.is_jump]
    ctrls = allrows[~allrows.is_jump]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color="#e9e8e4", lw=0.8)
        ax.set_axisbelow(True)

    # 1. every jump, priced at the exit a real system could actually aim at
    ax = axes[0]
    sub = jumps.sample(min(40000, len(jumps)), random_state=0)
    win = sub.pnl_peak > 0
    paid = sub.cost_peak + sub.fee_peak
    ax.scatter(paid[~win], sub.gross_peak[~win], s=3, alpha=0.10,
               color=BAD, linewidths=0, label="loses")
    ax.scatter(paid[win], sub.gross_peak[win], s=10, alpha=0.75,
               color=GOOD, linewidths=0, label="pays")
    lim = 55
    ax.plot([0, lim], [0, lim], color=INK2, lw=1.5, ls="--", label="break even")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("round trip actually paid: spread both ends + fee (ticks)",
                  color=INK2, fontsize=9.5)
    ax.set_ylabel("move captured, exiting at the peak (ticks)", color=INK2, fontsize=9.5)
    npay = int((jumps.pnl_peak > 0).sum())
    ax.set_title(f"Every durable jump: {npay:,} of {len(jumps):,} pay "
                 f"({npay / len(jumps):.1%})", color=INK, fontsize=11.5, loc="left")
    leg = ax.legend(markerscale=3, frameon=False, loc="upper left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)

    # 2. does entering at a jump beat entering at a random moment?
    ax = axes[1]
    j = jumps.groupby("entry_bucket", observed=True).pnl_best.mean().reindex(BUCKET_NAMES)
    c = ctrls.groupby("entry_bucket", observed=True).pnl_best.mean().reindex(BUCKET_NAMES)
    x = np.arange(len(BUCKET_NAMES))
    ax.bar(x - 0.21, j.values, width=0.4, color=S1, label="entered at a jump")
    ax.bar(x + 0.21, c.values, width=0.4, color=S2, label="entered at random")
    ax.axhline(0, color=INK2, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_NAMES)
    ax.set_xlabel("spread at entry (ticks) - the one thing visible beforehand",
                  color=INK2, fontsize=9.5)
    ax.set_ylabel("mean P&L given a perfect exit (ticks)", color=INK2, fontsize=9.5)
    ax.set_title("A jump must beat a random moment", color=INK, fontsize=11.5, loc="left")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)

    # 3. the one slice with any life, and what actually kills it
    ax = axes[2]
    tight = jumps[jumps.entry_spread <= 1.01]
    mt = ["moneyline", "spread", "total"]
    g = tight.groupby("market_type").pnl_peak_nofee.mean().reindex(mt)
    f = tight.groupby("market_type").pnl_peak.mean().reindex(mt)
    nn = tight.groupby("market_type").size().reindex(mt)
    x = np.arange(len(mt))
    ax.bar(x - 0.21, g.values, width=0.4, color=S1, label="before the fee")
    ax.bar(x + 0.21, f.values, width=0.4, color=S2, label="after the Polymarket fee")
    ax.axhline(0, color=INK2, lw=1.2)
    for xi, (gv, fv) in enumerate(zip(g.values, f.values)):
        ax.annotate(f"{gv:+.2f}", (xi - 0.21, gv), ha="center", fontsize=9, color=INK,
                    va="bottom" if gv >= 0 else "top")
        ax.annotate(f"{fv:+.2f}", (xi + 0.21, fv), ha="center", fontsize=9, color=INK,
                    va="bottom" if fv >= 0 else "top")
    ax.set_xticks(x)
    labels = [m + "\nn=" + format(int(v), ",") for m, v in zip(mt, nn.values)]
    ax.set_xticklabels(labels)
    ax.set_xlabel("market type, entering only at a 1-tick spread", color=INK2, fontsize=9.5)
    ax.set_ylabel("oracle P&L per jump, exit at the peak (ticks)", color=INK2, fontsize=9.5)
    ax.set_title("Only moneyline clears its spread; the fee sinks it",
                 color=INK, fontsize=11.5, loc="left")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
