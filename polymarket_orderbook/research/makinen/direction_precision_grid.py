"""
Two dials: how well you call direction, and how tightly you select.

Justin's question: if I can catch the direction, what precision do I need to
turn a profit? `selector_realistic_exit.py` answered the mirror image -- at the
model's operating points, what directional accuracy would be needed. This puts
both on one grid.

The trade, on a fixed clock with direction accuracy `a`, is worth

    EV = (2a - 1) * E|move| - E[spread both ends + fee]

over whatever set you select. Selecting harder raises E|move| and lowers the
cost, so it lowers the accuracy you need; it also cuts volume fast. The grid
shows the trade-off and marks where the selector's own precision sits at each
operating point, so "what precision do I need" can be read off directly.

The assumption in `(2a - 1) * E|move|` is that directional accuracy does not
depend on the size of the move. If a real direction model were better on big
moves this understates it, and worse on big moves would overstate it. It is the
neutral assumption, and it is the same one used in RULED_OUT section C.

Outputs, under results/makinen/oracle_jumps/
  direction_precision_grid.csv   EV for every (accuracy, operating point)
  direction_precision_need.csv   the loosest profitable cut per accuracy
  direction_precision_grid.png   the picture
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
sys.path.insert(0, str(HERE))

from selector_realistic_exit import (  # noqa: E402
    FEATURES, TEST_SESSIONS, TICK, fee_ticks,
)

HOLD_S = 30
FRACS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.00]
ACCURACIES = [0.50, 0.55, 0.60, 0.625, 0.65, 0.675, 0.70, 0.75, 0.80, 0.90, 1.00]
N_SESSIONS_TEST = 2


def prepare(outdir: Path) -> pd.DataFrame:
    """Selector scores plus the components of a 30s fixed-clock trade."""
    import lightgbm as lgb

    ev = pd.read_parquet(outdir / "events.parquet")
    ev = ev[ev.is_jump & np.isfinite(ev.pnl_peak)].copy()

    parts = []
    for sess, g in ev.groupby("session"):
        feat = pd.read_parquet(
            ROOT / "data" / "jump" / f"feat_{sess}_trimmed.parquet",
            columns=FEATURES + ["series", "ts"])
        parts.append(g.merge(feat, on=["series", "ts"], how="left", suffixes=("", "_f")))
    df = pd.concat(parts, ignore_index=True).dropna(subset=FEATURES).reset_index(drop=True)
    df["y"] = (df.pnl_peak > 0).astype(int)

    te = df.session.isin(TEST_SESSIONS)
    assert not set(df.slug[~te]) & set(df.slug[te]), "game leaked across the split"

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        class_weight="balanced", random_state=0, verbosity=-1)
    model.fit(df.loc[~te, FEATURES], df.loc[~te, "y"])

    sub = df.loc[te].copy()
    sub["p"] = model.predict_proba(sub[FEATURES])[:, 1]

    rows = []
    for sess, g in sub.groupby("session"):
        f = pd.read_parquet(ROOT / "data" / "jump" / f"feat_{sess}_trimmed.parquet",
                            columns=["mid", "spread_ticks", "ts", "series", "valid"])
        f = f[f.valid].sort_values(["series", "ts"])
        for ser, r in g.groupby("series"):
            s = f[f.series == ser]
            if len(s) == 0:
                continue
            ts, mid, spr = s.ts.to_numpy(), s.mid.to_numpy(), s.spread_ticks.to_numpy()
            pos = np.clip(np.searchsorted(ts, r.ts.to_numpy()), 0, len(ts) - 1)
            want = r.ts.to_numpy() + HOLD_S * 1000
            j = np.clip(np.searchsorted(ts, want), 0, len(ts) - 1)
            good = np.abs(ts[j] - want) <= 2000
            rows.append(r.assign(
                absmove=np.where(good, np.abs(mid[j] - mid[pos]) / TICK, np.nan),
                cost=(spr[pos] + spr[j]) / 2.0 + fee_ticks(mid[pos]) + fee_ticks(mid[j]),
            ))
    return pd.concat(rows).dropna(subset=["absmove"]).sort_values("p", ascending=False)


def boot_ci(absmove, cost, slugs, acc, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    games = np.unique(slugs)
    idx = {g: np.flatnonzero(slugs == g) for g in games}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(games, size=len(games), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        m = np.sort(absmove[sel])[::-1]
        m = m[int(0.05 * len(m)):int(0.95 * len(m))] if len(m) >= 20 else m[1:]
        draws[b] = (2 * acc - 1) * (m.mean() if len(m) else np.nan) - cost[sel].mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    args = ap.parse_args()
    outdir = Path(args.outdir)

    R = prepare(outdir)
    n = len(R)
    print(f"held-out jumps with a valid {HOLD_S}s exit: {n:,} "
          f"over {N_SESSIONS_TEST} sessions, {R.slug.nunique()} games")

    cells, ops = [], []
    for frac in FRACS:
        k = max(1, int(round(frac * n)))
        x = R.head(k)
        m = np.sort(x.absmove.to_numpy())[::-1]
        c = x.cost.mean()
        # The raw mean of |move| is outlier-driven at small k: in this data a
        # single 75-tick move is 38% of the top-34 set's total. Report the
        # robust versions alongside, and use the trimmed one for the grid.
        em = m.mean()
        em_drop = m[1:].mean() if len(m) > 1 else np.nan
        em_trim = m[int(0.05 * len(m)):int(0.95 * len(m))].mean() if len(m) >= 20 else np.nan
        use = em_trim if em_trim == em_trim else em_drop
        ops.append({
            "top_frac": frac, "n": k, "precision": float(x.y.mean()),
            "trades_per_session": k / N_SESSIONS_TEST,
            "mean_abs_move": em, "mean_abs_move_trimmed": em_trim, "mean_cost": c,
            "accuracy_needed_raw": (1 + c / em) / 2 if em > 0 else np.nan,
            "accuracy_needed_drop_largest": (1 + c / em_drop) / 2 if em_drop > 0 else np.nan,
            "accuracy_needed": (1 + c / use) / 2 if use > 0 else np.nan,
        })
        em = use  # the grid below uses the robust estimate
        for a in ACCURACIES:
            cells.append({"top_frac": frac, "n": k, "precision": float(x.y.mean()),
                          "accuracy": a, "ev": (2 * a - 1) * em - c})

    opsdf = pd.DataFrame(ops)
    grid = pd.DataFrame(cells)
    grid.to_csv(outdir / "direction_precision_grid.csv", index=False)

    print()
    print("OPERATING POINTS (30s hold, held out)")
    print(opsdf.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    piv = grid.pivot(index="top_frac", columns="accuracy", values="ev")
    print()
    print("EV PER TRADE (ticks): rows = how tightly you select, cols = direction accuracy")
    print(piv.to_string(float_format=lambda v: f"{v:+.2f}"))

    # The answer to the question as asked: at each accuracy, the LOOSEST cut
    # that still profits, and the selector precision you need there.
    need = []
    for a in ACCURACIES:
        g = grid[(grid.accuracy == a) & (grid.ev > 0)]
        if len(g) == 0:
            need.append({"accuracy": a, "feasible": False})
            continue
        loosest = g.sort_values("top_frac").iloc[-1]
        row = opsdf[opsdf.top_frac == loosest.top_frac].iloc[0]
        lo, hi = boot_ci(R.head(int(row.n)).absmove.to_numpy(),
                         R.head(int(row.n)).cost.to_numpy(),
                         R.head(int(row.n)).slug.to_numpy(), a)
        need.append({
            "accuracy": a, "feasible": True,
            "loosest_top_frac": loosest.top_frac,
            "n_trades": int(row.n),
            "trades_per_session": row.trades_per_session,
            "precision_needed": row.precision,
            "ev": loosest.ev, "ci_lo": lo, "ci_hi": hi,
            "ticks_per_session": loosest.ev * row.trades_per_session,
        })
    needdf = pd.DataFrame(need)
    needdf.to_csv(outdir / "direction_precision_need.csv", index=False)
    print()
    print("WHAT YOU NEED: loosest profitable cut at each direction accuracy")
    print(needdf.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    draw(opsdf, grid, outdir / "direction_precision_grid.png")


def draw(opsdf: pd.DataFrame, grid: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    S1, S2 = "#2a78d6", "#eb6834"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(color="#e9e8e4", lw=0.8)
        ax.set_axisbelow(True)

    # left: the frontier -- accuracy needed vs the precision you get there
    ax = axes[0]
    o = opsdf[opsdf.n >= 20]
    ax.plot(o.precision * 100, o.accuracy_needed * 100, "o-", color=S1, lw=2, ms=7)
    for _, r in o.iterrows():
        ax.annotate(f"top {r.top_frac:.1%}\nn={int(r.n)}",
                    (r.precision * 100, r.accuracy_needed * 100),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=7.5, color=INK2)
    ax.axhline(65.9, color=S2, lw=1.8, ls="--")
    ax.annotate("ROC-AUC 0.659, the best direction\nresult in this project (a weaker\n"
                "quantity than accuracy)",
                (o.precision.min() * 100, 66.5), fontsize=8.5, color=S2)
    ax.set_xlabel("selector precision at that cut (%)", color=INK2, fontsize=9.5)
    ax.set_ylabel("directional accuracy needed to break even (%)", color=INK2, fontsize=9.5)
    ax.set_title("Selecting harder buys down the direction you need",
                 color=INK, fontsize=11.5, loc="left")

    # right: EV curves
    ax = axes[1]
    for a, col in [(0.65, "#9AA5B1"), (0.75, S1), (0.90, "#1baf7a"), (1.00, S2)]:
        g = grid[(grid.accuracy == a) & (grid.n >= 20)].sort_values("top_frac")
        ax.plot(g.top_frac * 100, g.ev, "o-", lw=2, ms=6, color=col,
                label=f"direction {a:.0%}")
    ax.axhline(0, color=INK2, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("how tightly you select: top % of jumps taken", color=INK2, fontsize=9.5)
    ax.set_ylabel("EV per trade (ticks), 30s hold", color=INK2, fontsize=9.5)
    ax.set_title("Only the tightest cut survives, and only at 75% direction or better",
                 color=INK, fontsize=11.5, loc="left")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
