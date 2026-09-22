"""
Re-price the selector's picks on an exit you could actually run.

`can_you_select_payers.py` showed a model reaching 52.9% precision on the paying
jumps, and showed those picks earning +3.04 ticks when exited "3s after the
peak". That number is not real: the peak is located with future information, so
"3s after the peak" is an oracle clock, not a strategy.

This re-prices the same picks on a fixed clock -- exit at entry + T seconds --
which is implementable. Direction is still granted for free, so this remains an
upper bound, just a much tighter one.

Three things are needed to make a trade, and this separates them:

  1. pick a jump that pays        -- the model, 52.9% precision at the top
  2. know which way it goes       -- granted here; best measured elsewhere
                                     in this project is ROC 0.659
  3. get out at the right moment  -- a fixed clock, priced honestly here

Every exit is charged the spread actually standing at entry + T, plus the
Polymarket sports taker fee on both legs. Confidence intervals are bootstrapped
over whole games.

Outputs, under results/makinen/oracle_jumps/
  selector_fixed_clock.csv   P&L by operating point and holding time
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

TICK = 0.01
GRID_MS = 200
SPORTS_FEE_RATE = 0.05
HOLDS_S = [5, 10, 20, 30, 60, 120, 300]
FEATURES = [
    "spread_ticks", "imb1", "imb3", "imb5", "imb10",
    "log_bid_usd", "log_ask_usd", "conc_bid", "conc_ask",
    "reach_bid", "reach_ask", "microprice_dev",
    "rv_5", "rv_25", "rv_150", "ofi_5", "dmid_5", "ofi_25", "dmid_25",
    "stale", "book_age_ms", "mid",
]
def _test_sessions(n_held_out=2):
    """The chronologically LAST `n_held_out` sessions, discovered not listed.

    This was the hardcoded pair ["books_2026-09-12", "books_2026-09-13"], and
    it silently stopped being a chronological split the moment later sessions
    were built: 09-12 and 09-13 stayed the test set while 09-17 through 09-20
    -- recorded AFTER them -- joined TRAINING. The "no game straddles the
    split" assertion still passed, because no game does, so nothing announced
    it. A model fitted on the future of its own test period is not a held-out
    result, and every score it produces is suspect upwards.

    Session names are `books_YYYY-MM-DD`, so lexical order is chronological.
    """
    suffix = "_trimmed.parquet"
    built = sorted(q.name[len("feat_"):-len(suffix)]
                   for q in (ROOT / "data" / "jump").glob("feat_books_*" + suffix))
    return built[-n_held_out:]


TEST_SESSIONS = _test_sessions()


def fee_ticks(price):
    p = np.clip(np.asarray(price, dtype=np.float64), 0.0, 1.0)
    return SPORTS_FEE_RATE * p * (1.0 - p) / TICK


def add_fixed_clock(ev: pd.DataFrame) -> pd.DataFrame:
    """P&L of holding each jump for a fixed wall-clock time, direction granted."""
    out = []
    for sess, g in ev.groupby("session"):
        src = ROOT / "data" / "jump" / f"feat_{sess}_trimmed.parquet"
        feat = pd.read_parquet(src, columns=["mid", "spread_ticks", "ts", "series", "valid"])
        feat = feat[feat.valid].sort_values(["series", "ts"])
        g = g.copy()
        for T in HOLDS_S:
            g[f"pnl_T{T}"] = np.nan
        # keep the 30s components so the direction controls can be rebuilt
        g["raw_T30"] = np.nan
        g["cost_T30"] = np.nan
        for ser, rows in g.groupby("series"):
            s = feat[feat.series == ser]
            if len(s) == 0:
                continue
            ts = s.ts.to_numpy()
            mid = s.mid.to_numpy()
            spr = s.spread_ticks.to_numpy()
            pos = np.searchsorted(ts, rows.ts.to_numpy())
            pos = np.clip(pos, 0, len(ts) - 1)
            entry_mid = mid[pos]
            entry_spr = spr[pos]
            dirn = np.sign(rows.peak_mid.to_numpy() - rows.entry_mid.to_numpy())
            for T in HOLDS_S:
                want = rows.ts.to_numpy() + T * 1000
                j = np.clip(np.searchsorted(ts, want), 0, len(ts) - 1)
                # the exit must really exist: no bridging a gap or the game end
                good = np.abs(ts[j] - want) <= 2000
                move = dirn * (mid[j] - entry_mid) / TICK
                cost = (entry_spr + spr[j]) / 2.0
                fee = fee_ticks(entry_mid) + fee_ticks(mid[j])
                g.loc[rows.index, f"pnl_T{T}"] = np.where(good, move - cost - fee, np.nan)
                if T == 30:
                    g.loc[rows.index, "raw_T30"] = np.where(
                        good, (mid[j] - entry_mid) / TICK, np.nan)
                    g.loc[rows.index, "cost_T30"] = cost + fee
        out.append(g)
    return pd.concat(out, ignore_index=True)


def boot_ci(frame: pd.DataFrame, col: str, n_boot: int = 4000, seed: int = 0):
    v = frame[[col, "slug"]].dropna()
    if len(v) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    games = v.slug.unique()
    by = {g: v[col].to_numpy()[v.slug.to_numpy() == g] for g in games}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(games, size=len(games), replace=True)
        cat = np.concatenate([by[g] for g in pick])
        draws[b] = cat.mean() if len(cat) else np.nan
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "oracle_jumps"))
    args = ap.parse_args()
    outdir = Path(args.outdir)

    import lightgbm as lgb

    ev = pd.read_parquet(outdir / "events.parquet")
    ev = ev[ev.is_jump & np.isfinite(ev.pnl_peak)].copy()

    parts = []
    for sess, g in ev.groupby("session"):
        src = ROOT / "data" / "jump" / f"feat_{sess}_trimmed.parquet"
        feat = pd.read_parquet(src, columns=FEATURES + ["series", "ts"])
        parts.append(g.merge(feat, on=["series", "ts"], how="left", suffixes=("", "_f")))
    df = pd.concat(parts, ignore_index=True).dropna(subset=FEATURES).reset_index(drop=True)
    df["y"] = (df.pnl_peak > 0).astype(int)

    print("pricing fixed-clock exits ...")
    df = add_fixed_clock(df)

    te = df.session.isin(TEST_SESSIONS)
    tr = ~te
    assert not set(df.slug[tr]) & set(df.slug[te])

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, class_weight="balanced",
        random_state=0, verbosity=-1,
    )
    model.fit(df.loc[tr, FEATURES], df.loc[tr, "y"])
    sub = df.loc[te].copy()
    sub["p"] = model.predict_proba(sub[FEATURES])[:, 1]
    order = sub.sort_values("p", ascending=False)

    rows = []
    for frac in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 1.00]:
        k = max(1, int(round(frac * len(order))))
        sel = order.head(k)
        rec = {
            "top_frac": frac,
            "n": k,
            "precision": float(sel.y.mean()),
            "oracle_exit_at_peak": float(sel.pnl_peak.mean()),
        }
        for T in HOLDS_S:
            c = f"pnl_T{T}"
            rec[f"T{T}s"] = float(sel[c].mean())
            lo, hi = boot_ci(sel, c)
            rec[f"T{T}s_lo"] = lo
            rec[f"T{T}s_hi"] = hi
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "selector_fixed_clock.csv", index=False)

    print()
    print("REALISED P&L ON A CLOCK YOU CAN RUN (direction still granted free)")
    show = ["top_frac", "n", "precision", "oracle_exit_at_peak"] + [f"T{T}s" for T in HOLDS_S]
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print()
    print("game-clustered 95% CIs at the best operating point")
    best = res.iloc[0]
    for T in HOLDS_S:
        print(f"  hold {T:4d}s: {best[f'T{T}s']:+7.3f} t  "
              f"[{best[f'T{T}s_lo']:+.3f}, {best[f'T{T}s_hi']:+.3f}]"
              f"{'   <-- CI excludes zero' if best[f'T{T}s_lo'] > 0 else ''}")

    # --- the controls that decide it ----------------------------------------
    # Everything above grants direction for free. Take that away and see what
    # survives; then state what directional skill would actually be required.
    print()
    print("CONTROL: the same picks without a free direction (30s hold)")
    rng = np.random.default_rng(0)
    ctrl_rows = []
    for frac in [0.001, 0.005, 0.01, 0.02]:
        k = max(1, int(round(frac * len(order))))
        x = order.head(k).dropna(subset=["pnl_T30"]).copy()
        dirn = np.sign(x.peak_mid.to_numpy() - x.entry_mid.to_numpy())
        # pnl = dirn*raw - cost, so raw and cost come back out of the stored pnl
        raw = x.raw_T30.to_numpy()
        cost = x.cost_T30.to_numpy()
        coin = np.mean([(rng.choice([-1, 1], len(x)) * raw - cost).mean() for _ in range(2000)])
        em, c = np.abs(raw).mean(), cost.mean()
        ctrl_rows.append({
            "top_frac": frac, "n": len(x), "precision": float(x.y.mean()),
            "oracle_direction": float((dirn * raw - cost).mean()),
            "always_long": float((raw - cost).mean()),
            "always_short": float((-raw - cost).mean()),
            "coin_flip": float(coin),
            "mean_abs_move": em, "mean_cost_plus_fee": c,
            "perfect_direction_ev": em - c,
            "accuracy_needed": (1 + c / em) / 2 if em > 0 else np.nan,
        })
    ctrl = pd.DataFrame(ctrl_rows)
    ctrl.to_csv(outdir / "selector_direction_controls.csv", index=False)
    print(ctrl.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print()
    print("  Every operating point is negative once direction is not free.")
    print("  'accuracy_needed' is the directional hit rate that would make the")
    print("  picks break even. Note it is an ACCURACY; the best direction result")
    print("  in this project is ROC-AUC 0.659, which is a weaker thing.")

    # Is any of it concentrated? A handful of games producing all of it is the
    # signature of an artefact, not an edge.
    k = max(1, int(round(0.005 * len(order))))
    sel = order.head(k)
    bg = sel.groupby("slug").agg(n=("pnl_T30", "size"), ticks=("pnl_T30", "sum")).sort_values(
        "ticks", ascending=False)
    print()
    print(f"top 0.5% ({k} trades), 30s hold, by game -- top 6 of {len(bg)}")
    print(bg.head(6).round(2).to_string())
    tot = bg.ticks.sum()
    print(f"  total {tot:,.1f} t; top game contributes {bg.ticks.iloc[0] / tot:.0%}"
          if tot else "  total zero")


if __name__ == "__main__":
    main()
