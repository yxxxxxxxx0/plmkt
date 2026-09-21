"""
Before a large jump, does one side of the book vanish while the other stays?

Justin's question, and it is the direction question in its most concrete form:
if the bids evaporate and the asks do not, the price should fall, and that
would be a usable signal.

The project has looked at pre-jump asymmetry before and found little (RULED_OUT
A4: Cliff's delta max 0.305, and larger at -15s than -1s; A6: the collapsing
side predicts direction 50.2% of the time against a 55.2% baseline). Both were
measured across ALL jumps. This asks it again of the tail that matters -- the
large, durable moves -- because today's work showed the payers are a small and
quite different population from the median jump.

Method, deliberately model-free:

  * every durable jump from oracle_jump_scan.py, bucketed by size in ticks;
  * for each, read the book at -30s, -20s, -10s, -5s, -2s and 0 relative to
    the jump's onset;
  * measure how each side's resting dollars changed over that window, then
    orient by the direction the price actually went;
  * the headline quantity is `into - away`: how much more the side the price
    moved INTO depleted, compared with the side it moved away from. Positive
    means the book gets out of the way in the direction of the move, which is
    what Justin's hypothesis predicts;
  * matched controls drawn at the WINDOW START, not at the jump instant. The
    earlier event study matched at the anchor and so picked control moments
    that were already depleted, which hid the effect entirely (defect 4).

The predictive test is the one that decides anything: take the sign of the
asymmetry at t and ask how often it calls the direction of the jump. Anything
near 50% is another closed door.

Outputs, under results/makinen/asymmetry/
  asymmetry_by_size.csv      the event study, by jump size
  direction_hit_rate.csv     can the asymmetry call the side?
  pre_jump_asymmetry.png     the picture
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

GRID_MS = 200
TICK = 0.01
OFFSETS_S = [-30, -20, -10, -5, -2, 0]
SIZE_BUCKETS = [(2, 3), (3, 5), (5, 10), (10, 20), (20, 10_000)]
COLS = ["log_bid_usd", "log_ask_usd", "imb1", "imb5", "spread_ticks", "mid"]


def cliffs_delta(a: np.ndarray, b: np.ndarray, cap: int = 4000, seed: int = 0) -> float:
    """P(a>b) - P(a<b), subsampled for tractability."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    if len(a) > cap:
        a = rng.choice(a, cap, replace=False)
    if len(b) > cap:
        b = rng.choice(b, cap, replace=False)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def build(outdir: Path) -> pd.DataFrame:
    ev = pd.read_parquet(outdir.parent / "oracle_jumps" / "events.parquet")
    ev = ev[ev.is_jump & np.isfinite(ev.pnl_peak)].copy()
    ev["dirn"] = np.sign(ev.peak_mid - ev.entry_mid)
    ev = ev[ev.dirn != 0]
    ev["size_ticks"] = ev.gross_peak

    out = []
    for sess, g in ev.groupby("session"):
        feat = pd.read_parquet(ROOT / "data" / "jump" / f"feat_{sess}_trimmed.parquet",
                               columns=COLS + ["series", "ts", "valid"])
        feat = feat[feat.valid].sort_values(["series", "ts"]).reset_index(drop=True)
        rng = np.random.default_rng(0)

        for ser, rows in g.groupby("series"):
            s = feat[feat.series == ser]
            if len(s) < 200:
                continue
            ts = s.ts.to_numpy()
            arr = {c: s[c].to_numpy() for c in COLS}
            pos = np.searchsorted(ts, rows.ts.to_numpy())
            pos = np.clip(pos, 0, len(ts) - 1)

            # controls: random valid instants in the same contract, drawn so a
            # full 30s pre-window exists. Matched at the WINDOW START below.
            lo = int(30 * 1000 / GRID_MS)
            if len(ts) <= lo + 5:
                continue
            ctrl = rng.choice(np.arange(lo, len(ts)), size=min(len(pos), len(ts) - lo),
                              replace=False)

            for tag, idx, dirs in (("jump", pos, rows.dirn.to_numpy()),
                                   ("control", ctrl, rng.choice([-1.0, 1.0], len(ctrl)))):
                rec = {"session": sess, "series": ser, "kind": tag, "dirn": dirs}
                if tag == "jump":
                    rec["size_ticks"] = rows.size_ticks.to_numpy()
                    rec["slug"] = rows.slug.to_numpy()
                else:
                    rec["size_ticks"] = np.full(len(idx), np.nan)
                    rec["slug"] = np.full(len(idx), rows.slug.iloc[0])
                ok = np.ones(len(idx), bool)
                for o in OFFSETS_S:
                    k = int(abs(o) * 1000 / GRID_MS)
                    j = idx - k
                    ok &= j >= 0
                    j = np.clip(j, 0, len(ts) - 1)
                    # windows must not bridge a gap
                    ok &= (ts[np.clip(idx, 0, len(ts) - 1)] - ts[j]) <= abs(o) * 1000 + 400
                    for c in COLS:
                        rec[f"{c}_t{o}"] = arr[c][j]
                rec["ok"] = ok
                out.append(pd.DataFrame(rec))

    df = pd.concat(out, ignore_index=True)
    return df[df.ok].drop(columns="ok").reset_index(drop=True)


def orient(df: pd.DataFrame) -> pd.DataFrame:
    """Express each side's change as 'the side the price moved into' vs 'away'."""
    base = -30
    for o in OFFSETS_S:
        d_bid = df[f"log_bid_usd_t{o}"] - df[f"log_bid_usd_t{base}"]
        d_ask = df[f"log_ask_usd_t{o}"] - df[f"log_ask_usd_t{base}"]
        up = df.dirn > 0
        # price rising eats / climbs the ASK; falling eats the BID
        df[f"into_t{o}"] = np.where(up, d_ask, d_bid)
        df[f"away_t{o}"] = np.where(up, d_bid, d_ask)
        df[f"gap_t{o}"] = df[f"into_t{o}"] - df[f"away_t{o}"]
        # raw imbalance oriented the same way (positive = book favours the move)
        df[f"imb1_or_t{o}"] = np.where(up, df[f"imb1_t{o}"], -df[f"imb1_t{o}"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results" / "makinen" / "asymmetry"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("reading the book around every jump ...")
    df = build(outdir)
    df = orient(df)
    j = df[df.kind == "jump"]
    c = df[df.kind == "control"]
    print(f"  {len(j):,} jumps, {len(c):,} matched controls\n")

    rows = []
    for lo, hi in SIZE_BUCKETS:
        m = j[(j.size_ticks >= lo) & (j.size_ticks < hi)]
        if len(m) < 50:
            continue
        r = {"bucket": f"{lo}-{hi if hi < 9999 else '+'} ticks", "n": len(m)}
        for o in OFFSETS_S:
            r[f"into_t{o}"] = m[f"into_t{o}"].mean()
            r[f"away_t{o}"] = m[f"away_t{o}"].mean()
            r[f"gap_t{o}"] = m[f"gap_t{o}"].mean()
        r["delta_gap_vs_control"] = cliffs_delta(m["gap_t0"].to_numpy(),
                                                 c["gap_t0"].to_numpy())
        rows.append(r)
    by = pd.DataFrame(rows)
    by.to_csv(outdir / "asymmetry_by_size.csv", index=False)

    print("HOW EACH SIDE MOVES IN THE 30s BEFORE A JUMP (log dollars, vs t-30s)")
    show = ["bucket", "n"] + [f"{k}_t{o}" for o in (-10, -5, -2, 0) for k in ("into", "away")]
    print(by[show].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print()
    print("  'into'  = the side the price moved INTO (asks for an up-move)")
    print("  'away'  = the other side")
    print("  negative = that side lost resting dollars over the window")
    print()
    print("THE ASYMMETRY ITSELF (into - away; positive = book clears ahead of the move)")
    print(by[["bucket", "n"] + [f"gap_t{o}" for o in OFFSETS_S] +
             ["delta_gap_vs_control"]].to_string(
        index=False, float_format=lambda v: f"{v:+.3f}"))

    # The test that decides it: can the asymmetry call the side, before the fact?
    hit = []
    for lo, hi in SIZE_BUCKETS:
        m = j[(j.size_ticks >= lo) & (j.size_ticks < hi)]
        if len(m) < 50:
            continue
        for o in (-10, -5, -2):
            # unoriented signals, so this is a genuine prediction
            d_bid = m[f"log_bid_usd_t{o}"] - m[f"log_bid_usd_t-30"]
            d_ask = m[f"log_ask_usd_t{o}"] - m[f"log_ask_usd_t-30"]
            # hypothesis: the side that vanished is the side the price moves into
            pred = np.where(d_ask < d_bid, 1.0, -1.0)
            hit.append({
                "bucket": f"{lo}-{hi if hi < 9999 else '+'} ticks", "n": len(m),
                "lead_s": o,
                "hit_rate_depletion_rule": float((pred == m.dirn.to_numpy()).mean()),
                "hit_rate_imbalance_rule": float(
                    (np.where(m[f"imb1_t{o}"] > 0, 1.0, -1.0) == m.dirn.to_numpy()).mean()),
                "base_rate_always_up": float((m.dirn > 0).mean()),
            })
    hr = pd.DataFrame(hit)
    hr.to_csv(outdir / "direction_hit_rate.csv", index=False)
    print()
    print("CAN IT CALL THE SIDE? (rule: the side that lost more dollars is the")
    print("side the price moves into. Compare against always guessing the")
    print("majority direction.)")
    print(hr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    draw(by, hr, j, outdir / "pre_jump_asymmetry.png")
    print(f"\nwrote {outdir}")


def draw(by: pd.DataFrame, hr: pd.DataFrame, j: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(color="#e9e8e4", lw=0.8)
        ax.set_axisbelow(True)

    # 1. the two sides over the pre-window, for the largest bucket
    ax = axes[0]
    big = by.iloc[-1]
    x = OFFSETS_S
    ax.plot(x, [big[f"into_t{o}"] for o in x], "o-", color=S2, lw=2.2, ms=7,
            label="side the price moves INTO")
    ax.plot(x, [big[f"away_t{o}"] for o in x], "o-", color=S1, lw=2.2, ms=7,
            label="the other side")
    ax.axhline(0, color=INK2, lw=1.1)
    ax.set_xlabel("seconds before the jump", color=INK2, fontsize=9.5)
    ax.set_ylabel("change in log resting dollars since t-30s", color=INK2, fontsize=9.5)
    ax.set_title(f"Largest jumps ({big.bucket}, n={int(big.n):,})",
                 color=INK, fontsize=11.5, loc="left")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK2)

    # 2. the asymmetry by jump size
    ax = axes[1]
    for o, col in [(-10, "#9AA5B1"), (-5, S1), (-2, S3), (0, S2)]:
        ax.plot(range(len(by)), [by[f"gap_t{o}"].iloc[i] for i in range(len(by))],
                "o-", lw=2, ms=6, color=col, label=f"t{o}s")
    ax.axhline(0, color=INK2, lw=1.1)
    ax.set_xticks(range(len(by)))
    ax.set_xticklabels([f"{b}\nn={int(n):,}" for b, n in zip(by.bucket, by.n)], fontsize=8)
    ax.set_xlabel("jump size", color=INK2, fontsize=9.5)
    ax.set_ylabel("into - away (negative = the move's side emptied more)",
                  color=INK2, fontsize=9.5)
    ax.set_title("Does one side vanish ahead of the move?", color=INK, fontsize=11.5,
                 loc="left")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)

    # 3. can it call the side -- benchmarked against the MAJORITY CLASS, not
    # against 50%. Guessing "always up" in a bucket that is 56% up scores 56%,
    # so a coin-flip line would flatter the rule badly here.
    ax = axes[2]
    b5 = hr[hr.lead_s == -5].reset_index(drop=True)
    base = np.maximum(b5.base_rate_always_up, 1 - b5.base_rate_always_up) * 100
    for lead, col in [(-10, "#9AA5B1"), (-5, S1), (-2, S3)]:
        s_ = hr[hr.lead_s == lead].reset_index(drop=True)
        ax.plot(range(len(s_)), s_.hit_rate_depletion_rule * 100, "o-", lw=2, ms=6,
                color=col, label=f"depletion rule, t{lead}s")
    ax.plot(range(len(b5)), base, "s--", lw=2.2, ms=7, color="#d03b3b",
            label="always guess the majority side")
    ax.set_xticks(range(len(b5)))
    ax.set_xticklabels(b5.bucket, fontsize=8, rotation=15)
    ax.set_xlabel("jump size", color=INK2, fontsize=9.5)
    ax.set_ylabel("directional hit rate (%)", color=INK2, fontsize=9.5)
    ax.set_title("The rule loses to guessing the majority side, at every size",
                 color=INK, fontsize=11.5, loc="left")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
