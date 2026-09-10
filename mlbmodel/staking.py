"""Bankroll simulation across staking schemes.

Why this module exists
----------------------
`backtest.bet_sim` reports a `kelly_growth` figure that is wrong, and wrong
in the flattering direction. It compounds bets one at a time with
`prod(1 + r)`, which assumes each wager resolves before the next is placed.
Baseball plays ~15 games simultaneously. Staking 5% on eight of today's
games puts 40% of the bankroll at risk at once, not 5% eight times in a row,
and the sequential product overstates growth badly.

This module resolves bets in daily batches, caps total exposure per day, and
reports the numbers that decide whether a staking plan is survivable:
terminal wealth, max drawdown, and how often the path is ruined.

The theory is not in doubt. Expected profit is linear in stake,

    E[profit] = sum_i (stake_i * edge_i)

so if every edge_i is negative, no choice of positive stakes makes the sum
positive. Sizing controls the *variance and growth rate* of an edge that
already exists; it cannot manufacture one. Kelly itself says so: the optimal
fraction (p*d - 1)/(d - 1) goes negative exactly when p*d < 1, and the
correct action there is to stake nothing.

What sizing genuinely decides is how a real edge is harvested -- full Kelly
maximises log growth but rides brutal drawdowns, while a fraction of Kelly
gives up a little growth for a far smoother path. That trade-off is what the
simulation below measures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OUT


# --------------------------------------------------------------------------- #
def _select(p, dh, da, edge_thresh):
    """Pick a side per game and return (stake_side_prob, decimal_odds, on_home)."""
    ev_h, ev_a = p * dh - 1.0, (1 - p) * da - 1.0
    take_h = (ev_h > edge_thresh) & (ev_h >= ev_a)
    take_a = (ev_a > edge_thresh) & (ev_a > ev_h)
    bet = take_h | take_a
    prob = np.where(take_h, p, 1 - p)
    dec = np.where(take_h, dh, da)
    return bet, prob, dec, take_h


def kelly_fraction(prob, dec):
    """Full-Kelly stake as a fraction of bankroll. Negative edge -> 0."""
    b = dec - 1.0
    f = (prob * dec - 1.0) / b
    return np.clip(f, 0.0, 1.0)


def _day_blocks(df: pd.DataFrame, edge_thresh: float):
    """Pre-resolve the bet selection once, into per-day numpy blocks.

    Selection does not depend on the bankroll, so it is wasteful to redo it
    inside every bootstrap iteration. Each block is (kelly_fraction, decimal
    odds, won) for one game-day.
    """
    d = df.dropna(subset=["p_home", "dec_home", "dec_away"]).sort_values("date")
    if d.empty:
        return [], None, None
    p = d.p_home.to_numpy(dtype=float)
    dh, da = d.dec_home.to_numpy(dtype=float), d.dec_away.to_numpy(dtype=float)
    y = d.home_win.to_numpy(dtype=int)
    bet, prob, dec, on_home = _select(p, dh, da, edge_thresh)
    won = np.where(on_home, y == 1, y == 0)
    kf = kelly_fraction(prob, dec)

    dates = d.date.to_numpy()
    blocks = []
    for dt in pd.unique(dates):
        m = (dates == dt) & bet
        if m.any():
            blocks.append((kf[m], dec[m], won[m]))
    return blocks, d.date.min(), d.date.max()


def _run_blocks(blocks, scheme, flat_frac, max_per_bet, max_per_day, bankroll0):
    frac_of_kelly = {"full_kelly": 1.0, "half_kelly": 0.5,
                     "quarter_kelly": 0.25, "tenth_kelly": 0.1}.get(scheme)
    bank = peak = bankroll0
    max_dd = 0.0
    n_bets = 0
    staked = 0.0
    ruined = False

    for kf, dec, won in blocks:
        if scheme == "flat":
            frac = np.full(len(dec), flat_frac)
        elif scheme == "flat_stake":
            # flat fraction of the ORIGINAL bankroll, i.e. no compounding
            frac = np.full(len(dec), flat_frac * bankroll0 / max(bank, 1e-12))
        else:
            frac = frac_of_kelly * kf

        frac = np.minimum(frac, max_per_bet)
        tot = frac.sum()
        if tot > max_per_day:       # cap same-day exposure
            frac = frac * (max_per_day / tot)

        stake = frac * bank
        bank += np.where(won, stake * (dec - 1.0), -stake).sum()
        n_bets += len(dec)
        staked += stake.sum()

        if bank <= bankroll0 * 0.01:        # effectively wiped out
            ruined = True
            bank = max(bank, 0.0)
            break
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak)

    return bank, max_dd, ruined, n_bets, staked


def simulate(df: pd.DataFrame, scheme: str = "quarter_kelly",
             edge_thresh: float = 0.05, flat_frac: float = 0.01,
             max_per_bet: float = 0.05, max_per_day: float = 0.25,
             bankroll0: float = 1.0) -> dict:
    """Walk the bankroll forward day by day.

    Bets on the same date are placed together out of the bankroll standing at
    the start of that date, then all resolve at once. Per-bet and per-day
    exposure caps are applied the way any real staking plan would.
    """
    blocks, d0, d1 = _day_blocks(df, edge_thresh)
    if not blocks:
        return {}
    bank, max_dd, ruined, n_bets, staked = _run_blocks(
        blocks, scheme, flat_frac, max_per_bet, max_per_day, bankroll0)
    days = max((d1 - d0).days, 1)
    return dict(
        scheme=scheme, edge_thresh=edge_thresh, bets=n_bets,
        final=bank, total_return=bank / bankroll0 - 1.0,
        cagr=(bank / bankroll0) ** (365.0 / days) - 1.0 if bank > 0 else -1.0,
        max_drawdown=max_dd, ruined=ruined,
        roi_on_turnover=(bank - bankroll0) / staked if staked else np.nan,
    )


# --------------------------------------------------------------------------- #
def block_bootstrap(df: pd.DataFrame, scheme: str, edge_thresh: float = 0.05,
                    n_boot: int = 400, seed: int = 0, flat_frac: float = 0.01,
                    max_per_bet: float = 0.05, max_per_day: float = 0.25,
                    bankroll0: float = 1.0) -> pd.DataFrame:
    """Resample whole game-days with replacement.

    A single bankroll path is one sample. Resampling by day (not by bet)
    keeps same-day bets together, which matters because they compete for the
    same exposure cap and therefore are not independent.
    """
    blocks, _, _ = _day_blocks(df, edge_thresh)
    if not blocks:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(blocks), len(blocks))
        bank, max_dd, ruined, n_bets, staked = _run_blocks(
            [blocks[i] for i in idx], scheme, flat_frac,
            max_per_bet, max_per_day, bankroll0)
        rows.append(dict(total_return=bank / bankroll0 - 1.0,
                         max_drawdown=max_dd, ruined=ruined, bets=n_bets))
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, edge_thresh: float = 0.05,
            schemes=("flat_stake", "flat", "tenth_kelly", "quarter_kelly",
                     "half_kelly", "full_kelly"),
            n_boot: int = 400) -> pd.DataFrame:
    rows = []
    for s in schemes:
        r = simulate(df, s, edge_thresh=edge_thresh)
        if not r:
            continue
        b = block_bootstrap(df, s, edge_thresh=edge_thresh, n_boot=n_boot)
        rows.append(dict(
            scheme=s, bets=r["bets"],
            final_x=r["final"], total_return=r["total_return"],
            max_dd=r["max_drawdown"], ruined=r["ruined"],
            boot_median_return=b.total_return.median(),
            boot_p5=b.total_return.quantile(0.05),
            boot_p95=b.total_return.quantile(0.95),
            p_profit=float((b.total_return > 0).mean()),
            p_ruin=float(b.ruined.mean()),
            median_max_dd=float(b.max_drawdown.median()),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def per_season(df: pd.DataFrame, schemes=("flat_stake", "quarter_kelly"),
               edge_thresh: float = 0.05) -> pd.DataFrame:
    """Run each season as its own independent bankroll.

    The pooled block bootstrap silently assumes the sample is stationary. If
    one season carries the entire result, resampling days from the pool will
    report a healthy probability of profit that no real deployment could have
    achieved -- you cannot bet a season you have already seen.
    """
    rows = []
    for s, g in df.groupby("season"):
        for scheme in schemes:
            r = simulate(g, scheme, edge_thresh=edge_thresh)
            if r:
                rows.append(dict(season=int(s), scheme=scheme, bets=r["bets"],
                                 final_x=r["final"], total_return=r["total_return"],
                                 max_dd=r["max_drawdown"]))
    return pd.DataFrame(rows)


def deploy_forward(df: pd.DataFrame, first_live_season: int,
                   schemes=("flat_stake", "tenth_kelly", "quarter_kelly", "half_kelly"),
                   edge_thresh: float = 0.05, n_boot: int = 400) -> pd.DataFrame:
    """The only honest question: bet the seasons *after* the one that looked good.

    With a genuine edge, larger Kelly fractions raise the median return. With
    a negative edge the ordering inverts -- losses grow monotonically with
    stake size, which is the signature to look for.
    """
    live = df[df.season >= first_live_season]
    rows = []
    for scheme in schemes:
        r = simulate(live, scheme, edge_thresh=edge_thresh)
        if not r:
            continue
        b = block_bootstrap(live, scheme, edge_thresh=edge_thresh, n_boot=n_boot)
        rows.append(dict(scheme=scheme, bets=r["bets"], final_x=r["final"],
                         total_return=r["total_return"], max_dd=r["max_drawdown"],
                         p_profit=float((b.total_return > 0).mean()),
                         boot_p5=b.total_return.quantile(0.05),
                         boot_median=b.total_return.median()))
    return pd.DataFrame(rows)


def main():
    from .config import PROC

    pr = pd.read_parquet(OUT / "predictions.parquet")
    ds = pd.read_parquet(PROC / "dataset.parquet")
    fmt = lambda v: f"{v:,.4f}"

    model = "ensemble_elo"
    pr = pr[pr.model == model]

    clean_pks = set(ds.loc[ds.snap_lag_min <= 15, "game_pk"]) \
        if "snap_lag_min" in ds.columns else set()

    for label, sub in (("ALL ODDS (includes stale snapshots)", pr),
                       ("CLEAN SNAPSHOTS ONLY", pr[pr.game_pk.isin(clean_pks)])):
        if sub.empty:
            continue
        print(f"\n{'='*78}\n{label} -- model={model}, edge>5%, {len(sub)} games\n{'='*78}")
        t = compare(sub)
        print(t.to_string(index=False, float_format=fmt))

        print("\n  games per season (is the sample stationary?):")
        print("   ", sub.dropna(subset=["dec_home"]).groupby("season").size().to_dict())
        print("\n  each season as an independent bankroll:")
        print(per_season(sub).to_string(index=False, float_format=fmt))

    clean = pr[pr.game_pk.isin(clean_pks)]
    if not clean.empty:
        print(f"\n{'='*78}\nDEPLOY FORWARD: bet 2024+ only, having seen 2023 look good"
              f"\n{'='*78}")
        print(deploy_forward(clean, 2024).to_string(index=False, float_format=fmt))

    # The control that settles it: sizing applied to a strategy with a known
    # negative edge. If sizing could rescue a losing edge, this would profit.
    print(f"\n{'='*78}\nCONTROL: bet every favourite (known -5.7% edge), sized every way"
          f"\n{'='*78}")
    fav = pr.dropna(subset=["mkt_p_home", "dec_home"]).copy()
    # force the selection to 'always back the favourite' by handing the
    # simulator the market's own probability at a zero edge threshold
    fav["p_home"] = np.where(fav.mkt_p_home >= 0.5, 0.999, 0.001)
    t = compare(fav, edge_thresh=-1.0, n_boot=200)
    print(t.to_string(index=False, float_format=fmt))

    OUT.mkdir(exist_ok=True)
    compare(pr).to_csv(OUT / "staking_comparison.csv", index=False)
    print(f"\nwrote {OUT / 'staking_comparison.csv'}")


if __name__ == "__main__":
    main()
