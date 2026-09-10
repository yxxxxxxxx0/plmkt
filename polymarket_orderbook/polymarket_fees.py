"""Polymarket fee schedule, verified against the docs (checked 2026-09-10).

Why this file exists: `paper_maker_sim.py`'s config carried
`taker_fee_rate: 0.07` as a bare number, and reading it as "7% of notional"
made every taker strategy look impossible -- a fee of 2.98 ticks against a
mean move of 1.78 ticks, i.e. a break-even accuracy of 1.000. That was wrong
twice over: 0.07 is the CRYPTO rate (sports is 0.05), and the fee is not a
flat fraction of notional.

The actual formula, from https://docs.polymarket.com/trading/fees:

    fee = C * feeRate * p * (1 - p)          dollars, charged on shares traded

with C the share count and p the share price. It is symmetric about p = 0.5,
peaks there, and falls to nearly nothing at the extremes -- a trade at 0.05
costs 0.05 * 0.05 * 0.95 = 0.0024/share, twenty times cheaper per share than
the same trade at 0.50. Confirmed against the docs' worked example: 100 shares
at $0.50 on crypto = 100 * 0.07 * 0.5 * 0.5 = $1.75.

Makers are never charged. A category share of taker fees funds the Maker
Rebates Program (25% for most categories, 20% crypto, 50% finance), paid daily
in proportion to filled maker volume, and a tiered Taker Rebate Program (live
2026-05-29) returns part of taker fees by 30-day weighted volume. Neither is
modelled here; both can only make a taker strategy cheaper than these numbers.

The US CFTC-regulated venue is a DIFFERENT schedule -- flat 0.30% taker fee
and 0.20% maker rebate -- so `us_regulated_fee` is provided separately. Which
one applies depends on the jurisdiction you actually trade from, and
`maker_regime_scanner.py` already checks the geoblock endpoint on every run.
"""
from __future__ import annotations

import numpy as np

# https://docs.polymarket.com/trading/fees -- international (non-US) venue
TAKER_FEE_RATE = {
    "crypto": 0.07,
    "sports": 0.05,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "mentions": 0.04,
    "tech": 0.04,
    "geopolitics": 0.00,
}
MAKER_FEE_RATE = 0.0            # makers are never charged
US_TAKER_FEE_FRAC = 0.0030      # flat 0.30% on the CFTC-regulated venue
US_MAKER_REBATE_FRAC = 0.0020   # 0.20% rebate there


def taker_fee_per_share(price, category="sports"):
    """Dollars of taker fee per share at `price`.

    fee = feeRate * p * (1 - p). Vectorised over `price`.
    """
    rate = TAKER_FEE_RATE.get(str(category).lower(), 0.05)
    p = np.clip(np.asarray(price, dtype=np.float64), 0.0, 1.0)
    return rate * p * (1.0 - p)


def taker_fee_ticks(price, tick=0.01, category="sports"):
    """The same fee expressed in ticks, which is how the studies here count."""
    return taker_fee_per_share(price, category) / tick


def round_trip_cost_ticks(price_in, price_out, spread_in, spread_out,
                          tick=0.01, category="sports", exit_as_maker=False):
    """Total cost in ticks of entering and leaving a directional position.

    Entry always crosses the spread and pays a taker fee. The exit is the
    decisive lever and the reason it is a parameter rather than a constant:

      exit_as_maker=False  cross back out. Pay half the far spread AND a
                           second taker fee. This is the conservative case and
                           the only one this data can support.
      exit_as_maker=True   post a limit order and be filled passively. You
                           EARN half the spread instead of paying it, and pay
                           no fee at all -- but whether that fill happens, and
                           what it is adversely selected against, is exactly
                           what reports/FINDINGS.md flags as unmeasurable
                           without trade prints. Treat it as the optimistic
                           bound, not a plan.
    """
    fee_in = taker_fee_ticks(price_in, tick, category)
    half_in = np.asarray(spread_in, dtype=np.float64) / 2.0
    half_out = np.asarray(spread_out, dtype=np.float64) / 2.0
    if exit_as_maker:
        return fee_in + half_in - half_out
    return fee_in + half_in + half_out + taker_fee_ticks(price_out, tick,
                                                         category)


def break_even_accuracy(move_ticks, cost_ticks):
    """Directional accuracy needed for E[pnl] = 0 on a move of that size.

    E[pnl] = m(2p - 1) - cost  =>  p* = 0.5 + cost / (2m)
    """
    m = np.maximum(np.asarray(move_ticks, dtype=np.float64), 1e-9)
    return np.minimum(0.5 + np.asarray(cost_ticks) / (2.0 * m), 1.0)


if __name__ == "__main__":
    # reproduce the docs' worked example as a self-check
    got = taker_fee_per_share(0.50, "crypto") * 100
    print(f"100 shares @ $0.50, crypto: ${got:.2f}  (docs say $1.75)")
    assert abs(got - 1.75) < 1e-9, got
    for p in (0.05, 0.20, 0.425, 0.50, 0.80, 0.95):
        print(f"  sports @ p={p:.3f}: {taker_fee_ticks(p):.2f} ticks/share "
              f"({taker_fee_per_share(p)*100:.3f} cents)")
    print("self-check OK")
