"""Lead-lag ("causal") arbitrage between a game market and its futures market.

The idea
--------
A team's game result mechanically changes its championship probability. Game
markets are heavily traded and reprice in seconds; the World Series market is
thin and traded by people who are not watching every game. If the futures leg
lags, you can trade it on information already public in the game leg.

This needs no predictive model at all -- only that two mechanically linked
markets update at different speeds. That makes it the most promising remaining
idea after the outright-model work came up empty.

What decides it
---------------
Three quantities, in order of how likely each is to kill the trade:

1. SENSITIVITY -- how much *should* a futures price move per game? For a
   regular-season game this is tiny: one game in 162, filtered through a
   playoff structure. Call it dP_champ/dGame.
2. SPREAD on the futures leg -- you cross it to get in and again to get out.
   Championship markets sit at 1-cent ticks on prices near $0.10, so the
   spread is often 10%+ in relative terms.
3. LAG -- how long the futures leg actually takes to catch up, and whether
   anything is left after (1) and (2).

If sensitivity < spread, the trade is dead before latency even matters. This
module measures all three rather than assuming any of them.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from .config import RAW, PROC

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-research/1.0", "Accept": "application/json"})


def _get(url, params, cache_key=None, ttl_days=0.25):
    """GET with a short-lived disk cache (prices move, so the TTL is hours)."""
    if cache_key:
        fp = RAW / f"{cache_key}.json"
        if fp.exists() and (time.time() - fp.stat().st_mtime) < ttl_days * 86400:
            return json.loads(fp.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt == 3:
                return None
            time.sleep(2 * (attempt + 1))
    if cache_key:
        (RAW / f"{cache_key}.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def _tids(m):
    t = m.get("clobTokenIds")
    return json.loads(t) if isinstance(t, str) else t


def _jl(v):
    return json.loads(v) if isinstance(v, str) else v


# --------------------------------------------------------------------------- #
# 1. Find the futures (championship) markets and their live books
# --------------------------------------------------------------------------- #
def futures_markets(event_slug_contains="world-series") -> pd.DataFrame:
    """One row per team's 'will X win the title' market, with live book."""
    rows = []
    for off in (0, 100, 200):
        j = _get(f"{GAMMA}/events", {"tag_slug": "mlb", "closed": "false",
                                     "limit": 100, "offset": off},
                 f"ll_fut_events_{off}")
        if not j:
            break
        for e in j:
            if event_slug_contains not in str(e.get("slug", "")):
                continue
            for m in e.get("markets", []):
                ct = _tids(m)
                pr = _jl(m.get("outcomePrices"))
                if not ct or not pr:
                    continue
                rows.append(dict(
                    event=e["slug"], question=m.get("question", ""),
                    slug=m.get("slug", ""), token_yes=ct[0],
                    price=float(pr[0]),
                    best_bid=m.get("bestBid"), best_ask=m.get("bestAsk"),
                    tick=float(m.get("orderPriceMinTickSize") or 0.01),
                    vol=float(m.get("volume") or 0),
                    liq=float(m.get("liquidity") or 0)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["spread"] = pd.to_numeric(df.best_ask, errors="coerce") - \
                   pd.to_numeric(df.best_bid, errors="coerce")
    df["spread_rel"] = df.spread / df.price.clip(lower=1e-6)
    return df.sort_values("vol", ascending=False).reset_index(drop=True)


def order_book(token_id: str) -> dict:
    j = _get(f"{CLOB}/book", {"token_id": token_id}, None)
    if not j:
        return {}
    bids = sorted([(float(x["price"]), float(x["size"])) for x in j.get("bids", [])],
                  key=lambda t: -t[0])
    asks = sorted([(float(x["price"]), float(x["size"])) for x in j.get("asks", [])],
                  key=lambda t: t[0])
    if not bids or not asks:
        return {}
    return dict(bid=bids[0][0], bid_sz=bids[0][1], ask=asks[0][0], ask_sz=asks[0][1],
                spread=asks[0][0] - bids[0][0],
                depth5_bid=sum(s for _, s in bids[:5]),
                depth5_ask=sum(s for _, s in asks[:5]),
                n_levels=len(bids) + len(asks))


# --------------------------------------------------------------------------- #
# 2. Price history
# --------------------------------------------------------------------------- #
EMPTY_HIST = pd.DataFrame({"ts": pd.Series([], dtype="datetime64[ns, UTC]"),
                           "p": pd.Series([], dtype="float64")})


def price_history(token_id: str, interval="1d", fidelity=1,
                  cache_key=None) -> pd.DataFrame:
    """Tick history for one CLOB token.

    `fidelity` is the bucket size in minutes and `interval` the lookback.
    Only some combinations are served, so fall back through a few: a longer
    interval silently coarsens the spacing, which matters for a lag study.
    """
    tries = [(interval, fidelity)]
    for iv in ("1d", "1w", "1m", "max"):
        if (iv, fidelity) not in tries:
            tries.append((iv, fidelity))
    for iv, fid in tries:
        j = _get(f"{CLOB}/prices-history",
                 {"market": token_id, "interval": iv, "fidelity": fid},
                 f"{cache_key}_{iv}_{fid}" if cache_key else None)
        if j and j.get("history"):
            h = pd.DataFrame(j["history"])
            h["ts"] = pd.to_datetime(h.t, unit="s", utc=True)
            return h[["ts", "p"]].sort_values("ts").reset_index(drop=True)
    return EMPTY_HIST.copy()


# --------------------------------------------------------------------------- #
# 3. Theoretical sensitivity: dP(champ) / dP(game)
# --------------------------------------------------------------------------- #
def sensitivity_from_history(game_hist: pd.DataFrame, fut_hist: pd.DataFrame,
                             max_lag_min: int = 240) -> pd.DataFrame:
    """Regress futures returns on lagged game returns, minute by minute.

    A positive coefficient at positive lag means the futures leg is still
    catching up to the game leg -- that is the tradeable pattern. A
    coefficient concentrated at lag 0 means both legs move together and
    there is nothing to capture.
    """
    if game_hist.empty or fut_hist.empty:
        return pd.DataFrame()
    g = game_hist.set_index("ts").p.resample("1min").last().ffill()
    f = fut_hist.set_index("ts").p.resample("1min").last().ffill()
    idx = g.index.intersection(f.index)
    if len(idx) < 60:
        return pd.DataFrame()
    g, f = g.loc[idx], f.loc[idx]
    dg, df_ = g.diff(), f.diff()

    rows = []
    for lag in range(0, max_lag_min + 1, 5):
        x = dg.shift(lag)
        m = x.notna() & df_.notna()
        if m.sum() < 30 or x[m].std() == 0:
            continue
        beta = np.polyfit(x[m], df_[m], 1)[0]
        rows.append(dict(lag_min=lag, beta=beta,
                         corr=float(np.corrcoef(x[m], df_[m])[0, 1]),
                         n=int(m.sum())))
    return pd.DataFrame(rows)


def viability(sens_beta: float, game_move: float, fut_price: float,
              fut_spread: float, fut_depth_usd: float) -> dict:
    """Is the expected futures move bigger than the cost of capturing it?

    `sens_beta` is dP_futures per unit dP_game. A game swing of `game_move`
    therefore implies a futures move of beta*game_move. You pay roughly the
    full spread round-trip (cross to enter, cross to exit), so the trade only
    exists if the implied move clears that.
    """
    implied = abs(sens_beta) * abs(game_move)
    cost = fut_spread            # ~one full spread round-trip
    return dict(
        implied_move=implied, round_trip_cost=cost,
        net_per_share=implied - cost,
        ratio=implied / cost if cost > 0 else np.inf,
        tradeable=implied > cost,
        max_size_usd=fut_depth_usd,
        net_usd_at_depth=(implied - cost) / max(fut_price, 1e-6) * fut_depth_usd
        if fut_price else np.nan,
    )


# --------------------------------------------------------------------------- #
# 4. Small-market making survey
# --------------------------------------------------------------------------- #
def survey_books(max_markets: int = 120, vol_ceiling: float | None = None) -> pd.DataFrame:
    """Spread, depth and reward config across live sports moneylines.

    Answers the practical question directly: in the markets a small bankroll
    could actually compete in, how wide is the spread and how much size is
    already resting in front of you?
    """
    rows = []
    seen = set()
    for off in range(0, 1000, 100):
        j = _get(f"{GAMMA}/events", {"tag_slug": "sports", "closed": "false",
                                     "limit": 100, "offset": off,
                                     "order": "volume24hr", "ascending": "false"},
                 f"ll_survey_{off}")
        if not j:
            break
        for e in j:
            for m in e.get("markets", []):
                if m.get("sportsMarketType") != "moneyline":
                    continue
                ct = _tids(m)
                pr = _jl(m.get("outcomePrices"))
                if not ct or not pr or not (0 < float(pr[1]) < 1):
                    continue
                vol = float(m.get("volume") or 0)
                if vol_ceiling is not None and vol > vol_ceiling:
                    continue
                if ct[1] in seen:
                    continue
                seen.add(ct[1])
                rows.append(dict(slug=e.get("slug", ""),
                                 sport=str(e.get("slug", "")).split("-")[0],
                                 token=ct[1], mid=float(pr[1]), vol=vol,
                                 liq=float(m.get("liquidity") or 0),
                                 rmin=m.get("rewardsMinSize"),
                                 rewards=bool(m.get("clobRewards"))))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("vol", ascending=False).head(max_markets)

    books = []
    for _, r in df.iterrows():
        b = order_book(r.token)
        books.append(b if b else {})
    bd = pd.DataFrame(books)
    out = pd.concat([df.reset_index(drop=True), bd], axis=1)
    out = out[out.get("bid").notna()] if "bid" in out.columns else out
    if not out.empty:
        out["spread_rel"] = out.spread / out.mid.clip(lower=1e-6)
        out["bid_usd"] = out.bid * out.bid_sz
        out["ask_usd"] = out.ask * out.ask_sz
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Event study: what does a finished game do to the futures price?
# --------------------------------------------------------------------------- #
def game_event_study(days: int = 16) -> pd.DataFrame:
    """Daily futures move for each live team, tagged by that day's game result.

    Compares the size of the win-minus-loss effect against the tick size and
    the quoted spread. If the effect is smaller than one tick, the market
    cannot express the update even in principle -- which makes this a
    granularity problem, not a latency one, and no amount of speed helps.
    """
    from . import collect

    fut = futures_markets("world-series")
    if fut.empty:
        return pd.DataFrame()
    fut["team"] = (fut.question.str.replace("Will the ", "", regex=False)
                   .str.replace(" win the 2026 World Series?", "", regex=False))
    alive = fut[fut.price > 0.01]

    games = collect.load("games")
    recent = games[games.date >= games.date.max() - pd.Timedelta(days=days)]
    j = collect._get("/teams", dict(sportId=1, season=2026), "teams_2026")
    names = {t["id"]: t["name"] for t in j["teams"]}

    rows = []
    for _, fr in alive.iterrows():
        tid = next((k for k, v in names.items() if v == fr.team), None)
        if tid is None:
            continue
        h = price_history(fr.token_yes, interval="max", fidelity=60,
                          cache_key=f"lle_{fr.team[:8]}")
        if h.empty or len(h) < 20:
            continue
        daily = h.set_index("ts").p.resample("1D").last().dropna().diff().dropna()
        tg = recent[(recent.home_id == tid) | (recent.away_id == tid)].copy()
        tg["won"] = np.where(tg.home_id == tid, tg.home_win == 1, tg.home_win == 0)
        per_day = tg.groupby(tg.date.dt.normalize()).won.agg(["sum", "count"])
        book = order_book(fr.token_yes)
        for dt, dv in daily.items():
            key = pd.Timestamp(dt.date())
            if key not in per_day.index:
                continue
            rows.append(dict(team=fr.team, date=key.date(), price=fr.price,
                             wins=int(per_day.loc[key, "sum"]),
                             games=int(per_day.loc[key, "count"]),
                             d_fut=float(dv), tick=fr.tick,
                             spread=book.get("spread", np.nan)))
    return pd.DataFrame(rows)


def summarize_event_study(R: pd.DataFrame) -> dict:
    if R.empty:
        return {}
    won, lost = R[R.wins > 0].d_fut, R[R.wins == 0].d_fut
    if len(won) < 4 or len(lost) < 4:
        return {}
    diff = won.mean() - lost.mean()
    se = float(np.sqrt(won.var(ddof=1) / len(won) + lost.var(ddof=1) / len(lost)))
    tick, spread = R.tick.median(), R.spread.median()
    return dict(
        n_team_days=len(R), n_won=len(won), n_lost=len(lost),
        mean_move_won=float(won.mean()), mean_move_lost=float(lost.mean()),
        win_minus_loss=float(diff), se=se, t_stat=float(diff / se) if se else np.nan,
        median_tick=float(tick), median_spread=float(spread),
        signal_over_tick=float(abs(diff) / tick) if tick else np.nan,
        signal_over_spread=float(abs(diff) / spread) if spread else np.nan,
        tradeable=bool(abs(diff) > spread),
    )


def main():
    fmt = lambda v: f"{v:,.4f}"

    print("=" * 78)
    print("1. FUTURES LEG -- the cost of trading the slow market")
    print("=" * 78)
    f = futures_markets("world-series")
    if not f.empty:
        f["team"] = (f.question.str.replace("Will the ", "", regex=False)
                     .str.replace(" win the 2026 World Series?", "", regex=False))
        books = pd.DataFrame([order_book(t) or {} for t in f.token_yes])
        v = pd.concat([f[["team", "price", "tick", "vol"]].reset_index(drop=True),
                       books], axis=1).dropna(subset=["spread"])
        v["spread_rel"] = v.spread / v.price.clip(lower=1e-6)
        v["bid_usd"] = v.bid * v.bid_sz
        print(v[["team", "price", "bid", "ask", "spread", "spread_rel",
                 "bid_usd", "vol"]].to_string(index=False, float_format=fmt))
        print(f"\n  median relative spread {v.spread_rel.median()*100:.1f}%, "
              f"median $ at best bid ${v.bid_usd.median():,.0f}")

    print("\n" + "=" * 78)
    print("2. EVENT STUDY -- does a finished game move the futures price?")
    print("=" * 78)
    R = game_event_study()
    s = summarize_event_study(R)
    if s:
        for k, val in s.items():
            print(f"  {k:>20}: {val}")
        print(f"\n  VERDICT: signal is {s['signal_over_tick']:.2f} ticks and "
              f"{s['signal_over_spread']:.2f} spreads -> "
              f"{'TRADEABLE' if s['tradeable'] else 'NOT TRADEABLE'}")
        R.to_csv(PROC / "leadlag_event_study.csv", index=False)

    print("\n" + "=" * 78)
    print("3. SMALL-MARKET MAKING -- does a thinner market pay a wider spread?")
    print("=" * 78)
    sb = survey_books(max_markets=140)
    if not sb.empty:
        sb = sb[sb.spread.notna()]
        sb["vol_bucket"] = pd.cut(sb.vol, [-1, 1e3, 1e4, 1e5, 1e6, 1e12],
                                  labels=["<$1k", "$1k-10k", "$10k-100k",
                                          "$100k-1M", ">$1M"])
        agg = sb.groupby("vol_bucket", observed=True).agg(
            n=("spread", "size"), med_vol=("vol", "median"),
            med_spread=("spread", "median"),
            med_spread_rel=("spread_rel", "median"),
            med_bid_usd=("bid_usd", "median"),
            rewards_frac=("rewards", "mean")).reset_index()
        agg["share_of_queue_with_50"] = 50.0 / agg.med_bid_usd
        print(agg.to_string(index=False, float_format=fmt))
        sb.to_csv(PROC / "book_survey.csv", index=False)
        print(f"\n  wrote {PROC/'book_survey.csv'}")


if __name__ == "__main__":
    main()
