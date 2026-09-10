"""Lead-lag between markets on the SAME game: moneyline vs spread vs NRFI.

Why this is a better question than game -> futures
--------------------------------------------------
The earlier futures test failed on granularity: one regular-season game moves
a championship price by about a fifth of a tick, so there is nothing to trade
however fast you are. Same-game markets do not have that problem. A run
scored moves the moneyline, the run line, the total and the first-five
markets all at once, by amounts that are *large* relative to the tick -- and
they are mechanically linked, so any lag between them is a real dislocation
rather than a difference of opinion.

The measurement is a cross-correlation of mid returns at lags of tens to
thousands of milliseconds. A peak at a non-zero lag says one market leads;
a peak at zero says they move together and there is nothing to capture.

Because ties within a millisecond are common in this feed, everything is
resampled to a fixed grid before correlating, and the grid step is a
parameter -- a lead that only exists below the resample step is invisible.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ADV = os.path.join(BASE, "data", "adverse")

# Only the YES leg of each market is used; the NO leg is its exact mirror.
PAIRS = [
    ("moneyline", "spread"),
    ("moneyline", "total"),
    ("moneyline", "first_inning_run"),
    ("spread", "total"),
    ("moneyline", "first_five_total"),
]


def extract_raw(path: str, out_parquet: str, progress_every=4_000_000):
    """Stream a raw books.jsonl into per-(game, market) top-of-book.

    The compacted touch tables carry only an asset index, not the event slug,
    so they cannot tell which game a market belongs to -- grouping by file
    would happily correlate one game's moneyline against another game's
    total. This pass keeps `slug` and the market's `line`, which is what makes
    the pairing meaningful.
    """
    import orjson

    rows = {c: [] for c in ("ts", "slug", "mt", "line", "bid", "ask")}
    last = {}
    n = kept = 0
    t0 = time.time()
    with open(path, "rb") as fh:
        for raw in fh:
            n += 1
            try:
                r = orjson.loads(raw)
            except Exception:
                continue
            if r.get("outcome") != "YES":       # the NO leg is a mirror
                continue
            bids, asks = r.get("bids"), r.get("asks")
            if not bids or not asks:
                continue
            b0, a0 = bids[0][0], asks[0][0]
            if a0 <= b0:
                continue
            key = (r.get("slug"), r.get("market_type"), r.get("line"))
            if last.get(key) == (b0, a0):
                continue
            last[key] = (b0, a0)
            rows["ts"].append(r["ts"]); rows["slug"].append(key[0])
            rows["mt"].append(key[1]); rows["line"].append(key[2])
            rows["bid"].append(b0); rows["ask"].append(a0)
            kept += 1
            if n % progress_every == 0:
                print(f"    {n:,} lines, {kept:,} kept, "
                      f"{n/(time.time()-t0):,.0f}/s", flush=True)
    df = pd.DataFrame(rows)
    df["slug"] = df.slug.astype("category")
    df["mt"] = df.mt.astype("category")
    df.to_parquet(out_parquet, index=False)
    print(f"    {n:,} lines -> {len(df):,} rows, "
          f"{df.slug.nunique()} games, {time.time()-t0:.0f}s", flush=True)
    return df


def mid_series(df: pd.DataFrame, step_ms: int) -> dict:
    """asset -> mid resampled onto a fixed grid, forward-filled."""
    out = {}
    for asset, g in df.groupby("asset", sort=False):
        g = g.sort_values("ts")
        if len(g) < 200:
            continue
        mid = (g.bid.to_numpy(np.float64) + g.ask.to_numpy(np.float64)) / 2.0
        s = pd.Series(mid, index=pd.to_datetime(g.ts.to_numpy(), unit="ms", utc=True))
        s = s[~s.index.duplicated(keep="last")]
        r = s.resample(f"{step_ms}ms").last().ffill()
        if r.notna().sum() > 200:
            out[asset] = r
    return out


def xcorr(a: pd.Series, b: pd.Series, max_lag_steps: int):
    """Correlate d(a) against d(b) shifted by +/- lag. Positive lag = a leads."""
    idx = a.index.intersection(b.index)
    if len(idx) < 500:
        return None
    da = a.loc[idx].diff()
    db = b.loc[idx].diff()
    if da.std() == 0 or db.std() == 0:
        return None
    rows = []
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        x = da.shift(lag)          # a moved `lag` steps earlier
        m = x.notna() & db.notna()
        if m.sum() < 200:
            continue
        xv, yv = x[m].to_numpy(), db[m].to_numpy()
        if xv.std() == 0 or yv.std() == 0:
            continue
        rows.append((lag, float(np.corrcoef(xv, yv)[0, 1]), int(m.sum())))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["lag", "xc", "n"])


def analyse(df: pd.DataFrame, step_ms=250, max_lag_steps=20):
    """Correlate market-type pairs WITHIN each game, then pool across games."""
    results = []
    for slug, gs in df.groupby("slug", observed=True):
        if len(gs) < 1000:
            continue
        # one representative series per market type: the most active line.
        # dropna=False matters -- the moneyline carries line=None, and the
        # default would silently drop every moneyline row from the grouping.
        by_type = {}
        for (mt, line), g in gs.groupby(["mt", "line"], observed=True,
                                        dropna=False):
            if mt not in by_type or len(g) > len(by_type[mt]):
                by_type[mt] = g
        series = {}
        for mt, g in by_type.items():
            g = g.sort_values("ts")
            mid = (g.bid.to_numpy(np.float64) + g.ask.to_numpy(np.float64)) / 2.0
            s = pd.Series(mid, index=pd.to_datetime(g.ts.to_numpy(), unit="ms",
                                                    utc=True))
            s = s[~s.index.duplicated(keep="last")]
            r = s.resample(f"{step_ms}ms").last().ffill()
            if r.notna().sum() > 400:
                series[mt] = r

        for ta, tb in PAIRS:
            if ta not in series or tb not in series:
                continue
            r = xcorr(series[ta], series[tb], max_lag_steps)
            if r is None:
                continue
            r["pair"] = f"{ta} -> {tb}"
            r["slug"] = slug
            results.append(r)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def jump_response(df: pd.DataFrame, step_ms=250, jump_ticks=2.0, tick=0.01,
                  pre_s=3, post_s=15):
    """Event study: when market A jumps, how and WHEN does market B move?

    Correlating sparse mid changes understates the linkage badly -- at 250ms
    most bins contain no change in either market, so the correlation is
    dominated by joint inactivity. Conditioning on jumps is both a fairer
    test and the question actually being asked: a run scores, one market
    reprices, and we want to know whether the others follow later.

    Returns the mean cumulative move of B, per offset, relative to A's jump.
    A response concentrated at positive offsets is a genuine lag.
    """
    thresh = jump_ticks * tick
    rows = []
    for slug, gs in df.groupby("slug", observed=True):
        by_type = {}
        for (mt, line), g in gs.groupby(["mt", "line"], observed=True,
                                        dropna=False):
            if mt not in by_type or len(g) > len(by_type[mt]):
                by_type[mt] = g
        series = {}
        for mt, g in by_type.items():
            g = g.sort_values("ts")
            mid = (g.bid.to_numpy(np.float64) + g.ask.to_numpy(np.float64)) / 2.0
            s = pd.Series(mid, index=pd.to_datetime(g.ts.to_numpy(), unit="ms",
                                                    utc=True))
            s = s[~s.index.duplicated(keep="last")]
            r = s.resample(f"{step_ms}ms").last().ffill()
            if r.notna().sum() > 400:
                series[mt] = r

        pre_n, post_n = int(pre_s * 1000 / step_ms), int(post_s * 1000 / step_ms)
        for ta, tb in PAIRS:
            if ta not in series or tb not in series:
                continue
            idx = series[ta].index.intersection(series[tb].index)
            if len(idx) < 500:
                continue
            A = series[ta].loc[idx].to_numpy()
            B = series[tb].loc[idx].to_numpy()
            dA = np.diff(A, prepend=A[0])
            hits = np.where(np.abs(dA) >= thresh)[0]
            hits = hits[(hits > pre_n) & (hits < len(A) - post_n - 1)]
            if hits.size < 5:
                continue
            sgn = np.sign(dA[hits])
            for off in range(-pre_n, post_n + 1):
                # B's move from the instant before A jumped, signed by A
                mv = (B[hits + off] - B[hits - 1]) * sgn
                rows.append(dict(pair=f"{ta} -> {tb}", slug=slug,
                                 offset_ms=off * step_ms,
                                 mean_move=float(mv.mean()), n=int(hits.size)))
    return pd.DataFrame(rows)


def summarize_jump(J: pd.DataFrame, tick=0.01) -> pd.DataFrame:
    if J.empty:
        return pd.DataFrame()
    g = (J.groupby(["pair", "offset_ms"], observed=True)
          .agg(move=("mean_move", "mean"), games=("n", "size"),
               events=("n", "sum")).reset_index())
    rows = []
    for pair, gg in g.groupby("pair", observed=True):
        gg = gg.sort_values("offset_ms")
        at0 = gg.loc[gg.offset_ms == 0, "move"]
        m0 = float(at0.iloc[0]) if len(at0) else np.nan
        fin = float(gg.move.iloc[-1])
        # how much of the eventual response had already happened at offset 0
        frac_immediate = (m0 / fin) if fin not in (0, np.nan) else np.nan
        pos = gg[gg.offset_ms > 0]
        half = np.nan
        if len(pos) and fin != 0:
            reach = pos[np.sign(fin) * pos.move >= 0.5 * abs(fin)]
            if len(reach):
                half = float(reach.offset_ms.iloc[0])
        rows.append(dict(pair=pair, events=int(gg.events.iloc[0]),
                         move_at_0=m0, move_final=fin,
                         move_final_ticks=fin / tick,
                         frac_immediate=frac_immediate,
                         half_response_ms=half))
    return pd.DataFrame(rows)


def summarize(R: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if R.empty:
        return pd.DataFrame()
    g = R.groupby(["pair", "lag"], observed=True).agg(
        xc=("xc", "mean"), n_games=("xc", "size")).reset_index()
    g["lag_ms"] = g.lag * step_ms
    rows = []
    for pair, gg in g.groupby("pair", observed=True):
        gg = gg.sort_values("lag")
        peak = gg.loc[gg.xc.abs().idxmax()]
        at0 = gg.loc[gg.lag == 0, "xc"]
        rows.append(dict(
            pair=pair, peak_lag_ms=int(peak.lag_ms), peak_corr=float(peak.xc),
            corr_at_lag0=float(at0.iloc[0]) if len(at0) else np.nan,
            lead_gain=float(abs(peak.xc) - abs(at0.iloc[0])) if len(at0) else np.nan,
            n_games=int(peak.n_games)))
    return pd.DataFrame(rows).sort_values("lead_gain", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="*", default=None,
                    help="books.jsonl file(s) to extract per-game top-of-book from")
    ap.add_argument("--step-ms", type=int, default=250)
    ap.add_argument("--max-lag-steps", type=int, default=20)
    a = ap.parse_args()

    os.makedirs(ADV, exist_ok=True)
    if a.raw:
        for src in a.raw:
            name = os.path.basename(src).replace(".jsonl", "").replace(" ", "_")
            dst = os.path.join(ADV, f"xm_{name}.parquet")
            if os.path.exists(dst):
                print(f"skip {name} (already extracted)")
                continue
            print(f"=== extracting {src} ===")
            extract_raw(src, dst)

    files = sorted(glob.glob(os.path.join(ADV, "xm_*.parquet")))
    if not files:
        raise SystemExit("nothing extracted; pass --raw <books.jsonl>")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"\n{len(df):,} top-of-book rows, {df.slug.nunique()} games, "
          f"market types: {sorted(df.mt.dropna().unique())}")

    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:,.4f}"
    R = analyse(df, step_ms=a.step_ms, max_lag_steps=a.max_lag_steps)
    if R.empty:
        raise SystemExit("no overlapping market pairs found")

    print("=" * 92)
    print(f"CROSS-MARKET LEAD-LAG, same game  (grid {a.step_ms}ms, "
          f"+/-{a.max_lag_steps * a.step_ms}ms)")
    print("  positive peak lag  => the FIRST market leads the second")
    print("  lead_gain          => how much correlation beats simultaneous (lag 0)")
    print("=" * 92)
    print(summarize(R, a.step_ms).to_string(index=False, float_format=fmt))

    print("\n=== full lag profile per pair (mean corr across games) ===")
    g = R.groupby(["pair", "lag"], observed=True).xc.mean().reset_index()
    g["lag_ms"] = g.lag * a.step_ms
    for pair, gg in g.groupby("pair", observed=True):
        gg = gg.sort_values("lag")
        top = gg.reindex(gg.xc.abs().sort_values(ascending=False).index).head(5)
        print(f"\n  {pair}")
        print(top[["lag_ms", "xc"]].to_string(index=False, float_format=fmt))

    R.to_parquet(os.path.join(ADV, "crossmarket_xcorr.parquet"), index=False)

    print("\n" + "=" * 92)
    print(f"JUMP EVENT STUDY -- when A moves >= 2 ticks, when does B follow?")
    print("  frac_immediate   share of B's eventual response already done at offset 0")
    print("  half_response_ms time for B to complete half its response")
    print("=" * 92)
    J = jump_response(df, step_ms=a.step_ms)
    S = summarize_jump(J)
    if S.empty:
        print("  (no pairs with enough jumps)")
    else:
        print(S.to_string(index=False, float_format=fmt))
        print("\n=== response profile (mean signed move of B, in ticks) ===")
        g = (J.groupby(["pair", "offset_ms"], observed=True).mean_move.mean()
             .reset_index())
        for pair, gg in g.groupby("pair", observed=True):
            gg = gg[gg.offset_ms.isin([-1000, -500, 0, 250, 500, 1000, 2000,
                                       5000, 10000, 15000])]
            prof = "  ".join(f"{int(r.offset_ms):>+6}ms:{r.mean_move/0.01:+6.2f}"
                             for r in gg.itertuples())
            print(f"\n  {pair}\n    {prof}")
        J.to_parquet(os.path.join(ADV, "crossmarket_jumps.parquet"), index=False)


if __name__ == "__main__":
    main()

