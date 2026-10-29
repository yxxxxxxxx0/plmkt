"""Maker bot test, $100 bankroll: quote at the touch on both venues, cancel on book collapse.

Per venue, per outcome: rest a bid at the best bid and an ask at the best ask (SIZE shares each),
re-pegging to the touch whenever it moves (re-peg = back of the queue). Fills come from the
trade tape: a taker print at our price eats the queue ahead of us first; a print through our
price fills us outright. Inventory is capped at CAP shares either way (short = holding the other
side, collateral 1-p). Every position is marked at the game's last mid (close to settlement).

Cancel switch (the joint-collapse detector, causal): total top-5 depth below 0.5x its own median
over the previous 60 s -> 5 s. Modes: none / own venue / either venue / both venues. After a
trigger we stay out PAUSE seconds. Cancel latency: Polymarket football takers are delayed 1 s,
so a cancel at the trigger second beats every taker print in the next second; elsewhere we
assume CANCEL_MS of exposure after the trigger.

Data limits: the book is a 1 s snapshot (top 5 levels) and the queue model is approximate, so
treat fill counts as optimistic for joining the queue and pessimistic for re-pegs.

    python research/venue_jumps/maker_cancel.py      (from polymarket_orderbook/)
"""
import json
import os

import numpy as np
import pandas as pd

VIEW = "data/view"
SIZE, CAP = 5, 20                 # shares per quote (Polymarket minimum is 5), max |inventory|
PAUSE, CANCEL_MS = 30, 500
KS_MAKER_FEE = 0.0175             # assumption: Kalshi maker fee rate x p(1-p); Polymarket makers pay 0
RATIO = 0.5


def ratio(D):
    out = np.full(len(D), np.nan)
    for i in range(60, len(D)):
        w = D[i - 60:i - 4]
        w = w[~np.isnan(w)]
        if len(w) >= 28 and np.median(w) > 0 and not np.isnan(D[i]):
            out[i] = D[i] / np.median(w)
    return out


def simulate(t, book, trades, collapsed, delay_protect, final_mid, venue):
    """One venue, one outcome. book: dict of arrays bid,ask,bid_sz,ask_sz; trades: (ms, price, size, side)."""
    n = len(t)
    inv, cash, fills, peak = 0, 0.0, [], 0.0
    bid = ask = None                          # [price, queue_ahead, remaining]
    paused_until = -1
    tr_t = np.array([x[0] for x in trades]) if trades else np.zeros(0)
    for i in range(80, n - 1):
        b, a = book["bid"][i], book["ask"][i]
        if collapsed[i]:
            paused_until = i + PAUSE
        live = i > paused_until and not np.isnan(b) and not np.isnan(a) and 0.03 <= b and a <= 0.97 and a > b
        if not live:
            bid = ask = None
        else:
            if inv < CAP and (bid is None or bid[0] != b):
                bid = [b, book["bid_sz"][i], SIZE]
            if inv > -CAP and (ask is None or ask[0] != a):
                ask = [a, book["ask_sz"][i], SIZE]
            if inv >= CAP:
                bid = None
            if inv <= -CAP:
                ask = None
        # trades until the next snapshot; a trigger at i+1 cancels from t[i+1] (+ latency unless protected)
        lo, hi = np.searchsorted(tr_t, [t[i], t[i + 1]], side="right")
        stop = t[i + 1] if (i + 1 <= n - 1 and collapsed[i + 1] and delay_protect) else None
        for ms, p, sz, side in trades[lo:hi]:
            if stop is not None and ms > stop:
                break
            if side == "sell" and bid is not None and p <= bid[0]:
                q = sz if p < bid[0] else sz - bid[1]
                if p == bid[0]:
                    bid[1] = max(0.0, bid[1] - sz)
                take = min(bid[2], max(0.0, q))
                if take > 0:
                    inv += take; cash -= take * bid[0]; bid[2] -= take
                    fills.append((i, +1, bid[0], take))
                    if bid[2] <= 0:
                        bid = None
            elif side == "buy" and ask is not None and p >= ask[0]:
                q = sz if p > ask[0] else sz - ask[1]
                if p == ask[0]:
                    ask[1] = max(0.0, ask[1] - sz)
                take = min(ask[2], max(0.0, q))
                if take > 0:
                    inv -= take; cash += take * ask[0]; ask[2] -= take
                    fills.append((i, -1, ask[0], take))
                    if ask[2] <= 0:
                        ask = None
        mid = (b + a) / 2 if not (np.isnan(b) or np.isnan(a)) else 0.5
        peak = max(peak, inv * mid if inv > 0 else -inv * (1 - mid))
    fee = sum(KS_MAKER_FEE * p * (1 - p) * q for _, _, p, q in fills) if venue == "ks" else 0.0
    return cash + inv * final_mid - fee, fills, peak


def main():
    idx = json.load(open(f"{VIEW}/index.json"))
    modes = ("none", "own", "either", "both")
    res = {m: [] for m in modes}
    mark = {m: [] for m in modes}
    for tag, sess in idx.items():
        for gi in sess["games"]:
            slug = gi["slug"]
            d = pd.read_csv(f"{VIEW}/{tag}/aligned/{slug}.csv.gz")
            g = json.load(open(f"{VIEW}/{gi['file']}", encoding="utf-8"))
            t = d["t_ms"].to_numpy(np.int64)
            keys = sorted({c.split("_")[0] for c in d.columns if c.endswith("_pm_mid")} & set(g["series"]))
            col = lambda c: d[c].to_numpy(float) if c in d else np.full(len(t), np.nan)
            tot = {}
            for v in ("pm", "ks"):
                s = sum(np.nan_to_num(col(f"{k}_{v}_{sd}{l}_sz")) for k in keys for sd in ("bid", "ask") for l in range(1, 6))
                s = s.astype(float); s[s <= 0] = np.nan
                tot[v] = ratio(s) < RATIO
            for v in ("pm", "ks"):
                other = "ks" if v == "pm" else "pm"
                trig = {"none": np.zeros(len(t), bool), "own": tot[v], "either": tot[v] | tot[other],
                        "both": tot[v] & (np.convolve(tot[other], np.ones(5))[:len(t)] > 0)}
                protect = v == "pm" and not slug.startswith("mlb")
                tape = g[f"{v}_trades"]
                for k in keys:
                    book = {x: pd.Series(col(f"{k}_{v}_{x}1")).ffill().to_numpy() for x in ("bid", "ask")}
                    book["bid_sz"], book["ask_sz"] = np.nan_to_num(col(f"{k}_{v}_bid1_sz")), np.nan_to_num(col(f"{k}_{v}_ask1_sz"))
                    m = pd.Series(col(f"{k}_{v}_mid")).ffill().dropna()
                    if m.empty:
                        continue
                    final = float(m.iloc[-1])
                    mids = pd.Series(col(f"{k}_{v}_mid")).ffill().to_numpy()
                    trades = sorted((int(x[0]), float(x[2]), float(x[3]), x[4]) for x in tape if x[1] == k)
                    for mode in modes:
                        pnl, fills, peak = simulate(t, book, trades, trig[mode], protect, final, v)
                        res[mode].append(dict(game=slug, venue=v, k=k, pnl=pnl, fills=len(fills),
                                              shares=sum(f[3] for f in fills), peak=peak))
                        for i, sg, p, q in fills:   # 30 s markout: did the mid move against the fill?
                            j = min(i + 30, len(t) - 1)
                            mark[mode].append((v, q * sg * (mids[j] - p), q * abs(p - mids[i])))
            print(f"  {slug} done", flush=True)
    print(f"\nSize {SIZE} shares per quote, inventory cap {CAP}, pause {PAUSE}s after a collapse trigger")
    print(f"{'cancel switch':14s} {'venue':5s} {'fills':>6s} {'shares':>7s} {'P&L $':>8s} {'$/share':>8s} "
          f"{'spread got':>10s} {'30s edge':>11s} {'peak $ in use':>13s}")
    for mode in modes:
        R = pd.DataFrame(res[mode])
        M = pd.DataFrame(mark[mode], columns=["v", "mk", "hs"])
        for v in ("pm", "ks"):
            r = R[R.venue == v]
            mm = M[M.v == v]
            sh = r.shares.sum()
            peak_game = r.groupby("game").peak.sum().max()
            print(f"{mode:14s} {v:5s} {int(r.fills.sum()):6d} {sh:7.0f} {r.pnl.sum():+8.2f} {r.pnl.sum() / max(sh, 1):+8.4f} "
                  f"{mm.hs.sum() / max(sh, 1):+10.4f} {mm.mk.sum() / max(sh, 1):+11.4f} {peak_game:13.2f}")
    best = pd.DataFrame(res["either"])
    print("\nper game, cancel on either venue (both venues summed, $):")
    print(best.groupby("game").pnl.sum().round(2).to_string())


if __name__ == "__main__":
    main()
