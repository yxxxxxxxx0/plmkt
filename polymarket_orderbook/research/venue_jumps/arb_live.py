"""Continuous arbitrage watcher for every Polymarket market with no taker delay.

Universe (refreshed every REFRESH_MIN): every open event from Gamma, paged with /events/keyset
(plain offsets stop at 2,100 events). A market counts as delay-free when Gamma's `secondsDelay`
is unset or 0. Arbitrage checked, all on delay-free markets only:

  buy all YES   negRisk event, one YES per outcome; pays $1 if exactly one outcome wins.
                Safe only if the outcomes are exhaustive: READ THE RULES before trading.
  buy all NO    negRisk event, one NO per outcome; pays n-1 if one wins (n if none), always safe.
  nested        team champion NO + same team conference YES; pays >= $1 whatever happens.

Books come from the CLOB market websocket (full snapshot, then price_change deltas), so every
change is seen. A group is evaluated only once every leg has had a snapshot. A gap OPENS when the
locked-in profit (after a taker fee of 0.05*p(1-p) a share, walking the ask ladders, cost capped at
$100) exceeds MIN_PROFIT, and CLOSES when profit falls to 0. Each opening is re-checked against
the REST /books endpoint, so a gap caused by a missed websocket update is marked unconfirmed.

Output:
  data/arb/live_gaps.jsonl   one row per closed gap: start, end, seconds open, profit at open and at
                             best, sets and cost, confirmed on REST or not, why it closed
  data/arb/live_open.json    gaps open right now (rewritten every status tick)
  stdout                     OPEN / CLOSE lines and a status line every STATUS_S

    python research/venue_jumps/arb_live.py            run until Ctrl+C
    python research/venue_jumps/arb_live.py --report   summary of live_gaps.jsonl
No orders are sent.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import datetime as dt

import websockets

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "data", "arb")
GAPS = os.path.join(OUT, "live_gaps.jsonl")
OPEN = os.path.join(OUT, "live_open.json")
WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
KEYSET = "https://gamma-api.polymarket.com/events/keyset?closed=false&active=true&limit=100"
NESTED = [("Pro Football: 2027 Champion", ["Pro Football: 2027 NFC Champion", "Pro Football: 2027 AFC Champion"]),
          ("NBA: 2027 Champion", ["NBA: 2027 Western Conference Champion", "NBA: 2027 Eastern Conference Champion"])]
BUDGET, MIN_PROFIT = 100.0, 0.01
CHUNK, REFRESH_MIN, STATUS_S = 400, 30, 60
fee = lambda p: 0.05 * p * (1 - p)


def per_year(profit, cost, end_date):
    """Locked-in return per year until the event's end date (None if no date)."""
    try:
        yrs = (dt.date.fromisoformat(end_date) - dt.date.today()).days / 365.25
    except (TypeError, ValueError):
        return None
    return round(profit / cost / max(yrs, 1 / 365.25), 4) if cost > 0 else None


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def curl(url, body=None):
    cmd = ["curl", "-s", "--max-time", "60", url]
    if body is not None:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(body)]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
    try:
        return json.loads(out or "null")
    except json.JSONDecodeError:
        return None


def all_events():
    out, cur = {}, None
    for _ in range(500):
        d = curl(KEYSET + (f"&after_cursor={cur}" if cur else ""))
        if not isinstance(d, dict) or not d.get("events"):
            break
        out.update({e["id"]: e for e in d["events"]})
        cur = d.get("next_cursor")
        if not cur:
            break
    return list(out.values())


def toks(m):
    t = json.loads(m.get("clobTokenIds") or "[]")
    return (t[0], t[1]) if len(t) == 2 else (None, None)


delay_free = lambda m: not (m.get("secondsDelay") or 0)


def build_groups(E):
    """group id -> dict(kind, title, slug, end, legs=[token...], payout)."""
    G, n_free, n_all = {}, 0, 0
    for e in E:
        ms = [m for m in e.get("markets", []) if not m.get("closed")]
        n_all += len(ms); n_free += sum(map(delay_free, ms))
        if not e.get("negRisk") or len(ms) < 2 or not all(m.get("acceptingOrders") and delay_free(m) for m in ms):
            continue
        T = [toks(m) for m in ms]
        if any(None in t for t in T):
            continue
        base = dict(title=e.get("title", ""), slug=e.get("slug", ""), end=(e.get("endDate") or "")[:10], n=len(ms),
                    has_other=any((m.get("groupItemTitle") or "").strip().lower().startswith("other") or m.get("negRiskOther") for m in ms))
        G[f"{e['id']}:YES"] = dict(base, kind="buy all YES", legs=[t[0] for t in T], payout=1.0)
        G[f"{e['id']}:NO"] = dict(base, kind="buy all NO", legs=[t[1] for t in T], payout=len(ms) - 1.0)
    by_title = {(e.get("title") or "").strip(): e for e in E}
    for champ, confs in NESTED:
        C = {(m.get("groupItemTitle") or "").strip(): m for m in by_title.get(champ, {}).get("markets", [])
             if not m.get("closed") and delay_free(m)}
        for conf in confs:
            for m in by_title.get(conf, {}).get("markets", []):
                t = (m.get("groupItemTitle") or "").strip()
                if t in C and not m.get("closed") and delay_free(m):
                    a, b = toks(C[t])[1], toks(m)[0]
                    if a and b:
                        G[f"nested:{conf}:{t}"] = dict(kind="nested", title=f"{t}: {champ} NO + {conf} YES", slug="", end="",
                                                       n=2, has_other=False, legs=[a, b], payout=1.0)
    return G, n_free, n_all


def fill(ladders, payout):
    """Buy one share of every leg per set, walking each ask ladder. Returns (sets, cost, profit, gross)."""
    legs = [list(l) for l in ladders]
    if not legs or any(not l for l in legs):
        return 0.0, 0.0, 0.0, 0.0
    sets = cost = profit = gross = 0.0
    while all(legs):
        prices = [l[0][0] for l in legs]
        unit = sum(p + fee(p) for p in prices)
        if payout - unit <= 0:
            break
        q = min(min(l[0][1] for l in legs), (BUDGET - cost) / unit)
        if q <= 1e-9:
            break
        sets += q; cost += q * unit; profit += q * (payout - unit); gross += q * (payout - sum(prices))
        for l in legs:
            l[0] = (l[0][0], l[0][1] - q)
            if l[0][1] <= 1e-9:
                l.pop(0)
    return sets, cost, profit, gross


class Watcher:
    def __init__(self):
        self.asks, self.ready = {}, set()        # token -> {price: size}; tokens with a snapshot
        self.groups, self.by_tok = {}, {}
        self.subscribed, self.conns = set(), []
        self.open = {}                           # gid -> gap state
        self.msgs = 0

    def set_universe(self, G):
        self.groups = G
        self.by_tok = {}
        for gid, g in G.items():
            for t in g["legs"]:
                self.by_tok.setdefault(t, []).append(gid)
        for gid in [g for g in self.open if g not in G]:
            self.close(gid, "left universe")

    def ladder(self, t):
        return sorted(self.asks.get(t, {}).items())

    def evaluate(self, gid):
        g = self.groups.get(gid)
        if not g or not all(t in self.ready for t in g["legs"]):
            return
        sets, cost, profit, gross = fill([self.ladder(t) for t in g["legs"]], g["payout"])
        now = time.time(); o = self.open.get(gid)
        if o is None and profit > MIN_PROFIT:
            o = self.open[gid] = dict(gid=gid, kind=g["kind"], title=g["title"], slug=g["slug"], end_date=g["end"], n=g["n"],
                                      has_other=g["has_other"], start=now, profit_open=round(profit, 4), sets_open=round(sets, 2),
                                      cost_open=round(cost, 2), gross_open=round(gross, 4), best=round(profit, 4), confirmed=None,
                                      per_year=per_year(profit, cost, g["end"]))
            py = o["per_year"]
            log(f"OPEN   {g['kind']:11s} ${profit:6.2f} on ${cost:6.2f} ({sets:.1f} sets)  "
                f"{'%6.1f%%/yr' % (100 * py) if py is not None else '   ?%/yr'} to {g['end'] or '?'}  {g['title'][:60]}")
            asyncio.get_running_loop().run_in_executor(None, self.confirm, gid, list(g["legs"]), g["payout"])
        elif o is not None:
            if profit <= 0:
                self.close(gid, "taken or repriced")
            else:
                o["best"] = round(max(o["best"], profit), 4)

    def confirm(self, gid, legs, payout):
        books = curl("https://clob.polymarket.com/books", [{"token_id": t} for t in legs]) or []
        L = {b["asset_id"]: sorted((float(x["price"]), float(x["size"])) for x in b.get("asks", [])) for b in books}
        _, _, p, _ = fill([L.get(t, []) for t in legs], payout)
        if gid in self.open:
            self.open[gid]["confirmed"] = bool(p > 0)
            self.open[gid]["profit_rest"] = round(p, 4)

    def close(self, gid, why):
        o = self.open.pop(gid, None)
        if not o:
            return
        o["end"] = time.time(); o["seconds_open"] = round(o["end"] - o["start"], 2); o["closed_by"] = why
        os.makedirs(OUT, exist_ok=True)
        with open(GAPS, "a", encoding="utf-8") as f:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
        log(f"CLOSE  {o['kind']:11s} open {o['seconds_open']:8.1f}s  best ${o['best']:.2f}  confirmed={o['confirmed']}  "
            f"({why})  {o['title'][:60]}")

    def on_msg(self, m):
        et = m.get("event_type")
        touched = set()
        if et == "book":
            t = m.get("asset_id")
            if t in self.by_tok:
                self.asks[t] = {float(l["price"]): float(l["size"]) for l in m.get("asks", []) if float(l["size"]) > 0}
                self.ready.add(t); touched.add(t)
        elif et == "price_change":
            for ch in m.get("price_changes") or m.get("changes") or ([m] if "price" in m else []):
                t = ch.get("asset_id") or m.get("asset_id")
                if t not in self.by_tok or ch.get("side") not in ("SELL", "ASK"):
                    continue
                p, s = float(ch["price"]), float(ch["size"])
                lv = self.asks.setdefault(t, {})
                lv.pop(p, None) if s <= 0 else lv.__setitem__(p, s)
                touched.add(t)
        for t in touched:
            for gid in self.by_tok.get(t, []):
                self.evaluate(gid)

    async def conn(self, ids):
        backoff = 1
        while True:
            try:
                async with websockets.connect(WS, ping_interval=20, ping_timeout=60, open_timeout=45,
                                              max_queue=None, max_size=None, close_timeout=5) as ws:
                    await ws.send(json.dumps({"assets_ids": ids, "type": "market"}))
                    backoff = 1
                    while True:
                        raw = await ws.recv()
                        try:
                            P = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        for m in (P if isinstance(P, list) else [P]):
                            self.msgs += 1
                            if isinstance(m, dict):
                                self.on_msg(m)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # books on this socket are stale until the resubscribe snapshot: drop them, close their gaps
                for t in ids:
                    self.ready.discard(t)
                for gid in [g for g in self.open if any(t in ids for t in self.groups.get(g, {}).get("legs", []))]:
                    self.close(gid, "feed dropped")
                log(f"socket of {len(ids)} assets dropped ({type(e).__name__}: {e}); retry in {backoff}s")
                await asyncio.sleep(backoff); backoff = min(backoff * 2, 60)

    async def refresh_loop(self):
        while True:
            E = await asyncio.get_running_loop().run_in_executor(None, all_events)
            if E:
                G, nf, na = build_groups(E)
                self.set_universe(G)
                new = sorted(set(self.by_tok) - self.subscribed)
                for i in range(0, len(new), CHUNK):
                    self.conns.append(asyncio.create_task(self.conn(new[i:i + CHUNK])))
                self.subscribed |= set(new)
                log(f"universe: {len(E)} events, {nf}/{na} open markets delay-free, {len(G)} arbitrage groups, "
                    f"{len(self.by_tok)} tokens ({len(new)} new), {len(self.conns)} sockets")
            await asyncio.sleep(REFRESH_MIN * 60)

    async def status_loop(self):
        while True:
            await asyncio.sleep(STATUS_S)
            rdy = sum(all(t in self.ready for t in g["legs"]) for g in self.groups.values())
            log(f"status: {rdy}/{len(self.groups)} groups live, {self.msgs} msgs, {len(self.open)} gaps open")
            os.makedirs(OUT, exist_ok=True)
            with open(OPEN, "w", encoding="utf-8") as f:
                json.dump(list(self.open.values()), f, ensure_ascii=False, indent=1)

    async def run(self):
        await asyncio.gather(self.refresh_loop(), self.status_loop())


def report():
    rows = [json.loads(l) for l in open(GAPS, encoding="utf-8")] if os.path.exists(GAPS) else []
    real = [r for r in rows if r["closed_by"] == "taken or repriced"]
    print(f"{len(rows)} gaps closed ({len(real)} by the market, {len(rows) - len(real)} by a feed drop or universe change)")
    if not real:
        return
    import statistics as st
    d = sorted(r["seconds_open"] for r in real)
    print(f"seconds open: median {st.median(d):.1f}, 90th pct {d[int(.9 * (len(d) - 1))]:.1f}, max {d[-1]:.1f}")
    print(f"confirmed on REST: {sum(r['confirmed'] is True for r in real)}/{len(real)}\n")
    for r in sorted(real, key=lambda r: -r["best"])[:30]:
        py = r.get("per_year")
        print(f"  {'%6.1f%%/yr' % (100 * py) if py is not None else '   ?%/yr'}  ${r['best']:6.2f} best (${r['profit_open']:.2f} at open, cost ${r['cost_open']:.2f})  open {r['seconds_open']:8.1f}s  "
              f"conf={str(r['confirmed']):5s} {r['kind']:11s} ends {r['end_date'] or '?':10s} {r['title'][:55]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    if ap.parse_args().report:
        report()
    else:
        try:
            asyncio.run(Watcher().run())
        except KeyboardInterrupt:
            pass
