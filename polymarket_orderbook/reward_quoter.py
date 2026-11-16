"""Paper quoter for Polymarket liquidity rewards. Nothing is sent to the exchange.

Question it answers: does a small two-sided quote earn more in rewards than it loses to the
traders who fill it? The 2026-11-10 snapshot screen said a ~$19 quote earns a median ~1%/day
of its capital in rewards; this measures the other half, the fills, live.

Selection (once, saved in the state file so a restart resumes the same markets):
  every market in /rewards/markets/current with min size <= --max-min-size, no taker delay,
  not sports, 9-120 days to its end date, traded in the last 24 h, mid 15-85c, another maker
  already scoring (an empty book is usually a new or decided market), book spread within 2 x the
  reward band (a wider book's mid is no fair value: on 2026-11-10 other makers re-centred 10-25c
  away from it and ran over every quote), then the --markets best
  by estimated reward per dollar that fit in the bankroll.

Quoting, in YES space (an ask on YES is a bid on NO): size = max(min size, 20) shares, half way
out in the reward band (score 0.25 of the best), re-priced when the target moves a tick, pulled
for 60 s when the mid moves 5c within 30 s. Inventory cap 2 x size per side, with quotes leaned
against the position (sigbot/pmmm.py's skew) so it is worked off by the opposite quote, not held. Fills use the queue
model from arb_scan.py (join the back of the level; prints at our price eat the queue first; a
print or the opposite book through our price fills us). Rewards are scored once a minute with the
docs' formula: S = ((v - s) / v)^2 per share, min of the two sides (single side / 3 when the mid
is 10-90c), pro rata against everyone else resting in the band.

Writes data/rewards/minutes.csv (one row per market per minute), data/rewards/fills.csv and
data/rewards/state.json.

    python reward_quoter.py                 run (resumes data/rewards/state.json if present)
    python reward_quoter.py --fresh         pick new markets (keeps the old logs)
    python reward_quoter.py --report        rewards vs fill P&L so far, with markouts
    python reward_quoter.py --survey        test many market types side by side (data/rewards_survey): the best
                                            --per-cat markets of every fee type, sports and short-dated ones
                                            included, each with its own budget; reselects every --hours
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import datetime as dt
import json
import math
import os
import random
import time

import requests
import websockets

from arb_scan import Account, SimOrder, on_book, on_trade, tick_ceil, tick_floor
from live_recorder import apply_change, apply_snapshot, new_book
from polymarket_fees import TAKER_FEE_RATE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rewards")
STATE, MIN_CSV, FILL_CSV = (os.path.join(OUT, f) for f in ("state.json", "minutes.csv", "fills.csv"))


Q15_CSV = os.path.join(OUT, "q15.csv")


def set_out(d):
    global OUT, STATE, MIN_CSV, FILL_CSV, Q15_CSV
    OUT = d
    STATE, MIN_CSV, FILL_CSV, Q15_CSV = (os.path.join(OUT, f) for f in ("state.json", "minutes.csv", "fills.csv", "q15.csv"))


def category(m):
    """Market type for the survey: the fee schedule without its version, plus whether takers are delayed."""
    f = (m.get("feeType") or "none").replace("_fees", "")
    f = "sports" if f.startswith("sports") else f.split("_v")[0]
    return f + ("+delay" if m.get("secondsDelay") else "")


def budget(c):
    """Cash a market can tie up: both quotes resting plus a full inventory cap (2 x size) at the worse price."""
    return c["size"] * (1 - c["v"]) + 2 * c["size"] * max(c["mid"], 1 - c["mid"])


WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CLOB, GAMMA = "https://clob.polymarket.com", "https://gamma-api.polymarket.com"
LATENCY, PULL_MOVE, PULL_S = 0.25, 0.05, 60   # LATENCY: s for an order or a cancel to take effect (--latency)
FEED_SILENT = 900       # s without a message: resubscribe. A quiet market can be silent for minutes; the 10 s ping
                        # catches a dead link, this only a connection that answers pings but has stopped sending
# Per-signer limits, lowest (Standard) tier, docs.polymarket.com/api-reference/trading-rate-limits (read 2026-11-14):
# orders refill 40/s with a burst of 60, cancels 80/s burst 120 and may run negative. --all splits them over --shards.
ORDER_RATE, ORDER_BURST = 40, 60
# Risk controls ported from sig-predictions-cup sigbot/live/dynrisk.py. They gate *opening* quotes only; a quote
# that closes inventory is always allowed.
VOL_MAX = 0.01          # EWMA sd of 1-min mid changes above this: no new positions (dynrisk vol scaling)
VOL_HALFLIFE = 10       # samples (minutes)
TREND_TAU = 600         # s, slow fair; mid this far from it (in reward bands x TREND_BANDS) = trending, no openings
TREND_BANDS = 0.5
ADD_STEP = 0.01         # no stacking: a further opening fill on a side needs a price this much better
MAX_HOLD = 6 * 3600     # time stop: inventory older than this is sold at the best price
# Where the quotes sit. The defaults are the tested behaviour; the others are for replays of the recorded feed.
QUOTE_FRAC = 0.5        # distance of a quote from the (skewed) mid, as a share of the reward band v
QUOTE_FRONT = 0         # 1: an opening quote joins the best price on its side when that is closer; 2: one tick better
FROZEN_EXIT = True      # a closing quote is never worse than the fair at entry
EXIT_FRONT = 0          # as QUOTE_FRONT, for the quote that closes a position
MARKOUT_S = 600         # a fill "won" if the mid this long after it is on our side of the fill price
HIT_WINDOW, HIT_MIN, HIT_RATE, PAUSE_S = 6, 4, 0.5, 86400   # pause a market whose fills keep losing
# Swapping markets (the default run): every SWAP_CHECK s the screen is rerun, and a flat market is replaced when it
# was paused, its pool ended, it was quoted under MIN_QUOTED of the last SWAP_WINDOW s or earned under MIN_YIELD a day
# of its capital there, it no longer passes the screen, or the screen rates a free market SWAP_GAIN x higher. A
# market swapped out is not picked again for EXCLUDE_S. A market has also levelled off, and is swapped when a free one
# exists, once it has been held MIN_STAY s and its reward over the last LEVEL_S s is under LEVEL_FRAC of its best LEVEL_S.
SWAP_WINDOW, SWAP_CHECK = 2 * 3600, 1800
LEVEL_S, LEVEL_FRAC, MIN_STAY = 1800, 0.5, 3600
MIN_QUOTED, MIN_YIELD, SWAP_GAIN, EXCLUDE_S = 0.5, 0.01, 2.0, 86400


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def get(url, **kw):
    for i in range(4):
        try:
            r = requests.get(url, params=kw, timeout=60)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 + 3 * i)
    return None


def score(levels, mid, v, minsz):
    """Reward points of resting (price, size) levels within v of the mid."""
    return sum(((v - abs(p - mid)) / v) ** 2 * q for p, q in levels if abs(p - mid) < v and q >= minsz)


def q_min(q_bid, q_ask, mid):
    if 0.10 <= mid <= 0.90:
        return max(min(q_bid, q_ask), max(q_bid, q_ask) / 3)
    return min(q_bid, q_ask)


def books(tokens):
    out = {}
    for i in range(0, len(tokens), 100):
        try:
            r = requests.post(f"{CLOB}/books", json=[{"token_id": t} for t in tokens[i:i + 100]], timeout=60).json()
            out.update({b["asset_id"]: b for b in r})
        except (requests.RequestException, ValueError, KeyError):
            pass
    return out


def candidates(a):
    """Every market passing the filters, best estimated reward per dollar first."""
    R, cur = [], ""
    while cur != "LTE=":
        d = get(f"{CLOB}/rewards/markets/current", next_cursor=cur) or {}
        R += d.get("data") or []
        cur = d.get("next_cursor") or "LTE="
    R = {r["condition_id"]: r for r in R if 0 < r["rewards_min_size"] <= a.max_min_size and r["rewards_max_spread"] > 0
         and r["total_daily_rate"] > 0}
    log(f"{len(R)} reward markets with min size <= {a.max_min_size}")
    ids, M = list(R), []
    for i in range(0, len(ids), 50):
        M += get(f"{GAMMA}/markets", condition_ids=ids[i:i + 50], limit=50) or []
    now = dt.datetime.now(dt.timezone.utc)
    C = []
    for m in M:
        try:
            days = (dt.datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")) - now).days
            yes, no = json.loads(m["clobTokenIds"])
        except (KeyError, ValueError, TypeError):
            continue
        if not a.survey and ((m.get("secondsDelay") or 0) or "sports" in (m.get("feeType") or "")):
            continue
        if not m.get("acceptingOrders") or not (1 if a.survey else 9) <= days <= 120 \
                or float(m.get("volume24hrClob") or 0) <= 0:
            continue
        r = R[m["conditionId"]]
        C.append(dict(cond=m["conditionId"], q=m["question"][:80], yes=yes, no=no, rate=r["total_daily_rate"],
                      v=r["rewards_max_spread"] / 100, minsz=r["rewards_min_size"], tick=float(m.get("orderPriceMinTickSize") or 0.01),
                      end=m["endDate"][:10], fee=m.get("feeType") or "", cat=category(m),
                      start=m.get("gameStartTime")))
    if a.shards > 1:                                   # --all runs one process per shard of the markets
        C = [c for c in C if int(c["cond"], 16) % a.shards == a.shard]
    log(f"{len(C)} pass the market filters; reading their books")
    B = books([c["yes"] for c in C])
    for c in C:
        b = B.get(c["yes"]) or {}
        bid = [(float(x["price"]), float(x["size"])) for x in b.get("bids", [])]
        ask = [(float(x["price"]), float(x["size"])) for x in b.get("asks", [])]
        c["est"] = 0.0
        if not bid or not ask:
            continue
        mid = (max(p for p, _ in bid) + min(p for p, _ in ask)) / 2
        others = q_min(score(bid, mid, c["v"], c["minsz"]), score(ask, mid, c["v"], c["minsz"]), mid)
        n = max(c["minsz"], 20)
        c.update(mid=mid, size=n, capital=n * (1 - c["v"]), others=others)   # both sides rest: n*(mid-v/2) + n*(1-mid-v/2)
        spread = min(p for p, _ in ask) - max(p for p, _ in bid)
        if 0.15 <= mid <= 0.85 and others > 0 and spread <= 2 * c["v"]:
            mine = 0.25 * n
            c["est"] = c["rate"] * mine / (mine + others) / c["capital"]
    C.sort(key=lambda c: -c["est"])
    return C


def select(a, keep=(), exclude=(), C=None):
    """The markets held (keep), topped up with the best of the rest (C, or a fresh screen) that fit the bankroll."""
    C = candidates(a) if C is None else C
    if a.survey:
        return survey_pick(C, a, keep)
    have = {c["cond"] for c in keep}
    pick, spent = list(keep), sum(c["capital"] for c in keep)
    for c in C:
        if c["est"] <= 0 or len(pick) >= a.markets:
            break
        if c["cond"] not in have and c["cond"] not in exclude and spent + c["capital"] <= a.bankroll:
            pick.append(c); spent += c["capital"]
    for c in pick[len(keep):]:
        log(f"  {100 * c['est']:5.2f}%/day est  ${c['rate']:>6.1f}/day pool  mid {c['mid']:.2f}  v {100 * c['v']:.1f}c  "
            f"{c['size']} sh  ends {c['end']}  {c['q']}")
    return pick


def reselect(a, q):
    """After a run stopped to swap: drop the markets q.swap names from the state file and fill their places."""
    st, now = json.load(open(STATE)), time.time()
    ex = {c: x for c, x in st.get("excluded", {}).items() if now - x < EXCLUDE_S}
    seen = st.setdefault("seen", {})
    for m in st["markets"]:
        if m["cond"] in q.swap:
            log(f"swapping out {m['q'][:50]}: {q.swap[m['cond']]}")
            ex[m["cond"]], seen[m["cond"]] = now, m["q"][:50]
    keep = [m for m in st["markets"] if m["cond"] not in q.swap]
    st.update(excluded=ex, markets=select(a, keep, set(ex), q.cands))
    if len(st["markets"]) == len(keep):
        log("no replacement fits the bankroll; quoting what is left")
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def survey_pick(C, a, keep=()):
    """The best a.per_cat markets of each type, after the markets still held (keep), each with its own budget."""
    pick, n = list(keep), collections.Counter(c["cat"] for c in keep)
    have = {c["cond"] for c in keep}
    for c in C:
        if c["est"] > 0 and c["cond"] not in have and (a.per_cat <= 0 or n[c["cat"]] < a.per_cat):
            c["budget"] = round(budget(c), 2)
            pick.append(c); n[c["cat"]] += 1
    for c in sorted(pick, key=lambda c: c["cat"]) if a.per_cat > 0 else ():
        log(f"  {c['cat']:14s} budget ${c.get('budget', 0):7.2f}  {100 * c['est']:5.2f}%/day est  ${c['rate']:>6.1f}/day pool  "
            f"{c['size']} sh  ends {c['end']}  {c['q']}")
    log(f"{len(pick)} markets in {len(n)} types, total budget ${sum(c.get('budget', 0) for c in pick):.0f}")
    return pick


class Quoter:
    def __init__(self, a, st):
        self.a, self.mk = a, {m["yes"]: m for m in st["markets"]}
        self.acct = Account(cash=st["cash"], net=st["net"])
        self.reward = st["reward"]
        self.book = {t: new_book() for t in self.mk}
        self.orders = {t: {} for t in self.mk}        # token -> side -> SimOrder
        self.hist = {t: [] for t in self.mk}          # (time, mid) for the pull rule
        self.pulled = {t: 0.0 for t in self.mk}
        self.n_fill = 0
        self.var, self.vn, self.last = {}, {}, {}     # token -> EWMA variance, samples, (time, mid) of last sample
        self.slow = {}                                # token -> (time, slow EWMA mid)
        self.entry = st.get("entry", {})              # token -> {side, fair, price, opened}
        self.outcomes = st.get("outcomes", {})        # token -> last fills: 1 won / 0 lost at the markout
        self.paused = st.get("paused", {})            # token -> [until, reason]
        self.pending = []                             # (due, token, side, price) markouts to score
        self.bulk = a.all                             # --all: quote on book changes, log 15-min sums
        self.agg = {}                                 # cond -> [minutes, minutes quoted, reward] since the last q15 row
        self.dying = {t: [] for t in self.mk}         # (cancel lands at, order): still fillable until then
        self.blind = set()                            # feed down: orders rest unseen until the resubscribe snapshot
        self.done = set()                             # resolved and settled
        self.recent = {t: collections.deque() for t in self.mk}   # (time, reward, quoted) per sampled minute
        self.excluded = st.get("excluded", {})        # cond -> when it was swapped out
        self.t_check = self.t0 = time.time()
        self.peak = {t: 0.0 for t in self.mk}         # best reward over LEVEL_S since this run started
        self.swap, self.cands = None, None            # set when sample_loop stops to swap markets
        k = max(1, a.shards) if self.bulk else 1
        self.rate, self.burst = ORDER_RATE / k, ORDER_BURST / k
        self.tokens, self.t_tok, self.n_limited = self.burst, time.time(), 0
        os.makedirs(OUT, exist_ok=True)
        self.fw = self._csv(FILL_CSV, ["ts", "cond", "side", "price", "qty", "mid", "net_after", "how"])
        if self.bulk:
            self.qw = self._csv(Q15_CSV, ["ts", "cond", "mid", "best_bid", "best_ask", "mins", "quoted_mins", "reward",
                                          "net", "cash"])
        else:
            self.mw = self._csv(MIN_CSV, ["ts", "cond", "mid", "best_bid", "best_ask", "my_bid", "my_ask", "my_q",
                                          "others_q", "share", "reward_min", "net", "cash"])

    @staticmethod
    def _csv(path, header):
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a", newline="", encoding="utf-8")
        w = csv.writer(f)
        if new:
            w.writerow(header)
        return f, w

    def take_order_token(self, now):
        """Order bucket for this process's share of the signer's limit. Cancels are not limited (they may go
        negative on the Standard tier)."""
        self.tokens = min(self.burst, self.tokens + (now - self.t_tok) * self.rate)
        self.t_tok = now
        if self.tokens < 1:
            self.n_limited += 1
            return False
        self.tokens -= 1
        return True

    def drop(self, t, side=None):
        """Cancel. The order stays at the exchange, fillable and holding cash, until the cancel lands."""
        now = time.time()
        for s in [side] if side else list(self.orders[t]):
            o = self.orders[t].pop(s, None)
            if o and o.size > 0:
                self.dying[t].append((now + LATENCY, o))

    def resting(self, t, now):
        d = [(x, o) for x, o in self.dying[t] if x > now and o.size > 0]
        self.dying[t] = d
        return list(self.orders[t].values()) + [o for _, o in d]

    def others_book(self, t):
        """The book the quotes are priced and scored against (reward_live.py takes our own orders out of it)."""
        return self.book[t]

    def place(self, t, side, p, size, ahead, now):
        self.orders[t][side] = SimOrder(side, p, size, ahead, now + LATENCY)

    def lost_feed(self, toks):
        self.blind.update(toks)                              # orders stay live at the exchange, unseen

    def mid(self, t):
        b = self.others_book(t)
        return (max(b["bids"]) + min(b["asks"])) / 2 if b["bids"] and b["asks"] else None

    def fill(self, t, side, price, qty, how):
        if qty <= 0:
            return
        before = self.acct.net.get(t, 0.0)
        self.acct.fill(t, side, price, qty)
        self.n_fill += 1
        now, after, mid = time.time(), self.acct.net.get(t, 0.0), self.mid(t)
        if how in ("print", "cross", "live"):                # a time stop or a settlement is not a maker fill to judge
            self.pending.append((now + MARKOUT_S, t, side, price))
        if abs(after) < 1:
            self.entry.pop(t, None)
        elif abs(after) > abs(before) or before * after < 0:      # opened or flipped: track the entry
            e = self.entry.get(t)
            if e is None or e["side"] != side or before * after < 0:
                self.entry[t] = dict(side=side, fair=mid if mid is not None else price, price=price, opened=now)
            else:
                e["price"] = price
        f, w = self.fw
        w.writerow([round(time.time(), 3), self.mk[t]["cond"], side, price, round(qty, 4), self.mid(t),
                    round(self.acct.net.get(t, 0), 4), how]); f.flush()

    def on_trade(self, t, price, size):
        now = time.time()
        for o in self.resting(t, now):
            self.fill(t, o.side, o.price, on_trade(o, price, size, now), "print")

    def on_book(self, t):
        b, now = self.book[t], time.time()
        for o in self.resting(t, now):
            lvl = (b["bids"] if o.side == "bid" else b["asks"]).get(o.price, 0.0)
            opp = (min(b["asks"]) if b["asks"] else None) if o.side == "bid" else (max(b["bids"]) if b["bids"] else None)
            self.fill(t, o.side, o.price, on_book(o, lvl, opp, now), "cross")

    def requote(self, t, now):
        m, b = self.mk[t], self.others_book(t)
        mid = (max(b["bids"]) + min(b["asks"])) / 2 if b["bids"] and b["asks"] else None
        if t in self.blind or t in self.done:          # blind: we cannot see the book, so we cannot act on it
            return
        if mid is None:
            self.drop(t)
            return
        h = self.hist[t]
        h.append((now, mid))
        while h and h[0][0] < now - 30:
            h.pop(0)
        if max(x for _, x in h) - min(x for _, x in h) >= PULL_MOVE:
            self.pulled[t] = now + PULL_S
        if now < self.pulled[t] or min(b["asks"]) - max(b["bids"]) > 2 * m["v"]:   # no anchor for the mid
            self.drop(t)
            return
        net, n = self.acct.net.get(t, 0.0), m["size"]
        e = self.entry.get(t)
        if e is None and abs(net) >= 1:                # a position from a state file older than the entry log
            e = self.entry[t] = dict(side="bid" if net > 0 else "ask", fair=mid, price=mid, opened=now)
        if e and now - e["opened"] >= MAX_HOLD:
            self.time_stop(t, net)
            return
        why = self.no_open(t, mid, now)
        # inventory skew, as in sigbot/pmmm.py: lean both quotes against the position so the other side
        # fills and flattens it instead of holding to resolution. At the cap the closing quote sits at the mid.
        half = m["v"] * QUOTE_FRAC
        c = mid - half * net / (2 * n)
        tgt = {"bid": tick_floor(c - half, m["tick"]), "ask": tick_ceil(c + half, m["tick"])}
        for side, k in (("bid", EXIT_FRONT if net <= -1 else 0 if net >= 1 else QUOTE_FRONT),
                        ("ask", EXIT_FRONT if net >= 1 else 0 if net <= -1 else QUOTE_FRONT)):
            if k:                                            # at, or a tick ahead of, the best other quote
                if side == "bid":
                    tgt["bid"] = max(tgt["bid"], round(max(b["bids"]) + (k - 1) * m["tick"], 6))
                else:
                    tgt["ask"] = min(tgt["ask"], round(min(b["asks"]) - (k - 1) * m["tick"], 6))
        if e and FROZEN_EXIT:   # frozen exit (dynrisk.exit_fair): never close worse than the fair at entry
            if net > 0:
                tgt["ask"] = max(tgt["ask"], tick_ceil(e["fair"], m["tick"]))
            else:
                tgt["bid"] = min(tgt["bid"], tick_floor(e["fair"], m["tick"]))
        tgt["bid"] = min(tgt["bid"], round(min(b["asks"]) - m["tick"], 6))   # never cross
        tgt["ask"] = max(tgt["ask"], round(max(b["bids"]) + m["tick"], 6))
        for side in ("bid", "ask"):
            p = tgt[side]
            closing = max(0.0, -net) if side == "bid" else max(0.0, net)
            room = 2 * n - net if side == "bid" else 2 * n + net          # shares left under the cap
            stacked = e and e["side"] == side and (p > e["price"] - ADD_STEP if side == "bid"
                                                   else p < e["price"] + ADD_STEP)
            opening = not (why or stacked)
            size = min(n, room) if opening else min(n, closing)
            # an opening quote must score (reward min size); a closing one only needs the exchange's 5 shares
            if size < (m["minsz"] if opening else 5) or not 0 < p < 1:
                self.drop(t, side)
                continue
            o = self.orders[t].get(side)
            if o and abs(o.price - p) < 1e-9 and 0 < o.size <= size:
                continue
            if self.acct.cost(t, side, p, size) > self.free_cash(t, side):
                self.drop(t, side)
                continue
            if not self.take_order_token(now):             # rate limited: the old quote (if any) stays where it is
                continue
            self.drop(t, side)                             # a replace is a cancel plus a new order
            ahead = (b["bids"] if side == "bid" else b["asks"]).get(p, 0.0)
            self.place(t, side, p, size, ahead, now)

    def no_open(self, t, mid, now):
        """Why no new position may be opened in this market right now (None if it may)."""
        p = self.paused.get(t)
        if p and p[0] > now:
            return p[1]
        last = self.last.get(t)
        if last is None:
            self.last[t] = (now, mid)
        elif now - last[0] >= 60:
            a = 1 - 0.5 ** (1 / VOL_HALFLIFE)
            d2 = (mid - last[1]) ** 2
            self.var[t] = d2 if t not in self.var else (1 - a) * self.var[t] + a * d2
            self.vn[t] = self.vn.get(t, 0) + 1
            self.last[t] = (now, mid)
        s = self.slow.get(t)
        slow = mid if s is None else s[1] + (1 - math.exp(-(now - s[0]) / TREND_TAU)) * (mid - s[1])
        self.slow[t] = (now, slow)
        if self.vn.get(t, 0) < 3:
            return "warming up"
        if math.sqrt(self.var[t]) > VOL_MAX:
            return f"volatile ({math.sqrt(self.var[t]) * 100:.1f}c/min)"
        if abs(mid - slow) > TREND_BANDS * self.mk[t]["v"]:
            return f"trending ({(mid - slow) * 100:+.1f}c from slow fair)"
        return None

    def time_stop(self, t, net):
        """dynrisk.timed_out: sell what is held at the best price (top level only), then forget the entry. A taker
        trade: the fee, rate * p * (1 - p) per share, is folded into the recorded price."""
        b = self.book[t]
        f = self.mk[t].get("fee", "").replace("_fees", "")
        f = "sports" if f.startswith("sports") else f.split("_v")[0]
        rate = 0.0 if f in ("", "zero") else TAKER_FEE_RATE.get(f, 0.05)
        if net > 0 and b["bids"]:
            p = max(b["bids"]); self.fill(t, "ask", p - rate * p * (1 - p), min(net, b["bids"][p]), "time_stop")
        elif net < 0 and b["asks"]:
            p = min(b["asks"]); self.fill(t, "bid", p + rate * p * (1 - p), min(-net, b["asks"][p]), "time_stop")
        self.drop(t)
        if t in self.entry:
            self.entry[t]["opened"] = time.time() - MAX_HOLD + 60     # what the top level didn't take: retry in 1 min

    def resolved(self):
        """YES payout of every held market that has resolved, by token. Runs in a thread."""
        held = [m for t, m in self.mk.items() if abs(self.acct.net.get(t, 0)) >= 1 and t not in self.done]
        out = {}
        for i in range(0, len(held), 50):
            for g in get(f"{GAMMA}/markets", condition_ids=[m["cond"] for m in held[i:i + 50]], closed="true",
                         limit=50) or []:
                try:
                    px, yes = float(json.loads(g["outcomePrices"])[0]), json.loads(g["clobTokenIds"])[0]
                except (KeyError, ValueError, TypeError, IndexError):
                    continue
                if g.get("closed") and (px >= 0.99 or px <= 0.01):
                    out[yes] = round(px)
        return out

    def settle(self, t, payout):
        """Resolution: what is held pays 1 or 0 a YES share, and the market stops quoting."""
        net = self.acct.net.get(t, 0.0)
        self.drop(t)
        self.dying[t] = []
        if net > 0:
            self.fill(t, "ask", payout, net, "settle")
        elif net < 0:
            self.fill(t, "bid", payout, -net, "settle")
        self.done.add(t)
        log(f"settled {self.mk[t]['q'][:50]} at {payout}: {net:+.0f} YES")

    def score_markouts(self):
        now = time.time()
        due, self.pending = [x for x in self.pending if x[0] <= now], [x for x in self.pending if x[0] > now]
        for _, t, side, price in due:
            mid = self.mid(t)
            if mid is None:
                continue
            won = mid > price if side == "bid" else mid < price
            h = self.outcomes.setdefault(t, [])
            h.append(int(won))
            del h[:-HIT_WINDOW]
            if len(h) >= HIT_MIN and sum(h) / len(h) < HIT_RATE:
                self.paused[t] = [now + PAUSE_S, f"fills won {sum(h)}/{len(h)}"]
                self.outcomes.pop(t)
                log(f"paused 24h: {self.mk[t]['q'][:50]} (fills won {sum(h)}/{len(h)} at +{MARKOUT_S // 60} min)")

    def judge(self, C, now):
        """cond -> why, for each flat market that should be swapped out. C: a fresh screen."""
        est = {c["cond"]: c["est"] for c in C}
        held = {m["cond"] for m in self.mk.values()}
        free = [c for c in C if c["est"] > 0 and c["cond"] not in held and now - self.excluded.get(c["cond"], 0) >= EXCLUDE_S]
        spare = self.a.bankroll - sum(m["capital"] for m in self.mk.values())
        best, out = (free[0]["est"] if free else 0.0), {}
        if len(self.mk) < self.a.markets and any(c["capital"] <= spare for c in free):
            out[None] = "a slot is empty"                  # nothing fitted at the last swap; something does now
        for t, m in self.mk.items():
            if t in self.done:
                out[m["cond"]] = "resolved"
                continue
            if abs(self.acct.net.get(t, 0.0)) >= 1:        # never leave a position behind
                continue
            d, p = self.recent[t], self.paused.get(t)
            quoted = sum(x[2] for x in d) / max(1, len(d))
            yld = sum(x[1] for x in d) * 1440 / max(1, len(d)) / m["capital"]
            if p and p[0] > now:
                out[m["cond"]] = f"paused: {p[1]}"
            elif m["rate"] <= 0:
                out[m["cond"]] = "reward pool ended"
            elif quoted < MIN_QUOTED:
                out[m["cond"]] = f"quoted {quoted:.0%} of the last {SWAP_WINDOW // 60} min"
            elif yld < MIN_YIELD:
                out[m["cond"]] = f"earned {yld:.2%}/day of its capital over the last {SWAP_WINDOW // 3600} h"
            elif m["cond"] not in est:
                out[m["cond"]] = "no longer passes the screen"
            elif best > SWAP_GAIN * est[m["cond"]]:
                out[m["cond"]] = f"screen rates another market {best:.0%}/day against {est[m['cond']]:.0%}/day here"
            elif free and now - self.t0 >= MIN_STAY and (r := self.earned(t, now)) < LEVEL_FRAC * self.peak[t]:
                out[m["cond"]] = (f"levelled off: ${r * 86400 / LEVEL_S:.2f}/day over the last {LEVEL_S // 60} min "
                                  f"against ${self.peak[t] * 86400 / LEVEL_S:.2f}/day at best")
        return out

    def earned(self, t, now):
        """Reward on t over the last LEVEL_S s."""
        return sum(x[1] for x in self.recent[t] if x[0] >= now - LEVEL_S)

    def free_cash(self, t, side):
        """Cash not reserved by the other resting orders (all of them could fill)."""
        used = sum(self.acct.cost(u, s, o.price, o.size) for u, os_ in self.orders.items() for s, o in os_.items()
                   if (u, s) != (t, side))
        used += sum(self.acct.cost(u, o.side, o.price, o.size) for u, d in self.dying.items() for _, o in d)
        return self.acct.cash - used

    def sample(self):
        now = time.time()
        for t, m in self.mk.items():
            b = self.others_book(t)
            mid = (max(b["bids"]) + min(b["asks"])) / 2 if b["bids"] and b["asks"] else None
            if mid is None:
                continue
            others = q_min(score(b["bids"].items(), mid, m["v"], m["minsz"]), score(b["asks"].items(), mid, m["v"], m["minsz"]), mid)
            live = {s: o for s, o in self.orders[t].items() if now >= o.live_at and o.size > 0}
            mine = q_min(*(score([(o.price, o.size)], mid, m["v"], m["minsz"]) if (o := live.get(s)) else 0.0
                           for s in ("bid", "ask")), mid)
            share = mine / (mine + others) if mine > 0 else 0.0
            r = m["rate"] / 1440 * share
            self.reward += r
            d = self.recent[t]
            d.append((now, r, mine > 0))
            while d[0][0] < now - SWAP_WINDOW:
                d.popleft()
            if now - self.t0 >= LEVEL_S:
                self.peak[t] = max(self.peak[t], self.earned(t, now))
            if self.bulk:
                g = self.agg.setdefault(m["cond"], [0, 0, 0.0])
                g[0] += 1; g[1] += mine > 0; g[2] += r
                if int(now // 60) % 15 == 0:
                    f, w = self.qw
                    w.writerow([round(now), m["cond"], mid, max(b["bids"]), min(b["asks"]), g[0], g[1], round(g[2], 5),
                                round(self.acct.net.get(t, 0), 4), round(self.acct.cash, 2)])
                    del self.agg[m["cond"]]
                continue
            f, w = self.mw
            w.writerow([round(now), m["cond"], mid, max(b["bids"]), min(b["asks"]),
                        live["bid"].price if "bid" in live else "", live["ask"].price if "ask" in live else "",
                        round(mine, 2), round(others, 2), round(share, 4), round(r, 5),
                        round(self.acct.net.get(t, 0), 4), round(self.acct.cash, 4)])
            f.flush()
        if self.bulk:
            self.qw[0].flush()
        self.score_markouts()
        try:                                           # write beside it and swap: a crash mid-write keeps the old state
            with open(STATE) as f:
                st = json.load(f)
            st.update(cash=self.acct.cash, net=self.acct.net, reward=self.reward, entry=self.entry,
                      outcomes=self.outcomes, paused=self.paused)
            with open(STATE + ".tmp", "w") as f:
                json.dump(st, f, indent=1)
            os.replace(STATE + ".tmp", STATE)
        except OSError as e:                           # e.g. antivirus holding it
            log(f"state save failed ({e}), retrying next minute")

    def handle(self, msg):
        et, t = msg.get("event_type"), msg.get("asset_id")
        if et == "book" and t in self.book:
            apply_snapshot(self.book[t], msg); self.on_book(t)   # anything the market moved through while blind fills
            self.blind.discard(t)
            if self.bulk:
                self.requote(t, time.time())
        elif et == "price_change":
            seen = set()
            for ch in msg.get("price_changes") or msg.get("changes") or []:
                u = ch.get("asset_id") or t
                if u in self.book:
                    apply_change(self.book[u], float(ch["price"]), ch["side"], float(ch["size"])); self.on_book(u)
                    seen.add(u)
            if self.bulk:                                    # quote on the change, not on a 1 s sweep of every market
                for u in seen:
                    self.requote(u, time.time())
        # ponytail: YES-token prints only; NO-token prints could double-count a minting match. Fills
        # from NO-side flow still arrive through on_book's cross check.
        elif et == "last_trade_price" and t in self.book:
            self.on_trade(t, float(msg["price"]), float(msg.get("size") or 0))
        elif et == "tick_size_change" and t in self.mk:
            self.mk[t]["tick"] = float(msg["new_tick_size"])

    async def feed(self, toks):
        while True:
            try:
                async with websockets.connect(WS, ping_interval=10, ping_timeout=20, max_size=None) as ws:
                    await ws.send(json.dumps({"assets_ids": toks, "type": "market"}))
                    while True:
                        P = json.loads(await asyncio.wait_for(ws.recv(), FEED_SILENT))
                        for m in P if isinstance(P, list) else [P]:
                            if isinstance(m, dict):
                                self.handle(m)
            except Exception as e:                           # noqa: BLE001
                self.lost_feed(toks)
                log(f"feed reconnect ({type(e).__name__}: {e})")
                await asyncio.sleep(3)

    async def quote_loop(self):
        while True:
            now = time.time()
            for t in self.mk:
                self.requote(t, now)
            await asyncio.sleep(30 if self.bulk else 1)     # bulk: a slow sweep for the time stop and quiet books

    def refresh_rates(self):
        # the pool changes daily; selection-time total_daily_rate went stale (Super Bowl 200 -> 50/day)
        if self.bulk:                                        # one paged listing, not a request per market
            R, cur = {}, ""
            while cur != "LTE=":
                d = get(f"{CLOB}/rewards/markets/current", next_cursor=cur)
                if d is None:                                # keep the old rates rather than zero them
                    return
                R.update({r["condition_id"]: r["total_daily_rate"] for r in d.get("data") or []})
                cur = d.get("next_cursor") or "LTE="
            for m in self.mk.values():
                m["rate"] = R.get(m["cond"], 0)
            return
        today = dt.date.today().isoformat()
        for m in self.mk.values():
            d = get(f"{CLOB}/rewards/markets/{m['cond']}")
            if d is None:                                    # keep the old rate rather than zero it
                continue
            cfg = (d.get("data") or [{}])[0].get("rewards_config") or []
            m["rate"] = sum(c["rate_per_day"] for c in cfg if c["start_date"] <= today <= c["end_date"])

    async def sample_loop(self):
        await asyncio.to_thread(self.refresh_rates)
        scan = None
        while True:
            if int(time.time() // 60) % (120 if self.bulk else 30) == 0:
                await asyncio.to_thread(self.refresh_rates)
                for t, px in (await asyncio.to_thread(self.resolved)).items():
                    self.settle(t, px)
            await asyncio.sleep(60 - time.time() % 60 + random.uniform(0, 59))   # one random instant per minute
            self.sample()
            if getattr(self.a, "swap", False) and scan is None and time.time() - self.t_check >= SWAP_CHECK:
                self.t_check = time.time()
                scan = asyncio.create_task(asyncio.to_thread(candidates, self.a))   # minutes; sampling goes on
            if scan and scan.done():
                try:
                    C = scan.result()
                except Exception as e:
                    log(f"market scan failed ({type(e).__name__}: {e}); keeping the markets")
                    C = []
                scan = None
                why = self.judge(C, time.time()) if C else {}
                if why:                                      # run() returns; the caller reselects and restarts
                    self.swap, self.cands = why, C
                    return
            if int(time.time() // 60) % 15 == 0:
                log(f"reward ${self.reward:.2f}  fills {self.n_fill}  cash ${self.acct.cash:.2f}  "
                    f"positions {{{', '.join(f'{self.mk[t]['q'][:20]}: {n:+.0f}' for t, n in self.acct.net.items() if abs(n) > 1e-6)}}}")

    async def run(self):
        toks = list(self.mk)
        tasks = [asyncio.create_task(c) for c in (*(self.feed(toks[i:i + 200]) for i in range(0, len(toks), 200)),
                                                  self.quote_loop(), self.sample_loop())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)   # only sample_loop returns (a swap)
        for x in tasks:
            x.cancel()
        for x in done:
            x.result()                                       # an exception stops the run, as gather did


def report():
    import pandas as pd
    st = json.load(open(STATE))
    M = pd.read_csv(MIN_CSV)
    F = pd.read_csv(FILL_CSV) if os.path.exists(FILL_CSV) and os.path.getsize(FILL_CSV) else pd.DataFrame()
    days = (M.ts.max() - M.ts.min()) / 86400
    print(f"{len(st['markets'])} markets, {days:.2f} days logged, bankroll ${st['bankroll']:.0f}")
    by = M.groupby("cond").agg(minutes=("ts", "size"), quoted=("my_q", lambda x: (x > 0).mean()), share=("share", "mean"),
                               reward=("reward_min", "sum"))
    q = {**st.get("seen", {}), **{m["cond"]: m["q"][:50] for m in st["markets"]}}
    by.index = [q.get(c, c[:12]) for c in by.index]
    print(by.round(3).to_string())
    daily = M.assign(day=pd.to_datetime(M.ts, unit="s").dt.date).groupby("day").reward_min.sum()
    paid = daily.where(daily >= 1, 0)                  # payouts under $1 a day are not paid (assumed account-wide)
    print(f"\nrewards accrued ${daily.sum():.2f}; paid after the $1/day minimum ${paid.sum():.2f}")
    last = M.sort_values("ts").groupby("cond").tail(1).set_index("cond")
    cond = {m["yes"]: m["cond"] for m in st["markets"]}
    for how, col in (("mid", "mid"), ("price you could sell or cover at", None)):
        px = lambda t, n: last.loc[cond[t], "mid"] if col else last.loc[cond[t], "best_bid" if n > 0 else "best_ask"]
        pos = sum(n * px(t, n) if n > 0 else -n * (1 - px(t, n)) for t, n in st["net"].items() if t in cond)   # short YES = long NO
        fill_pnl = st["cash"] + pos - st["bankroll"]
        print(f"fill P&L marked at the last {how}: ${fill_pnl:+.2f} ({len(F)} fills)  ->  net ${paid.sum() + fill_pnl:+.2f}")
    if len(F):
        rt = F.assign(sgn=F.side.map({"bid": -1, "ask": 1}))
        print(f"  fills: {(F.side == 'bid').sum()} buys, {(F.side == 'ask').sum()} sells; "
              f"shares still held {sum(abs(n) for n in st['net'].values()):.0f}")
        M2 = M.sort_values("ts").astype({"ts": float})
        for h in (60, 600, 3600):
            x = pd.merge_asof(F.assign(t2=F.ts + h).sort_values("t2"), M2[["ts", "cond", "mid"]].rename(columns={"ts": "t2", "mid": "mid_h"}),
                              on="t2", by="cond", direction="forward").dropna(subset=["mid_h"])
            sgn = x.side.map({"bid": 1, "ask": -1})
            mk = sgn * (x.mid_h - x.price)
            print(f"  markout +{h // 60:>3} min: {100 * (mk * x.qty).sum() / max(x.qty.sum(), 1):+.2f}c/share over {len(x)} fills "
                  f"(spread captured at fill {100 * (sgn * (x.mid - x.price) * x.qty).sum() / max(x.qty.sum(), 1):+.2f}c)")


def survey_start(a):
    """Keep the markets still holding a position, drop the flat ones (their budget is withdrawn, their P&L stays in
    cash), add fresh picks (their budget is deposited). bankroll = money in, so cash + positions - bankroll = fill P&L."""
    os.makedirs(OUT, exist_ok=True)
    st = json.load(open(STATE)) if os.path.exists(STATE) else dict(started=time.time(), bankroll=0.0, cash=0.0, net={},
                                                                   reward=0.0, markets=[], seen={})
    keep = [m for m in st["markets"] if abs(st["net"].get(m["yes"], 0)) >= 1]
    for m in st["markets"]:
        if m not in keep:
            st["cash"] -= m["budget"]; st["bankroll"] -= m["budget"]
    pick = select(a, keep)
    for m in pick[len(keep):]:
        st["cash"] += m["budget"]; st["bankroll"] += m["budget"]
    st["markets"] = pick
    st.setdefault("seen", {}).update({m["cond"]: dict(q=m["q"], cat=m["cat"], budget=m["budget"]) for m in pick})
    json.dump(st, open(STATE, "w"), indent=1)


def report_survey(roots=None):
    """Rewards and fill P&L per market and per type. A market's fill P&L is its fill cash flows plus what it
    still holds at the last mid it was logged at. roots: --all's parent folder, whose shard* folders are combined."""
    import glob
    import pandas as pd
    dirs = sorted(glob.glob(os.path.join(roots[0], "shard*"))) if roots else [OUT]
    rows = []
    for d in dirs:
        st = json.load(open(os.path.join(d, "state.json")))
        q15 = os.path.join(d, "q15.csv")
        if os.path.exists(q15):
            if os.path.getsize(q15) == 0:                 # header not flushed yet: no 15-min row
                continue
            M = pd.read_csv(q15).rename(columns={"reward": "reward_min"})
            if M.empty:
                continue
        else:
            M = pd.read_csv(os.path.join(d, "minutes.csv")).assign(mins=1)
        fp = os.path.join(d, "fills.csv")
        F = pd.read_csv(fp) if os.path.exists(fp) and os.path.getsize(fp) else pd.DataFrame(
            columns=["ts", "cond", "side", "price", "qty", "net_after"])
        last = M.sort_values("ts").groupby("cond").tail(1).set_index("cond")
        for cond, info in st["seen"].items():
            m = M[M.cond == cond]
            f = F[F.cond == cond].sort_values("ts")
            flow = (f.qty * f.price * f.side.map({"ask": 1, "bid": -1})).sum()
            held = f.net_after.iloc[-1] if len(f) else 0.0
            fill = flow + held * (last.loc[cond, "mid"] if cond in last.index else 0.0)
            rows.append(dict(cat=info["cat"], market=info["q"][:45], budget=info["budget"], hours=m.mins.sum() / 60,
                             reward=m.reward_min.sum(), fills=len(f), fill_pnl=fill, held=held))
    R = pd.DataFrame(rows)
    R = R[R.hours > 0]
    R["net"] = R.reward + R.fill_pnl
    R["net_per_day_per_100"] = R.net / (R.hours / 24).clip(lower=1 / 24) / R.budget * 100
    pd.set_option("display.width", 200)
    if len(R) > 60:                                       # --all: per-market rows go to a file, not the screen
        R.to_csv(os.path.join(dirs[0], "..", "by_market.csv"), index=False)
        print(f"{len(R)} markets; per-market rows in by_market.csv")
    else:
        print(R.sort_values(["cat", "net"]).round(2).to_string(index=False))
    C = R.groupby("cat").agg(markets=("market", "size"), budget=("budget", "sum"), hours=("hours", "sum"),
                             reward=("reward", "sum"), fill_pnl=("fill_pnl", "sum"), fills=("fills", "sum"), net=("net", "sum"))
    C["net_per_day_per_100"] = C.net / (C.hours / 24).clip(lower=1 / 24) / (C.budget / C.markets) * 100
    print("\nby type (net per day per $100 of budget):")
    print(C.sort_values("net_per_day_per_100", ascending=False).round(2).to_string())
    print(f"\ntotal: reward ${R.reward.sum():.2f}  fill P&L ${R.fill_pnl.sum():+.2f}  net ${R.net.sum():+.2f}")


def selftest():
    assert abs(score([(0.49, 100)], 0.50, 0.03, 20) - 100 * (2 / 3) ** 2) < 1e-9
    assert score([(0.49, 10)], 0.50, 0.03, 20) == 0                 # under the min size
    assert q_min(9, 0, 0.5) == 3 and q_min(9, 0, 0.95) == 0
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--markets", type=int, default=5)
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--max-min-size", type=float, default=50)
    ap.add_argument("--survey", action="store_true")
    ap.add_argument("--per-cat", type=int, default=4)
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--all", action="store_true", help="survey every quotable market, split over --shards processes")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", help="folder for state and logs instead of data/rewards (a separate paper run)")
    ap.add_argument("--select-only", action="store_true", help="--all: write the shard's state.json and exit")
    ap.add_argument("--latency", type=float, default=None,
                    help="s for an order or cancel to take effect; --all defaults to 0.05 (AWS Ireland -> the CLOB)")
    ap.add_argument("--quote-frac", type=float, default=None,
                    help="distance of a quote from the mid as a share of the reward band (default 0.5)")
    a = ap.parse_args()
    if a.quote_frac is not None:
        QUOTE_FRAC = a.quote_frac
    if a.latency is not None or a.all:
        LATENCY = a.latency if a.latency is not None else 0.05
    if a.out:
        set_out(os.path.abspath(a.out))
    if a.all:
        a.survey, a.per_cat = True, 0
        set_out(os.path.join(OUT + "_all", f"shard{a.shard}"))
        if a.report:
            report_survey([os.path.dirname(OUT)])
            raise SystemExit
    elif a.survey:
        set_out(OUT + "_survey")
    if a.selftest:
        selftest()
    elif a.report:
        report_survey() if a.survey else report()
    elif a.survey:
        survey_start(a)
        if a.select_only:                              # cpp/reward_quoter.exe quotes the shard from here
            raise SystemExit
        st = json.load(open(STATE))
        log(f"surveying {len(st['markets'])} markets, cash ${st['cash']:.2f}, reward so far ${st['reward']:.2f}")
        try:
            asyncio.run(asyncio.wait_for(Quoter(a, st).run(), a.hours * 3600))
        except (asyncio.TimeoutError, TimeoutError):
            log("reselecting")
    else:
        if a.fresh or not os.path.exists(STATE):
            os.makedirs(OUT, exist_ok=True)
            json.dump(dict(started=time.time(), bankroll=a.bankroll, cash=a.bankroll, net={}, reward=0.0, markets=select(a)),
                      open(STATE, "w"), indent=1)
        a.swap = True
        while True:
            st = json.load(open(STATE))
            log(f"quoting {len(st['markets'])} markets, cash ${st['cash']:.2f}, reward so far ${st['reward']:.2f}")
            q = Quoter(a, st)
            asyncio.run(q.run())
            q.fw[0].close(); q.mw[0].close()
            reselect(a, q)
