"""Build the dataset for short-horizon jump prediction from raw order books.

Target: P(max_{s<=H} |mid(t+s) - mid(t)| >= J ticks | book state up to t).
The max over the path, not the endpoint -- a resting quote is picked off by
the path.

Two representations come out of one pass:

  LOB tensor      top-K levels as [bid_px, bid_sz, ask_px, ask_sz], prices
                  expressed as ticks away from the mid and sizes in log
                  dollars. This is what the CNN reads.
  hand features   spread, multi-level depth imbalance, liquidity
                  concentration, realised volatility over several windows,
                  order-flow imbalance, staleness. This is what logistic
                  regression and the GBM read.

These are NOT equivalent, and the first version of this file wrongly claimed
they were. Expressing every level relative to that row's own mid makes each
frame translation-invariant, so a window of frames carries the book's internal
geometry but **not the mid's trajectory** -- exactly, not approximately. At
level 0 it is degenerate by construction: since mid = (bid+ask)/2, every row
has bid_px = -spread/2 and ask_px = +spread/2, so the touch price channel
holds no information beyond the spread.

Consequence: the mid-path family of features -- realised volatility over any
window, dmid, staleness -- is unreachable from the tensor alone, and those are
81% of the GBM's gain. `mid` is therefore kept as a column here so a model
reading the tensor can be handed the path separately; `jump_model.py` builds
its per-frame scalars from it. Any future consumer of the tensor must do the
same or it is training on a strictly weaker input than the baselines.

Sampling is on a fixed time grid rather than per update, because per-update
sampling would weight busy markets and busy moments enormously and make the
label base rate meaningless.

Usage:
    python jump_data.py --raw ../data/live/books_2026-08-30.jsonl
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import orjson
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "jump")
os.makedirs(OUT, exist_ok=True)

K_LEVELS = 10
GRID_MS = 200
# Longest silence that is forward-filled, in seconds. Must comfortably exceed
# the deep models' lookback (200 slots = 40s) so a fresh prediction point
# still has a fully contiguous window behind it.
MAX_FILL_S = 60.0
KEEP_TYPES = {"moneyline", "spread", "total", "first_inning_run"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def open_recording(path):
    """Open a raw recording, transparently handling xz archives.

    `compress_raw.py` deletes the original after proving a bit-for-bit
    round-trip, so a slate collected with compression enabled exists only as
    books_<tag>.jsonl.xz. Streaming it through lzma costs CPU but no disk,
    which is what makes collecting several nights on one machine possible --
    a slate is 25-45 GB raw and about 10x smaller archived.
    """
    if path.endswith(".xz"):
        import lzma
        return lzma.open(path, "rb")
    return open(path, "rb")


def session_tag(path):
    """books_<tag>.jsonl[.xz] -> <tag>, so archives name their outputs the
    same way the raw file would have."""
    name = os.path.basename(path)
    for ext in (".xz", ".jsonl"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name.replace(" ", "_")


def _levels(b, a, k):
    """One book's top-k levels as fixed-width arrays."""
    bp = np.full(k, np.nan); bs = np.zeros(k)
    ap = np.full(k, np.nan); asz = np.zeros(k)
    for i, (p, s) in enumerate(b):
        bp[i], bs[i] = p, s
    for i, (p, s) in enumerate(a):
        ap[i], asz[i] = p, s
    return bp, bs, ap, asz


class _RowBuffer:
    """Accumulates grid rows into preallocated blocks, not per-row arrays.

    The original version appended a tuple per row holding four separate
    numpy arrays (bid px, bid sz, ask px, ask sz). At ~10M rows that is ~40M
    tiny array objects, and numpy's per-object overhead (~100 bytes each,
    dwarfing the 40 bytes of actual payload) runs to several GB on its own --
    which is what made building the 2026-09-12 session OOM with 5GB free.

    Writing straight into preallocated (BLOCK, k) arrays removes that
    overhead entirely: memory becomes the data itself plus one partly filled
    block. Blocks are concatenated once at the end.
    """

    BLOCK = 500_000

    def __init__(self, k):
        self.k = k
        self.blocks = []
        self._new_block()

    def _new_block(self):
        b = self.BLOCK
        self.i = 0
        self.slug = np.empty(b, dtype=object)
        self.mt = np.empty(b, dtype=object)
        self.line = np.empty(b, np.float32)
        self.ts = np.empty(b, np.int64)
        self.tsb = np.empty(b, np.int64)
        self.BP = np.full((b, self.k), np.nan, np.float32)
        self.BS = np.zeros((b, self.k), np.float32)
        self.AP = np.full((b, self.k), np.nan, np.float32)
        self.AS = np.zeros((b, self.k), np.float32)

    def _seal(self):
        if self.i == 0:
            return
        self.blocks.append(dict(
            slug=self.slug[:self.i].copy(), mt=self.mt[:self.i].copy(),
            line=self.line[:self.i].copy(), ts=self.ts[:self.i].copy(),
            tsb=self.tsb[:self.i].copy(),
            BP=self.BP[:self.i].copy(), BS=self.BS[:self.i].copy(),
            AP=self.AP[:self.i].copy(), AS=self.AS[:self.i].copy()))

    def add(self, slug, mt, line, slot, ts_book, bids, asks):
        if self.i >= self.BLOCK:
            self._seal()
            self._new_block()
        i = self.i
        self.slug[i] = slug
        self.mt[i] = mt
        self.line[i] = np.nan if line is None else line
        self.ts[i] = slot
        self.tsb[i] = ts_book
        # write levels directly into the block rows; the row was preset to
        # NaN price / 0 size by _new_block, so short books need no padding
        self.BP[i, :] = np.nan
        self.BS[i, :] = 0.0
        self.AP[i, :] = np.nan
        self.AS[i, :] = 0.0
        for j, (p, s) in enumerate(bids[:self.k]):
            self.BP[i, j] = p
            self.BS[i, j] = s
        for j, (p, s) in enumerate(asks[:self.k]):
            self.AP[i, j] = p
            self.AS[i, j] = s
        self.i += 1

    def __len__(self):
        return sum(len(b["ts"]) for b in self.blocks) + self.i

    def finish(self):
        self._seal()
        if not self.blocks:
            raise SystemExit("no grid rows produced")
        cat = lambda key: np.concatenate([b[key] for b in self.blocks], axis=0)
        out = dict(slug=cat("slug"), mt=cat("mt"), line=cat("line"),
                   ts=cat("ts"), ts_book=cat("tsb"),
                   BP=cat("BP"), BS=cat("BS"), AP=cat("AP"), AS=cat("AS"))
        self.blocks.clear()
        return out


def extract(path: str, grid_ms=GRID_MS, k=K_LEVELS, max_fill_s=MAX_FILL_S):
    """Raw JSONL -> one row per (series, grid tick) with top-k depth.

    Forward-fill direction is the subtle part, and the first version of this
    function had it backwards -- see the comment in the emit loop.

    `max_fill_s` caps how long a silence is forward-filled. Without it the
    2026-09-09 session expands 2.8M messages into 22.6M grid rows, because a
    market quiet for 500s still emits 2,500 slots holding one book. Those rows
    carry no additional information, they dominate the objective, and the
    resulting tensor did not fit in RAM. Capping leaves GAPS in the slot
    sequence, so jump_split.load() checks that a sample's lookback window is
    actually contiguous rather than assuming it -- assuming it would bend the
    time axis, the same class of error as the back-fill bug.
    """
    max_fill_slots = max(1, int(round(max_fill_s * 1000 / grid_ms)))
    capped = 0
    # series key -> last snapshot seen, so the grid can be forward-filled
    cur: dict[tuple, tuple] = {}
    rows = _RowBuffer(k)
    n = 0
    t0 = time.time()
    next_emit: dict[tuple, int] = {}

    with open_recording(path) as fh:
        for raw in fh:
            n += 1
            try:
                r = orjson.loads(raw)
            except Exception:
                continue
            if r.get("outcome") != "YES":
                continue
            mt = r.get("market_type")
            if mt not in KEEP_TYPES:
                continue
            bids, asks = r.get("bids"), r.get("asks")
            if not bids or not asks:
                continue
            if asks[0][0] <= bids[0][0]:
                continue

            key = (r.get("slug"), mt, r.get("line"))
            ts = r["ts"]

            slot = next_emit.get(key)
            if slot is None:
                cur[key] = (ts, bids[:k], asks[:k])
                next_emit[key] = ts - (ts % grid_ms) + grid_ms
                continue

            # Every grid slot strictly before this message is described by the
            # PREVIOUS book. That is the only thing "carrying the last book"
            # can mean, and the first version of this loop did the opposite:
            # it assigned cur[key] = new book first and then back-filled the
            # pending slots with it, stamping a book from time `ts` onto rows
            # labelled seconds earlier.
            #
            # Measured cost of that bug on the 2026-08-30 build: 82.5% of rows
            # sat inside a run of identical mid longer than the entire 25-row
            # label window, 55% inside runs of 500+ rows, and `stale` became a
            # near-deterministic predictor -- P(jump | stale > 200) = 1.0% over
            # 5.2M rows versus 73% at stale = 0 -- purely as an artifact. With
            # a long enough silence the leak could also cross the train/test
            # boundary, since the gap is only 120s.
            pts, pb, pa = cur[key]
            filled = 0
            while slot < ts and filled < max_fill_slots:
                rows.add(key[0], mt, key[2], slot, pts, pb, pa)
                slot += grid_ms
                filled += 1
            if slot < ts:
                # silence longer than the cap: jump to the first boundary at
                # or after this message and leave the gap
                capped += 1
                slot = ts - (ts % grid_ms)
                if slot < ts:
                    slot += grid_ms
            cur[key] = (ts, bids[:k], asks[:k])
            # a slot landing exactly on this message does see the new book
            if slot == ts:
                rows.add(key[0], mt, key[2], slot, ts, bids[:k], asks[:k])
                slot += grid_ms
            next_emit[key] = slot

            if n % 5_000_000 == 0:
                log(f"  {n:,} lines, {len(rows):,} grid rows, "
                    f"{n/(time.time()-t0):,.0f}/s")

    log(f"  {n:,} lines -> {len(rows):,} grid rows in {time.time()-t0:.0f}s "
        f"({capped:,} silences truncated at {max_fill_s:.0f}s)")
    # ts_book is the exchange time of the book each row carries; ts - ts_book
    # is how stale the row genuinely is, which makes the forward-fill
    # auditable instead of something you have to infer from run lengths.
    d = rows.finish()
    assert (d["ts_book"] <= d["ts"]).all(), \
        "a row carries a book from its own future"
    return d


# --------------------------------------------------------------------------- #
def build_features(d: dict, tick=0.01, horizon_s=5, jump_ticks=2,
                   vol_windows=(5, 25, 150)):
    """Per-series features, labels and the normalised LOB tensor."""
    slug, mt, ts = d["slug"], d["mt"], d["ts"]
    BP, BS, AP, AS = d["BP"], d["BS"], d["AP"], d["AS"]
    series_id = pd.Series([f"{s}|{m}|{l}" for s, m, l in
                           zip(slug, mt, d["line"])]).astype("category")

    bid, ask = BP[:, 0], AP[:, 0]
    mid = (bid + ask) / 2.0
    spread = ask - bid

    bs_tot = np.nansum(BS, axis=1)
    as_tot = np.nansum(AS, axis=1)
    bid_usd = np.nansum(BS * np.nan_to_num(BP), axis=1)
    ask_usd = np.nansum(AS * np.nan_to_num(AP), axis=1)

    def imb(n):
        b = np.nansum(BS[:, :n], axis=1)
        a = np.nansum(AS[:, :n], axis=1)
        return (b - a) / np.maximum(b + a, 1e-9)

    feats = {
        "mid": mid, "spread_ticks": spread / tick,
        "imb1": imb(1), "imb3": imb(3), "imb5": imb(5), "imb10": imb(10),
        "log_bid_usd": np.log1p(bid_usd), "log_ask_usd": np.log1p(ask_usd),
        # liquidity concentration: how much of the book sits at the touch
        "conc_bid": BS[:, 0] / np.maximum(bs_tot, 1e-9),
        "conc_ask": AS[:, 0] / np.maximum(as_tot, 1e-9),
        # how far the book reaches, in ticks
        "reach_bid": (bid - np.nanmin(BP, axis=1)) / tick,
        "reach_ask": (np.nanmax(AP, axis=1) - ask) / tick,
        "microprice_dev": ((bid * AS[:, 0] + ask * BS[:, 0])
                           / np.maximum(BS[:, 0] + AS[:, 0], 1e-9) - mid) / tick,
    }

    out_rows, labels, tensors, meta = [], [], [], []
    codes = series_id.cat.codes.to_numpy()
    F = pd.DataFrame(feats)

    for sid in np.unique(codes):
        m = codes == sid
        if m.sum() < 400:
            continue
        idx = np.where(m)[0]
        sub = F.iloc[idx].reset_index(drop=True)
        t = ts[idx]
        md = mid[idx]

        # ---- realised volatility and order-flow imbalance, causal ----
        dm = np.diff(md, prepend=md[0])
        for w in vol_windows:
            sub[f"rv_{w}"] = (pd.Series(dm).pow(2).rolling(w, min_periods=1)
                              .mean().pow(0.5).shift(1).fillna(0).to_numpy() / tick)
        for w in (5, 25):
            sub[f"ofi_{w}"] = (pd.Series(sub.imb1).rolling(w, min_periods=1)
                               .mean().shift(1).fillna(0).to_numpy())
            sub[f"dmid_{w}"] = (pd.Series(dm).rolling(w, min_periods=1)
                                .sum().shift(1).fillna(0).to_numpy() / tick)
        # staleness: grid steps since the mid last moved
        chg = np.where(dm != 0)[0]
        stale = np.zeros(len(md))
        last = -1
        for i in range(len(md)):
            if dm[i] != 0:
                last = i
            stale[i] = i - last if last >= 0 else i
        sub["stale"] = stale

        # ---- label: does the mid move >= jump_ticks within horizon? ----
        h = int(horizon_s * 1000 / GRID_MS)
        fut = np.minimum(np.arange(len(md)) + h, len(md) - 1)
        # Max absolute excursion over [t, t+H] INCLUSIVE -- a quote is picked
        # off by the path, not by where the price ends up.
        #
        # The window has to contain the endpoint md[i+h]. The original
        # rolling(h).shift(-h+1) spanned only i..i+h-1, so it covered 24
        # forward steps instead of 25 and excluded the endpoint entirely,
        # which let |signed_ticks| exceed the supposed path maximum. Part of
        # what looked like an "AUC ranks well but P&L ranks badly" objective
        # mismatch was this off-by-one: the label and the P&L were measuring
        # windows that did not even contain each other.
        fwd_max = pd.Series(md).rolling(h + 1, min_periods=1).max().shift(-h)
        fwd_min = pd.Series(md).rolling(h + 1, min_periods=1).min().shift(-h)
        exc = np.maximum((fwd_max.to_numpy() - md), (md - fwd_min.to_numpy()))
        y = (exc >= jump_ticks * tick).astype(np.int8)
        signed = (md[fut] - md) / tick

        valid = np.arange(len(md)) < len(md) - h
        sub["y"] = y
        sub["signed_ticks"] = signed
        # The path max, persisted. `y` was always defined on this -- a resting
        # quote is picked off by the path, not by where the price ends up --
        # but only the endpoint (`signed_ticks`) used to be saved, so the P&L
        # evaluation in jump_model.py silently scored a different, smaller
        # quantity: path max averages 1.78x the endpoint. Label and P&L now
        # measure the same thing.
        sub["path_exc_ticks"] = exc / tick
        # how stale the carried book genuinely is, in ms
        sub["book_age_ms"] = (ts - d["ts_book"])[idx]
        sub["valid"] = valid
        sub["ts"] = t
        sub["series"] = series_id.cat.categories[sid]
        out_rows.append(sub)

        # ---- LOB tensor: prices in ticks from mid, sizes in log dollars ----
        # Mid-centering is deliberate -- it makes book shape comparable across
        # markets trading at 0.05 and at 0.95 -- but it erases the mid path
        # (see the module docstring). `mid` stays in `sub` for that reason;
        # a tensor consumer must read the path from there.
        bpt = (np.nan_to_num(BP[idx]) - md[:, None]) / tick
        apt = (np.nan_to_num(AP[idx]) - md[:, None]) / tick
        bsl = np.log1p(BS[idx] * np.nan_to_num(BP[idx]))
        asl = np.log1p(AS[idx] * np.nan_to_num(AP[idx]))
        tensors.append(np.stack([bpt, bsl, apt, asl], axis=1).astype(np.float32))
        meta.append(np.full(len(idx), sid))

    if not out_rows:
        raise SystemExit(
            f"no series survived the >=400-row minimum "
            f"({len(np.unique(codes))} series seen, longest "
            f"{max((int((codes == s).sum()) for s in np.unique(codes)), default=0)} "
            f"rows). Nothing to build.")
    Fall = pd.concat(out_rows, ignore_index=True)
    T = np.concatenate(tensors, axis=0)
    S = np.concatenate(meta, axis=0)
    return Fall, T, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--horizon-s", type=int, default=5)
    ap.add_argument("--jump-ticks", type=int, default=2)
    ap.add_argument("--max-fill-s", type=float, default=MAX_FILL_S,
                    help="longest silence to forward-fill. Must exceed the "
                         "deep models' lookback (40s) for samples to have a "
                         "contiguous window, but a session whose books update "
                         "every ~8s needs a tighter cap to fit in memory -- "
                         "and that tightness is itself the honest statement "
                         "that the session has little dense book activity")
    a = ap.parse_args()

    for src in a.raw:
        name = session_tag(src)
        fp = os.path.join(OUT, f"feat_{name}.parquet")
        tp = os.path.join(OUT, f"lob_{name}.npy")
        if os.path.exists(fp) and os.path.exists(tp):
            log(f"skip {name}")
            continue
        log(f"=== {src} ===")
        d = extract(src, max_fill_s=a.max_fill_s)
        F, T, S = build_features(d, horizon_s=a.horizon_s,
                                 jump_ticks=a.jump_ticks)
        F["sid"] = S
        F.to_parquet(fp, index=False)
        np.save(tp, T)
        log(f"  features {F.shape}, tensor {T.shape}")
        v = F[F.valid]
        log(f"  jump base rate: {v.y.mean():.4f}  "
            f"({v.y.sum():,} of {len(v):,} samples)")
        # Forward-fill audit. Every row now carries a book from its own past,
        # so this is the honest measure of how much of the grid is real
        # observation and how much is a held-over book.
        age = v.book_age_ms
        log(f"  book age ms: p50 {age.median():,.0f} p90 "
            f"{age.quantile(0.90):,.0f} p99 {age.quantile(0.99):,.0f} "
            f"max {age.max():,.0f}")
        log(f"  rows carrying a book older than the {a.horizon_s}s horizon: "
            f"{(age > a.horizon_s * 1000).mean():.1%}")
        # The grid is 200ms but these books do not update every 200ms, so most
        # rows are held-over state. This ratio is the real oversampling factor
        # and it bounds how many independent observations the file contains.
        log(f"  grid is {GRID_MS}ms; median book age {age.median():,.0f}ms "
            f"=> ~{max(age.median(), 1) / GRID_MS:.0f}x oversampled")
        log(f"  excursion ticks: endpoint mean "
            f"{v.signed_ticks.abs().mean():.3f}  path-max mean "
            f"{v.path_exc_ticks.mean():.3f}")
        log(f"  by market type:")
        v = v.copy()
        v["mt"] = v.series.str.split("|").str[1]
        print(v.groupby("mt").y.agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
