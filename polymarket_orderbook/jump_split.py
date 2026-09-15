"""Shared data loading, splitting and economic scoring for the jump study.

Split
-----
Three temporal blocks with a discarded gap on both interior boundaries, since
labels look H seconds forward and adjacent grid rows overlap almost entirely:

    |------ train ------|gap|-- val --|gap|------ test ------|

Validation exists for one reason: to choose the quote/no-quote threshold. The
first version of this study had no validation block at all and selected the
threshold by argmax P&L over 21 grid points **on the test set**, so every
reported gain was a best-of-21 on the evaluation data -- the trivial
single-feature gates included, which is what made them look competitive with
the learned models. Test is now touched exactly once per model, at a threshold
fixed beforehand.

What a "sample" is worth
------------------------
A 5s label on a 200ms grid means consecutive rows share 24/25 of their forward
window. 9.5M rows is roughly 380k independent observations, so `n_eff` is
reported alongside n and every t-like quantity should be read against it, not
against the row count.

Economics
---------
A maker quoting both sides at the touch is filled on whichever side the market
moves toward, so per quote:

    pnl(t) = half_spread(t) - max_{s<=H} |mid(t+s) - mid(t)|

The excursion is the path maximum, not the endpoint change: a resting quote is
picked off by the path. `jump_data.py` now persists it (`path_exc_ticks`);
before that only the endpoint was saved and the P&L silently scored a quantity
averaging 1.78x smaller than the one the label was defined on.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.join(BASE, "data", "jump")
TICK = 0.01
GRID_MS = 200

HAND = ["spread_ticks", "imb1", "imb3", "imb5", "imb10",
        "log_bid_usd", "log_ask_usd", "conc_bid", "conc_ask",
        "reach_bid", "reach_ask", "microprice_dev",
        "rv_5", "rv_25", "rv_150", "ofi_5", "ofi_25",
        "dmid_5", "dmid_25", "stale"]


# --------------------------------------------------------------------------- #
# artifact persistence
#
# The first run's CNN-Transformer result was lost outright: no checkpoint was
# saved, and model_comparison.csv was written once at the end of main(), so a
# later --skip-deep run overwrote the only record of it with a table that did
# not contain it. Rows are now merged by model name as each model finishes.
# --------------------------------------------------------------------------- #
def save_row(row: dict, path=None):
    path = path or os.path.join(JD, "model_comparison.csv")
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old.model != row["model"]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return path


def save_preds(name: str, p: np.ndarray, idx: np.ndarray):
    np.save(os.path.join(JD, f"preds_{name}.npy"), p.astype(np.float32))
    np.save(os.path.join(JD, f"testidx_{name}.npy"), idx)


class TensorView:
    """Indexes the on-disk LOB tensor in feature-frame row space.

    The obvious `T = T[keep]` materialises a ~1.4GB filtered copy into process
    memory, and on a 15.6GB machine that was enough to get training killed
    before the first epoch finished. Held as a memmap instead, the same pages
    live in the OS page cache: shared, evictable under pressure, and just as
    fast once warm, because a batch touches only B x L scattered rows.

    Indexing accepts the same (B, L) integer arrays the batchers build, and
    remaps them through `idx` so callers keep working in F's row space.
    """

    def __init__(self, T, idx):
        self._T = T
        self._idx = np.asarray(idx, np.int64)
        self.shape = (len(self._idx),) + tuple(T.shape[1:])
        self.dtype = T.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, J):
        return np.asarray(self._T[self._idx[J]])

    def channel(self, ch, chunk=1_000_000):
        """One (N, K) channel, read in chunks so the peak stays bounded."""
        out = np.empty((len(self), self.shape[2]), np.float32)
        for s in range(0, len(self), chunk):
            e = min(s + chunk, len(self))
            out[s:e] = self._T[self._idx[s:e], ch, :]
        return out


# --------------------------------------------------------------------------- #
def group_split(F, ok, q_train=0.60, q_val=0.80, seed=0, log=print):
    """Split by MATCH, not by time. The fix for the overfitting we measured.

    The temporal split put 79 of 85 series on both sides, so the model saw the
    same games in training and test and only had to generalise across hours.
    It duly overfit: sign AUC 0.9001 on train against 0.7883 on test, a +0.112
    gap, with 93.5% of taker P&L coming from 10 of 49 series and only 15 of 49
    profitable. Neither number is visible to a temporal split.

    Holding out whole matches makes the question the honest one: does this work
    on a game it has never seen? Matches are assigned to blocks by a hash of
    the slug, so the split is deterministic and stable as sessions are added.
    """
    slug = F.series.astype(str).str.split("|").str[0].to_numpy()
    uniq = np.array(sorted(set(slug[ok])))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_tr = int(len(uniq) * q_train)
    n_va = int(len(uniq) * q_val)
    tr_s = set(uniq[order[:n_tr]])
    va_s = set(uniq[order[n_tr:n_va]])
    te_s = set(uniq[order[n_va:]])
    log(f"  group split by match: {len(tr_s)} train / {len(va_s)} val / "
        f"{len(te_s)} test matches, disjoint")
    return (ok & np.isin(slug, list(tr_s)),
            ok & np.isin(slug, list(va_s)),
            ok & np.isin(slug, list(te_s)))


def load(market_types=None, lookback=1, gap_s=120, target="jump",
         q_train=0.60, q_val=0.80, max_book_age_s=5.0, sessions=None,
         in_game_only=False, drop_broken=False, split="time",
         max_rows_per_session=None, use_trimmed=False, log=print):
    """Load the jump dataset.

    `sessions` restricts which feat_*.parquet files are read (substring
    match). Worth using whenever you only need one session: with a single
    unfiltered file the tensor stays a memmap via TensorView, whereas
    concatenating several sessions -- or filtering/subsampling one -- has to
    materialise the kept rows into RAM. This machine has 15.6GB total and has
    OOM'd loading 4 full sessions (7.51GB for the tensor alone) more than
    once, so `max_rows_per_session` exists to bound that: filtering happens
    PER FILE before concatenation, so trimming to kickoff..final-out (which
    drops 53-83% of rows on its own) or capping rows shrinks what needs to be
    concatenated instead of shrinking it after the expensive part already
    happened.
    """
    if use_trimmed:
        # Read the small caches trim_sessions.py already produced (one file
        # at a time, filtered, freed) instead of the full raw parquets. This
        # is the memory-safe path: reading several full feat_*.parquet at
        # once measured free RAM down to 0.2GB on this machine even before
        # any tensor concat, whereas the trimmed files are already 5-8x
        # smaller (in-game trimming alone drops 53-83% of rows).
        fs = sorted(glob.glob(os.path.join(JD, "feat_*_trimmed.parquet")))
        if not fs:
            raise SystemExit(
                "no trimmed caches found -- run trim_sessions.py first")
        in_game_only = drop_broken = False   # already applied when trimmed
    else:
        fs = sorted(f for f in glob.glob(os.path.join(JD, "feat_*.parquet"))
                   if "_trimmed" not in f)
    if sessions:
        want = [s for s in ([sessions] if isinstance(sessions, str)
                            else sessions)]
        fs = [f for f in fs if any(w in os.path.basename(f) for w in want)]
        if not fs:
            raise SystemExit(f"no feat file matches {want}")
    if not fs:
        raise SystemExit("run jump_data.py first")
    # Filter and subsample EACH file BEFORE concatenating, not after. Reading
    # all sessions in full and then filtering (the previous version) needs
    # every session's tensor resident in RAM at once -- 4 sessions hit 7.51GB
    # for the tensor alone, and this machine has crashed under that load more
    # than once (an OOM here, a killed forward_test job there, a crashed
    # editor here). in-game trimming alone drops 53-83% of rows per session,
    # so applying it first shrinks what ever needs to be concatenated instead
    # of shrinking it after the expensive part already happened.
    from match_filter import masks as _match_masks

    import gc

    parts, tvecs, tmaps = [], [], []
    off = 0
    any_filtered = False
    for f in fs:
        p = pd.read_parquet(f)
        n0 = len(p)
        # series arrives as a mix of dtypes across files (object in one
        # parquet, category in another), which breaks .str operations on the
        # merged column with an opaque overflow deep inside pandas' type
        # inference. Force it to plain string per-file, before any concat.
        p["series"] = p["series"].astype(str)
        p["mt"] = p["series"].str.split("|").str[1]

        # Downcast float64 -> float32 and series -> category IMMEDIATELY per
        # file, before this file's DataFrame is held alongside the others in
        # `parts` for the eventual concat. Doing this only after the concat
        # (the previous version) meant every session's frame sat at full
        # float64 width simultaneously -- 4 sessions of even the already
        # in-game-trimmed caches (2.38GB on disk) drove free RAM to 0.22GB,
        # because the resting pandas representation of 29 float64 columns
        # runs close to 2x the on-disk (already-compressed) parquet size.
        keep_int = {"ts", "ts_book", "sid", "pos", "y", "y_jump", "y_econ"}
        for c in p.columns:
            if c in keep_int or p[c].dtype.name in ("category", "object", "bool"):
                continue
            if p[c].dtype == np.float64:
                p[c] = p[c].astype(np.float32)
        p["series"] = p["series"].astype("category")

        # keep_idx tracks, at every step, which ORIGINAL row (0..n0-1) each
        # surviving row of `p` came from -- required to gather the matching
        # tensor rows correctly. Each filter below is applied as a boolean
        # mask over the CURRENT `p` and used to subset keep_idx in the same
        # step, before p itself is reduced, so the two never drift apart.
        keep_idx = np.arange(n0)
        if market_types:
            m = p["mt"].isin(market_types).to_numpy()
            p = p[m].reset_index(drop=True)
            keep_idx = keep_idx[m]
        if in_game_only or drop_broken:
            keep, _rep = _match_masks(p, in_game_only=in_game_only,
                                      drop_broken=drop_broken, log=log)
            p = p[keep].reset_index(drop=True)
            keep_idx = keep_idx[keep]
        if max_rows_per_session and len(p) > max_rows_per_session:
            # Cap PER MATCH (per sid), keeping each match's rows as one
            # CONTIGUOUS prefix -- not a random sample of individual rows.
            # The deep models need `lookback` (200) contiguous grid steps of
            # history per sample, verified later by an exact-timestamp-
            # spacing check. A random single-row sample very rarely has 200
            # of its neighbours also survive the same random draw, so nearly
            # every row fails that check -- this is what silently produced
            # 0 usable rows the first time this was tried, not a crash but a
            # wrong answer. Truncating each match's own contiguous block
            # keeps full history within the kept portion and represents
            # every match evenly, which random subsampling of the whole file
            # would not (this parquet stores each match's rows as one
            # contiguous run, not time-interleaved across matches).
            per_sid_cap = max(max_rows_per_session // max(p["sid"].nunique(), 1),
                              lookback + 50)
            keep_mask = np.zeros(len(p), bool)
            for _, idxs in p.groupby("sid", sort=False).groups.items():
                a = np.asarray(idxs)
                keep_mask[a[:per_sid_cap]] = True
            p = p[keep_mask].reset_index(drop=True)
            keep_idx = keep_idx[keep_mask]

        # sid is per-file and parquet stores it narrowly (int8 when a session
        # has under 128 series), so offsetting across sessions overflows
        # unless it is widened first.
        p["sid"] = p["sid"].to_numpy().astype(np.int64) + off
        off = int(p["sid"].max()) + 1 if len(p) else off

        tm = np.load(f.replace("feat_", "lob_").replace(".parquet", ".npy"),
                    mmap_mode="r")
        trivial = len(keep_idx) == n0 and np.array_equal(
            keep_idx, np.arange(n0))
        any_filtered = any_filtered or not trivial
        tmaps.append((tm, keep_idx, trivial))
        parts.append(p)
        log(f"  {os.path.basename(f)}: kept {len(p):,} of {n0:,} rows")

    F = pd.concat(parts, ignore_index=True)
    F["mt"] = F["mt"].astype("category")

    if len(fs) == 1 and not any_filtered:
        # the common case: one session, nothing dropped. Keep the tensor as a
        # memmap so its pages live in the evictable OS cache instead of
        # process RSS -- see TensorView's docstring for why this matters.
        T = TensorView(tmaps[0][0], np.arange(len(F)))
        log(f"  1 session(s) -> {len(F):,} rows (tensor stays memory-mapped)")
    else:
        # filtered and/or multi-session: gather only the KEPT rows from each
        # file's memmap. Indexing a memmap with an integer array copies just
        # those rows rather than the whole file, so a session trimmed to 20%
        # of its rows costs 20% of its tensor size in RAM, not 100% -- this is
        # what makes in-game trimming a genuine memory fix rather than only a
        # data-quality one.
        tvecs = [np.asarray(tm[idx]) for tm, idx, _ in tmaps]
        T = tvecs[0] if len(tvecs) == 1 else np.concatenate(tvecs, axis=0)
        log(f"  {len(fs)} session(s) -> {len(F):,} rows total "
            f"({T.nbytes/1e9:.2f} GB tensor materialised)")
        T = TensorView(T, np.arange(len(F)))
    assert len(F) == len(T), f"feature/tensor length mismatch {len(F)} vs {len(T)}"

    # F is ~2GB at float64 across 29 columns and every model reads it while
    # the tensor pages are also live. float32 is ample for book features and
    # halves it; ts/sid must stay integral.
    keep_int = {"ts", "ts_book", "sid", "pos", "y", "y_jump", "y_econ"}
    for c in F.columns:
        if c in keep_int or F[c].dtype.name in ("category", "object", "bool"):
            continue
        if F[c].dtype == np.float64:
            F[c] = F[c].astype(np.float32)
    # the per-row series label is an 8.6M-entry object column used only to
    # derive `mt`, which is already extracted above
    if "series" in F.columns:
        F["series"] = F["series"].astype("category")

    if "path_exc_ticks" not in F.columns:
        raise SystemExit(
            "this dataset predates the path-max excursion fix -- rebuild with "
            "jump_data.py before scoring anything economic")

    F["half_spread"] = F.spread_ticks / 2.0
    F["excursion"] = F.path_exc_ticks             # path max: what a quote pays
    F["excursion_endpoint"] = F.signed_ticks.abs()   # kept for comparison only

    # The economic label: was quoting actually unprofitable? That is the
    # decision the strategy makes; "did the mid move 2 ticks" is a proxy for it.
    F["y_jump"] = F.y.astype(np.int8)
    F["y_econ"] = (F.excursion > F.half_spread).astype(np.int8)
    F["y"] = F[{"jump": "y_jump", "econ": "y_econ"}[target]]

    # a sample needs `lookback` steps of history from the SAME series
    F["pos"] = F.groupby("sid").cumcount()
    ok = F.valid.to_numpy() & (F.pos.to_numpy() >= lookback)

    # Window contiguity. jump_data.py caps how long a silence is forward
    # filled, so the slot sequence within a series can contain gaps. The deep
    # models window backwards by ROW INDEX and assume each step is one grid
    # interval, so a sample spanning a gap would silently mix timescales.
    #
    # Slots within a series are strictly increasing multiples of the grid, so
    # the whole window is contiguous exactly when the endpoints differ by
    # lookback grid steps -- any gap makes the difference larger.
    if lookback > 0:
        ts_all = F.ts.to_numpy()
        prev = np.full(len(F), -1, np.int64)
        prev[lookback:] = ts_all[:-lookback]
        contig = ok & (ts_all - prev == lookback * GRID_MS)
        log(f"  window contiguity: {contig.sum():,} of {ok.sum():,} eligible "
            f"samples have an unbroken {lookback}-slot "
            f"({lookback * GRID_MS / 1000:.0f}s) history")
        ok = contig

    # Staleness eligibility. The 200ms grid vastly oversamples these books --
    # median carried book age is ~4s on one session, and the Sep 9 session
    # expands 2.8M messages into 22.6M grid rows because a market silent for
    # 500s still emits 2,500 slots. Rows whose book has been unobserved for
    # longer than the horizon we predict over are not observations of anything
    # current, and there are enough of them to dominate the objective.
    #
    # This restricts which rows may serve as PREDICTION POINTS. It must not
    # drop rows from F or T: the deep models window backwards by row index and
    # assume slots are contiguous 200ms apart, so removing rows would silently
    # bend the time axis -- the same class of error as the back-fill bug.
    if max_book_age_s is not None and "book_age_ms" in F.columns:
        fresh = F.book_age_ms.to_numpy() <= max_book_age_s * 1000
        log(f"  staleness filter: {fresh.mean():.1%} of rows carry a book "
            f"<= {max_book_age_s}s old; eligible samples "
            f"{ok.sum():,} -> {(ok & fresh).sum():,}")
        ok = ok & fresh
    elif max_book_age_s is not None:
        log("  no book_age_ms column (pre-fix dataset); staleness filter off")

    # NOTE: in-game trimming and broken-match exclusion already happened
    # PER FILE, before concatenation (see the load loop above) -- doing it
    # here as well would be redundant (every row is already 100% in-game by
    # this point) and, worse, would re-materialise the whole `series` column
    # again. Left as a comment rather than silently removed so it is obvious
    # this is deliberate, not an oversight.

    ts = F.ts.to_numpy()
    if split == "group":
        tr, va, te = group_split(F, ok, q_train, q_val, log=log)
    else:
        c1, c2 = F.ts.quantile(q_train), F.ts.quantile(q_val)
        g = gap_s * 1000
        tr = ok & (ts < c1 - g)
        va = ok & (ts >= c1) & (ts < c2 - g)
        te = ok & (ts >= c2)

    h = int(5000 / GRID_MS)
    log(f"loaded {len(F):,} rows from {len(fs)} session(s); "
        f"usable {ok.sum():,} (lookback {lookback}); target={target}")
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        log(f"  {nm:<5} {m.sum():>9,} rows (~{m.sum()//h:>7,} independent)  "
            f"base rate {F.y.to_numpy()[m].mean():.4f}  "
            f"series {F.sid[m].nunique()}")
    log(f"  discarded to gaps: {(ok & ~tr & ~va & ~te).sum():,} rows "
        f"({gap_s}s either side of each boundary)")
    return F, T, np.where(tr)[0], np.where(va)[0], np.where(te)[0]


def subsample(idx, n, seed=0):
    if n is None or len(idx) <= n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, n, replace=False))


def pnl_weights(F, idx, cap=20.0):
    """How much the quote/no-quote decision is worth on each sample.

    Quoting earns half_spread - excursion, so |half_spread - excursion| is the
    regret of getting the decision wrong. Samples where the two are close are
    nearly free either way and should not dominate the fit; samples where a
    quote loses several ticks should. Normalised to mean 1 so the loss scale
    matches an unweighted run, and capped because the excursion tail is heavy.
    """
    hs = F.half_spread.to_numpy()[idx] * TICK
    exc = F.excursion.to_numpy()[idx] * TICK
    w = np.abs(hs - exc)
    w = w / max(w.mean(), 1e-12)
    return np.clip(w, 0.0, cap).astype(np.float64)


# --------------------------------------------------------------------------- #
def econ_eval(p, F, idx, thresholds=None):
    """P&L per quote when the strategy pulls quotes above a jump threshold."""
    if thresholds is None:
        thresholds = np.arange(0.0, 1.001, 0.05)
    hs = F.half_spread.to_numpy()[idx] * TICK
    exc = F.excursion.to_numpy()[idx] * TICK
    pnl_if_quote = hs - exc
    base = float(pnl_if_quote.mean())
    rows = []
    for th in thresholds:
        quote = p < th
        n = int(quote.sum())
        if n == 0:
            rows.append(dict(threshold=float(th), quote_frac=0.0,
                             pnl_per_quote=0.0, pnl_per_opportunity=0.0,
                             adverse=np.nan, half_spread=np.nan, n_quotes=0))
            continue
        kept = pnl_if_quote[quote]
        rows.append(dict(
            threshold=float(th), quote_frac=float(quote.mean()),
            pnl_per_quote=float(kept.mean()),
            pnl_per_opportunity=float(kept.sum() / len(idx)),
            adverse=float(exc[quote].mean()),
            half_spread=float(hs[quote].mean()), n_quotes=n))
    R = pd.DataFrame(rows)
    R["vs_always_quote"] = R.pnl_per_quote - base
    return R, base


def bootstrap_pnl_opp(p, F, idx, th, n_boot=400, block=1000, seed=0):
    """Moving-block bootstrap CI on P&L per opportunity.

    Necessary rather than decorative. A 5s label on a 200ms grid leaves ~8,000
    independent observations in a 200k-row test block, and the effects here are
    fractions of a cent -- the difference between the best gate and the GBM is
    0.00115 vs 0.00060. Comparing those without an interval is exactly the
    mistake reports/FINDINGS.md is about: a +9.2% ROI that survived until
    someone checked it.

    Blocks are contiguous in the test index, which is time-ordered, so local
    dependence between overlapping labels is preserved inside a block.
    """
    hs = F.half_spread.to_numpy()[idx] * TICK
    exc = F.excursion.to_numpy()[idx] * TICK
    # contribution per OPPORTUNITY: zero when the gate declines to quote
    contrib = np.where(p < th, hs - exc, 0.0)
    n = len(contrib)
    if n < 2:
        return (float("nan"), float("nan"))
    block = min(block, max(1, n // 4))
    nb = max(1, n // block)
    starts = np.arange(0, max(1, n - block + 1))
    rng = np.random.default_rng(seed)
    off = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(starts, nb)
        boots[b] = contrib[(s[:, None] + off).ravel()].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate(name, p_val, p_te, F, va, te, out, persist=True, extra=None):
    """Choose the gating threshold on validation; report it on test.

    This is the only place validation is consumed, and the only place test is.
    """
    y_val = F.y.to_numpy()[va]
    y_te = F.y.to_numpy()[te]
    auc = lambda y, p: (roc_auc_score(y, p) if len(np.unique(y)) > 1
                        else float("nan"))

    Rv, base_v = econ_eval(p_val, F, va)
    bv = Rv.loc[Rv.pnl_per_opportunity.idxmax()]
    th = float(bv.threshold)

    Rt, base_t = econ_eval(p_te, F, te, thresholds=np.array([th]))
    r = Rt.iloc[0]

    h = int(5000 / GRID_MS)
    lo, hi = bootstrap_pnl_opp(p_te, F, te, th)
    # A gate that picked threshold 0 never quotes. That earns exactly zero,
    # which is better than quoting indiscriminately but is not a "gain" --
    # pnl_per_quote is undefined with no quotes, and reporting 0 for it made
    # never-quoting look like the best strategy in the table. pnl_per_opp is
    # the metric with a consistent denominator, and it is what the threshold
    # is selected on.
    degenerate = float(r.quote_frac) == 0.0
    row = dict(model=name,
               auc=auc(y_te, p_te), brier=brier_score_loss(y_te, p_te),
               base_rate=float(y_te.mean()), pnl_always=base_t,
               threshold=th, quote_frac=float(r.quote_frac),
               pnl_per_quote=float("nan") if degenerate else float(r.pnl_per_quote),
               pnl_per_opp=float(r.pnl_per_opportunity),
               pnl_opp_lo=lo, pnl_opp_hi=hi,
               beats_zero=bool(lo > 0),
               gain_per_quote=(float("nan") if degenerate
                               else float(r.pnl_per_quote) - base_t),
               never_quotes=degenerate,
               adverse=float("nan") if degenerate else float(r.adverse),
               n_test=len(te), n_eff_test=len(te) // h,
               val_auc=auc(y_val, p_val),
               val_pnl_opp=float(bv.pnl_per_opportunity))
    if extra:
        row.update(extra)
    out.append(row)
    if persist:
        save_row(row)
        save_preds(name, p_te, te)
    return row


def show(rows, title="MODEL COMPARISON"):
    R = pd.DataFrame(rows).sort_values("pnl_per_opp", ascending=False)
    cols = ["model", "auc", "brier", "threshold", "quote_frac",
            "pnl_always", "pnl_per_opp", "pnl_opp_lo", "pnl_opp_hi",
            "beats_zero", "pnl_per_quote", "adverse", "n_eff_test",
            "val_auc", "val_pnl_opp"]
    cols = [c for c in cols if c in R.columns]
    print("\n" + "=" * 130)
    print(f"{title}  (P&L in dollars per share)")
    print("  ranked by pnl_per_opp -- total P&L over ALL opportunities, the "
          "only denominator that is the same for every row")
    print("  threshold    chosen on VALIDATION, applied once to test")
    print("  pnl_always   quote every opportunity, no gating")
    print("  [lo, hi]     95% moving-block bootstrap CI on pnl_per_opp")
    print("  beats_zero   CI lower bound above 0, i.e. better than not "
          "quoting at all")
    print("=" * 130)
    print(R[cols].to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    nq = R[R.get("never_quotes", False) == True] if "never_quotes" in R else R.iloc[:0]
    if len(nq):
        print(f"\n  {len(nq)} model(s) chose never to quote on validation "
              f"({', '.join(nq.model)}): no threshold made gating profitable, "
              f"so the optimal action was to stay out. pnl_per_quote is blank "
              f"for those rather than 0.")
    return R
