"""Forward test: take a freshly recorded slate and report what the models earn.

Runs the whole chain unattended, on games that did not exist when the models
were trained:

    1. wait for the recording       collect_days.py (plan -> record -> verify)
    2. health check                 the manifest verdict must not be FAIL
    3. build the dataset            jump_data.py, with a fill cap chosen from
                                    the session's own message rate
    4. score the magnitude model    the one that survived its controls
    5. score the directional model  the one that did not
    6. walk the clock               simulate_taker.py -> dollar P&L

Why this matters more than any number produced so far: every earlier result
was fit and evaluated inside one recording, split only by time, with 79 of 85
series on both sides. The diagnostics found 93.5% of taker P&L coming from 10
of 49 series with only 15 profitable, and a +0.112 train-to-test AUC gap. A
temporal split cannot see either problem. New games can.

Expectations, recorded here so the result is not read as a surprise:

  * The DIRECTIONAL model is expected to fail. 35% of its 5s move lands
    within 200ms, and on the 2026-09-09 holdout it made +$293 only at zero
    latency (with ~100% drawdown), -$17 at 200ms, and lost the whole bankroll
    at 1s. The simulator sweeps entry delay so this is visible rather than
    assumed.
  * The MAGNITUDE model is the live question. It held ~0.88 AUC within a
    single market type and improved monotonically with touch depth, reaching
    0.9608 on books backed by over $5,000, with a CI above zero in every
    depth bucket. Whether that survives on new games is the thing worth
    learning.

    python forward_test.py --hkt-date 2026-09-11
    python forward_test.py --session-tag 2026-09-10 --skip-collect
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(BASE), ".venv", "Scripts", "python.exe")
LIVE = os.path.join(BASE, "data", "live")
JD = os.path.join(BASE, "data", "jump")
LOGS = os.path.join(BASE, "logs")


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "forward_test.txt"), "a",
              encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(args, label, timeout=None):
    log(f"--- {label}: {' '.join(str(x) for x in args[1:])}")
    p = subprocess.run([str(x) for x in args], cwd=BASE, timeout=timeout)
    log(f"--- {label}: exit {p.returncode}")
    return p.returncode


def newest_books(before=None):
    """Most recent recording, raw or archived, optionally newer than `before`.

    Both extensions are searched because run_slate.py compresses each slate
    when compression is enabled, and compress_raw.py deletes the original once
    it has proven a bit-for-bit round-trip -- so a finished slate may exist
    only as books_<tag>.jsonl.xz.
    """
    cands = []
    for pat in ("books_*.jsonl", "books_*.jsonl.xz"):
        for f in glob.glob(os.path.join(LIVE, pat)):
            m = os.path.getmtime(f)
            if before is None or m > before:
                cands.append((m, f))
    if not cands:
        return None
    return max(cands)[1]


def message_rate(path, sample_lines=200_000):
    """Rough messages/second, used to pick the forward-fill cap.

    A 200ms grid oversamples a book that updates every ~8s by ~40x, and one
    session expanded 2.8M messages into 22.6M grid rows and would not fit in
    memory. Sparse sessions therefore need a tighter cap.
    """
    import orjson
    from jump_data import open_recording
    first = last = None
    n = 0
    with open_recording(path) as fh:
        for raw in fh:
            n += 1
            if n > sample_lines:
                break
            try:
                ts = orjson.loads(raw).get("ts")
            except Exception:
                continue
            if ts:
                first = ts if first is None else first
                last = ts
    if not (first and last) or last <= first:
        return None
    return n / ((last - first) / 1000.0)


def one_night(a, hkt_date=None, session_tag_arg=None):
    """Record (or reuse) one slate, then build, score, simulate and archive.

    Returns the session tag on success, None on failure. Failures are logged
    and returned rather than raised so a bad night does not abort the rest of
    the schedule.
    """
    t_start = time.time()
    log("=" * 78)
    log(f"NIGHT {hkt_date or session_tag_arg}")
    log("=" * 78)

    # ---- 1. record --------------------------------------------------------
    if session_tag_arg:
        books = next((p for p in (
            os.path.join(LIVE, f"books_{session_tag_arg}.jsonl"),
            os.path.join(LIVE, f"books_{session_tag_arg}.jsonl.xz"))
            if os.path.exists(p)), None)
        if not books:
            log(f"  no recording for tag '{session_tag_arg}'")
            return None
    else:
        # --no-compress deliberately: building from the raw file is far faster
        # than streaming 45 GB through lzma. The slate is archived at the END
        # of this function instead, which reclaims the disk just as well.
        # The timeout must cover the WAIT as well as the recording.
        # collect_days sleeps until 45 min before first pitch, which can be up
        # to ~24h away when a night finishes in the morning and the next slate
        # starts late that evening. The first version allowed duration + 8h =
        # 22h, so nights 2-5 of the 2026-09-11 schedule were killed mid-
        # recording at exactly 79,200s. The recordings survived only because
        # live_recorder is a grandchild and outlived the kill -- three full
        # slates sat unprocessed on disk as a result.
        #
        # run_slate and live_recorder both enforce their own duration limits,
        # so this outer timeout is only a backstop against a wedged process.
        # It should be generous: up to 24h of sleep + the recording + settle
        # and archive.
        rc = run([PY, "-u", "collect_days.py", "--hkt-dates", hkt_date,
                  "--duration-hours", a.duration_hours, "--no-compress"],
                 "collect", timeout=(a.duration_hours + 30) * 3600)
        if rc != 0:
            log("  collection failed; skipping this night")
            return None
        books = newest_books(before=t_start)
        if not books:
            log("  collection produced no new recording; skipping")
            return None

    sys.path.insert(0, BASE)
    from jump_data import session_tag as _stag
    tag = _stag(books).replace("books_", "", 1)
    log(f"recording: {books}  ({os.path.getsize(books)/1e9:.2f} GB"
        f"{' archived' if books.endswith('.xz') else ''}, tag '{tag}')")

    # ---- 2. health --------------------------------------------------------
    hp = books + ".health.json"
    if not os.path.exists(hp):
        run([PY, "-u", "verify_recording.py", books, "--json", hp],
            "verify", timeout=4 * 3600)
    if os.path.exists(hp):
        with open(hp, encoding="utf-8") as fh:
            rep = json.load(fh)[0]
        log(f"health: {rep['verdict']}   usable_for={rep.get('usable_for')}")
        for f in rep.get("fails", []):
            log(f"  [FAIL] {f}")
        for w in rep.get("warns", []):
            log(f"  [warn] {w}")
        if rep["verdict"] == "FAIL":
            log("  recording failed its health check; not scoring it")
            return None

    # ---- 3. build ---------------------------------------------------------
    feat = os.path.join(JD, f"feat_books_{tag}.parquet")
    if os.path.exists(feat):
        log(f"dataset already built: {os.path.basename(feat)}")
    else:
        cap = a.max_fill_s
        if cap is None:
            r = message_rate(books)
            cap = 60.0 if (r or 0) > 300 else (30.0 if (r or 0) > 100 else 15.0)
            log(f"message rate ~{(r or 0):,.0f}/s -> forward-fill cap {cap:.0f}s")
        rc = run([PY, "-u", "jump_data.py", "--raw", books,
                  "--max-fill-s", cap], "jump_data", timeout=6 * 3600)
        if rc != 0 or not os.path.exists(feat):
            log("  dataset build failed; skipping this night")
            return None

    # ---- 4/5. score -------------------------------------------------------
    for m in a.models:
        if run([PY, "-u", "score_session.py", "--session", tag,
                "--model", m, "--min-backing", a.min_backing],
               f"score {m}", timeout=4 * 3600) != 0:
            log(f"  scoring {m} failed; continuing")

    # ---- 6. walk the clock ------------------------------------------------
    for m in a.models:
        if not os.path.exists(os.path.join(JD, f"takeredge_{m}_{tag}.npy")):
            log(f"no directional edges for {m}; magnitude models pick no side")
            continue
        run([PY, "-u", "simulate_taker.py", "--model", f"{m}_{tag}",
             "--session-file", tag, "--capital", a.capital,
             "--flat-stake", a.flat_stake, "--min-backing", a.min_backing],
            f"simulate {m}", timeout=3 * 3600)

    # ---- 7. archive -------------------------------------------------------
    # Now that the dataset is built, the raw file is redundant: ~42x smaller
    # archived, and compress_raw only deletes the original after proving a
    # bit-for-bit round-trip. Every reader handles .xz, so nothing downstream
    # breaks.
    if not a.no_archive and books.endswith(".jsonl"):
        run([PY, "-u", "compress_raw.py", books], f"archive {tag}",
            timeout=8 * 3600)

    log(f"NIGHT {tag} DONE in {(time.time()-t_start)/3600:.2f}h")
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hkt-dates", nargs="*", default=None,
                    help="slates to record and analyse, in order")
    ap.add_argument("--hkt-date", default=None,
                    help="single slate; omit with --skip-collect")
    ap.add_argument("--duration-hours", type=float, default=14.0)
    ap.add_argument("--skip-collect", action="store_true",
                    help="use an existing recording instead of recording one")
    ap.add_argument("--session-tag", default=None,
                    help="with --skip-collect: the books_<tag>.jsonl to use")
    ap.add_argument("--max-fill-s", type=float, default=None,
                    help="override the automatic forward-fill cap")
    ap.add_argument("--models", nargs="*",
                    default=["cnn_depth_raw", "cnn_direction"])
    ap.add_argument("--capital", type=float, default=50.0)
    ap.add_argument("--flat-stake", type=float, default=5.0)
    ap.add_argument("--min-backing", type=float, default=200.0)
    ap.add_argument("--no-archive", action="store_true",
                    help="keep the raw recording instead of compressing it "
                         "after the dataset is built")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)

    log("=" * 78)
    log("FORWARD TEST")
    log("=" * 78)

    dates = list(a.hkt_dates or ([a.hkt_date] if a.hkt_date else []))
    if a.skip_collect:
        if not a.session_tag:
            raise SystemExit("--skip-collect needs --session-tag")
        done = [one_night(a, session_tag_arg=a.session_tag)]
    else:
        if not dates:
            raise SystemExit("pass --hkt-dates, --hkt-date, or --skip-collect")
        log(f"schedule: {len(dates)} night(s) -- {', '.join(dates)}")
        log(f"models: {', '.join(a.models)}")
        done = []
        for d in dates:
            # a failed night must not abort the schedule -- the remaining
            # slates are independent and each is a separate chance to see
            # whether the models generalise
            try:
                done.append(one_night(a, hkt_date=d))
            except Exception as e:
                log(f"NIGHT {d} raised {type(e).__name__}: {e}")
                done.append(None)

    ok = [t for t in done if t]
    log("=" * 78)
    log(f"ALL NIGHTS DONE: {len(ok)} of {len(done)} produced results")
    for t in ok:
        log(f"  {t}")
    log(f"  P&L summaries    data/jump/sim_summary_*.json")
    log(f"  OOS scores       data/jump/oos_*.json")
    log(f"  magnitude table  data/jump/model_comparison.csv")
    log(f"  this log         logs/forward_test.txt")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
