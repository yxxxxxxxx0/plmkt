"""
Cleans a Kalshi recording after it has been downloaded: removes the
one-reply flickers caused by REST replies arriving out of order. The raw
files are never modified; the result is written beside them.

Why this is possible offline: the recorder writes a book only when it
changes (in arrival order), plus one kpolls line per reply, also in arrival
order. So every reply's full book can be rebuilt -- a ticker's book at a
reply is its last written change -- and any rule applied afterwards.

The rule: a book state counts only once TWO CONSECUTIVE replies show it.
Measured on live matches (2026-10-18) this cuts A->B->A flickers from
~15% of changes to under 5% (the rest is plausibly real quote churn), at
the cost of ignoring states that last only one poll (~100 ms).

Each cleaned record carries:
    t_first_send, t_first_recv  the first reply that showed the state --
                                the state existed at some moment in
                                [t_first_send, t_first_recv]
    t_confirm_recv              when the second reply confirmed it
Use t_first_recv as the state's time in analysis; t_confirm_recv is when a
live system could have acted on it.

Usage:
    python clean_kalshi.py ~/plmkt-recordings/2026-10-19_0000_football
    python clean_kalshi.py data/outbox/test4h --confirm 3
Writes kalshi/kbooks_clean_<tag>.jsonl.xz in that session folder.
"""
import argparse
import collections
import glob
import json
import lzma
import os


def _open(path):
    return lzma.open(path, "rt", encoding="utf-8") if path.endswith(".xz") else open(path, encoding="utf-8")


def load(kbooks_path, kpolls_path):
    """Returns (replies, books, meta):
    replies  successful replies in arrival order: [{poll_id, t_send, t_recv}]
    books    poll_id -> {ticker: (bids, asks)} for the tickers that changed
    meta     ticker -> {event, outcome}"""
    books = collections.defaultdict(dict)
    meta = {}
    with _open(kbooks_path) as f:
        for line in f:
            r = json.loads(line)
            books[r["poll_id"]][r["ticker"]] = (r["bids"], r["asks"])
            meta[r["ticker"]] = {"event": r["event"], "outcome": r["outcome"]}
    replies = []
    with _open(kpolls_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "poll" and r.get("status") == 200:
                replies.append(r)
    return replies, books, meta


def confirmed(replies, books, n=2):
    """Yields (ticker, state, first_reply, confirm_reply) for every state that
    n consecutive replies agree on, in arrival order."""
    cur = {}                                   # ticker -> current raw state
    run = {}                                   # ticker -> (state, first_reply, count)
    accepted = {}                              # ticker -> last accepted state
    for rep in replies:
        for tk, st in books.get(rep["poll_id"], {}).items():
            cur[tk] = st
        for tk, st in cur.items():
            prev = run.get(tk)
            if prev and prev[0] == st:
                run[tk] = (st, prev[1], prev[2] + 1)
            else:
                run[tk] = (st, rep, 1)
            s, first, cnt = run[tk]
            if cnt == n and accepted.get(tk) != s:
                accepted[tk] = s
                yield tk, s, first, rep


def clean_session(folder, n=2):
    kb = glob.glob(os.path.join(folder, "**", "kbooks_*.jsonl*"), recursive=True)
    kb = [p for p in kb if "kbooks_clean_" not in p]
    if not kb:
        raise SystemExit(f"no kbooks file under {folder}")
    kb = kb[0]
    tag = os.path.basename(kb).split("kbooks_", 1)[1].split(".jsonl")[0]
    kp = glob.glob(os.path.join(os.path.dirname(kb), f"kpolls_{tag}.jsonl*"))
    if not kp:
        raise SystemExit(f"no kpolls_{tag} beside {kb}")
    replies, books, meta = load(kb, kp[0])
    out = os.path.join(os.path.dirname(kb), f"kbooks_clean_{tag}.jsonl.xz")
    raw_changes = sum(len(v) for v in books.values())
    kept = 0
    with lzma.open(out, "wt", encoding="utf-8", preset=3) as f:
        for tk, (bids, asks), first, conf in confirmed(replies, books, n):
            kept += 1
            f.write(json.dumps({
                "ticker": tk, **meta[tk],
                "t_first_send": first["t_send"], "t_first_recv": first["t_recv"],
                "t_confirm_recv": conf["t_recv"], "poll_id": first["poll_id"],
                "bids": bids, "asks": asks,
            }, separators=(",", ":")))
            f.write("\n")
    print(f"{tag}: {len(replies):,} replies, {raw_changes:,} raw changes -> "
          f"{kept:,} confirmed states (n={n})")
    print(f"  -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="session folder(s), e.g. ~/plmkt-recordings/<tag>")
    ap.add_argument("--confirm", type=int, default=2, help="consecutive replies required")
    a = ap.parse_args()
    for s in a.sessions:
        clean_session(os.path.expanduser(s), a.confirm)


if __name__ == "__main__":
    main()
