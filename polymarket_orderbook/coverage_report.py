"""Do the built datasets cover the match, and what else are they carrying?

jump_data.py uses every recorded row. The recorder starts 45 minutes before
first pitch and runs until the MLB API reports every game Final, so a dataset
contains three regimes, not one:

    PREGAME    thin books, almost nothing trades
    IN-GAME    kickoff -> final out, from data/game_windows.json
    POSTGAME   the tail after the last out

That matters beyond tidiness. Rows where nothing can happen are free negatives
for any classifier: it gets credit for telling a dead book from a live one,
which is not tradeable and not what the model is supposed to be learning. The
per-market-type slice already showed pooled AUC collapsing from 0.88 to 0.67
once market mixing was removed. Pregame/postgame mixing is the same confound
along a different axis, and this measures how much of it each session carries.

It also answers the plain question directly: is the whole match present, or
did the recording start late or stop early?

    python coverage_report.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.join(BASE, "data", "jump")
WIN = os.path.join(BASE, "data", "game_windows.json")


def main():
    pd.set_option("display.width", 220)
    if not os.path.exists(WIN):
        raise SystemExit("data/game_windows.json missing")
    win = json.load(open(WIN, encoding="utf-8"))
    W = {k: (pd.Timestamp(v["start_utc"]).value // 10**6,
             pd.Timestamp(v["end_utc"]).value // 10**6)
         for k, v in win.items()
         if v.get("start_utc") and v.get("end_utc")}
    print(f"{len(W)} games have a kickoff/final-out window\n")

    for f in sorted(glob.glob(os.path.join(JD, "feat_*.parquet"))):
        d = pd.read_parquet(f, columns=["series", "ts", "y", "valid"])
        d["slug"] = d.series.astype(str).str.split("|").str[0]
        sess = os.path.basename(f)
        print("=" * 112)
        print(f"{sess}   {len(d):,} rows")
        print("=" * 112)

        rows = []
        for slug, g in d.groupby("slug", sort=True):
            w = W.get(slug)
            ts = g.ts.to_numpy()
            if w is None:
                rows.append(dict(slug=slug, n=len(g), window="MISSING",
                                 pre=np.nan, ingame=np.nan, post=np.nan,
                                 br_in=np.nan, br_out=np.nan,
                                 starts_before=np.nan, ends_after=np.nan))
                continue
            s, e = w
            pre = ts < s
            post = ts > e
            ing = ~pre & ~post
            y = g.y.to_numpy().astype(bool) & g.valid.to_numpy().astype(bool)
            rows.append(dict(
                slug=slug, n=len(g), window="ok",
                pre=pre.mean(), ingame=ing.mean(), post=post.mean(),
                br_in=(y[ing].mean() if ing.any() else np.nan),
                br_out=(y[~ing].mean() if (~ing).any() else np.nan),
                # negative = recording started AFTER first pitch (missed play)
                starts_before=(s - ts.min()) / 60000.0,
                ends_after=(ts.max() - e) / 60000.0))
        R = pd.DataFrame(rows)
        print(R.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

        ok = R[R.window == "ok"]
        if len(ok):
            tot = ok.n.sum()
            wpre = float((ok.pre * ok.n).sum() / tot)
            wing = float((ok.ingame * ok.n).sum() / tot)
            wpost = float((ok.post * ok.n).sum() / tot)
            print(f"\n  weighted: pregame {wpre:.1%}   in-game {wing:.1%}   "
                  f"postgame {wpost:.1%}")
            bi = float(np.nansum(ok.br_in * ok.ingame * ok.n)
                       / max(np.nansum(ok.ingame * ok.n), 1))
            bo = float(np.nansum(ok.br_out * (1 - ok.ingame) * ok.n)
                       / max(np.nansum((1 - ok.ingame) * ok.n), 1))
            print(f"  jump base rate IN-GAME {bi:.4f}   OUT-OF-GAME {bo:.4f}")
            if bo > 0 and bi > 0:
                print(f"  -> out-of-game rows are {bi/max(bo,1e-9):.1f}x less "
                      f"likely to jump; every one is a free negative")
            late = ok[ok.starts_before < 0]
            if len(late):
                print(f"  WARNING: {len(late)} match(es) recorded AFTER first "
                      f"pitch: {', '.join(late.slug)}")
            missing = R[R.window == "MISSING"]
            if len(missing):
                print(f"  {len(missing)} match(es) have no window entry "
                      f"(run fetch_game_windows.py --merge)")
        print()


if __name__ == "__main__":
    main()
