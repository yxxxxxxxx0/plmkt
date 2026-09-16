"""Kaggle-side targeted extraction for match-to-season lead/lag research.

The full daily files remain in Kaggle.  Output contains only main match markets
and longer-term team markets needed for the cross-market event study.
"""

from pathlib import Path
import gc
import re
import shutil

import polars as pl

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/cross_market_sports")

FUTURE_CONTEXT = re.compile(
    r"(?ix)(NBA|Eastern Conference|Western Conference|World Series|"
    r"American League|National League|AL East|AL Central|AL West|NL East|NL Central|NL West|"
    r"Premier League|Champions League|La Liga|Serie A|Bundesliga|Ligue 1|MLS|FA Cup|Carabao Cup|"
    r"ESL Pro League|Counter-Strike|LCK|LPL|LoL|League of Legends)"
)
FUTURE_FORM = re.compile(
    r"(?ix)^Will\s+.+?\s+(win|make|finish|reach|qualify|be relegated)\b"
)
PLAIN_MATCH = re.compile(r"(?i)^[^:\n?]+\s+vs\.?\s+[^:\n?]+$")
ESPORTS_MATCH = re.compile(
    r"(?i)^(Counter-Strike|LoL|League of Legends):\s+.+?\s+vs\.?\s+.+?\s+\(BO\d\)\s+-"
)
DATED_TEAM_WIN = re.compile(r"(?i)^Will\s+.+?\s+win on 2026-\d\d-\d\d\??$")


def locate() -> Path:
    roots = []
    for labels in INPUT.rglob("labels"):
        root = labels.parent
        if (labels / "market_targets.parquet").exists() and (root / "orderbook").is_dir():
            roots.append(root)
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one attached source dataset, found {roots}")
    return roots[0]


def classify(question: str) -> tuple[bool, str]:
    text = question or ""
    if FUTURE_FORM.search(text) and FUTURE_CONTEXT.search(text):
        return True, "long_term_team_market"
    if PLAIN_MATCH.search(text):
        return True, "plain_main_match"
    if ESPORTS_MATCH.search(text) and not re.search(r"(?i)Map \d|Handicap|Total", text):
        return True, "esports_main_match"
    if DATED_TEAM_WIN.search(text):
        return True, "football_team_win"
    return False, "excluded"


def main() -> None:
    root = locate()
    metadata = pl.read_parquet(root / "labels" / "market_targets.parquet")
    decisions = [classify(x) for x in metadata.get_column("question").fill_null("").to_list()]
    selected = metadata.with_columns(
        pl.Series("cross_market_selected", [x[0] for x in decisions]),
        pl.Series("cross_market_role", [x[1] for x in decisions]),
    ).filter(pl.col("cross_market_selected"))
    if selected.is_empty():
        raise RuntimeError("Candidate classifier selected zero markets")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "metadata").mkdir(parents=True)
    (OUT / "orderbook").mkdir(parents=True)
    selected.write_parquet(OUT / "metadata" / "cross_market_candidates.parquet")
    selected.group_by("cross_market_role").len().sort("cross_market_role").write_csv(
        OUT / "metadata" / "candidate_counts.csv"
    )
    ids = selected.get_column("condition_id").drop_nulls().unique()
    print(f"Selected {selected.height:,} candidate conditions and {ids.len():,} IDs")
    print(selected.group_by("cross_market_role").len().sort("cross_market_role"))

    files = sorted((root / "orderbook").glob("orderbook_*.parquet"))
    report = []
    for number, source in enumerate(files, 1):
        target = OUT / "orderbook" / source.name
        print(f"[{number}/{len(files)}] {source.name}", flush=True)
        query = pl.scan_parquet(source).filter(pl.col("market_id").is_in(ids))
        query.sink_parquet(target, compression="zstd", maintain_order=True)
        rows = pl.scan_parquet(target).select(pl.len()).collect().item()
        if rows == 0:
            target.unlink()
        report.append({"file": source.name, "rows": rows, "size_mb": target.stat().st_size / 2**20 if target.exists() else 0})
        del query
        gc.collect()

    pl.DataFrame(report).write_csv(OUT / "extraction_report.csv")
    archive = shutil.make_archive("/kaggle/working/cross_market_sports", "zip", OUT.parent, OUT.name)
    print(f"Archive: {archive} ({Path(archive).stat().st_size / 2**30:.3f} GiB)")


if __name__ == "__main__":
    main()
