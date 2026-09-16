from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path("polymarket_sports")
META = ROOT / "metadata" / "sports_markets.parquet"


def main() -> None:
    table = pq.read_table(META)
    print(table.schema)
    frame = table.to_pandas()
    mask = frame["question"].fillna("").str.contains(
        r"Gen\.G|FUT Esports", case=False, regex=True
    )
    columns = [
        c
        for c in (
            "condition_id",
            "question",
            "end_date",
            "token_id",
            "outcome",
            "title",
            "slug",
        )
        if c in frame.columns
    ]
    print(frame.loc[mask, columns].to_string(index=False))

    for day in ("2026-03-13", "2026-03-14", "2026-03-15", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-21"):
        path = ROOT / "orderbook" / f"orderbook_{day}.parquet"
        dataset = ds.dataset(path, format="parquet")
        ids = set(frame.loc[mask, "condition_id"].dropna().astype(str))
        found = dataset.to_table(
            columns=["market_id"], filter=ds.field("market_id").isin(ids)
        )
        if found.num_rows:
            print(day, found.num_rows, len(set(found.column("market_id").to_pylist())))


if __name__ == "__main__":
    main()
