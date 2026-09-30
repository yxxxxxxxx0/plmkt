#!/bin/sh
# Pull every recorded trade (last_trade_price) out of the raw recordings, one CSV per session.
# The raw .jsonl.xz files are read, never written. Sessions before 09-10 recorded no trades.
#     sh research/large_jumps/extract_trades.sh      (from polymarket_orderbook/)
OUT=results/large_jumps/inputs/trades
mkdir -p "$OUT"
for f in data/live/books_2026-*.jsonl.xz; do
  s=$(basename "$f" .jsonl.xz)
  (xz -dc "$f" | grep -F '"et":"trade"' | ../.venv/Scripts/python.exe research/large_jumps/trade_parse.py > "$OUT/$s.csv") &
done
wait
wc -l "$OUT"/*.csv
