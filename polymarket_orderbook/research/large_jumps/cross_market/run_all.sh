#!/bin/sh
# Run the whole cross-market study. From polymarket_orderbook/:
#     sh research/large_jumps/cross_market/run_all.sh            full run (several hours on a CPU)
#     sh research/large_jumps/cross_market/run_all.sh --smoke    2 sessions, a few games, ~5 minutes
# Set PY to use another interpreter, e.g. PY=python sh ...   Full logs: research/large_jumps/cache/xm/logs/
PY=${PY:-../.venv/Scripts/python.exe}
D=research/large_jumps/cross_market
LOG=research/large_jumps/cache/xm/logs; mkdir -p "$LOG"
S=$1
run() {
  echo; echo "=== $1  $(date +%H:%M:%S)"
  $PY -W ignore $D/$1.py $S > "$LOG/$1.log" 2>&1; st=$?
  grep -v -E '^(books_| scored)' "$LOG/$1.log"
  [ $st -eq 0 ] || { echo "!!! $1 failed (exit $st), see $LOG/$1.log"; exit $st; }
}
run build_panel         # ~15 min full: one panel per game in research/large_jumps/cache/xm/
run m1_trees            # ~20 min
run m5_unsupervised     # ~20 min
run m4_leadlag          # ~45 min (Hawkes fits are a Python loop)
run m2_cnn              # ~45 min, most of it scoring every test second
run m3_attention        # ~90 min
echo; echo "=== all reports"
$PY -c "import pandas as pd, sys, os; sys.argv += '$S'.split(); sys.path.insert(0, '$D'); from common import OUT; \
print(pd.read_csv(os.path.join(OUT, 'reports.csv')).round(3).to_string(index=False))"
