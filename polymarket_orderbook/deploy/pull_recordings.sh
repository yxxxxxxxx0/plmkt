#!/usr/bin/env bash
# Runs on the Mac (launchd, every 30 min): downloads finished sessions from
# the server's data/outbox/ and deletes them there ONLY once verified here,
# so the server's disk never fills with recordings that already exist locally.
#
#   1. rsync copy outbox/ -> $DEST            (nothing deleted yet)
#   2. xz -t every newly arrived archive      (CRC of every block)
#   3. rsync --checksum --remove-source-files (re-compares every file by
#      full checksum; the server copy is removed only for files that match)
#
#   4. clean_kalshi.py on each new session -> kalshi/kbooks_clean_<tag>.jsonl.xz
#
# A failed xz -t stops the run before step 3, so a corrupt download never
# costs the server copy. Sessions still being written live under .tmp-* on
# the server and are skipped.
#
# Install as a launchd job:  bash deploy/pull_recordings.sh --install
set -euo pipefail

HOST="${PLMKT_HOST:-ubuntu@18.144.70.77}"
KEY="${PLMKT_KEY:-$HOME/.ssh/sigbot-key.pem}"
REMOTE="${PLMKT_REMOTE:-/opt/plmkt/polymarket_orderbook/data/outbox/}"
DEST="${PLMKT_DEST:-$HOME/plmkt-recordings}"
LABEL=com.plmkt.pull-recordings
SSH="ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

if [ "${1:-}" = "--install" ]; then
    PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
    SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$SCRIPT</string></array>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/plmkt-pull.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/plmkt-pull.log</string>
</dict></plist>
EOF
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "installed $PLIST (every 30 min; log ~/Library/Logs/plmkt-pull.log)"
    exit 0
fi

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$DEST"
echo "[$(ts)] pull from $HOST:$REMOTE -> $DEST"

STAMP="$DEST/.pull-stamp"
touch "$STAMP.new"
# 1. copy
rsync -a --partial --exclude='.tmp-*' -e "$SSH" "$HOST:$REMOTE" "$DEST/"

# 2. verify what just arrived
bad=0
while IFS= read -r -d '' f; do
    if ! xz -t "$f"; then
        echo "[$(ts)] CORRUPT: $f -- leaving the server copy in place"
        bad=1
    fi
done < <(find "$DEST" -name '*.xz' -newer "$STAMP" -print0 2>/dev/null || find "$DEST" -name '*.xz' -print0)
if [ "$bad" = 1 ]; then
    rm -f "$STAMP.new"
    exit 1
fi

# 3. delete from the server only what matches here byte for byte
rsync -a --checksum --remove-source-files --exclude='.tmp-*' -e "$SSH" "$HOST:$REMOTE" "$DEST/"
$SSH "$HOST" "find '$REMOTE' -mindepth 1 -type d -empty -not -name '.tmp-*' -delete; df -h / | tail -1"
mv "$STAMP.new" "$STAMP"

# 4. clean Kalshi's out-of-order replies (clean_kalshi.py; raw files untouched)
CLEANER="$(cd "$(dirname "$0")/.." && pwd)/clean_kalshi.py"
for d in "$DEST"/*/; do
    if ls "$d"kalshi/kbooks_*.jsonl.xz >/dev/null 2>&1 && ! ls "$d"kalshi/kbooks_clean_* >/dev/null 2>&1; then
        python3 "$CLEANER" "$d" || echo "[$(ts)] clean_kalshi failed for $d (raw data kept)"
    fi
done
echo "[$(ts)] done; $(find "$DEST" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') session(s) in $DEST"
