#!/usr/bin/env bash
# One-shot setup on an EC2 instance (Ubuntu 22.04+/24.04 or Amazon Linux 2023).
#
#   1. copy the code up (from the repo root on the Mac):
#        rsync -a --exclude data --exclude logs --exclude .venv --exclude '*.xz' \
#            -e "ssh -i ~/.ssh/sigbot-key.pem" polymarket_orderbook ubuntu@HOST:/opt/plmkt/
#   2. ssh in and run:   bash /opt/plmkt/polymarket_orderbook/deploy/setup_ec2.sh [s3://bucket/prefix]
#   3. on the Mac:       bash polymarket_orderbook/deploy/pull_recordings.sh --install
#
# Without an S3 argument, finished sessions wait in data/outbox/ until the
# Mac pulls them (step 3), which also deletes them from the server.
set -euo pipefail

S3="${1:-}"
ROOT=/opt/plmkt
APP=$ROOT/polymarket_orderbook
ME="$(id -un)"

if command -v apt-get >/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-venv python3-pip xz-utils chrony rsync
    PY=python3
else
    sudo dnf install -y python3.11 python3.11-pip xz chrony rsync
    PY=python3.11
fi
CHRONY=$(systemctl list-unit-files | grep -oE '^chronyd?\.service' | head -1)
sudo systemctl enable --now "$CHRONY"

# Order-book timing across two venues depends on the clock. Use Amazon Time
# Sync (169.254.169.123) -- AL2023 does by default, Ubuntu may not.
if ! chronyc sources | grep -q 169.254.169.123; then
    CONF=$( [ -d /etc/chrony/sources.d ] && echo /etc/chrony/sources.d/amazon.sources || echo /etc/chrony.d/amazon.sources )
    echo "server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4" | sudo tee "$CONF"
    sudo systemctl restart "$CHRONY"
fi
chronyc tracking | grep -E "Reference|System time"

sudo chown -R "$ME:$ME" "$ROOT"
$PY -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q --upgrade pip
"$ROOT/.venv/bin/pip" install -q requests websockets aiohttp orjson
mkdir -p "$APP/logs" "$APP/data/live/kalshi" "$APP/data/sessions" "$APP/data/outbox"

# Polymarket blocks US users from trading; confirm its public feed is still
# reachable from this region before relying on it.
echo "--- 30 s Polymarket smoke test"
(cd "$APP" && timeout 90 "$ROOT/.venv/bin/python" live_recorder.py --live-discover 6 --duration 30 --session setupcheck | tail -4)
echo "--- 30 s Kalshi smoke test"
EV=$(curl -s "https://api.elections.kalshi.com/trade-api/v2/events?status=open&limit=3&series_ticker=KXEPLGAME" \
     | "$ROOT/.venv/bin/python" -c "import json,sys;print(' '.join(e['event_ticker'] for e in json.load(sys.stdin)['events']))")
(cd "$APP" && timeout 90 "$ROOT/.venv/bin/python" kalshi_recorder.py --events $EV --duration 30 --session setupcheck | tail -3)
rm -f "$APP"/data/live/*setupcheck* "$APP"/data/live/kalshi/*setupcheck*

UNIT=/etc/systemd/system/plmkt-recorder.service
sudo cp "$APP/deploy/plmkt-recorder.service" "$UNIT"
sudo sed -i "s#User=RECORDER_USER#User=$ME#" "$UNIT"
if [ -n "$S3" ]; then
    sudo sed -i "s#run_session.py --daemon#run_session.py --daemon --s3 $S3#" "$UNIT"
fi
sudo systemctl daemon-reload
sudo systemctl enable --now plmkt-recorder
systemctl --no-pager status plmkt-recorder | head -5
echo "Done. Follow it with:  tail -f $APP/logs/daemon.log"
