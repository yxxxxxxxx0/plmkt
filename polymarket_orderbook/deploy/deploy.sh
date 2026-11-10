#!/bin/sh
# Launch a t4g.nano (~$3/mo) running arb_scan.py as a service.
# Needs: aws CLI configured. Run from polymarket_orderbook/:  sh deploy/deploy.sh
# Logs:  aws ssm start-session --target <id>, then: journalctl -u arb-scan -f
# Trades: /opt/plmkt/polymarket_orderbook/data/arb/trades.csv
set -e
REGION=${REGION:-us-east-1}   # Polymarket's matching engine is in AWS London (eu-west-2); use that if allowed
tar czf /tmp/arb.tgz arb_scan.py live_recorder.py discovery.py matches.py polymarket_fees.py deploy/arb-scan.service
B64=$(base64 -w0 /tmp/arb.tgz)
AMI=$(aws ssm get-parameter --region $REGION --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 --query Parameter.Value --output text)
cat > /tmp/ud.sh <<UD
#!/bin/sh
dnf install -y python3.12 python3.12-pip
mkdir -p /opt/plmkt/polymarket_orderbook && cd /opt/plmkt/polymarket_orderbook
echo $B64 | base64 -d | tar xz
python3.12 -m venv /opt/plmkt/venv && /opt/plmkt/venv/bin/pip install requests websockets numpy
chown -R ec2-user /opt/plmkt
cp deploy/arb-scan.service /etc/systemd/system/ && systemctl enable --now arb-scan
UD
aws ec2 run-instances --region $REGION --image-id $AMI --instance-type t4g.nano \
  --user-data file:///tmp/ud.sh --metadata-options HttpTokens=required \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=arb-scan}]' \
  --query 'Instances[0].InstanceId' --output text
