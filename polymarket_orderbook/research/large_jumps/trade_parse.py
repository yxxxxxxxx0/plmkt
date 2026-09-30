import os
import sys, json, csv
w = csv.writer(sys.stdout)
w.writerow(['ts', 'slug', 'market_type', 'line', 'asset_id', 'price', 'size', 'aggressor'])
for ln in sys.stdin:
    d = json.loads(ln)
    w.writerow([d['ts'], d['slug'], d['market_type'], d['line'], d['asset_id'], d['price'], d['size'], d.get('aggressor')])
