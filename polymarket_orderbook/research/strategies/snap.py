# Snapshot of every open Polymarket event (Gamma keyset paging) -> cache/open_events.json
import json, subprocess
URL = "https://gamma-api.polymarket.com/events/keyset?closed=false&active=true&limit=100"
out, cur = {}, None
for _ in range(600):
    r = subprocess.run(["curl", "-s", "--max-time", "60", URL + (f"&after_cursor={cur}" if cur else "")], capture_output=True, text=True, encoding="utf-8").stdout
    try: d = json.loads(r)
    except Exception: break
    if not d.get("events"): break
    out.update({e["id"]: e for e in d["events"]}); cur = d.get("next_cursor")
    if not cur: break
json.dump(list(out.values()), open("research/strategies/cache/open_events.json", "w"))
print(len(out), "events", sum(len(e.get("markets", [])) for e in out.values()), "markets")
