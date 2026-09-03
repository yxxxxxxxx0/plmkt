import requests

slug = "mlb-min-oak-2026-08-26"

event = requests.get(
    "https://gamma-api.polymarket.com/events",
    params={"slug": slug}
).json()[0]

for m in event["markets"]:
    cid = m["conditionId"]

    r = requests.get(
        f"https://clob.polymarket.com/markets/{cid}"
    ).json()

    print(m["question"])
    print("condition_id:", cid)
    print("seconds_delay:", r.get("seconds_delay"))
    print()