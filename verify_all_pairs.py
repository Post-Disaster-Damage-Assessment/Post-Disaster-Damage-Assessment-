import json
from pathlib import Path

label_dir = Path("data/train/labels")

total = 0
matched = 0
unmatched = 0
missing_pairs = 0

for post_file in label_dir.glob("*_post_disaster.json"):
    pre_file = post_file.with_name(
        post_file.name.replace("_post_disaster", "_pre_disaster")
    )

    if not pre_file.exists():
        missing_pairs += 1
        continue

    with open(pre_file, encoding="utf-8") as f:
        pre = json.load(f)["features"]["xy"]

    with open(post_file, encoding="utf-8") as f:
        post = json.load(f)["features"]["xy"]

    pre_ids = {
        x.get("properties", {}).get("uid")
        for x in pre
    }

    post_ids = {
        x.get("properties", {}).get("uid")
        for x in post
    }

    pre_ids.discard(None)
    post_ids.discard(None)

    total += len(post_ids)
    matched += len(pre_ids & post_ids)
    unmatched += len(post_ids - pre_ids)

print("TOTAL POST-DISASTER BUILDINGS:", total)
print("MATCHED WITH PRE-DISASTER UID:", matched)
print("POST BUILDINGS WITHOUT MATCH:", unmatched)
print("MISSING PRE-DISASTER FILES:", missing_pairs)
print("MATCH RATE:", round(matched / total * 100, 2) if total else 0, "%")
