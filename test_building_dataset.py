from collections import Counter
from building_dataset import (
    collect_buildings,
    PairedBuildingDataset,
    CLASS_NAMES
)

print("Scanning building annotations...")

records = collect_buildings("data/train")

print("\nTOTAL BUILDING SAMPLES:", len(records))

counts = Counter(x["label"] for x in records)

print("\nCLASS DISTRIBUTION:")

for label_id, name in enumerate(CLASS_NAMES):
    print(f"{name:15}: {counts[label_id]}")

dataset = PairedBuildingDataset(records)

print("\nTESTING ONE SAMPLE PER CLASS:")

for label_id, name in enumerate(CLASS_NAMES):

    index = next(
        (i for i, x in enumerate(records)
         if x["label"] == label_id),
        None
    )

    if index is None:
        print(name, "- NO SAMPLE FOUND")
        continue

    pre, post, label = dataset[index]

    print(f"\n{name}")
    print("Pre image shape:", tuple(pre.shape))
    print("Post image shape:", tuple(post.shape))
    print("Label:", label.item())
    print("UID:", records[index]["uid"])
    print("Scene:", records[index]["scene"])

print("\nDATASET TEST COMPLETE")
