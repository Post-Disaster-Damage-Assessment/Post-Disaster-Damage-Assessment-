
import csv
from pathlib import Path
from collections import defaultdict

from building_dataset import collect_buildings, CLASS_NAMES
from train_siamese import create_scene_split

ROOT = Path("data/train")
OUTPUT = Path("validation_results/training_exposure")

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("TRAINING VS VALIDATION EXPOSURE ANALYSIS")
print("=" * 75)

print("\nCollecting building records...")
records = collect_buildings(ROOT)

print("Total buildings:", len(records))

print("\nReconstructing original scene-level split...")
train_records, val_records = create_scene_split(records)

print("Training buildings:", len(train_records))
print("Validation buildings:", len(val_records))

def get_event(record):
    return record["scene"].rsplit("_", 1)[0]


def get_scene_counts(records):
    result = defaultdict(set)

    for record in records:
        result[get_event(record)].add(record["scene"])

    return result


train_scenes = get_scene_counts(train_records)
val_scenes = get_scene_counts(val_records)

events = sorted(
    set(train_scenes) | set(val_scenes)
)

rows = []

for event in events:

    train_event = [
        r for r in train_records
        if get_event(r) == event
    ]

    val_event = [
        r for r in val_records
        if get_event(r) == event
    ]

    train_total = len(train_event)
    val_total = len(val_event)

    for class_id, class_name in enumerate(CLASS_NAMES):

        train_count = sum(
            r["label"] == class_id
            for r in train_event
        )

        val_count = sum(
            r["label"] == class_id
            for r in val_event
        )

        rows.append({
            "event": event,
            "class": class_name,
            "training_scenes": len(train_scenes[event]),
            "validation_scenes": len(val_scenes[event]),
            "training_buildings": train_total,
            "validation_buildings": val_total,
            "training_class_count": train_count,
            "validation_class_count": val_count,
            "training_class_percentage": round(
                train_count / train_total * 100, 3
            ) if train_total else 0,
            "validation_class_percentage": round(
                val_count / val_total * 100, 3
            ) if val_total else 0
        })


# Save detailed CSV
csv_path = OUTPUT / "event_training_exposure.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)



# Print event-level summary
print("\n" + "=" * 75)
print("EVENT-LEVEL TRAINING EXPOSURE")
print("=" * 75)

print(
    f"{'Event':26s} {'Train scenes':>12s} "
    f"{'Val scenes':>10s} {'Train buildings':>16s} "
    f"{'Val buildings':>14s}"
)

print("-" * 85)

train_by_event = defaultdict(list)
val_by_event = defaultdict(list)

for record in train_records:
    train_by_event[get_event(record)].append(record)

for record in val_records:
    val_by_event[get_event(record)].append(record)

for event in events:

    train_total = len(train_by_event[event])
    val_total = len(val_by_event[event])

    print(
        f"{event:26s} "
        f"{len(train_scenes[event]):12d} "
        f"{len(val_scenes[event]):10d} "
        f"{train_total:16d} "
        f"{val_total:14d}"
    )


# Detailed class exposure for difficult events
for event in ["hurricane-matthew", "hurricane-michael"]:

    print("\n" + "=" * 75)
    print(event.upper())
    print("=" * 75)

    print(
        f"{'Class':18s} {'Train count':>12s} "
        f"{'Val count':>10s} {'Train %':>10s} "
        f"{'Val %':>10s}"
    )

    print("-" * 75)

    for row in rows:

        if row["event"] != event:
            continue

        print(
            f"{row['class']:18s} "
            f"{row['training_class_count']:12d} "
            f"{row['validation_class_count']:10d} "
            f"{row['training_class_percentage']:9.2f}% "
            f"{row['validation_class_percentage']:9.2f}%"
        )


print("\n" + "=" * 75)
print("ANALYSIS COMPLETED")
print("Saved:", csv_path.resolve())
print("=" * 75)
# Detailed class exposure for difficult events
for event in ["hurricane-matthew", "hurricane-michael"]:

    print("\n" + "=" * 75)
    print(event.upper())
    print("=" * 75)

    print(
        f"{'Class':18s} {'Train count':>12s} "
        f"{'Val count':>10s} {'Train %':>10s} "
        f"{'Val %':>10s}"
    )

    print("-" * 75)

    for row in rows:

        if row["event"] != event:
            continue

        print(
            f"{row['class']:18s} "
            f"{row['training_class_count']:12d} "
            f"{row['validation_class_count']:10d} "
            f"{row['training_class_percentage']:9.2f}% "
            f"{row['validation_class_percentage']:9.2f}%"
        )


print("\n" + "=" * 75)
print("ANALYSIS COMPLETED")
print("Saved:", csv_path.resolve())
print("=" * 75)
