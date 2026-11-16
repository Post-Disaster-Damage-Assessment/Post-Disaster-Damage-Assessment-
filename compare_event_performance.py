
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

ORIGINAL_FILE = Path(
    "comparison_results/original/validation_predictions.csv"
)

EXPERIMENT_FILE = Path(
    "comparison_results/photo_aug/validation_predictions.csv"
)

OUTPUT_DIR = Path("comparison_results/event_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]


def load_predictions(path):
    records = {}

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            key = (row["scene"], row["building_uid"])

            records[key] = {
                "scene": row["scene"],
                "event": row["scene"].rsplit("_", 1)[0],
                "true": row["true_damage"],
                "predicted": row["predicted_damage"]
            }

    return records


def calculate_metrics(records):
    confusion = np.zeros((4, 4), dtype=np.int64)

    for record in records:
        true_index = CLASSES.index(record["true"])
        pred_index = CLASSES.index(record["predicted"])
        confusion[true_index, pred_index] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0

    class_metrics = []

    for i, name in enumerate(CLASSES):
        tp = int(confusion[i, i])
        support = int(confusion[i].sum())
        predicted = int(confusion[:, i].sum())

        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        class_metrics.append({
            "class": name,
            "support": support,
            "predicted_count": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    macro_f1 = float(np.mean([m["f1"] for m in class_metrics]))

    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in class_metrics) / total
        if total else 0.0
    )

    return {
        "total": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_metrics": class_metrics,
        "confusion": confusion
    }


def save_csv(filename, rows):
    if not rows:
        return

    path = OUTPUT_DIR / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


print("=" * 75)
print("ORIGINAL VS PHOTOMETRIC AUGMENTATION")
print("EVENT-WISE PERFORMANCE COMPARISON")
print("=" * 75)

print("\nLoading original predictions...")
original = load_predictions(ORIGINAL_FILE)

print("Loading experimental predictions...")
experimental = load_predictions(EXPERIMENT_FILE)

print("Original buildings:", len(original))
print("Experimental buildings:", len(experimental))

# Verify identical validation records
if set(original) != set(experimental):
    raise RuntimeError(
        "Validation building IDs do not match between models."
    )

for key in original:
    if original[key]["true"] != experimental[key]["true"]:
        raise RuntimeError(
            f"Ground-truth mismatch detected: {key}"
        )

print("Validation records and ground-truth labels match.")

# Group predictions by event
original_events = defaultdict(list)
experimental_events = defaultdict(list)

for key in original:
    event = original[key]["event"]
    original_events[event].append(original[key])
    experimental_events[event].append(experimental[key])

if set(original_events) != set(experimental_events):
    raise RuntimeError("Event sets do not match.")

event_rows = []
class_rows = []
confusion_rows = []
global_rows = []

# Global comparison
global_original = calculate_metrics(list(original.values()))
global_experimental = calculate_metrics(list(experimental.values()))

for name, metrics in [
    ("original", global_original),
    ("photo_aug", global_experimental)
]:
    global_rows.append({
        "model": name,
        "total_buildings": metrics["total"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"]
    })

# Event-wise comparison
for event in sorted(original_events):

    old = calculate_metrics(original_events[event])
    new = calculate_metrics(experimental_events[event])

    event_rows.append({
        "event": event,
        "buildings": old["total"],
        "original_accuracy": old["accuracy"],
        "photo_aug_accuracy": new["accuracy"],
        "accuracy_change": new["accuracy"] - old["accuracy"],
        "original_macro_f1": old["macro_f1"],
        "photo_aug_macro_f1": new["macro_f1"],
        "macro_f1_change": new["macro_f1"] - old["macro_f1"],
        "original_weighted_f1": old["weighted_f1"],
        "photo_aug_weighted_f1": new["weighted_f1"],
        "weighted_f1_change": new["weighted_f1"] - old["weighted_f1"]
    })

    # Class-wise comparison
    for old_class, new_class in zip(
        old["class_metrics"],
        new["class_metrics"]
    ):
        class_rows.append({
            "event": event,
            "class": old_class["class"],
            "support": old_class["support"],
            "original_precision": old_class["precision"],
            "photo_aug_precision": new_class["precision"],
            "precision_change": (
                new_class["precision"] - old_class["precision"]
            ),
            "original_recall": old_class["recall"],
            "photo_aug_recall": new_class["recall"],
            "recall_change": (
                new_class["recall"] - old_class["recall"]
            ),
            "original_f1": old_class["f1"],
            "photo_aug_f1": new_class["f1"],
            "f1_change": new_class["f1"] - old_class["f1"]
        })

    # Event confusion matrix comparison
    for i, true_class in enumerate(CLASSES):
        for j, pred_class in enumerate(CLASSES):
            old_count = int(old["confusion"][i, j])
            new_count = int(new["confusion"][i, j])

            confusion_rows.append({
                "event": event,
                "true_class": true_class,
                "predicted_class": pred_class,
                "original_count": old_count,
                "photo_aug_count": new_count,
                "count_change": new_count - old_count
            })


# Save all outputs
save_csv("global_metrics_comparison.csv", global_rows)
save_csv("event_metrics_comparison.csv", event_rows)
save_csv("event_class_metrics_comparison.csv", class_rows)
save_csv("event_confusion_comparison.csv", confusion_rows)


# Print global comparison
print("\n" + "=" * 75)
print("GLOBAL COMPARISON")
print("=" * 75)

print(
    f"{'Metric':20s} {'Original':>12s} "
    f"{'Photo Aug':>12s} {'Change':>12s}"
)

for metric in ["accuracy", "macro_f1", "weighted_f1"]:
    old_value = global_original[metric]
    new_value = global_experimental[metric]

    print(
        f"{metric:20s} {old_value:12.4f} "
        f"{new_value:12.4f} {new_value-old_value:+12.4f}"
    )


# Print event ranking
print("\n" + "=" * 75)
print("EVENT-WISE PERFORMANCE")
print("=" * 75)

print(
    f"{'Event':27s} {'Original F1':>12s} "
    f"{'New F1':>12s} {'Change':>12s}"
)

for row in event_rows:
    print(
        f"{row['event']:27s} "
        f"{row['original_macro_f1']:12.4f} "
        f"{row['photo_aug_macro_f1']:12.4f} "
        f"{row['macro_f1_change']:+12.4f}"
    )


# Focus on difficult events
for focus in ["hurricane-matthew", "hurricane-michael"]:

    print("\n" + "=" * 75)
    print(f"{focus.upper()} - CLASS-WISE COMPARISON")
    print("=" * 75)

    matching = [
        row for row in class_rows
        if row["event"] == focus
    ]

    if not matching:
        print("Event not found.")
        continue

    print(
        f"{'Class':18s} {'Original F1':>12s} "
        f"{'New F1':>12s} {'Change':>12s}"
    )

    for row in matching:
        print(
            f"{row['class']:18s} "
            f"{row['original_f1']:12.4f} "
            f"{row['photo_aug_f1']:12.4f} "
            f"{row['f1_change']:+12.4f}"
        )


print("\n" + "=" * 75)
print("EVENT COMPARISON COMPLETED")
print("Results directory:", OUTPUT_DIR.resolve())
print("=" * 75)
