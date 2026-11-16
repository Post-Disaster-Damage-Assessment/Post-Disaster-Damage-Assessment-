import csv
import json
from pathlib import Path
from collections import defaultdict

INPUT_FILE = Path("validation_results/validation_predictions.csv")
OUTPUT_DIR = Path("validation_results/event_class_analysis")

CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load validation predictions
records = []

with INPUT_FILE.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        scene = row["scene"]
        event = scene.rsplit("_", 1)[0]

        records.append({
            "event": event,
            "scene": scene,
            "true": row["true_damage"],
            "predicted": row["predicted_damage"],
            "correct": row["correct"].strip().lower() == "true",
            "confidence": float(row["confidence"])
        })

print("=" * 70)
print("EVENT-WISE CLASS PERFORMANCE ANALYSIS")
print("=" * 70)
print(f"Total validation buildings: {len(records)}")

# Group records by event
events = defaultdict(list)

for record in records:
    events[record["event"]].append(record)

class_metrics = []
confusion_rows = []
event_summary = []

for event, event_records in sorted(events.items()):

    total = len(event_records)
    correct = sum(r["correct"] for r in event_records)
    errors = total - correct

    event_summary.append({
        "event": event,
        "total_buildings": total,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / total if total else 0,
        "error_rate": errors / total if total else 0
    })

    for true_class in CLASSES:

        actual = [
            r for r in event_records
            if r["true"] == true_class
        ]

        predicted = [
            r for r in event_records
            if r["predicted"] == true_class
        ]

        tp = sum(
            1 for r in actual
            if r["predicted"] == true_class
        )

        fp = sum(
            1 for r in predicted
            if r["true"] != true_class
        )

        fn = len(actual) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0
        )

        class_metrics.append({
            "event": event,
            "class": true_class,
            "support": len(actual),
            "predicted_count": len(predicted),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "f1_score": round(f1, 5)
        })

        # Full confusion counts for this event
        for predicted_class in CLASSES:
            count = sum(
                1 for r in event_records
                if r["true"] == true_class
                and r["predicted"] == predicted_class
            )

            confusion_rows.append({
                "event": event,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "count": count
            })


def save_csv(filename, rows):
    if not rows:
        return

    with (OUTPUT_DIR / filename).open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


save_csv("event_class_metrics.csv", class_metrics)
save_csv("event_class_confusion.csv", confusion_rows)
save_csv("event_summary.csv", event_summary)

# Save a combined JSON report
report = {
    "total_validation_buildings": len(records),
    "total_events": len(events),
    "event_summary": event_summary,
    "class_metrics": class_metrics
}

with (OUTPUT_DIR / "event_class_report.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(report, file, indent=4)

# Print overall event ranking
print("\nEVENTS RANKED BY ERROR RATE")
print("-" * 70)

for item in sorted(
    event_summary,
    key=lambda x: x["error_rate"],
    reverse=True
):
    print(
        f"{item['event']:28s} "
        f"Errors: {item['errors']:5d}/{item['total_buildings']:5d} "
        f"Error rate: {item['error_rate'] * 100:6.2f}%"
    )

# Focused analysis
FOCUS_EVENTS = [
    "hurricane-matthew",
    "hurricane-michael"
]

for event in FOCUS_EVENTS:

    print("\n" + "=" * 70)
    print(event.upper())
    print("=" * 70)

    matching = [
        r for r in class_metrics
        if r["event"] == event
    ]

    if not matching:
        print("Event not found.")
        continue

    print(
        f"{'Class':18s} {'Support':>9s} "
        f"{'Predicted':>10s} {'Precision':>11s} "
        f"{'Recall':>9s} {'F1':>9s}"
    )

    print("-" * 70)

    for row in matching:
        print(
            f"{row['class']:18s} "
            f"{row['support']:9d} "
            f"{row['predicted_count']:10d} "
            f"{row['precision']:11.4f} "
            f"{row['recall']:9.4f} "
            f"{row['f1_score']:9.4f}"
        )

    # Most common errors for this event
    errors = [
        r for r in confusion_rows
        if r["event"] == event
        and r["true_class"] != r["predicted_class"]
        and r["count"] > 0
    ]

    errors.sort(key=lambda x: x["count"], reverse=True)

    print("\nMOST COMMON MISCLASSIFICATIONS")

    for row in errors[:8]:
        print(
            f"{row['true_class']:18s} -> "
            f"{row['predicted_class']:18s}: "
            f"{row['count']}"
        )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print(f"Results saved in: {OUTPUT_DIR.resolve()}")
print("=" * 70)