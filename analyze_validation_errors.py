
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path("validation_results/validation_predictions.csv")
OUTPUT = Path("validation_results/error_analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]


def write_csv(path, rows):
    if not rows:
        print("No rows to save:", path)
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if not INPUT.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT}")

with open(INPUT, newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

if not records:
    raise RuntimeError("Prediction CSV is empty.")

total = len(records)
errors = [
    r for r in records
    if r["true_damage"] != r["predicted_damage"]
]

print("=" * 65)
print("VALIDATION MISCLASSIFICATION ANALYSIS")
print("=" * 65)
print("Total buildings:", total)
print("Correct predictions:", total - len(errors))
print("Incorrect predictions:", len(errors))
print("Error rate:", round(len(errors) / total * 100, 2), "%")

# 1. Class-wise performance and error counts
class_rows = []

for name in CLASSES:
    actual = [r for r in records if r["true_damage"] == name]
    class_errors = [r for r in actual if r["true_damage"] != r["predicted_damage"]]

    class_rows.append({
        "class": name,
        "support": len(actual),
        "correct": len(actual) - len(class_errors),
        "misclassified": len(class_errors),
        "error_rate_percent": round(
            len(class_errors) / len(actual) * 100, 2
        ) if actual else 0,
    })

write_csv(OUTPUT / "class_error_summary.csv", class_rows)

# 2. Actual-to-predicted error combinations
confusion_errors = Counter(
    (r["true_damage"], r["predicted_damage"])
    for r in errors
)

confusion_rows = [
    {
        "true_class": true_class,
        "predicted_class": predicted_class,
        "count": count,
    }
    for (true_class, predicted_class), count
    in confusion_errors.most_common()
]

write_csv(OUTPUT / "misclassification_pairs.csv", confusion_rows)

# 3. Confidence analysis of incorrect predictions
for r in errors:
    r["confidence"] = float(r["confidence"])

errors_by_confidence = sorted(
    errors,
    key=lambda r: float(r["confidence"]),
    reverse=True
)

write_csv(
    OUTPUT / "all_misclassified_buildings.csv",
    errors_by_confidence
)

high_confidence_errors = [
    r for r in errors
    if float(r["confidence"]) >= 0.80
]

write_csv(
    OUTPUT / "high_confidence_errors.csv",
    high_confidence_errors
)

# 4. Event-wise analysis
event_stats = defaultdict(lambda: {"total": 0, "errors": 0})

for r in records:
    event = r["scene"].rsplit("_", 1)[0]
    event_stats[event]["total"] += 1

    if r["true_damage"] != r["predicted_damage"]:
        event_stats[event]["errors"] += 1

event_rows = []

for event, stats in event_stats.items():
    event_rows.append({
        "event": event,
        "buildings": stats["total"],
        "errors": stats["errors"],
        "correct": stats["total"] - stats["errors"],
        "error_rate_percent": round(
            stats["errors"] / stats["total"] * 100, 2
        ) if stats["total"] else 0,
    })

event_rows.sort(
    key=lambda r: r["error_rate_percent"],
    reverse=True
)

write_csv(OUTPUT / "event_error_summary.csv", event_rows)

# 5. Confidence bands for all predictions and errors
bands = [
    ("0.00-0.49", 0.00, 0.50),
    ("0.50-0.69", 0.50, 0.70),
    ("0.70-0.79", 0.70, 0.80),
    ("0.80-0.89", 0.80, 0.90),
    ("0.90-1.00", 0.90, 1.01),
]

confidence_rows = []

for label, low, high in bands:
    selected = [
        r for r in records
        if low <= float(r["confidence"]) < high
    ]

    wrong = [
        r for r in selected
        if r["true_damage"] != r["predicted_damage"]
    ]

    confidence_rows.append({
        "confidence_band": label,
        "predictions": len(selected),
        "errors": len(wrong),
        "error_rate_percent": round(
            len(wrong) / len(selected) * 100, 2
        ) if selected else 0,
    })

write_csv(OUTPUT / "confidence_error_summary.csv", confidence_rows)

# 6. Save machine-readable summary
summary = {
    "total_buildings": total,
    "correct_predictions": total - len(errors),
    "misclassified_buildings": len(errors),
    "accuracy_percent": round(
        (total - len(errors)) / total * 100, 2
    ),
    "error_rate_percent": round(
        len(errors) / total * 100, 2
    ),
    "high_confidence_errors_80_percent_or_more": len(
        high_confidence_errors
    ),
    "class_summary": class_rows,
    "most_common_misclassifications": confusion_rows[:10],
    "highest_event_error_rates": event_rows[:10],
}

with open(
    OUTPUT / "error_analysis_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)

# 7. Terminal report
print("\nCLASS-WISE ERRORS")
for r in class_rows:
    print(
        f"{r['class']:15s} "
        f"Errors: {r['misclassified']:5d} / {r['support']:5d} "
        f"({r['error_rate_percent']:.2f}%)"
    )

print("\nMOST COMMON MISCLASSIFICATIONS")
for r in confusion_rows[:10]:
    print(
        f"{r['true_class']:15s} -> "
        f"{r['predicted_class']:15s}: {r['count']}"
    )

print("\nHIGH-CONFIDENCE ERRORS")
print("Incorrect predictions with confidence >= 0.80:",
      len(high_confidence_errors))

print("\nEVENTS WITH HIGHEST ERROR RATES")
for r in event_rows[:10]:
    print(
        f"{r['event']:30s} "
        f"Errors: {r['errors']:4d}/{r['buildings']:5d} "
        f"({r['error_rate_percent']:.2f}%)"
    )

print("\nOUTPUT DIRECTORY:")
print(OUTPUT.resolve())

print("\nERROR ANALYSIS COMPLETED")
