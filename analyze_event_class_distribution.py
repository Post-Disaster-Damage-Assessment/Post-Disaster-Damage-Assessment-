
import pandas as pd
from pathlib import Path

INPUT = Path("validation_results/validation_predictions.csv")
OUTPUT = Path("validation_results/event_class_distribution")

CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]

FOCUS = ["hurricane-matthew", "hurricane-michael"]

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

df["event"] = df["scene"].str.rsplit("_", n=1).str[0]
df["correct"] = (
    df["correct"].astype(str).str.lower() == "true"
)

results = []

for event, group in df.groupby("event"):

    total = len(group)

    for damage_class in CLASSES:

        actual_count = (
            group["true_damage"] == damage_class
        ).sum()

        predicted_count = (
            group["predicted_damage"] == damage_class
        ).sum()

        actual_pct = actual_count / total * 100
        predicted_pct = predicted_count / total * 100

        results.append({
            "event": event,
            "class": damage_class,
            "total_buildings": total,
            "actual_count": actual_count,
            "predicted_count": predicted_count,
            "actual_percentage": round(actual_pct, 3),
            "predicted_percentage": round(predicted_pct, 3),
            "percentage_difference": round(
                predicted_pct - actual_pct, 3
            )
        })

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT / "event_class_distribution.csv",
    index=False
)

# Event-level summary
summary = (
    df.groupby("event")
    .agg(
        total_buildings=("correct", "size"),
        correct_predictions=("correct", "sum")
    )
    .reset_index()
)

summary["accuracy"] = (
    summary["correct_predictions"] /
    summary["total_buildings"]
)

summary.to_csv(
    OUTPUT / "event_accuracy.csv",
    index=False
)

# Print focused reports
for event in FOCUS:

    print("\n" + "=" * 75)
    print(event.upper())
    print("=" * 75)

    subset = result_df[result_df["event"] == event]

    print(
        f"{'Class':18s} {'Actual':>9s} "
        f"{'Predicted':>10s} {'Actual %':>10s} "
        f"{'Pred %':>9s} {'Difference':>12s}"
    )

    print("-" * 75)

    for _, row in subset.iterrows():

        print(
            f"{row['class']:18s} "
            f"{row['actual_count']:9d} "
            f"{row['predicted_count']:10d} "
            f"{row['actual_percentage']:9.2f}% "
            f"{row['predicted_percentage']:8.2f}% "
            f"{row['percentage_difference']:+11.2f}%"
        )

    accuracy = summary[summary["event"] == event].iloc[0]

    print(
        f"\nAccuracy: "
        f"{accuracy['correct_predictions']}/"
        f"{accuracy['total_buildings']} "
        f"({accuracy['accuracy'] * 100:.2f}%)"
    )

print("\n" + "=" * 75)
print("EVENT CLASS DISTRIBUTION ANALYSIS COMPLETED")
print("Output:", OUTPUT.resolve())
print("=" * 75)
