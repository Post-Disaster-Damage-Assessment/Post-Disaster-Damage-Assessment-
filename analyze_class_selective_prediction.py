
import pandas as pd
from pathlib import Path

INPUT = Path("validation_results/validation_predictions.csv")
OUTPUT = Path("validation_results/selective_prediction")

OUTPUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]

THRESHOLDS = [0.60, 0.70, 0.80, 0.90]

df = pd.read_csv(INPUT)

df["event"] = df["scene"].str.rsplit("_", n=1).str[0]

df["correct"] = (
    df["correct"].astype(str).str.lower() == "true"
)

prob_cols = [f"prob_{cls}" for cls in CLASSES]
df["confidence"] = df[prob_cols].max(axis=1)

rows = []

for event in ["overall", "hurricane-matthew", "hurricane-michael"]:

    if event == "overall":
        event_data = df
    else:
        event_data = df[df["event"] == event]

    for cls in CLASSES:

        class_data = event_data[
            event_data["true_damage"] == cls
        ]

        for threshold in THRESHOLDS:

            accepted = class_data[
                class_data["confidence"] >= threshold
            ]

            total = len(class_data)
            retained = len(accepted)
            correct = int(accepted["correct"].sum())

            rows.append({
                "event": event,
                "actual_class": cls,
                "threshold": threshold,
                "total": total,
                "accepted": retained,
                "reviewed": total - retained,
                "coverage": retained / total if total else 0,
                "accepted_accuracy": (
                    correct / retained if retained else 0
                ),
                "accepted_errors": retained - correct
            })

result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT / "class_selective_prediction.csv",
    index=False
)

for event in ["overall", "hurricane-matthew", "hurricane-michael"]:

    print("\n" + "=" * 85)
    print(event.upper())
    print("=" * 85)

    subset = result[result["event"] == event]

    for threshold in THRESHOLDS:

        print(f"\nCONFIDENCE THRESHOLD: {threshold:.0%}")

        table = subset[subset["threshold"] == threshold]

        print(
            table[
                [
                    "actual_class",
                    "total",
                    "accepted",
                    "reviewed",
                    "coverage",
                    "accepted_accuracy",
                    "accepted_errors"
                ]
            ].to_string(
                index=False,
                formatters={
                    "coverage": lambda x: f"{x * 100:.2f}%",
                    "accepted_accuracy": lambda x: f"{x * 100:.2f}%"
                }
            )
        )

print("\nAnalysis completed.")
print("Saved:", (OUTPUT / "class_selective_prediction.csv").resolve())
