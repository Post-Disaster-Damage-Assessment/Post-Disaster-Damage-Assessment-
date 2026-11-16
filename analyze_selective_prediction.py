
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("validation_results/validation_predictions.csv")
OUTPUT = Path("validation_results/selective_prediction")

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

df["event"] = df["scene"].str.rsplit("_", n=1).str[0]

df["correct"] = (
    df["correct"].astype(str).str.lower() == "true"
)

classes = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]

prob_cols = [f"prob_{c}" for c in classes]

df["confidence"] = df[prob_cols].max(axis=1)

thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

rows = []

def calculate(group, event, threshold):

    accepted = group[group["confidence"] >= threshold]

    total = len(group)
    retained = len(accepted)

    correct = int(accepted["correct"].sum())
    errors = retained - correct

    return {
        "event": event,
        "threshold": threshold,
        "total": total,
        "accepted": retained,
        "reviewed": total - retained,
        "coverage": retained / total if total else 0,
        "accepted_accuracy": correct / retained if retained else 0,
        "accepted_errors": errors,
        "missed_errors": errors
    }

# Overall
for threshold in thresholds:

    rows.append(
        calculate(df, "overall", threshold)
    )

# Individual events
for event, group in df.groupby("event"):

    for threshold in thresholds:

        rows.append(
            calculate(group, event, threshold)
        )

result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT / "selective_prediction_metrics.csv",
    index=False
)

print("\n" + "=" * 85)
print("OVERALL SELECTIVE PREDICTION")
print("=" * 85)

overall = result[result["event"] == "overall"]

print(
    overall[
        [
            "threshold",
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

print("\n" + "=" * 85)
print("HURRICANE EVENT RESULTS")
print("=" * 85)

events = result[
    result["event"].isin(
        ["hurricane-matthew", "hurricane-michael"]
    )
]

print(
    events[
        [
            "event",
            "threshold",
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
print("Output:", OUTPUT.resolve())
