
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("validation_results/validation_predictions.csv")
OUTPUT = Path("validation_results/confidence_analysis")

OUTPUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]

df = pd.read_csv(INPUT)

df["event"] = df["scene"].str.rsplit("_", n=1).str[0]

df["correct"] = (
    df["correct"].astype(str).str.lower() == "true"
)

# Confidence of the predicted class
probability_columns = [
    f"prob_{cls}" for cls in CLASSES
]

df["confidence"] = df[probability_columns].max(axis=1)

df["confidence"] = df["confidence"].clip(0, 1)

df["result"] = np.where(
    df["correct"], "correct", "incorrect"
)

# -------------------------------------------------
# 1. Overall confidence comparison
# -------------------------------------------------

print("\n" + "=" * 75)
print("OVERALL CONFIDENCE ANALYSIS")
print("=" * 75)

for result in ["correct", "incorrect"]:

    values = df.loc[
        df["result"] == result, "confidence"
    ]

    print(f"\n{result.upper()}")
    print("Count:", len(values))
    print("Mean confidence:", round(values.mean(), 4))
    print("Median confidence:", round(values.median(), 4))
    print("Below 50%:", int((values < 0.50).sum()))
    print("Above 80%:", int((values >= 0.80).sum()))
    print("Above 90%:", int((values >= 0.90).sum()))

# -------------------------------------------------
# 2. Confidence buckets
# -------------------------------------------------

bins = [0, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]

labels = [
    "<50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%"
]

df["confidence_bucket"] = pd.cut(
    df["confidence"],
    bins=bins,
    labels=labels,
    right=False
)

bucket_summary = (
    df.groupby("confidence_bucket", observed=False)
    .agg(
        buildings=("correct", "size"),
        correct=("correct", "sum")
    )
    .reset_index()
)

bucket_summary["incorrect"] = (
    bucket_summary["buildings"] -
    bucket_summary["correct"]
)

bucket_summary["error_rate"] = (
    bucket_summary["incorrect"] /
    bucket_summary["buildings"].replace(0, np.nan)
)

bucket_summary.to_csv(
    OUTPUT / "confidence_bucket_summary.csv",
    index=False
)

print("\nCONFIDENCE BUCKETS")
print(bucket_summary.to_string(index=False))

# -------------------------------------------------
# 3. Confidence by actual class
# -------------------------------------------------

print("\n" + "=" * 75)
print("CONFIDENCE BY ACTUAL CLASS")
print("=" * 75)

class_rows = []

for cls in CLASSES:

    group = df[df["true_damage"] == cls]

    correct = group[group["correct"]]
    incorrect = group[~group["correct"]]

    row = {
        "class": cls,
        "total": len(group),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "error_rate": len(incorrect) / len(group),
        "mean_correct_confidence": (
            correct["confidence"].mean()
            if len(correct) else np.nan
        ),
        "mean_incorrect_confidence": (
            incorrect["confidence"].mean()
            if len(incorrect) else np.nan
        )
    }

    class_rows.append(row)

class_summary = pd.DataFrame(class_rows)

class_summary.to_csv(
    OUTPUT / "class_confidence_summary.csv",
    index=False
)

print(class_summary.to_string(index=False))

# -------------------------------------------------
# 4. Confidence by event
# -------------------------------------------------

print("\n" + "=" * 75)
print("CONFIDENCE BY EVENT")
print("=" * 75)

event_rows = []

for event in [
    "hurricane-matthew",
    "hurricane-michael"
]:

    group = df[df["event"] == event]

    correct = group[group["correct"]]
    incorrect = group[~group["correct"]]

    row = {
        "event": event,
        "total": len(group),
        "errors": len(incorrect),
        "error_rate": len(incorrect) / len(group),
        "mean_correct_confidence": correct["confidence"].mean(),
        "mean_incorrect_confidence": incorrect["confidence"].mean(),
        "high_confidence_errors": int(
            (incorrect["confidence"] >= 0.80).sum()
        ),
        "very_high_confidence_errors": int(
            (incorrect["confidence"] >= 0.90).sum()
        )
    }

    event_rows.append(row)

event_summary = pd.DataFrame(event_rows)

event_summary.to_csv(
    OUTPUT / "event_confidence_summary.csv",
    index=False
)

print(event_summary.to_string(index=False))

# -------------------------------------------------
# 5. High-confidence incorrect predictions
# -------------------------------------------------

high_errors = df[
    (~df["correct"]) &
    (df["confidence"] >= 0.80)
].copy()

high_errors = high_errors.sort_values(
    "confidence",
    ascending=False
)

high_errors.to_csv(
    OUTPUT / "high_confidence_errors.csv",
    index=False
)

print("\n" + "=" * 75)
print("HIGH-CONFIDENCE ERRORS")
print("=" * 75)

print("Total:", len(high_errors))

print("\nBy event:")
print(
    high_errors.groupby("event").size()
    .sort_values(ascending=False)
    .to_string()
)

print("\nMost common actual -> predicted mistakes:")

print(
    high_errors.groupby(
        ["true_damage", "predicted_damage"]
    ).size()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

print("\nAnalysis completed.")
print("Output:", OUTPUT.resolve())
