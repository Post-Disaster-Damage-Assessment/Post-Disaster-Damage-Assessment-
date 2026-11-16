import pandas as pd
from pathlib import Path

original_file = Path(
    "comparison_results/original/validation_predictions.csv"
)

new_file = Path(
    "comparison_results/photo_aug/validation_predictions.csv"
)

old = pd.read_csv(original_file)
new = pd.read_csv(new_file)

keys = ["scene", "building_uid"]

df = old.merge(
    new,
    on=keys,
    suffixes=("_original", "_new"),
    validate="one_to_one"
)

if not (
    df["true_damage_original"] ==
    df["true_damage_new"]
).all():
    raise ValueError("Ground-truth mismatch!")

df["original_correct"] = (
    df["true_damage_original"] ==
    df["predicted_damage_original"]
)

df["new_correct"] = (
    df["true_damage_new"] ==
    df["predicted_damage_new"]
)

both_correct = (
    df["original_correct"] & df["new_correct"]
).sum()

both_wrong = (
    ~df["original_correct"] & ~df["new_correct"]
).sum()

improved = (
    ~df["original_correct"] & df["new_correct"]
).sum()

regressed = (
    df["original_correct"] & ~df["new_correct"]
).sum()

changed = (
    df["predicted_damage_original"] !=
    df["predicted_damage_new"]
).sum()

print("=" * 65)
print("BUILDING-LEVEL PREDICTION CHANGE ANALYSIS")
print("=" * 65)

print("Total buildings:", len(df))
print("Both models correct:", both_correct)
print("Both models incorrect:", both_wrong)
print("Original wrong, new correct:", improved)
print("Original correct, new wrong:", regressed)
print("Predictions changed:", changed)

print("\nNET CHANGE IN CORRECT PREDICTIONS:", improved - regressed)

print("\nPREDICTION TRANSITIONS")

transitions = pd.crosstab(
    df["predicted_damage_original"],
    df["predicted_damage_new"]
)

print(transitions)

output = Path("comparison_results/prediction_changes")
output.mkdir(parents=True, exist_ok=True)

df.to_csv(
    output / "building_prediction_changes.csv",
    index=False
)

print("\nSaved detailed results:", output.resolve())