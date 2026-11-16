
import pandas as pd
from pathlib import Path

INPUT = Path("validation_results/validation_predictions.csv")
OUTPUT = Path("validation_results/scene_analysis")

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

rows = []

for (event, scene), group in df.groupby(["event", "scene"]):

    if event not in ["hurricane-matthew", "hurricane-michael"]:
        continue

    row = {
        "event": event,
        "scene": scene,
        "buildings": len(group)
    }

    for cls in CLASSES:

        mask = group["true_damage"] == cls

        support = int(mask.sum())
        errors = int((mask & ~group["correct"]).sum())

        row[f"{cls}_support"] = support
        row[f"{cls}_errors"] = errors

        row[f"{cls}_error_rate"] = (
            round(errors / support, 4)
            if support else None
        )

    rows.append(row)

result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT / "scene_class_error_rates.csv",
    index=False
)

for event in ["hurricane-matthew", "hurricane-michael"]:

    data = result[result["event"] == event]

    print("\n" + "=" * 85)
    print(event.upper())
    print("=" * 85)

    # Focus on scenes with at least 30 buildings
    data = data[data["buildings"] >= 30]

    print("Scenes with at least 30 buildings:", len(data))

    for cls in CLASSES:

        support_col = f"{cls}_support"
        error_col = f"{cls}_errors"

        total_support = int(data[support_col].sum())
        total_errors = int(data[error_col].sum())

        rate = (
            total_errors / total_support * 100
            if total_support else 0
        )

        print(
            f"\n{cls.upper()}"
            f"\nBuildings: {total_support}"
            f"\nErrors: {total_errors}"
            f"\nError rate: {rate:.2f}%"
        )

        ranked = data.sort_values(
            error_col,
            ascending=False
        ).head(5)

        print("Top scenes by error count:")

        print(
            ranked[
                ["scene", "buildings", support_col, error_col]
            ].to_string(index=False)
        )

print("\nAnalysis completed.")
print("Saved:", (OUTPUT / "scene_class_error_rates.csv").resolve())
