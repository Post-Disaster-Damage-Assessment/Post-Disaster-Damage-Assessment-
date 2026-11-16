
import pandas as pd
from pathlib import Path

INPUT = Path("validation_results/validation_predictions.csv")
OUTPUT = Path("validation_results/error_concentration")

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

df["error"] = ~df["correct"]

for event in ["hurricane-matthew", "hurricane-michael"]:

    data = df[df["event"] == event].copy()

    scene_rows = []

    for scene, group in data.groupby("scene"):

        total = len(group)
        errors = int(group["error"].sum())

        row = {
            "event": event,
            "scene": scene,
            "buildings": total,
            "errors": errors,
            "accuracy": round(1 - errors / total, 5),
            "error_rate": round(errors / total, 5)
        }

        for damage_class in CLASSES:

            actual = group["true_damage"] == damage_class

            row[f"actual_{damage_class}"] = int(actual.sum())

            row[f"errors_{damage_class}"] = int(
                (actual & group["error"]).sum()
            )

        scene_rows.append(row)

    result = pd.DataFrame(scene_rows)

    result = result.sort_values(
        ["errors", "buildings"],
        ascending=[False, False]
    )

    result.to_csv(
        OUTPUT / f"{event}_error_concentration.csv",
        index=False
    )

    # Only use scenes with at least 30 buildings
    substantial = result[result["buildings"] >= 30].copy()

    total_event_errors = int(data["error"].sum())

    print("\n" + "=" * 85)
    print(event.upper())
    print("=" * 85)

    print("Total event errors:", total_event_errors)
    print("Scenes with at least 30 buildings:", len(substantial))

    print("\nTOP 10 SCENES BY ABSOLUTE ERROR COUNT")

    print(
        result.head(10)[
            ["scene", "buildings", "errors", "accuracy"]
        ].to_string(
            index=False,
            formatters={
                "accuracy": lambda x: f"{x * 100:.2f}%"
            }
        )
    )

    print("\nTOP 10 SUBSTANTIAL SCENES")

    print(
        substantial.head(10)[
            ["scene", "buildings", "errors", "error_rate"]
        ].to_string(
            index=False,
            formatters={
                "error_rate": lambda x: f"{x * 100:.2f}%"
            }
        )
    )

    # Error concentration in largest-error scenes
    for n in [3, 5, 10]:

        top_errors = int(result.head(n)["errors"].sum())

        print(
            f"Top {n} scenes account for "
            f"{top_errors}/{total_event_errors} errors "
            f"({top_errors / total_event_errors * 100:.2f}%)"
        )

    # Which actual classes contribute most to errors?
    print("\nERRORS BY ACTUAL DAMAGE CLASS")

    for damage_class in CLASSES:

        column = f"errors_{damage_class}"
        count = int(result[column].sum())

        print(
            f"{damage_class:18s}: "
            f"{count:5d} errors "
            f"({count / total_event_errors * 100:.2f}%)"
        )

print("\nAnalysis completed.")
print("Output:", OUTPUT.resolve())
