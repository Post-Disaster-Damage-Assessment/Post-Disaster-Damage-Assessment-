
import pandas as pd
from pathlib import Path

INPUT = Path("validation_results/validation_predictions.csv")
OUTPUT = Path("validation_results/scene_analysis")

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

df["event"] = df["scene"].str.rsplit("_", n=1).str[0]
df["correct"] = (
    df["correct"].astype(str).str.lower() == "true"
)

# Calculate performance for every validation scene
scene_results = (
    df.groupby(["event", "scene"])
    .agg(
        buildings=("correct", "size"),
        correct=("correct", "sum")
    )
    .reset_index()
)

scene_results["errors"] = (
    scene_results["buildings"] - scene_results["correct"]
)

scene_results["accuracy"] = (
    scene_results["correct"] / scene_results["buildings"]
)

scene_results["error_rate"] = (
    scene_results["errors"] / scene_results["buildings"]
)

scene_results.to_csv(
    OUTPUT / "scene_performance.csv",
    index=False
)

for event in ["hurricane-matthew", "hurricane-michael"]:

    subset = scene_results[
        scene_results["event"] == event
    ].copy()

    print("\n" + "=" * 75)
    print(event.upper())
    print("=" * 75)

    print("Validation scenes:", len(subset))
    print("Total buildings:", subset["buildings"].sum())

    print(
        "Mean scene accuracy:",
        round(subset["accuracy"].mean() * 100, 2), "%"
    )

    print(
        "Median scene accuracy:",
        round(subset["accuracy"].median() * 100, 2), "%"
    )

    print(
        "Scenes with accuracy below 50%:",
        (subset["accuracy"] < 0.50).sum()
    )

    print(
        "Scenes with accuracy below 70%:",
        (subset["accuracy"] < 0.70).sum()
    )

    print("\n10 WORST SCENES")

    worst = subset.sort_values(
        ["error_rate", "buildings"],
        ascending=[False, False]
    ).head(10)

    print(
        worst[
            ["scene", "buildings", "errors", "accuracy", "error_rate"]
        ].to_string(
            index=False,
            formatters={
                "accuracy": lambda x: f"{x * 100:.2f}%",
                "error_rate": lambda x: f"{x * 100:.2f}%"
            }
        )
    )

    print("\n10 BEST SCENES")

    best = subset.sort_values(
        ["accuracy", "buildings"],
        ascending=[False, False]
    ).head(10)

    print(
        best[
            ["scene", "buildings", "errors", "accuracy"]
        ].to_string(
            index=False,
            formatters={
                "accuracy": lambda x: f"{x * 100:.2f}%"
            }
        )
    )

print("\n" + "=" * 75)
print("SCENE ANALYSIS COMPLETED")
print("Saved:", (OUTPUT / "scene_performance.csv").resolve())
print("=" * 75)
