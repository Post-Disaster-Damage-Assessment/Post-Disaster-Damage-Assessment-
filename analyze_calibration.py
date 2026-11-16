
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("validation_results/validation_predictions.csv")
OUTPUT = Path("validation_results/calibration_analysis")

OUTPUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]

df = pd.read_csv(INPUT)

prob_cols = [f"prob_{c}" for c in CLASSES]

probs = df[prob_cols].to_numpy(dtype=float)
y_true = df["true_damage"].map(
    {name: i for i, name in enumerate(CLASSES)}
).to_numpy()

y_pred = np.argmax(probs, axis=1)
confidence = np.max(probs, axis=1)
correct = (y_pred == y_true).astype(int)

# ------------------------------------------------
# 1. Overall top-label calibration
# ------------------------------------------------

bins = np.linspace(0, 1, 11)

bin_rows = []
ece = 0.0

for i in range(len(bins) - 1):

    lower = bins[i]
    upper = bins[i + 1]

    if i == len(bins) - 2:
        mask = (confidence >= lower) & (confidence <= upper)
    else:
        mask = (confidence >= lower) & (confidence < upper)

    count = int(mask.sum())

    if count == 0:
        continue

    avg_conf = float(confidence[mask].mean())
    accuracy = float(correct[mask].mean())
    gap = abs(avg_conf - accuracy)

    ece += count / len(df) * gap

    bin_rows.append({
        "confidence_range": f"{lower:.1f}-{upper:.1f}",
        "count": count,
        "mean_confidence": avg_conf,
        "accuracy": accuracy,
        "calibration_gap": gap
    })

calibration = pd.DataFrame(bin_rows)

calibration.to_csv(
    OUTPUT / "reliability_bins.csv",
    index=False
)

print("\n" + "=" * 75)
print("OVERALL CALIBRATION")
print("=" * 75)

print(calibration.to_string(index=False))

print(f"\nExpected Calibration Error (ECE): {ece:.5f}")
print(f"ECE percentage: {ece * 100:.2f}%")

# ------------------------------------------------
# 2. Multiclass Brier score and log loss
# ------------------------------------------------

one_hot = np.eye(len(CLASSES))[y_true]

brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

clipped = np.clip(probs, 1e-12, 1.0)
nll = -np.mean(np.log(clipped[np.arange(len(df)), y_true]))

print("\nMulticlass Brier score:", round(brier, 5))
print("Multiclass log loss:", round(nll, 5))

# ------------------------------------------------
# 3. Calibration by event
# ------------------------------------------------

df["confidence"] = confidence
df["correct_prediction"] = correct

print("\n" + "=" * 75)
print("EVENT-WISE CALIBRATION")
print("=" * 75)

event_rows = []

for event in sorted(df["event"].unique()) if "event" in df.columns else sorted(
    df["scene"].str.rsplit("_", n=1).str[0].unique()
):

    if "event" not in df.columns:
        df["event"] = df["scene"].str.rsplit("_", n=1).str[0]

    group = df[df["event"] == event]

    event_ece = 0.0
    conf = group["confidence"].to_numpy()
    acc = group["correct_prediction"].to_numpy()

    for i in range(len(bins) - 1):

        if i == len(bins) - 2:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            mask = (conf >= bins[i]) & (conf < bins[i + 1])

        if mask.sum() == 0:
            continue

        gap = abs(conf[mask].mean() - acc[mask].mean())
        event_ece += mask.sum() / len(group) * gap

    event_rows.append({
        "event": event,
        "buildings": len(group),
        "accuracy": acc.mean(),
        "mean_confidence": conf.mean(),
        "ece": event_ece
    })

event_summary = pd.DataFrame(event_rows)

event_summary.to_csv(
    OUTPUT / "event_calibration.csv",
    index=False
)

print(event_summary.to_string(index=False))

# ------------------------------------------------
# 4. Class-wise confidence
# ------------------------------------------------

print("\n" + "=" * 75)
print("CLASS-WISE PREDICTION CONFIDENCE")
print("=" * 75)

class_rows = []

for i, cls in enumerate(CLASSES):

    mask = y_true == i

    class_rows.append({
        "actual_class": cls,
        "buildings": int(mask.sum()),
        "accuracy": float(correct[mask].mean()),
        "mean_confidence": float(confidence[mask].mean()),
        "mean_true_class_probability": float(probs[mask, i].mean())
    })

class_summary = pd.DataFrame(class_rows)

class_summary.to_csv(
    OUTPUT / "class_calibration.csv",
    index=False
)

print(class_summary.to_string(index=False))

print("\nCalibration analysis completed.")
print("Output:", OUTPUT.resolve())
