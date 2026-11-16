
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(".")
OUT = ROOT / "final_visualizations"
OUT.mkdir(exist_ok=True)

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
SIZE_GROUPS = ["Smallest", "Small-medium", "Large-medium", "Largest"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT / filename)


def read_confusion(path):
    rows = read_csv(path)
    matrix = []

    for row in rows:
        matrix.append([int(row[c]) for c in CLASSES])

    return np.array(matrix)


def class_f1(matrix):
    results = []

    for i in range(len(CLASSES)):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        results.append(f1)

    return results


def find_calibration_bins():
    folder = ROOT / "validation_results" / "calibration_analysis"

    for path in folder.glob("*.csv"):
        rows = read_csv(path)

        if not rows:
            continue

        columns = set(rows[0].keys())

        if {
            "confidence_range",
            "mean_confidence",
            "accuracy"
        }.issubset(columns):
            return rows

    raise FileNotFoundError(
        "Could not find calibration CSV containing "
        "confidence_range, mean_confidence and accuracy."
    )


def confidence_midpoint(label):
    label = label.replace("%", "").strip()

    if "-" in label:
        low, high = label.split("-", 1)
        return (float(low) + float(high)) / 2

    return float(label)


def main():

    print("\nGenerating project visualizations...\n")

    # ------------------------------------------------
    # 1. Global performance comparison
    # ------------------------------------------------

    global_rows = read_csv(
        ROOT / "comparison_results" /
        "event_comparison" / "global_metrics_comparison.csv"
    )

    metric_names = ["accuracy", "macro_f1", "weighted_f1"]

    original = [
        float(next(r for r in global_rows if r["model"] == "original")[m])
        for m in metric_names
    ]

    augmented = [
        float(next(r for r in global_rows if r["model"] == "photo_aug")[m])
        for m in metric_names
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width / 2, original, width, label="Original")
    ax.bar(x + width / 2, augmented, width, label="Photometric augmentation")

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro F1", "Weighted F1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Overall Model Performance Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, "01_global_performance.png")

    # ------------------------------------------------
    # 2. Confusion matrices
    # ------------------------------------------------

    original_matrix = read_confusion(
        ROOT / "comparison_results" /
        "original" / "confusion_matrix.csv"
    )

    augmented_matrix = read_confusion(
        ROOT / "comparison_results" /
        "photo_aug" / "confusion_matrix.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, matrix, title in [
        (axes[0], original_matrix, "Original Model"),
        (axes[1], augmented_matrix, "Photometric Augmentation")
    ]:
        image = ax.imshow(matrix)

        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CLASSES, rotation=35, ha="right")
        ax.set_yticklabels(CLASSES)

        for i in range(4):
            for j in range(4):
                ax.text(
                    j, i, str(matrix[i, j]),
                    ha="center", va="center"
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrix Comparison", fontsize=15)

    save_figure(fig, "02_confusion_matrices.png")

    # ------------------------------------------------
    # 3. Event-wise macro F1
    # ------------------------------------------------

    event_rows = read_csv(
        ROOT / "comparison_results" /
        "event_comparison" / "event_metrics_comparison.csv"
    )

    event_rows.sort(
        key=lambda r: float(r["original_macro_f1"]),
        reverse=True
    )

    events = [r["event"] for r in event_rows]
    old_f1 = [float(r["original_macro_f1"]) for r in event_rows]
    new_f1 = [float(r["photo_aug_macro_f1"]) for r in event_rows]

    x = np.arange(len(events))

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - width / 2, old_f1, width, label="Original")
    ax.bar(x + width / 2, new_f1, width, label="Photometric augmentation")

    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Event-wise Macro F1 Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, "03_event_wise_macro_f1.png")

    # ------------------------------------------------
    # 4. Class-wise F1
    # ------------------------------------------------

    old_class_f1 = class_f1(original_matrix)
    new_class_f1 = class_f1(augmented_matrix)

    x = np.arange(len(CLASSES))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width / 2, old_class_f1, width, label="Original")
    ax.bar(x + width / 2, new_class_f1, width, label="Photometric augmentation")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_title("Class-wise F1 Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, "04_class_wise_f1.png")

    # ------------------------------------------------
    # 5. Building-size accuracy
    # ------------------------------------------------

    size_rows = read_csv(
        ROOT / "building_size_comparison" /
        "building_size_metrics_comparison.csv"
    )

    size_accuracy = {}

    for row in size_rows:
        size_accuracy[(row["model"], row["size_group"])] = float(
            row["group_accuracy"]
        )

    old_accuracy = [
        size_accuracy[("original", group)]
        for group in SIZE_GROUPS
    ]

    new_accuracy = [
        size_accuracy[("photo_aug", group)]
        for group in SIZE_GROUPS
    ]

    x = np.arange(len(SIZE_GROUPS))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width / 2, old_accuracy, width, label="Original")
    ax.bar(x + width / 2, new_accuracy, width, label="Photometric augmentation")

    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_GROUPS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Across Building-size Groups")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, "05_building_size_accuracy.png")

    # ------------------------------------------------
    # 6. Class-wise F1 by building size
    # ------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for i, class_name in enumerate(CLASSES):
        old_values = []
        new_values = []

        for group in SIZE_GROUPS:
            old_row = next(
                r for r in size_rows
                if r["model"] == "original"
                and r["size_group"] == group
                and r["class"] == class_name
            )

            new_row = next(
                r for r in size_rows
                if r["model"] == "photo_aug"
                and r["size_group"] == group
                and r["class"] == class_name
            )

            old_values.append(float(old_row["f1"]))
            new_values.append(float(new_row["f1"]))

        x = np.arange(len(SIZE_GROUPS))
        ax = axes[i]

        ax.bar(x - width / 2, old_values, width, label="Original")
        ax.bar(x + width / 2, new_values, width, label="Photometric augmentation")

        ax.set_xticks(x)
        ax.set_xticklabels(SIZE_GROUPS, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(class_name)
        ax.set_ylabel("F1 score")
        ax.grid(axis="y", alpha=0.3)

        if i == 0:
            ax.legend()

    fig.suptitle("Class-wise F1 Across Building-size Groups", fontsize=15)

    save_figure(fig, "06_class_f1_by_building_size.png")

    # ------------------------------------------------
    # 7. Confidence calibration
    # ------------------------------------------------

    calibration_rows = find_calibration_bins()

    calibration_rows = [
        r for r in calibration_rows
        if float(r["mean_confidence"]) >= 0
    ]

    calibration_rows.sort(
        key=lambda r: confidence_midpoint(r["confidence_range"])
    )

    mean_confidence = [
        float(r["mean_confidence"])
        for r in calibration_rows
    ]

    accuracy = [
        float(r["accuracy"])
        for r in calibration_rows
    ]

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    ax.plot(
        mean_confidence,
        accuracy,
        marker="o",
        label="Original model"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean prediction confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Confidence Calibration Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, "07_confidence_calibration.png")

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    print("\n" + "=" * 55)
    print("VISUALIZATION GENERATION COMPLETED")
    print("=" * 55)

    for path in sorted(OUT.glob("*.png")):
        print(f"{path.name}: {path.stat().st_size:,} bytes")

    print("\nOutput folder:", OUT.resolve())


if __name__ == "__main__":
    main()
