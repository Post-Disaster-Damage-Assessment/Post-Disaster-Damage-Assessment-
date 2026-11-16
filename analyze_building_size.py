
import csv
import random
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader

from building_dataset import (
    collect_buildings,
    PairedBuildingDataset,
    CLASS_NAMES
)

from siamese_model import SiameseResNet18

from train_siamese import (
    create_scene_split,
    DEVICE,
    BATCH_SIZE
)


DATA_ROOT = "data/train"
CHECKPOINT_PATH = Path("checkpoints_siamese/best_model.pth")
OUTPUT_DIR = Path("building_size_analysis")

OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42


def calculate_size(record):
    min_x, min_y, max_x, max_y = record["bounds"]

    width = max_x - min_x
    height = max_y - min_y

    area = width * height

    aspect_ratio = (
        max(width, height) / max(min(width, height), 1e-6)
    )

    return width, height, area, aspect_ratio


def calculate_metrics(items):

    if not items:
        return None

    actual = np.array([x["actual"] for x in items])
    predicted = np.array([x["predicted"] for x in items])

    accuracy = float(np.mean(actual == predicted))

    f1_scores = []

    for class_id in range(4):

        tp = np.sum((actual == class_id) & (predicted == class_id))
        fp = np.sum((actual != class_id) & (predicted == class_id))
        fn = np.sum((actual == class_id) & (predicted != class_id))

        denominator = 2 * tp + fp + fn

        f1 = (
            2 * tp / denominator
            if denominator > 0 else 0.0
        )

        if np.sum(actual == class_id) > 0:
            f1_scores.append(float(f1))

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "samples": len(items),
        "accuracy": accuracy,
        "macro_f1_present_classes": macro_f1
    }


def main():

    print("DEVICE:", DEVICE)

    print("\nCollecting records...")

    records = collect_buildings(DATA_ROOT)
    _, val_records = create_scene_split(records)

    print("Validation buildings:", len(val_records))

    # Calculate building dimensions.
    for record in val_records:
        width, height, area, ratio = calculate_size(record)

        record["_width"] = width
        record["_height"] = height
        record["_area"] = area
        record["_ratio"] = ratio

    areas = np.array([r["_area"] for r in val_records])

    q1, q2, q3 = np.quantile(areas, [0.25, 0.50, 0.75])

    print("\nBUILDING AREA QUARTILES")
    print("Q1:", round(q1, 2))
    print("Q2:", round(q2, 2))
    print("Q3:", round(q3, 2))

    # Load dataset and original checkpoint.
    dataset = PairedBuildingDataset(val_records)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = SiameseResNet18(
        num_classes=4,
        pretrained=False
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nCheckpoint epoch:", checkpoint["epoch"])
    print("Evaluating predictions...")

    results = []

    with torch.inference_mode():

        for batch_idx, (pre, post, labels) in enumerate(loader):

            pre = pre.to(DEVICE)
            post = post.to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda")
            ):
                outputs = model(pre, post)

            predictions = outputs.argmax(dim=1).cpu().tolist()
            actuals = labels.tolist()

            start = batch_idx * BATCH_SIZE

            for i, (actual, predicted) in enumerate(
                zip(actuals, predictions)
            ):

                record = val_records[start + i]

                results.append({
                    "scene": record["scene"],
                    "uid": record["uid"],
                    "actual": actual,
                    "predicted": predicted,
                    "correct": int(actual == predicted),
                    "width": record["_width"],
                    "height": record["_height"],
                    "bbox_area": record["_area"],
                    "aspect_ratio": record["_ratio"]
                })

            if (batch_idx + 1) % 200 == 0:
                print(
                    f"Processed {batch_idx + 1}/{len(loader)} batches"
                )

    # Assign four area groups using validation-set quartiles.
    for item in results:

        area = item["bbox_area"]

        if area <= q1:
            item["size_group"] = "Smallest"
        elif area <= q2:
            item["size_group"] = "Small-medium"
        elif area <= q3:
            item["size_group"] = "Large-medium"
        else:
            item["size_group"] = "Largest"

        item["shape_group"] = (
            "Elongated"
            if item["aspect_ratio"] > 1.5
            else "Compact"
        )

    print("\nBUILDING SIZE RESULTS")
    print("=" * 75)

    summary_rows = []

    for group in [
        "Smallest",
        "Small-medium",
        "Large-medium",
        "Largest"
    ]:

        subset = [
            x for x in results
            if x["size_group"] == group
        ]

        metrics = calculate_metrics(subset)

        print(
            f"{group:15} | "
            f"Samples: {metrics['samples']:6,} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1_present_classes']:.4f}"
        )

        summary_rows.append({
            "group_type": "size",
            "group": group,
            **metrics
        })

    print("\nBUILDING SHAPE RESULTS")
    print("=" * 75)

    for group in ["Compact", "Elongated"]:

        subset = [
            x for x in results
            if x["shape_group"] == group
        ]

        metrics = calculate_metrics(subset)

        print(
            f"{group:15} | "
            f"Samples: {metrics['samples']:6,} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1_present_classes']:.4f}"
        )

        summary_rows.append({
            "group_type": "shape",
            "group": group,
            **metrics
        })

    # Save individual building predictions.
    with open(
        OUTPUT_DIR / "building_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    # Save group summary.
    with open(
        OUTPUT_DIR / "size_shape_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_type",
                "group",
                "samples",
                "accuracy",
                "macro_f1_present_classes"
            ]
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nAnalysis completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
