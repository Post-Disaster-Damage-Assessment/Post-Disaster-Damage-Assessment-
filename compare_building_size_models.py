
import csv
import json
from pathlib import Path
from collections import defaultdict

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

CHECKPOINTS = {
    "original": Path("checkpoints_siamese/best_model.pth"),
    "photo_aug": Path(
        "checkpoints_photo_aug/best_photo_aug_model.pth"
    )
}

OUTPUT_DIR = Path("building_size_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)


def calculate_size(record):
    min_x, min_y, max_x, max_y = record["bounds"]

    width = max_x - min_x
    height = max_y - min_y
    area = width * height

    return width, height, area


def assign_size_group(area, q1, q2, q3):
    if area <= q1:
        return "Smallest"
    elif area <= q2:
        return "Small-medium"
    elif area <= q3:
        return "Large-medium"
    return "Largest"


def evaluate_checkpoint(checkpoint_path, dataset, val_records):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = SiameseResNet18(
        num_classes=4,
        pretrained=False
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    predictions = []
    cursor = 0

    print("\nCheckpoint:", checkpoint_path)
    print("Epoch:", checkpoint["epoch"])

    with torch.inference_mode():

        for batch_idx, (pre, post, labels) in enumerate(loader):

            pre = pre.to(DEVICE)
            post = post.to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda")
            ):
                outputs = model(pre, post)

            predicted = outputs.argmax(dim=1).cpu().tolist()
            actual = labels.tolist()

            for true_label, predicted_label in zip(actual, predicted):

                record = val_records[cursor]

                predictions.append({
                    "scene": record["scene"],
                    "uid": record["uid"],
                    "actual": true_label,
                    "predicted": predicted_label,
                    "correct": int(true_label == predicted_label),
                    "size_group": record["_size_group"]
                })

                cursor += 1

            if (batch_idx + 1) % 200 == 0:
                print(
                    f"Processed {batch_idx + 1}/{len(loader)} batches"
                )

    if cursor != len(val_records):
        raise RuntimeError(
            f"Prediction count mismatch: {cursor} vs {len(val_records)}"
        )

    return predictions, checkpoint["epoch"]


def calculate_class_metrics(items):

    actual = np.array([x["actual"] for x in items])
    predicted = np.array([x["predicted"] for x in items])

    rows = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        tp = int(np.sum(
            (actual == class_id) & (predicted == class_id)
        ))

        fp = int(np.sum(
            (actual != class_id) & (predicted == class_id)
        ))

        fn = int(np.sum(
            (actual == class_id) & (predicted != class_id)
        ))

        support = int(np.sum(actual == class_id))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        rows.append({
            "class": class_name,
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return rows


def main():

    print("=" * 70)
    print("ORIGINAL VS PHOTOMETRIC AUGMENTATION")
    print("BUILDING-SIZE PERFORMANCE COMPARISON")
    print("=" * 70)

    print("\nCollecting validation records...")

    records = collect_buildings(DATA_ROOT)
    _, val_records = create_scene_split(records)

    print("Validation buildings:", len(val_records))

    areas = []

    for record in val_records:
        width, height, area = calculate_size(record)

        record["_width"] = width
        record["_height"] = height
        record["_area"] = area

        areas.append(area)

    q1, q2, q3 = np.quantile(areas, [0.25, 0.50, 0.75])

    print("\nFIXED BUILDING-SIZE QUARTILES")
    print("Q1:", q1)
    print("Q2:", q2)
    print("Q3:", q3)

    for record in val_records:
        record["_size_group"] = assign_size_group(
            record["_area"], q1, q2, q3
        )

    boundaries = {
        "q1": float(q1),
        "q2": float(q2),
        "q3": float(q3),
        "method": "Validation-set building bounding-box area quartiles"
    }

    with open(
        OUTPUT_DIR / "size_quartile_boundaries.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(boundaries, f, indent=4)

    dataset = PairedBuildingDataset(val_records)

    all_predictions = []
    all_metrics = []

    for model_name, checkpoint_path in CHECKPOINTS.items():

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        predictions, epoch = evaluate_checkpoint(
            checkpoint_path,
            dataset,
            val_records
        )

        for prediction in predictions:
            prediction["model"] = model_name
            prediction["checkpoint_epoch"] = epoch

        all_predictions.extend(predictions)

        print("\n", model_name.upper(), "RESULTS")
        print("-" * 70)

        for group in [
            "Smallest",
            "Small-medium",
            "Large-medium",
            "Largest"
        ]:

            subset = [
                x for x in predictions
                if x["size_group"] == group
            ]

            accuracy = np.mean([
                x["correct"] for x in subset
            ])

            class_metrics = calculate_class_metrics(subset)

            print(
                f"\n{group}: {len(subset)} buildings"
                f" | Accuracy: {accuracy:.4f}"
            )

            for metric in class_metrics:

                print(
                    f"  {metric['class']:15}"
                    f" Support: {metric['support']:5}"
                    f" Precision: {metric['precision']:.4f}"
                    f" Recall: {metric['recall']:.4f}"
                    f" F1: {metric['f1']:.4f}"
                )

                all_metrics.append({
                    "model": model_name,
                    "checkpoint_epoch": epoch,
                    "size_group": group,
                    "group_support": len(subset),
                    "group_accuracy": accuracy,
                    **metric
                })

    prediction_fields = [
        "model",
        "checkpoint_epoch",
        "scene",
        "uid",
        "actual",
        "predicted",
        "correct",
        "size_group"
    ]

    with open(
        OUTPUT_DIR / "building_size_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=prediction_fields
        )
        writer.writeheader()
        writer.writerows(all_predictions)

    metric_fields = list(all_metrics[0].keys())

    with open(
        OUTPUT_DIR / "building_size_metrics_comparison.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=metric_fields
        )
        writer.writeheader()
        writer.writerows(all_metrics)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)

    print("Predictions:", OUTPUT_DIR / "building_size_predictions.csv")
    print("Metrics:", OUTPUT_DIR / "building_size_metrics_comparison.csv")
    print("Boundaries:", OUTPUT_DIR / "size_quartile_boundaries.json")


if __name__ == "__main__":
    main()
