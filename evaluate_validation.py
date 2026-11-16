
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from building_dataset import (
    collect_buildings,
    PairedBuildingDataset,
    CLASS_NAMES,
)
from siamese_model import SiameseResNet18


SEED = 42
VAL_RATIO = 0.20
NUM_CLASSES = 4


def create_scene_split(records):
    """Reproduce the original event-wise, scene-level split."""
    event_scenes = defaultdict(set)

    for record in records:
        scene = record["scene"]
        event = scene.rsplit("_", 1)[0]
        event_scenes[event].add(scene)

    rng = random.Random(SEED)
    train_scenes = set()
    val_scenes = set()

    for event in sorted(event_scenes):
        scenes = sorted(event_scenes[event])
        rng.shuffle(scenes)

        val_count = max(1, round(len(scenes) * VAL_RATIO))

        val_scenes.update(scenes[:val_count])
        train_scenes.update(scenes[val_count:])

    train_records = [
        r for r in records if r["scene"] in train_scenes
    ]
    val_records = [
        r for r in records if r["scene"] in val_scenes
    ]

    return train_records, val_records


def load_model(checkpoint_path, device):
    model = SiameseResNet18(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Checkpoint epoch:", checkpoint.get("epoch", "Unknown"))
    print(
        "Checkpoint Macro F1:",
        checkpoint.get("macro_f1", "Unknown")
    )

    return model


def calculate_metrics(confusion):
    total = confusion.sum()
    correct = np.trace(confusion)

    accuracy = float(correct / total) if total else 0.0

    class_metrics = []
    f1_values = []
    weighted_f1_values = []
    supports = []

    for i, name in enumerate(CLASS_NAMES):
        tp = int(confusion[i, i])
        support = int(confusion[i].sum())
        predicted = int(confusion[:, i].sum())

        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        class_metrics.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

        f1_values.append(f1)
        weighted_f1_values.append(f1 * support)
        supports.append(support)

    macro_f1 = float(np.mean(f1_values))
    weighted_f1 = (
        float(sum(weighted_f1_values) / sum(supports))
        if sum(supports) else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_buildings": int(total),
        "class_metrics": class_metrics,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="data/train"
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints_siamese/best_model.pth"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=0,
        help="Optional quick test. 0 means all validation scenes."
    )
    parser.add_argument(
        "--output-dir",
        default="validation_results"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("SIAMESE MODEL - HELD-OUT VALIDATION")
    print("=" * 65)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nCollecting building records...")
    records = collect_buildings(args.root)
    print("Total valid buildings:", len(records))

    print("\nRecreating original scene split...")
    train_records, val_records = create_scene_split(records)

    train_scenes = {r["scene"] for r in train_records}
    val_scenes = {r["scene"] for r in val_records}

    print("Training scenes:", len(train_scenes))
    print("Validation scenes:", len(val_scenes))
    print("Training buildings:", len(train_records))
    print("Validation buildings:", len(val_records))

    overlap = train_scenes.intersection(val_scenes)

    if overlap:
        raise RuntimeError("Scene leakage detected!")

    if args.limit_scenes > 0:
        selected_scenes = sorted(val_scenes)[:args.limit_scenes]
        val_records = [
            r for r in val_records
            if r["scene"] in selected_scenes
        ]
        print("\nLIMITED RUN")
        print("Selected validation scenes:", len(selected_scenes))
        print("Selected buildings:", len(val_records))

    print("\nLoading checkpoint...")
    model = load_model(args.ckpt, device)

    dataset = PairedBuildingDataset(val_records, output_size=128)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    prediction_rows = []
    processed = 0

    print("\nStarting evaluation...")
    model.eval()

    with torch.inference_mode():
        for batch_index, (pre, post, labels) in enumerate(loader):

            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)

            logits = model(pre, post)
            probabilities = torch.softmax(logits, dim=1)

            predictions = probabilities.argmax(dim=1)

            probs_np = probabilities.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            labels_np = labels.numpy()

            batch_start = batch_index * args.batch_size

            for j in range(len(labels_np)):
                record = val_records[batch_start + j]

                true_label = int(labels_np[j])
                pred_label = int(predictions_np[j])

                confusion[true_label, pred_label] += 1

                row = {
                    "scene": record["scene"],
                    "building_uid": record["uid"],
                    "true_damage": CLASS_NAMES[true_label],
                    "predicted_damage": CLASS_NAMES[pred_label],
                    "correct": pred_label == true_label,
                    "confidence": float(probs_np[j].max()),
                }

                for class_index, class_name in enumerate(CLASS_NAMES):
                    row[f"prob_{class_name}"] = float(
                        probs_np[j, class_index]
                    )

                prediction_rows.append(row)

            processed += len(labels_np)

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Processed {processed}/{len(val_records)} buildings"
                )

    print("Processed:", processed)

    metrics = calculate_metrics(confusion)

    predictions_path = output_dir / "validation_predictions.csv"

    if prediction_rows:
        with open(
            predictions_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(prediction_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(prediction_rows)

    metrics["checkpoint"] = str(args.ckpt)
    metrics["validation_scenes"] = len(
        {r["scene"] for r in val_records}
    )
    metrics["class_names"] = CLASS_NAMES
    metrics["confusion_matrix"] = confusion.tolist()

    metrics_path = output_dir / "validation_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    confusion_path = output_dir / "confusion_matrix.csv"

    with open(
        confusion_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["True / Predicted"] + CLASS_NAMES)

        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + confusion[i].tolist())

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)

    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:  {metrics['weighted_f1']:.4f}")

    print("\nCLASS-WISE METRICS")

    for item in metrics["class_metrics"]:
        print(
            f"{item['class']:15s} "
            f"Precision: {item['precision']:.4f} | "
            f"Recall: {item['recall']:.4f} | "
            f"F1: {item['f1']:.4f} | "
            f"Support: {item['support']}"
        )

    print("\nCONFUSION MATRIX")
    print("Rows = True labels; Columns = Predicted labels")
    print(confusion)

    print("\nSaved files:")
    print(predictions_path.resolve())
    print(metrics_path.resolve())
    print(confusion_path.resolve())

    print("\nVALIDATION COMPLETED")


if __name__ == "__main__":
    main()
