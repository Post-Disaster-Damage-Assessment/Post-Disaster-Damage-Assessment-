import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, find_samples, XView2
from model import UNet

NUM_CLASSES = len(CLASS_NAMES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(confusion):
    metrics = {}
    class_ious = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)

        precision_denominator = tp + fp
        recall_denominator = tp + fn
        iou_denominator = tp + fp + fn
        dice_denominator = 2 * tp + fp + fn

        precision = tp / precision_denominator if precision_denominator else None
        recall = tp / recall_denominator if recall_denominator else None
        iou = tp / iou_denominator if iou_denominator else None
        dice = 2 * tp / dice_denominator if dice_denominator else None

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "dice": dice,
            "support": int(confusion[class_id, :].sum()),
        }
        if class_id > 0 and iou is not None:
            class_ious.append(iou)

    total = int(confusion.sum())
    metrics["overall_pixel_accuracy"] = (
        float(np.trace(confusion) / total) if total else None
    )
    metrics["macro_foreground_iou"] = (
        float(np.mean(class_ious)) if class_ious else None
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on labeled, held-out xView2 test data."
    )
    parser.add_argument("--data-root", default="data/test")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output", default="evaluation/test_metrics.json")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    samples = find_samples(args.data_root)
    if not samples:
        raise FileNotFoundError(
            f"No labeled pre/post samples found under '{args.data_root}'. "
            "Expected images/*_pre_disaster.png, matching post images, and "
            "labels/<post-image-name>.json."
        )

    dataset = XView2(samples, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    model = UNet(in_channels=6, num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint '{args.checkpoint}' does not contain "
            "'model_state_dict'; expected a checkpoint produced by train.py."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, (pre, post, mask) in enumerate(loader, start=1):
            inputs = torch.cat([pre, post], dim=1).to(DEVICE)
            targets = mask.to(DEVICE)
            predictions = model(inputs).argmax(dim=1)

            valid = targets != 255
            true_values = targets[valid].cpu().numpy()
            predicted_values = predictions[valid].cpu().numpy()
            if np.any((true_values < 0) | (true_values >= NUM_CLASSES)):
                raise ValueError("Test labels contain an unsupported class index.")

            encoded = true_values * NUM_CLASSES + predicted_values
            confusion += np.bincount(
                encoded, minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)

            if batch_idx % 50 == 0 or batch_idx == len(loader):
                print(f"Evaluated {batch_idx}/{len(loader)} samples")

    if confusion.sum() == 0:
        raise ValueError("Test set contains no evaluable pixels after ignoring class 255.")

    metrics = calculate_metrics(confusion)
    results = {
        "data_root": str(Path(args.data_root)),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": DEVICE,
        "sample_count": len(samples),
        "class_names": CLASS_NAMES,
        "confusion_matrix": confusion.tolist(),
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion)
    print("\nPer-class metrics:")
    for class_name in CLASS_NAMES:
        values = metrics[class_name]
        print(
            f"{class_name:12s} | "
            f"Precision: {values['precision'] if values['precision'] is not None else 'N/A'} | "
            f"Recall: {values['recall'] if values['recall'] is not None else 'N/A'} | "
            f"IoU: {values['iou'] if values['iou'] is not None else 'N/A'} | "
            f"Dice: {values['dice'] if values['dice'] is not None else 'N/A'}"
        )
    print(f"\nOverall Pixel Accuracy: {metrics['overall_pixel_accuracy']:.4f}")
    macro_iou = metrics["macro_foreground_iou"]
    print(
        "Macro Foreground IoU: "
        f"{macro_iou:.4f}" if macro_iou is not None else "Macro Foreground IoU: N/A"
    )
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
