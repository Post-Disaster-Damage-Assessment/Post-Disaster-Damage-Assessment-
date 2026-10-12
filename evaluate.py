import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import find_samples, split_samples, XView2
from model import UNet

DATA_ROOT = "data/train"
BATCH_SIZE = 2
CROP_SIZE = 512
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "background",
    "no-damage",
    "minor",
    "major",
    "destroyed"
]


def main():
    samples = find_samples(DATA_ROOT)

    _, val_samples = split_samples(
        samples,
        val_frac=0.1,
        seed=42
    )

    val_ds = XView2(
        val_samples,
        train=False,
        crop=CROP_SIZE
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = UNet(
        in_channels=6,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        "checkpoints/best_model.pth",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    with torch.no_grad():
        for batch_idx, (pre, post, mask) in enumerate(val_loader):

            pre = pre.to(DEVICE)
            post = post.to(DEVICE)
            mask = mask.to(DEVICE)

            x = torch.cat([pre, post], dim=1)

            output = model(x)
            prediction = output.argmax(dim=1)

            valid = mask != 255

            true_values = mask[valid].cpu().numpy()
            pred_values = prediction[valid].cpu().numpy()

            for true, pred in zip(true_values, pred_values):
                confusion[true, pred] += 1

            if batch_idx % 50 == 0:
                print(
                    f"Evaluated {batch_idx}/{len(val_loader)} batches"
                )

    print("\nConfusion Matrix:")
    print(confusion)

    print("\nPer-class metrics:")

    for c in range(NUM_CLASSES):

        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        dice = (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0
            else 0
        )

        print(
            f"{CLASS_NAMES[c]:12s} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"IoU: {iou:.4f} | "
            f"Dice: {dice:.4f}"
        )

    accuracy = np.trace(confusion) / confusion.sum()

    print(f"\nOverall Pixel Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
