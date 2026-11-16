
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter

from building_dataset import (
    collect_buildings,
    CLASS_NAMES
)
from siamese_model import SiameseResNet18
from train_siamese import (
    create_scene_split,
    TrainingDataset,
    calculate_metrics,
    SEED,
    VAL_RATIO,
    DATA_ROOT,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    USE_AMP
)

CHECKPOINT_PATH = "checkpoints_siamese/best_model.pth"


@torch.inference_mode()
def evaluate():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("DEVICE:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nCollecting building records...")
    records = collect_buildings(DATA_ROOT)

    train_records, val_records = create_scene_split(records)

    print("Total buildings:", len(records))
    print("Validation buildings:", len(val_records))

    val_dataset = TrainingDataset(
        val_records,
        augment=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=NUM_WORKERS > 0
    )

    # Recreate the same class weights used during training.
    train_counts = Counter(r["label"] for r in train_records)

    counts = torch.tensor(
        [train_counts[i] for i in range(4)],
        dtype=torch.float32
    )

    class_weights = torch.sqrt(
        counts.sum() / (4.0 * counts)
    )

    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("Class names do not match checkpoint.")

    model = SiameseResNet18(
        num_classes=4,
        pretrained=False
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint Macro F1:", checkpoint["macro_f1"])

    confusion = torch.zeros(
        4, 4,
        dtype=torch.int64,
        device=DEVICE
    )

    total_loss = 0.0

    print("\nStarting evaluation...")

    for batch_idx, (pre, post, labels) in enumerate(val_loader):
        pre = pre.to(DEVICE, non_blocking=True)
        post = post.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):
            outputs = model(pre, post)
            loss = criterion(outputs, labels)

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        indices = labels * 4 + predictions

        confusion += torch.bincount(
            indices,
            minlength=16
        ).reshape(4, 4)

        if (batch_idx + 1) % 200 == 0:
            print(
                f"Evaluated batches: "
                f"{batch_idx + 1}/{len(val_loader)}"
            )

    confusion = confusion.cpu().numpy()

    metrics, macro_f1 = calculate_metrics(confusion)

    avg_loss = total_loss / len(val_loader)

    supports = confusion.sum(axis=1)
    total = supports.sum()

    weighted_f1 = sum(
        metrics[i][2] * supports[i]
        for i in range(4)
    ) / total

    accuracy = np.trace(confusion) / total

    print("\n" + "=" * 65)
    print("FINAL CHECKPOINT EVALUATION")
    print("=" * 65)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)

    print(
        f"{'Class':15} {'Precision':>10} "
        f"{'Recall':>10} {'F1':>10} {'Support':>10}"
    )

    for i, name in enumerate(CLASS_NAMES):
        precision, recall, f1 = metrics[i]

        print(
            f"{name:15} "
            f"{precision:10.4f} "
            f"{recall:10.4f} "
            f"{f1:10.4f} "
            f"{supports[i]:10d}"
        )

    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print("Class order:", CLASS_NAMES)
    print(confusion)

    np.savetxt(
        "siamese_confusion_matrix.csv",
        confusion,
        delimiter=",",
        fmt="%d"
    )

    print("\nConfusion matrix saved to:")
    print("siamese_confusion_matrix.csv")

    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    evaluate()
