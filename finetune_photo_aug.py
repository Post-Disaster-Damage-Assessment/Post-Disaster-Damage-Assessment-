
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from building_dataset import (
    collect_buildings, PairedBuildingDataset, CLASS_NAMES, MEAN, STD
)
from train_siamese import (
    create_scene_split, TrainingDataset, calculate_metrics
)
from siamese_model import SiameseResNet18


# ---------------- CONFIGURATION ----------------

SEED = 42
DATA_ROOT = "data/train"
BASE_CHECKPOINT = Path("checkpoints_siamese/best_model.pth")
OUTPUT_DIR = Path("checkpoints_photo_aug")

OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5
PATIENCE = 2
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)


# ---------------- PHOTOMETRIC AUGMENTATION ----------------

def apply_photo_augmentation(pre, post):
    pre = (pre * std + mean).clamp(0, 1)
    post = (post * std + mean).clamp(0, 1)

    brightness = random.uniform(0.90, 1.10)
    contrast = random.uniform(0.90, 1.10)

    pre_mean = pre.mean(dim=(1, 2), keepdim=True)
    post_mean = post.mean(dim=(1, 2), keepdim=True)

    pre = (pre_mean + contrast * (pre - pre_mean)) * brightness
    post = (post_mean + contrast * (post - post_mean)) * brightness

    pre = pre.clamp(0, 1)
    post = post.clamp(0, 1)

    pre = (pre - mean) / std
    post = (post - mean) / std

    return pre, post


class PhotoAugmentedDataset(Dataset):
    def __init__(self, records):
        self.base = TrainingDataset(records, augment=True)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        pre, post, label = self.base[index]
        pre, post = apply_photo_augmentation(pre, post)
        return pre, post, label


# ---------------- EVALUATION ----------------

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    confusion = torch.zeros(4, 4, dtype=torch.int64, device=DEVICE)

    for pre, post, labels in loader:
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
            indices, minlength=16
        ).reshape(4, 4)

    confusion = confusion.cpu().numpy()
    metrics, macro_f1 = calculate_metrics(confusion)

    return total_loss / len(loader), confusion, metrics, macro_f1


# ---------------- TRAINING ----------------

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0

    for batch_idx, (pre, post, labels) in enumerate(loader):
        pre = pre.to(DEVICE, non_blocking=True)
        post = post.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):
            outputs = model(pre, post)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (batch_idx + 1) % 500 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(loader)}"
                f" | Loss: {loss.item():.4f}"
            )

    return total_loss / len(loader)


# ---------------- MAIN ----------------

def main():
    print("=" * 65)
    print("PHOTOMETRIC AUGMENTATION FINE-TUNING EXPERIMENT")
    print("=" * 65)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(BASE_CHECKPOINT)

    print("\nCollecting dataset...")
    records = collect_buildings(DATA_ROOT)

    train_records, val_records = create_scene_split(records)

    print("Training buildings:", len(train_records))
    print("Validation buildings:", len(val_records))

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

    print("Class weights:", class_weights.cpu().tolist())

    train_dataset = PhotoAugmentedDataset(train_records)
    val_dataset = TrainingDataset(val_records, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=NUM_WORKERS > 0
    )

    print("\nLoading original epoch-5 checkpoint...")

    saved = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model = SiameseResNet18(
        num_classes=4,
        pretrained=False
    )

    model.load_state_dict(saved["model_state_dict"])
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\nEvaluating original checkpoint...")
    baseline_loss, baseline_conf, baseline_metrics, baseline_f1 = (
        evaluate(model, val_loader, criterion)
    )

    print(f"Original validation Macro F1: {baseline_f1:.4f}")
    print("Original validation loss:", round(baseline_loss, 4))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        print("\n" + "=" * 65)
        print(f"EXPERIMENT EPOCH {epoch + 1}/{EPOCHS}")
        print("=" * 65)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        val_loss, confusion, metrics, macro_f1 = evaluate(
            model, val_loader, criterion
        )

        scheduler.step(macro_f1)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation Macro F1: {macro_f1:.4f}")
        print("Learning rate:", optimizer.param_groups[0]["lr"])

        for name, (precision, recall, f1) in zip(CLASS_NAMES, metrics):
            print(
                f"{name:15} | Precision: {precision:.4f}"
                f" | Recall: {recall:.4f} | F1: {f1:.4f}"
            )

        print("\nConfusion matrix:")
        print(confusion)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "macro_f1": float(best_f1),
                "baseline_macro_f1": float(baseline_f1),
                "class_names": CLASS_NAMES,
                "seed": SEED,
                "experiment": "shared_brightness_contrast",
                "learning_rate": LEARNING_RATE
            }, OUTPUT_DIR / "best_photo_aug_model.pth")

            print("Experimental checkpoint saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nExperimental early stopping triggered.")
            break

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETED")
    print("=" * 65)

    print(f"Original Macro F1:    {baseline_f1:.4f}")
    print(f"Best experiment F1:   {best_f1:.4f}")
    print(f"Difference:           {best_f1 - baseline_f1:+.4f}")

    if best_f1 > baseline_f1:
        print("Result: Experiment improved validation Macro F1.")
    else:
        print("Result: No validation Macro F1 improvement.")

    print("Experimental checkpoint:", OUTPUT_DIR / "best_photo_aug_model.pth")
    print("Original checkpoint was not modified.")


if __name__ == "__main__":
    main()
