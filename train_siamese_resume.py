import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader

from building_dataset import collect_buildings, PairedBuildingDataset, CLASS_NAMES
from siamese_model import SiameseResNet18


# ---------------- CONFIGURATION ----------------

SEED = 42
VAL_RATIO = 0.20

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
PATIENCE = 3

DATA_ROOT = "data/train"
CHECKPOINT_DIR = Path("checkpoints_siamese")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


# ---------------- REPRODUCIBILITY ----------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------- SCENE-LEVEL SPLIT ----------------

def create_scene_split(records):
    events = defaultdict(lambda: defaultdict(list))

    for record in records:
        scene = record["scene"]
        event = scene.rsplit("_", 1)[0]
        events[event][scene].append(record)

    rng = random.Random(SEED)

    train_records = []
    val_records = []
    train_scenes = set()
    val_scenes = set()

    for event in sorted(events):
        scenes = sorted(events[event])
        rng.shuffle(scenes)

        val_count = max(1, round(len(scenes) * VAL_RATIO))

        val_event_scenes = set(scenes[:val_count])
        train_event_scenes = set(scenes[val_count:])

        for scene in train_event_scenes:
            train_records.extend(events[event][scene])
            train_scenes.add(scene)

        for scene in val_event_scenes:
            val_records.extend(events[event][scene])
            val_scenes.add(scene)

    assert not (train_scenes & val_scenes)
    assert len(train_records) + len(val_records) == len(records)

    return train_records, val_records


# ---------------- SYNCHRONIZED AUGMENTATION ----------------

class TrainingDataset(Dataset):
    def __init__(self, records, augment=False):
        self.dataset = PairedBuildingDataset(records)
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pre, post, label = self.dataset[index]

        if self.augment:
            # Apply identical spatial transformations to both images.
            if random.random() < 0.5:
                pre = torch.flip(pre, dims=[2])
                post = torch.flip(post, dims=[2])

            if random.random() < 0.5:
                pre = torch.flip(pre, dims=[1])
                post = torch.flip(post, dims=[1])

            rotations = random.randint(0, 3)
            if rotations:
                pre = torch.rot90(pre, rotations, dims=[1, 2])
                post = torch.rot90(post, rotations, dims=[1, 2])

        return pre, post, label


# ---------------- METRICS ----------------

def calculate_metrics(confusion):
    results = []

    for i in range(len(CLASS_NAMES)):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results.append((precision, recall, f1))

    macro_f1 = np.mean([item[2] for item in results])
    return results, macro_f1


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


# ---------------- VALIDATION ----------------

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    confusion = torch.zeros(
        4, 4, dtype=torch.int64, device=DEVICE
    )

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


# ---------------- MAIN ----------------

def main():
    print("DEVICE:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nCollecting building records...")
    records = collect_buildings(DATA_ROOT)

    print("Total buildings:", len(records))

    train_records, val_records = create_scene_split(records)

    print("Training buildings:", len(train_records))
    print("Validation buildings:", len(val_records))

    train_counts = Counter(r["label"] for r in train_records)

    print("\nTraining class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:15}: {train_counts[i]:,}")

    # Moderate inverse-frequency class weighting.
    counts = torch.tensor(
        [train_counts[i] for i in range(4)],
        dtype=torch.float32
    )

    class_weights = torch.sqrt(
        counts.sum() / (4.0 * counts)
    )
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(DEVICE)

    print("\nClass weights:", class_weights.cpu().tolist())

    train_dataset = TrainingDataset(train_records, augment=True)
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

    print("\nLoading pretrained Siamese ResNet18...")
    model = SiameseResNet18(
        num_classes=4,
        pretrained=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

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

    # Resume from the best saved checkpoint.
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    print("\nLoading saved checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("Checkpoint class names do not match current classes.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_f1 = checkpoint["macro_f1"]
    epochs_without_improvement = 0

    print("Resuming from saved epoch:", start_epoch)
    print("Previous best Macro F1:", round(best_f1, 4))
    print("Remaining epochs:", EPOCHS - start_epoch)

    print("\nSTARTING SIAMESE TRAINING")
    print("=" * 65)

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        val_loss, confusion, metrics, macro_f1 = validate(
            model, val_loader, criterion
        )

        scheduler.step(macro_f1)

        print(f"\nTrain loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print("Current learning rate:", optimizer.param_groups[0]["lr"])

        print("\nClass metrics:")
        for name, (precision, recall, f1) in zip(CLASS_NAMES, metrics):
            print(
                f"{name:15} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"F1: {f1:.4f}"
            )

        print("\nConfusion matrix:")
        print(confusion)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_f1": best_f1,
                "class_names": CLASS_NAMES,
                "seed": SEED,
                "val_ratio": VAL_RATIO,
            }, CHECKPOINT_DIR / "best_model.pth")

            print("Best checkpoint saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\nTRAINING FINISHED")
    print("Best validation macro F1:", round(best_f1, 4))
    print("Checkpoint:", CHECKPOINT_DIR / "best_model.pth")


if __name__ == "__main__":
    main()
